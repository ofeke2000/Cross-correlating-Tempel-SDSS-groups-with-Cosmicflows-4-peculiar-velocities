"""
test_catalog_conversion.py
--------------------------
Unit tests for `dvcorr.pipeline.catalog_conversion`, on tiny synthetic CSVs
written to `tmp_path` -- never the real 11.2 GB catalog.

Covers the three properties the conversion is relied on for: the schema is the
same whatever the input columns, row order survives (which is what makes
`mass_min` a cheap row-group filter), and cache-and-skip keys on provenance
rather than on a filename existing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dvcorr import conventions
from dvcorr.pipeline.catalog_conversion import (
    IS_DISTINCT_COLUMN,
    PARQUET_COLUMNS,
    convert_catalog_to_parquet,
)


def _write_csv(path, n=500, seed=0, extra_columns=False):
    """A miniature halo catalog, mass-sorted like the real one."""
    rng = np.random.default_rng(seed)
    mvir = np.sort(conventions.PARTICLE_MASS * rng.integers(2, 10_000, size=n))
    frame = pd.DataFrame(
        {
            "mvir": mvir,
            "rvir": rng.uniform(20.0, 3000.0, size=n),
            "rs": rng.uniform(1.0, 900.0, size=n),
            "rockstarid": np.arange(n) + 12_000_000_000,
            "pid": np.where(rng.random(n) > 0.12, -1, 12_000_000_001),
            "x": rng.uniform(0.0, conventions.BOX_SIZE, size=n),
            "y": rng.uniform(0.0, conventions.BOX_SIZE, size=n),
            "z": rng.uniform(0.0, conventions.BOX_SIZE, size=n),
            "vx": rng.normal(scale=300.0, size=n),
            "vy": rng.normal(scale=300.0, size=n),
            "vz": rng.normal(scale=300.0, size=n),
        }
    )
    if extra_columns:
        # The older catalog carries four legacy derived columns this project
        # never reads; conversion must drop them without being told to.
        for name in ("delta_5", "delta_3", "near_virgo", "bulkflow_3"):
            frame[name] = rng.normal(size=n)
    frame.to_csv(path, index=False)
    return frame


class TestSchema:
    def test_both_column_layouts_convert_to_one_schema(self, tmp_path) -> None:
        """The 11-column and 15-column CSVs must produce identical schemas --
        that is what lets the loader have a single code path."""
        schemas = []
        for tag, extra in (("lean", False), ("legacy", True)):
            csv_path = tmp_path / f"{tag}.csv"
            _write_csv(csv_path, extra_columns=extra)
            report = convert_catalog_to_parquet(csv_path, tmp_path / f"{tag}.parquet")
            frame = pd.read_parquet(report.parquet_path)
            schemas.append(tuple(frame.columns))
        assert schemas[0] == schemas[1] == PARQUET_COLUMNS

    def test_dtypes_are_float32_and_bool(self, tmp_path) -> None:
        csv_path = tmp_path / "c.csv"
        _write_csv(csv_path)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        frame = pd.read_parquet(report.parquet_path)
        for name in PARQUET_COLUMNS[:-1]:
            assert frame[name].dtype == np.float32
        assert frame[IS_DISTINCT_COLUMN].dtype == np.bool_

    def test_is_distinct_matches_pid(self, tmp_path) -> None:
        csv_path = tmp_path / "c.csv"
        source = _write_csv(csv_path)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        frame = pd.read_parquet(report.parquet_path)
        np.testing.assert_array_equal(
            frame[IS_DISTINCT_COLUMN].to_numpy(), (source["pid"] == -1).to_numpy()
        )


class TestRowOrder:
    def test_row_order_is_preserved(self, tmp_path) -> None:
        """Order is what gives row groups narrow mass ranges, and so what makes
        a `mass_min` filter skip most of the file instead of reading it."""
        csv_path = tmp_path / "c.csv"
        source = _write_csv(csv_path)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        frame = pd.read_parquet(report.parquet_path)
        np.testing.assert_allclose(
            frame["mvir"].to_numpy(), source["mvir"].to_numpy().astype(np.float32)
        )
        assert bool(np.all(np.diff(frame["mvir"].to_numpy()) >= 0))

    def test_chunking_does_not_change_the_result(self, tmp_path) -> None:
        """Chunk size is a memory knob and must not be observable in the output."""
        csv_path = tmp_path / "c.csv"
        _write_csv(csv_path, n=500)
        one = convert_catalog_to_parquet(csv_path, tmp_path / "one.parquet")
        many = convert_catalog_to_parquet(
            csv_path, tmp_path / "many.parquet", chunk_rows=37, row_group_rows=17
        )
        assert one.n_rows == many.n_rows == 500
        pd.testing.assert_frame_equal(
            pd.read_parquet(one.parquet_path), pd.read_parquet(many.parquet_path)
        )


class TestCacheAndSkip:
    def test_second_call_skips(self, tmp_path) -> None:
        csv_path = tmp_path / "c.csv"
        _write_csv(csv_path)
        first = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        second = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        assert first.skipped is False
        assert second.skipped is True
        assert second.n_rows == first.n_rows
        assert second.n_distinct == first.n_distinct

    def test_force_reconverts(self, tmp_path) -> None:
        csv_path = tmp_path / "c.csv"
        _write_csv(csv_path)
        convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        forced = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet", force=True)
        assert forced.skipped is False

    def test_changed_source_is_not_reused(self, tmp_path) -> None:
        """Provenance, not mere existence: a Parquet file left over from a
        different or since-changed CSV must be rebuilt, not silently reused."""
        csv_path = tmp_path / "c.csv"
        _write_csv(csv_path, n=500, seed=1)
        convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        _write_csv(csv_path, n=700, seed=2)  # different size -> different bytes
        again = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        assert again.skipped is False
        assert again.n_rows == 700

    def test_stale_parquet_from_another_catalog_is_not_reused(self, tmp_path) -> None:
        a_csv, b_csv = tmp_path / "a.csv", tmp_path / "b.csv"
        _write_csv(a_csv, n=500, seed=1)
        _write_csv(b_csv, n=500, seed=2)
        shared = tmp_path / "shared.parquet"
        convert_catalog_to_parquet(a_csv, shared)
        # Same row count, different source: only the recorded source NAME
        # distinguishes them.
        assert convert_catalog_to_parquet(b_csv, shared).skipped is False


class TestGuards:
    def test_missing_csv_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_catalog_to_parquet(tmp_path / "absent.csv", tmp_path / "o.parquet")

    def test_report_summarizes_the_population(self, tmp_path) -> None:
        csv_path = tmp_path / "c.csv"
        source = _write_csv(csv_path)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        assert report.n_rows == len(source)
        assert report.n_distinct == int((source["pid"] == -1).sum())
        assert report.n_subhalos == report.n_rows - report.n_distinct
        assert report.n_nonfinite == 0
        assert report.min_particle_count >= 2.0
        assert "rows" in report.summary()
