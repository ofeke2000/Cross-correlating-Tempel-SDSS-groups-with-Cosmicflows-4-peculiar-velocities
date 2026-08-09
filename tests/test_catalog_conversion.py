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
    NUM_OF_SUBHALOS_COLUMN,
    NUM_OF_SUBHALOS_UNKNOWN,
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

    def test_dtypes_are_float32_bool_and_int32(self, tmp_path) -> None:
        csv_path = tmp_path / "c.csv"
        _write_csv(csv_path)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        frame = pd.read_parquet(report.parquet_path)
        # The float columns are named, not sliced off the end of
        # PARQUET_COLUMNS: that slice silently changes meaning every time a
        # non-float column is appended, which is how this test broke when
        # NUM_OF_SUBHALOS_COLUMN landed.
        float_columns = (
            *conventions.POSITION_COLUMNS,
            *conventions.VELOCITY_COLUMNS,
            conventions.HALO_COLUMNS["mass"],
        )
        assert set(float_columns) | {IS_DISTINCT_COLUMN, NUM_OF_SUBHALOS_COLUMN} == set(
            PARQUET_COLUMNS
        )
        for name in float_columns:
            assert frame[name].dtype == np.float32
        assert frame[IS_DISTINCT_COLUMN].dtype == np.bool_
        assert frame[NUM_OF_SUBHALOS_COLUMN].dtype == np.int32

    def test_is_distinct_matches_pid(self, tmp_path) -> None:
        csv_path = tmp_path / "c.csv"
        source = _write_csv(csv_path)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        frame = pd.read_parquet(report.parquet_path)
        np.testing.assert_array_equal(
            frame[IS_DISTINCT_COLUMN].to_numpy(), (source["pid"] == -1).to_numpy()
        )


class TestNumOfSubhalos:
    """The derived daughter count -- the one column that cannot be read off the
    row it describes, and so the one with a way to be subtly wrong."""

    @staticmethod
    def _catalog_with_known_family(path, n=40):
        """A catalog whose parentage is hand-built rather than random.

        Row 0 hosts three daughters, row 1 hosts one, and row 3 -- itself a
        subhalo of row 0 -- hosts a sub-subhalo. That last case is what
        separates "is a parent" from "is distinct".
        """
        ids = np.arange(n) + 12_000_000_000
        pid = np.full(n, -1, dtype=np.int64)
        pid[[3, 4, 5]] = ids[0]
        pid[6] = ids[1]
        pid[7] = ids[3]

        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            {
                "mvir": np.sort(
                    conventions.PARTICLE_MASS * rng.integers(2, 10_000, size=n)
                ).astype(float),
                "rvir": rng.uniform(20.0, 3000.0, size=n),
                "rs": rng.uniform(1.0, 900.0, size=n),
                "rockstarid": ids,
                "pid": pid,
                "x": rng.uniform(0.0, conventions.BOX_SIZE, size=n),
                "y": rng.uniform(0.0, conventions.BOX_SIZE, size=n),
                "z": rng.uniform(0.0, conventions.BOX_SIZE, size=n),
                "vx": rng.normal(scale=300.0, size=n),
                "vy": rng.normal(scale=300.0, size=n),
                "vz": rng.normal(scale=300.0, size=n),
            }
        )
        frame.to_csv(path, index=False)
        return frame, n

    def test_counts_the_daughters_of_every_halo(self, tmp_path) -> None:
        csv_path = tmp_path / "family.csv"
        _, n = self._catalog_with_known_family(csv_path)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "family.parquet")
        frame = pd.read_parquet(report.parquet_path)

        expected = np.zeros(n, dtype=np.int32)
        expected[0] = 3   # three daughters
        expected[1] = 1   # one daughter
        expected[3] = 1   # a subhalo that itself hosts a sub-subhalo
        np.testing.assert_array_equal(
            frame[NUM_OF_SUBHALOS_COLUMN].to_numpy(), expected
        )
        assert report.n_parents == 3
        assert report.num_of_subhalos_derived

    def test_a_subhalo_can_be_a_parent_in_this_column(self, tmp_path) -> None:
        """The column is a pure daughter COUNT and says nothing about `pid`.

        Combining the two into "is a host halo" is
        `dvcorr.pipeline.halo_class_comparison.center_class_mask`'s job, not
        this column's -- keeping them separate here is what lets that module
        define the classes in one place.
        """
        csv_path = tmp_path / "family.csv"
        source, _ = self._catalog_with_known_family(csv_path)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "family.parquet")
        frame = pd.read_parquet(report.parquet_path)

        assert not frame[IS_DISTINCT_COLUMN][3]              # row 3 is a subhalo
        assert frame[NUM_OF_SUBHALOS_COLUMN][3] == 1         # and still has a daughter
        assert (source["pid"][3] != -1)

    def test_a_catalog_without_subhalos_writes_the_unknown_sentinel(self, tmp_path) -> None:
        """Writing 0 there would assert "hosts none", which is not what a
        pre-cut, distinct-only catalog is able to say."""
        csv_path = tmp_path / "distinct_only.csv"
        source = _write_csv(csv_path)
        source["pid"] = -1
        source.to_csv(csv_path, index=False)

        report = convert_catalog_to_parquet(csv_path, tmp_path / "distinct_only.parquet")
        frame = pd.read_parquet(report.parquet_path)

        assert not report.num_of_subhalos_derived
        assert (frame[NUM_OF_SUBHALOS_COLUMN] == NUM_OF_SUBHALOS_UNKNOWN).all()

    def test_a_file_missing_the_column_is_reconverted_not_reused(self, tmp_path) -> None:
        """Cache-and-skip keys on the SCHEMA as well as on provenance.

        An old file matches its CSV perfectly and would otherwise be reused
        forever, leaving the new column permanently absent with no error until
        something asked for it.
        """
        import pyarrow.parquet as pq

        csv_path = tmp_path / "c.csv"
        _write_csv(csv_path)
        parquet_path = tmp_path / "c.parquet"
        report = convert_catalog_to_parquet(csv_path, parquet_path)
        assert not report.skipped
        assert convert_catalog_to_parquet(csv_path, parquet_path).skipped

        # Rewrite it without the new column, preserving the provenance
        # metadata -- exactly what an older conversion left on disk.
        table = pq.read_table(parquet_path)
        stale = table.drop_columns([NUM_OF_SUBHALOS_COLUMN]).replace_schema_metadata(
            table.schema.metadata
        )
        pq.write_table(stale, parquet_path)
        assert NUM_OF_SUBHALOS_COLUMN not in pq.read_schema(parquet_path).names

        rebuilt = convert_catalog_to_parquet(csv_path, parquet_path)
        assert not rebuilt.skipped
        assert NUM_OF_SUBHALOS_COLUMN in pq.read_schema(parquet_path).names


class TestPositionFolding:
    def test_box_face_coordinates_are_folded_to_the_origin(self, tmp_path) -> None:
        """A coordinate at exactly BOX_SIZE is the same point as 0.0 under PBC.

        Rockstar emits a handful of these (196 in the real full catalog), and
        the pre-cut catalog already stored them at 0.0 -- so the two files
        disagreed about shared halos until conversion adopted one convention.
        """
        csv_path = tmp_path / "c.csv"
        source = _write_csv(csv_path, n=50)
        source.loc[0, "x"] = conventions.BOX_SIZE
        source.loc[1, "z"] = conventions.BOX_SIZE
        source.to_csv(csv_path, index=False)

        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        frame = pd.read_parquet(report.parquet_path)

        assert report.n_folded == 2
        assert frame.loc[0, "x"] == 0.0
        assert frame.loc[1, "z"] == 0.0

    def test_every_position_lands_in_the_half_open_box(self, tmp_path) -> None:
        csv_path = tmp_path / "c.csv"
        source = _write_csv(csv_path, n=50)
        source.loc[0, ["x", "y", "z"]] = conventions.BOX_SIZE
        source.to_csv(csv_path, index=False)

        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        frame = pd.read_parquet(report.parquet_path)
        positions = frame[list(conventions.POSITION_COLUMNS)].to_numpy()
        assert positions.min() >= 0.0
        assert positions.max() < conventions.BOX_SIZE

    def test_interior_coordinates_are_untouched(self, tmp_path) -> None:
        """The fold must be a no-op everywhere else -- it is applied to every
        row, so a `mod` that perturbed interior values would silently shift the
        whole catalog rather than fix 196 rows."""
        csv_path = tmp_path / "c.csv"
        source = _write_csv(csv_path, n=200)
        report = convert_catalog_to_parquet(csv_path, tmp_path / "c.parquet")
        frame = pd.read_parquet(report.parquet_path)

        assert report.n_folded == 0
        for name in conventions.POSITION_COLUMNS:
            np.testing.assert_array_equal(
                frame[name].to_numpy(), source[name].to_numpy().astype(np.float32)
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
