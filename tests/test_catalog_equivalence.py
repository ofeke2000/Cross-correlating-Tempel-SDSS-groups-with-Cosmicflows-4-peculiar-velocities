"""
test_catalog_equivalence.py
---------------------------
The superset relation between the two halo catalogs, asserted against the real
files.

`CATALOG_FULL` filtered at `mvir >= MVIR12_CATALOG_FLOOR` with subhalos
excluded must reproduce `CATALOG_MVIR12` EXACTLY. That claim is what makes the
two catalogs interchangeable -- it is why a result measured on the older
catalog can be compared with one measured on the newer, and why the older one
can be returned to at any time. `dvcorr.config.catalog`'s docstring states it;
this file is what keeps it true.

These tests need the converted Parquet files on disk and are SKIPPED when they
are absent, so a checkout without the (gitignored, multi-GB) catalogs still
runs a green suite. Every other test in this project builds synthetic arrays
and never touches a catalog; these are the deliberate exception, because the
property under test is a property OF the files.

Build the inputs once with:

    python -m scripts.convert_mdpl2_catalog
"""

from __future__ import annotations

import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.config import (
    CATALOG_FULL,
    CATALOG_MVIR12,
    CatalogConfig,
    PathsConfig,
)
from dvcorr.config.catalog import MVIR12_CATALOG_FLOOR, MVIR12_CATALOG_ROWS
from dvcorr.pipeline.velocity_centered import (
    RunConfig,
    _load_all_halos,
    draw_candidates,
    load_and_carve,
)

_PATHS = PathsConfig()

#: Both catalogs must be present: a test that silently ran on one of them would
#: assert nothing about the relation BETWEEN them, which is the whole point.
_catalogs_available = all(
    _PATHS.halo_catalog(name).exists() for name in (CATALOG_FULL, CATALOG_MVIR12)
)

pytestmark = pytest.mark.skipif(
    not _catalogs_available,
    reason=(
        "converted catalog Parquet files not present; build them with "
        "`python -m scripts.convert_mdpl2_catalog`"
    ),
)


#: The cut that turns the full catalog into the older one.
_MVIR12_EQUIVALENT = CatalogConfig(
    name=CATALOG_FULL, mass_min=MVIR12_CATALOG_FLOOR, include_subhalos=False
)


def _sorted_rows(pos: np.ndarray) -> np.ndarray:
    """Rows of `pos` in a canonical, file-order-independent order.

    The two catalogs store the same halos in different row orders (both are
    mass-sorted, but ties break differently), so every comparison here is on
    lexicographically sorted rows. Comparing raw row order would fail for a
    reason that has nothing to do with which halos are present.

    No periodic folding is applied, and none is needed: conversion already
    folds every position into [0, BOX_SIZE) (see
    `dvcorr.pipeline.catalog_conversion`), so the two files now agree on which
    representative of a periodic image to store. Folding here as well would
    hide a regression in that, which is precisely what
    `test_box_coordinates_are_half_open` exists to catch.
    """
    return pos[np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))]


def test_full_catalog_filtered_has_the_mvir12_row_count() -> None:
    """The recorded row count is the superset relation in its simplest form."""
    halos = _load_all_halos(_PATHS, _MVIR12_EQUIVALENT)
    assert halos.n_total == MVIR12_CATALOG_ROWS


def test_mvir12_catalog_has_its_recorded_row_count() -> None:
    halos = _load_all_halos(_PATHS, CatalogConfig(name=CATALOG_MVIR12))
    assert halos.n_total == MVIR12_CATALOG_ROWS


def test_mvir12_catalog_is_distinct_halos_above_the_floor() -> None:
    """The cuts baked into the older catalog's file are the ones we ascribe to it.

    If this fails, `MVIR12_CATALOG_FLOOR` no longer describes that file and the
    equivalent cut applied to the full catalog is selecting a different
    population.
    """
    halos = _load_all_halos(_PATHS, CatalogConfig(name=CATALOG_MVIR12))
    assert bool(np.all(halos.is_distinct))
    assert float(halos.mvir.min()) >= MVIR12_CATALOG_FLOOR


def test_the_two_paths_select_the_same_halos() -> None:
    """Same halos, field for field -- not merely the same count.

    A row-count match alone would pass even if the two files disagreed about
    WHICH halos sit above the floor, so positions, velocities and masses are
    all compared, as a multiset over canonically ordered rows.
    """
    from_full = _load_all_halos(_PATHS, _MVIR12_EQUIVALENT)
    from_mvir12 = _load_all_halos(_PATHS, CatalogConfig(name=CATALOG_MVIR12))

    def canonical(halos) -> np.ndarray:
        pos = halos.pos
        record = np.empty(
            halos.n_total,
            dtype=[(name, "f8") for name in ("x", "y", "z", "vx", "vy", "vz", "mvir")],
        )
        for i, name in enumerate(("x", "y", "z")):
            record[name] = pos[:, i]
        for i, name in enumerate(("vx", "vy", "vz")):
            record[name] = halos.vel[:, i]
        record["mvir"] = halos.mvir
        record.sort()
        return record

    np.testing.assert_array_equal(canonical(from_full), canonical(from_mvir12))


def test_box_coordinates_are_half_open() -> None:
    """Every position lies in [0, BOX_SIZE) in BOTH catalogs.

    Rockstar emits a handful of halos with a coordinate at exactly `BOX_SIZE`
    (196 rows in the raw full catalog, none in the pre-cut one) -- the same
    point as `0.0` under PBC, but a different representative, which made the
    two files disagree on shared halos and left the half-open convention
    unsafe to assume. `dvcorr.pipeline.catalog_conversion` now folds at ingest.

    This is the test that keeps that true: if a future catalog is converted by
    some other route, or the fold is removed, downstream code that relies on
    half-open coordinates would otherwise fail silently and far from the cause.
    """
    for catalog in (CatalogConfig(name=CATALOG_MVIR12), CatalogConfig(name=CATALOG_FULL)):
        halos = _load_all_halos(_PATHS, catalog)
        assert float(halos.pos.min()) >= 0.0
        assert float(halos.pos.max()) < conventions.BOX_SIZE


def test_the_two_paths_carve_identical_sub_volumes() -> None:
    """The equivalence survives the spatial carve, which is what a run consumes.

    Asserted separately from the catalog-level comparison because the carve is
    where float32 positions meet a distance threshold: a halo sitting within
    float32 rounding of `sub_volume_radius` could in principle fall one way
    from one file and the other way from the other. It does not -- both files
    store identical float32 values, so the comparison is exact -- and this
    test is what would notice if that stopped being true.
    """
    from_full = load_and_carve(RunConfig(catalog=_MVIR12_EQUIVALENT), _PATHS)
    from_mvir12 = load_and_carve(
        RunConfig(catalog=CatalogConfig(name=CATALOG_MVIR12)), _PATHS
    )

    assert from_full.n_carved == from_mvir12.n_carved
    np.testing.assert_array_equal(
        _sorted_rows(from_full.pos), _sorted_rows(from_mvir12.pos)
    )


def test_the_two_paths_draw_the_same_candidate_centers() -> None:
    """The measurement, not just the population, is catalog-independent.

    This is what `draw_candidates_from_arrays`' canonical ordering exists for.
    Before it, the seeded draw picked ROW NUMBERS, so the same seed against the
    two files' different tie orders selected different halos -- measured at the
    time, 291 of 4000 in common -- and every comparison between the catalogs
    carried ~1 sigma of sampling scatter on top of whatever was being compared.

    A small candidate count keeps this test cheap; the property is independent
    of it.
    """
    n_candidates = 2000
    drawn = []
    for catalog in (CatalogConfig(name=CATALOG_MVIR12), _MVIR12_EQUIVALENT):
        cfg = RunConfig(catalog=catalog, n_candidate_centers=n_candidates)
        candidates = draw_candidates(cfg, load_and_carve(cfg, _PATHS))
        drawn.append(candidates)

    from_mvir12, from_full = drawn
    assert from_mvir12.s.shape == from_full.s.shape == (n_candidates, 3)
    np.testing.assert_array_equal(_sorted_rows(from_mvir12.s), _sorted_rows(from_full.s))
    # Same halos in the same order, not merely the same set: the canonical
    # ordering fixes the sequence too, which is what makes per-row arrays
    # (masses, velocities) line up between the two runs.
    np.testing.assert_array_equal(from_mvir12.s, from_full.s)
    np.testing.assert_array_equal(from_mvir12.mvir, from_full.mvir)


def test_full_catalog_reaches_below_the_resolution_threshold() -> None:
    """The full catalog really does contain the unresolved population.

    Guards the assumption the default `CatalogConfig` (no mass floor) is built
    on: if the file's floor rose above `RESOLVED_PARTICLE_COUNT`, "no mass cut"
    would quietly stop meaning what `dvcorr.config.catalog` says it means.
    """
    halos = _load_all_halos(
        _PATHS,
        CatalogConfig(
            name=CATALOG_FULL,
            mass_max=conventions.RESOLVED_PARTICLE_COUNT * conventions.PARTICLE_MASS,
        ),
    )
    assert halos.n_total > 0
    smallest_particle_count = float(halos.mvir.min()) / conventions.PARTICLE_MASS
    assert smallest_particle_count < conventions.RESOLVED_PARTICLE_COUNT
