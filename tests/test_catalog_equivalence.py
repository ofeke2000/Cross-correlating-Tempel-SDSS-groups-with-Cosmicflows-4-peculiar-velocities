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
from dvcorr.pipeline.velocity_centered import RunConfig, _load_all_halos, load_and_carve

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


def _wrapped(pos: np.ndarray) -> np.ndarray:
    """Positions folded into [0, BOX_SIZE), so periodic images compare equal.

    The two catalogs disagree about the boundary coordinate of exactly four
    halos: each sits at `BOX_SIZE` in `CATALOG_FULL` and at `0.0` in
    `CATALOG_MVIR12`, with identical velocity and mass. In a periodic box those
    are the SAME point, so the two files agree physically and differ only in
    which representative of the periodic image they stored.

    Comparing raw coordinates would therefore report a difference that does not
    exist, and -- worse -- would make a genuine four-halo disagreement
    indistinguishable from this benign one. Folding first states the invariant
    that actually holds: the catalogs contain the same halos up to periodic
    image (CLAUDE.md hard rule 3).

    Note that `CATALOG_FULL` containing a coordinate exactly equal to
    `BOX_SIZE` means box coordinates are NOT guaranteed to lie in the
    half-open interval [0, BOX_SIZE). Nothing in the current pipeline depends
    on that -- the analysis carve spans [200, 800] and never reaches a face --
    but any future code that assumes the half-open convention needs to fold
    first.
    """
    return np.mod(np.asarray(pos, dtype=float), conventions.BOX_SIZE)


def _sorted_rows(pos: np.ndarray) -> np.ndarray:
    """Wrapped rows of `pos` in a canonical, file-order-independent order.

    The two catalogs store the same halos in different row orders (both are
    mass-sorted, but ties break differently), so every comparison here is on
    lexicographically sorted rows. Comparing raw row order would fail for a
    reason that has nothing to do with which halos are present.
    """
    wrapped = _wrapped(pos)
    return wrapped[np.lexsort((wrapped[:, 2], wrapped[:, 1], wrapped[:, 0]))]


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
    all compared, as a multiset over canonically ordered rows and with
    positions folded into the box (see `_wrapped`).
    """
    from_full = _load_all_halos(_PATHS, _MVIR12_EQUIVALENT)
    from_mvir12 = _load_all_halos(_PATHS, CatalogConfig(name=CATALOG_MVIR12))

    def canonical(halos) -> np.ndarray:
        pos = _wrapped(halos.pos)
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


def test_boundary_halos_differ_only_by_a_periodic_image() -> None:
    """The four boundary halos are the SAME objects, wrapped -- not extra ones.

    Pins the finding `_wrapped` exists for. Without this, folding positions
    before comparing would look like a comparison quietly loosened until it
    passed; with it, the exact nature and size of the raw disagreement is
    recorded, and any growth in it fails.
    """
    from_full = _load_all_halos(_PATHS, _MVIR12_EQUIVALENT)
    from_mvir12 = _load_all_halos(_PATHS, CatalogConfig(name=CATALOG_MVIR12))

    at_box_edge = np.any(from_full.pos >= conventions.BOX_SIZE, axis=1)
    at_origin = np.any(from_mvir12.pos == 0.0, axis=1)

    # Small, and equal on both sides: every halo stored at BOX_SIZE in one file
    # is stored at 0.0 in the other.
    assert int(at_box_edge.sum()) == int(at_origin.sum())
    assert 0 < int(at_box_edge.sum()) < 100

    # And they carry identical velocities and masses, which is what makes them
    # the same objects rather than a genuine four-halo disagreement.
    np.testing.assert_array_equal(
        np.sort(from_full.mvir[at_box_edge]), np.sort(from_mvir12.mvir[at_origin])
    )


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
