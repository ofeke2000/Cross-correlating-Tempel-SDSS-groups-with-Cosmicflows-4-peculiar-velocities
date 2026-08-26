"""
test_dm_snapshot.py
-------------------
Tests for `dvcorr.pipeline.dm_snapshot`: the Gadget-4 particle-snapshot loaders
that let `velocity_frame_comparison` and `redshift_space_comparison` run on a
box that is not MDPL2.

Two groups, split by what they need:

  - The pure-arithmetic ones (`_subsample_mask`, `check_run_geometry`,
    `snapshot_number_density`) run on a synthetic `SnapshotBox` and always
    execute.
  - The rest need `data/DM_simulation.hdf5`, which is gitignored, and are
    skipped when it is absent -- the same pattern
    `tests/test_catalog_equivalence.py` uses for the MDPL2 catalogs.

What is worth pinning here is the set of facts that fail SILENTLY. A wrong box
size, a stale `n_bar`, a subsample that is a stride rather than a draw, or a
`check_run_geometry` call passed `sub_volume_radius` where it should have been
given the buffered radius -- none of these raises. They rescale or corrupt an
amplitude and leave a plausible-looking curve behind.
"""

from __future__ import annotations

import numpy as np
import pytest

from dvcorr.config import PathsConfig, SPACING_LOG, ShellConfig
from dvcorr.pipeline.dm_snapshot import (
    DEFAULT_SUBSAMPLE_SEED,
    SnapshotBox,
    _subsample_mask,
    check_run_geometry,
    load_and_carve_particles,
    load_and_carve_particles_buffered,
    peculiar_velocity_scale,
    read_snapshot_header,
    snapshot_number_density,
    snapshot_observer,
)
from dvcorr.pipeline.redshift_space_comparison import RedshiftSpaceRunConfig
from dvcorr.pipeline.velocity_centered import RunConfig

_PATHS = PathsConfig()

_snapshot_available = _PATHS.dm_snapshot.exists()

_needs_snapshot = pytest.mark.skipif(
    not _snapshot_available,
    reason=f"{_PATHS.dm_snapshot.name} not present in data/",
)


def _synthetic_box(box_size: float = 300.0, n_particles: int = 16777216) -> SnapshotBox:
    """A `SnapshotBox` with no file behind it, for the pure-arithmetic tests."""
    return SnapshotBox(
        box_size=box_size,
        n_particles=n_particles,
        particle_mass=1.37e11,
        redshift=0.0,
        scale_factor=1.0,
        hubble_param=0.678,
        omega_m=0.308,
        omega_lambda=0.692,
        omega_baryon=0.0482,
        softening=0.2,
        initial_conditions="synthetic",
    )


# ---------------------------------------------------------------------------
# Box arithmetic -- no file needed
# ---------------------------------------------------------------------------


def test_max_analysis_radius_is_half_the_box_not_the_frozen_mdpl2_value() -> None:
    """The ceiling must follow THIS box, not `conventions.MAX_ANALYSIS_RADIUS`.

    The failure this guards is the quiet one: MDPL2's frozen 500 is larger than
    anything a 300 h^-1 Mpc box can produce, so inheriting it means running
    with no effective ceiling at all rather than with a wrong one.
    """
    from dvcorr import conventions

    box = _synthetic_box(box_size=300.0)
    assert box.max_analysis_radius == 150.0
    assert box.max_analysis_radius < conventions.MAX_ANALYSIS_RADIUS


def test_observer_is_this_boxs_center_and_lies_inside_it() -> None:
    box = _synthetic_box(box_size=300.0)
    observer = snapshot_observer(box)
    assert observer.shape == (3,)
    assert np.all(observer == 150.0)
    assert np.all(observer < box.box_size)


def test_number_density_uses_the_count_it_is_given_not_the_snapshots() -> None:
    """n_bar must follow the THINNED count, or every amplitude is off by the factor.

    Feeding `box.n_particles` against a subsampled tracer field inflates n_bar
    by `subsample_factor` and suppresses every zeta_hat_1 by that factor, flat
    in r -- no shape change to reveal it. So the function must not be able to
    reach for the snapshot count itself.
    """
    box = _synthetic_box(box_size=300.0, n_particles=16777216)
    thinned = box.n_particles // 3
    assert snapshot_number_density(box.n_particles, box) == pytest.approx(box.number_density)
    assert snapshot_number_density(thinned, box) == pytest.approx(box.number_density / 3, rel=1e-6)


def test_mean_interparticle_spacing_exceeds_the_default_shell_floor() -> None:
    """The resolution floor is real and binds: it sits ABOVE `RunConfig`'s default.

    This is why a run in this box must raise `shells.min_radius` rather than
    inherit 1.0 -- pinned so that a future change to either number surfaces the
    collision instead of quietly measuring an unresolved bin.
    """
    box = _synthetic_box()
    assert box.mean_interparticle_spacing == pytest.approx(300.0 / 256.0, rel=1e-6)
    assert box.mean_interparticle_spacing > RunConfig().shells.min_radius


def test_peculiar_velocity_scale_is_one_at_a_equals_one() -> None:
    assert peculiar_velocity_scale(_synthetic_box()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# check_run_geometry
# ---------------------------------------------------------------------------


def _cfg(sub_volume_radius: float = 130.0, max_radius: float = 40.0) -> RunConfig:
    return RunConfig(
        sub_volume_radius=sub_volume_radius,
        shells=ShellConfig(
            min_radius=2.0,
            max_radius=max_radius,
            spacing=SPACING_LOG,
            n_bins=12,
            include_zero_bin=True,
        ),
    )


def test_check_run_geometry_accepts_a_configuration_that_fits() -> None:
    check_run_geometry(_cfg(), _synthetic_box(), 130.0)


def test_check_run_geometry_rejects_a_carve_past_half_the_box() -> None:
    with pytest.raises(ValueError, match="minimum-image ceiling"):
        check_run_geometry(_cfg(sub_volume_radius=160.0), _synthetic_box(), 160.0)


def test_check_run_geometry_rejects_a_buffered_radius_past_the_ceiling() -> None:
    """The redshift-space failure mode: the PLAIN radius fits and the buffer does not.

    `sub_volume_radius` = 130 is comfortably inside 150, so a check written
    against it passes while the tracer field is actually carved at 130 + 29 =
    159 and wraps the periodic faces. This is exactly why `outer_radius` is an
    argument rather than read off `cfg`.
    """
    cfg, box = _cfg(sub_volume_radius=130.0), _synthetic_box()
    check_run_geometry(cfg, box, cfg.sub_volume_radius)  # the plain radius is fine
    with pytest.raises(ValueError, match="minimum-image ceiling"):
        check_run_geometry(cfg, box, cfg.sub_volume_radius + 29.0)


def test_check_run_geometry_rejects_a_carve_that_leaves_no_core() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        check_run_geometry(_cfg(sub_volume_radius=40.0, max_radius=40.0), _synthetic_box(), 40.0)


# ---------------------------------------------------------------------------
# The subsample
# ---------------------------------------------------------------------------


def test_subsample_factor_one_keeps_everything() -> None:
    assert _subsample_mask(1000, 1, DEFAULT_SUBSAMPLE_SEED).all()


def test_subsample_keeps_the_requested_fraction_and_is_seeded() -> None:
    mask = _subsample_mask(9000, 3, DEFAULT_SUBSAMPLE_SEED)
    assert mask.sum() == 3000
    assert np.array_equal(mask, _subsample_mask(9000, 3, DEFAULT_SUBSAMPLE_SEED))
    assert not np.array_equal(mask, _subsample_mask(9000, 3, DEFAULT_SUBSAMPLE_SEED + 1))


def test_subsample_is_a_draw_not_a_stride() -> None:
    """A stride would be a sub-LATTICE of the Lagrangian grid, not a thinning.

    Gadget writes particles in Peano-Hilbert domain order and `ParticleIDs` is
    the Lagrangian grid index, so `pos[::N]` and an ID stride are both
    spatially structured subsets -- the one subsample that aliases against the
    initial conditions. Only a random mask is a fair Bernoulli thinning, and
    the difference is invisible in the kept COUNT, which is why it is asserted
    on the pattern instead.
    """
    mask = _subsample_mask(9000, 3, DEFAULT_SUBSAMPLE_SEED)
    stride = np.zeros(9000, dtype=bool)
    stride[::3] = True
    assert not np.array_equal(mask, stride)
    # A stride has a perfectly regular gap; a draw does not.
    gaps = np.diff(np.flatnonzero(mask))
    assert gaps.min() < gaps.max()


def test_subsample_rejects_a_factor_below_one() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        _subsample_mask(1000, 0, DEFAULT_SUBSAMPLE_SEED)


# ---------------------------------------------------------------------------
# Against the real snapshot
# ---------------------------------------------------------------------------


@_needs_snapshot
def test_header_matches_the_snapshot_this_module_was_written_for() -> None:
    """The box the loader assumes, read back from the file.

    Not a tautology: these are the numbers every downstream amplitude is
    normalized by, and pinning them means a swapped or re-run snapshot fails
    here rather than silently rescaling a result.
    """
    box = read_snapshot_header(_PATHS.dm_snapshot)
    assert box.box_size == 300.0
    assert box.n_particles == 256**3
    assert box.particle_mass == pytest.approx(1.37e11, rel=1e-6)
    assert box.redshift == pytest.approx(0.0, abs=1e-9)
    assert box.hubble_param == pytest.approx(0.678)
    # The provenance the observer-at-box-center reading depends on.
    assert "2mrs" in box.initial_conditions.lower()


@_needs_snapshot
def test_carve_folds_into_a_half_open_box_and_stays_inside_the_carve() -> None:
    """Positions land in [0, box_size), and every carved particle is really inside.

    Gadget emits a handful of coordinates at exactly `box_size`; the fold sends
    them to 0.0 (the same point under PBC), matching what
    `dvcorr.pipeline.catalog_conversion` does to the MDPL2 catalogs. Without
    it a particle at exactly 300.0 sits outside the half-open box every other
    part of the project assumes.
    """
    box = read_snapshot_header(_PATHS.dm_snapshot)
    cfg = _cfg()
    carved = load_and_carve_particles(cfg, box, _PATHS.dm_snapshot, subsample_factor=20)

    assert carved.pos.min() >= 0.0
    assert carved.pos.max() < box.box_size
    d = np.linalg.norm(carved.pos - snapshot_observer(box), axis=1)
    assert d.max() <= cfg.sub_volume_radius
    assert carved.pos.dtype == np.float64  # the estimators' documented contract
    assert carved.vel.dtype == np.float64


@_needs_snapshot
def test_carved_counts_are_consistent_with_the_thinning() -> None:
    box = read_snapshot_header(_PATHS.dm_snapshot)
    carved = load_and_carve_particles(_cfg(), box, _PATHS.dm_snapshot, subsample_factor=20)

    assert carved.n_snapshot == box.n_particles
    assert carved.n_total == box.n_particles // 20
    assert carved.n_carved == carved.pos.shape[0]
    assert carved.n_carved < carved.n_total


@_needs_snapshot
def test_thinning_leaves_the_number_density_of_the_carve_unchanged() -> None:
    """Bernoulli thinning is unbiased: it changes shot noise, not the field.

    The carved FRACTION is the quantity that must be invariant -- if it moved
    with `subsample_factor`, the draw would be spatially structured and every
    correlation function measured on it would be wrong.
    """
    box = read_snapshot_header(_PATHS.dm_snapshot)
    cfg = _cfg()
    fractions = [
        load_and_carve_particles(cfg, box, _PATHS.dm_snapshot, subsample_factor=f).n_carved
        / (box.n_particles // f)
        for f in (20, 40)
    ]
    assert fractions[0] == pytest.approx(fractions[1], rel=2e-3)


@_needs_snapshot
def test_buffered_carve_fits_the_box_and_contains_the_plain_one() -> None:
    """The buffer is a superset of the core, and still inside the PBC ceiling.

    `v_margin_statistic = "percentile"` rather than the default `"max"`: the
    fastest particle in the carve moves at ~2900 km/s, so a `"max"` margin
    buffers past this box's 150 h^-1 Mpc ceiling. That the percentile choice
    is load-bearing (not cosmetic) is the point of asserting the ceiling here.
    """
    box = read_snapshot_header(_PATHS.dm_snapshot)
    cfg = RedshiftSpaceRunConfig(
        sub_volume_radius=130.0, shells=_cfg().shells, v_margin_statistic="percentile"
    )
    buffer = load_and_carve_particles_buffered(
        cfg, box, _PATHS.dm_snapshot, subsample_factor=20
    )

    assert buffer.buffered_radius > cfg.sub_volume_radius
    assert buffer.buffered_radius <= box.max_analysis_radius
    assert buffer.n_buffer > buffer.n_core
    assert buffer.v_margin_mpc == pytest.approx(buffer.v_margin_kms / 100.0)
    # `BufferedCarve`'s halo fields are placeholders on this path (a particle
    # has no mvir and no subhalo flag) -- well-formed, uniform, and unread.
    assert buffer.mvir_core.shape == (buffer.n_core,)
    assert np.all(buffer.mvir_core == np.float32(box.particle_mass))
    assert buffer.is_distinct_core.all()


@_needs_snapshot
def test_buffered_max_margin_is_refused_by_the_geometry_check() -> None:
    """The `"max"` statistic really does overflow this box, and raises rather than wraps.

    The whole reason the notebook deviates from the pipeline default. If this
    ever stops raising, either the box grew or the second
    `check_run_geometry` call was dropped -- and the silent version of that
    bug is a tracer field wrapped around the periodic faces.
    """
    box = read_snapshot_header(_PATHS.dm_snapshot)
    cfg = RedshiftSpaceRunConfig(
        sub_volume_radius=130.0, shells=_cfg().shells, v_margin_statistic="max"
    )
    with pytest.raises(ValueError, match="minimum-image ceiling"):
        load_and_carve_particles_buffered(cfg, box, _PATHS.dm_snapshot, subsample_factor=20)
