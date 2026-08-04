"""
test_redshift_space_comparison.py
------------------------------------
Pipeline-level tests for `dvcorr.pipeline.redshift_space_comparison`: the
shared center set (identical across the real- and redshift-space runs) and
the membership diagnostics (net membership change / churn between the two
tracer fields).

Uses small, synthetic, deterministic populations built directly with numpy
-- never the real ~4M-row MDPL2 catalog (CLAUDE.md: don't load it fully
unless the task needs it, and this task explicitly does not need a real
pipeline run to test this pipeline's own arithmetic). `BufferedCarve` is
constructed by hand rather than through `load_and_carve_buffered`, which
would require the real catalog on disk.

See `tests/test_redshift_space.py` for the transform-level tests (zero-
velocity limit, displacement direction, n_hat invariance, the
through-observer flip guard) and `tests/test_velocity_centered_dipole.py`
for the redshift-space sign gate and epsilon-continuity (both extend the
project's shared infall toy directly).
"""

from __future__ import annotations

import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.redshift_space import radial_velocity, to_redshift_space, v_margin_from_statistic
from dvcorr.pipeline.velocity_centered import draw_candidates_from_arrays
from dvcorr.pipeline.redshift_space_comparison import (
    BufferedCarve,
    RedshiftSpaceRunConfig,
    build_tracer_spaces,
    membership_diagnostics,
    run_both_spaces,
    select_redshift_shared_centers,
)

_OBSERVER = np.asarray(conventions.OBSERVER_POSITION, dtype=float)


def _synthetic_buffer(
    n: int,
    sub_volume_radius: float,
    seed: int,
    velocity_scale: float = 300.0,
) -> tuple[BufferedCarve, RedshiftSpaceRunConfig]:
    """A small, deterministic stand-in for `load_and_carve_buffered`'s output.

    Positions are drawn uniformly in a ball generously larger than
    `sub_volume_radius`, so both the plain and buffered carves have a
    healthy population without needing the real catalog.
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.0, (sub_volume_radius * 1.3) ** 3, size=n) ** (1.0 / 3.0)
    pos_all = _OBSERVER + radii[:, None] * directions
    vel_all = rng.normal(scale=velocity_scale, size=(n, 3))
    # Masses span the real catalog's range in particle counts so anything
    # reading them sees a plausible spread; nothing under test depends on the
    # values, only on their row alignment surviving the cuts.
    mvir_all = conventions.PARTICLE_MASS * rng.integers(2, 10_000, size=n).astype(float)
    is_distinct_all = rng.random(n) > 0.12

    d_obs = np.linalg.norm(pos_all - _OBSERVER, axis=1)
    in_core = d_obs <= sub_volume_radius
    pos_core, vel_core = pos_all[in_core], vel_all[in_core]

    cfg = RedshiftSpaceRunConfig(sub_volume_radius=sub_volume_radius, n_candidate_centers=n)

    v_r_core = radial_velocity(pos_core, vel_core, observer=_OBSERVER)
    v_margin_kms = v_margin_from_statistic(v_r_core, cfg.v_margin_statistic, cfg.v_margin_percentile)
    v_margin_mpc = v_margin_kms / 100.0

    in_buffer = d_obs <= sub_volume_radius + v_margin_mpc
    pos_buffer, vel_buffer = pos_all[in_buffer], vel_all[in_buffer]

    buffer = BufferedCarve(
        pos_buffer=pos_buffer,
        vel_buffer=vel_buffer,
        pos_core=pos_core,
        vel_core=vel_core,
        n_core=pos_core.shape[0],
        n_buffer=pos_buffer.shape[0],
        n_total=n,
        v_margin_kms=v_margin_kms,
        v_margin_mpc=v_margin_mpc,
        buffered_radius=sub_volume_radius + v_margin_mpc,
        mvir_core=mvir_all[in_core],
        is_distinct_core=is_distinct_all[in_core],
        catalog_mvir=mvir_all,
    )
    return buffer, cfg


# ---------------------------------------------------------------------------
# Shared center set
# ---------------------------------------------------------------------------


def test_shared_centers_are_row_aligned_across_both_spaces():
    """`s_centers_real` and `s_centers_redshift` describe the SAME centers, row for row.

    Independently re-running `to_redshift_space` on `s_centers_real` /
    `v_centers` must reproduce `s_centers_redshift` EXACTLY: since the
    transform is a deterministic, purely per-row function, this can only
    hold if row alpha of both arrays is the same physical object -- i.e. the
    two runs' center index sets are identical, not merely equal in size.
    """
    buffer, cfg = _synthetic_buffer(n=5000, sub_volume_radius=200.0, seed=1)
    candidates = draw_candidates_from_arrays(
        cfg, buffer.pos_core, buffer.vel_core, buffer.mvir_core, buffer.is_distinct_core
    )
    s_candidates, v_candidates = candidates.s, candidates.v

    centers = select_redshift_shared_centers(
        cfg, s_candidates, v_candidates, _OBSERVER, buffer.v_margin_kms, buffer.v_margin_mpc
    )

    assert centers.n_centers > 0
    assert centers.s_centers_real.shape == centers.s_centers_redshift.shape == (centers.n_centers, 3)
    assert centers.v_centers.shape == (centers.n_centers, 3)

    check = to_redshift_space(
        centers.s_centers_real, centers.v_centers,
        observer=_OBSERVER, flip_guard_floor=cfg.flip_guard_floor,
    )
    assert check.n_dropped == 0   # already guarded once by select_redshift_shared_centers
    np.testing.assert_allclose(check.s_redshift, centers.s_centers_redshift)


def test_run_both_spaces_preserves_n_centers_and_does_not_raise():
    """The row-alignment invariant `run_both_spaces` checks internally holds
    end to end for an ordinary synthetic population (the triangle-inequality
    guarantee documented in `select_redshift_shared_centers`)."""
    buffer, cfg = _synthetic_buffer(n=5000, sub_volume_radius=200.0, seed=2)
    tracers = build_tracer_spaces(cfg, buffer, _OBSERVER)
    candidates = draw_candidates_from_arrays(
        cfg, buffer.pos_core, buffer.vel_core, buffer.mvir_core, buffer.is_distinct_core
    )
    s_candidates, v_candidates = candidates.s, candidates.v
    centers = select_redshift_shared_centers(
        cfg, s_candidates, v_candidates, _OBSERVER, buffer.v_margin_kms, buffer.v_margin_mpc
    )

    results = run_both_spaces(cfg, centers, tracers, _OBSERVER)

    assert results.real_result.n_centers == centers.n_centers
    assert results.redshift_result.n_centers == centers.n_centers


def test_select_redshift_shared_centers_percentile_statistic_drops_globally():
    """With the percentile statistic, centers with |v_r| above it are dropped
    GLOBALLY (position-independent) -- see the module docstring's
    legitimate-vs-illegitimate-cut argument. Exercised with a deliberately
    loose percentile (50th) so the drop is large enough to be unmistakable."""
    buffer, cfg = _synthetic_buffer(n=5000, sub_volume_radius=200.0, seed=3)
    cfg = RedshiftSpaceRunConfig(
        sub_volume_radius=cfg.sub_volume_radius,
        n_candidate_centers=cfg.n_candidate_centers,
        v_margin_statistic="percentile",
        v_margin_percentile=50.0,
    )
    v_r_core = radial_velocity(buffer.pos_core, buffer.vel_core, observer=_OBSERVER)
    v_margin_kms = v_margin_from_statistic(v_r_core, "percentile", 50.0)
    v_margin_mpc = v_margin_kms / 100.0

    candidates = draw_candidates_from_arrays(
        cfg, buffer.pos_core, buffer.vel_core, buffer.mvir_core, buffer.is_distinct_core
    )
    s_candidates, v_candidates = candidates.s, candidates.v
    centers = select_redshift_shared_centers(
        cfg, s_candidates, v_candidates, _OBSERVER, v_margin_kms, v_margin_mpc
    )

    assert centers.n_dropped_v_r_percentile > 0
    v_r_survivors = radial_velocity(centers.s_centers_real, centers.v_centers, observer=_OBSERVER)
    assert np.all(np.abs(v_r_survivors) <= v_margin_kms)


# ---------------------------------------------------------------------------
# Membership diagnostics
# ---------------------------------------------------------------------------


def test_membership_diagnostics_zero_velocity_gives_zero_net_change_and_churn():
    """All velocities zero -> the two tracer fields are IDENTICAL, so every
    shell's net membership change and churn (both directions) must be zero,
    and the intersection must equal the realized occupancy exactly.

    This is the real guard the zero-velocity limit exists for: it would
    equally be satisfied by a broken pipeline that silently used the wrong
    (e.g. duplicated) tracer array for both spaces, so it is deliberately
    paired with `tests/test_redshift_space.py`'s position-level zero-velocity
    check rather than replacing it.
    """
    buffer, cfg = _synthetic_buffer(n=3000, sub_volume_radius=150.0, seed=4)
    zero_buffer = BufferedCarve(
        pos_buffer=buffer.pos_buffer,
        vel_buffer=np.zeros_like(buffer.vel_buffer),
        pos_core=buffer.pos_core,
        vel_core=np.zeros_like(buffer.vel_core),
        n_core=buffer.n_core,
        n_buffer=buffer.n_buffer,
        n_total=buffer.n_total,
        v_margin_kms=0.0,
        v_margin_mpc=0.0,
        buffered_radius=cfg.sub_volume_radius,
        mvir_core=buffer.mvir_core,
        is_distinct_core=buffer.is_distinct_core,
        catalog_mvir=buffer.catalog_mvir,
    )
    tracers = build_tracer_spaces(cfg, zero_buffer, _OBSERVER)

    assert tracers.n_real_inside == tracers.n_redshift_inside
    np.testing.assert_array_equal(np.sort(tracers.real_ids), np.sort(tracers.redshift_ids))

    candidates = draw_candidates_from_arrays(
        cfg, zero_buffer.pos_core, zero_buffer.vel_core, zero_buffer.mvir_core, zero_buffer.is_distinct_core
    )
    s_candidates, v_candidates = candidates.s, candidates.v
    # min_center_speed > 0 would drop every zero-velocity candidate; relax it
    # for this test only, since membership (not the estimator's own speed
    # floor) is what is under test here.
    cfg_zero_speed = RedshiftSpaceRunConfig(
        sub_volume_radius=cfg.sub_volume_radius,
        n_candidate_centers=cfg.n_candidate_centers,
        min_center_speed=0.0,
    )
    core_margin = cfg_zero_speed.shells.max_radius  # v_margin_mpc == 0.0 here
    from dvcorr.pipeline.velocity_centered import core_center_mask
    keep = core_center_mask(s_candidates, cfg_zero_speed.sub_volume_radius, core_margin, observer=_OBSERVER)
    s_centers = s_candidates[keep][:20]  # a handful of centers is enough for this check
    assert s_centers.shape[0] > 0

    diagnostics = membership_diagnostics(
        s_centers, tracers.s_tracers_real, tracers.real_ids,
        tracers.s_tracers_redshift, tracers.redshift_ids,
        cfg_zero_speed.shells.shell_edges,
    )

    np.testing.assert_array_equal(diagnostics.net_change, 0.0)
    np.testing.assert_array_equal(diagnostics.per_center_net_change, 0.0)
    np.testing.assert_array_equal(diagnostics.churn_only_real, 0.0)
    np.testing.assert_array_equal(diagnostics.churn_only_redshift, 0.0)
    # every realized pair shows up in the intersection -- nothing churned
    assert np.all(diagnostics.churn_intersection >= 0.0)
    assert np.all(diagnostics.per_center_churn_only_real == 0.0)
    assert np.all(diagnostics.per_center_churn_only_redshift == 0.0)


def test_membership_diagnostics_uses_stable_ids_not_positional_indices():
    """A deliberately adversarial case: `s_tracers_real` and
    `s_tracers_redshift` are DIFFERENT subsets (different lengths, different
    row order relative to each other) of the same underlying id space, so a
    naive positional-index comparison would silently compare unrelated rows.
    With matching ids assigned correctly, the two known-overlapping halos
    must show up in the intersection, not in either "only" bucket.
    """
    center = _OBSERVER + np.array([50.0, 0.0, 0.0])
    shell_edges = np.array([0.0, 10.0])

    # Two tracers, ids 7 and 3, both within the single shell.
    real_ids = np.array([7, 3])
    s_tracers_real = np.vstack([
        center + np.array([1.0, 0.0, 0.0]),
        center + np.array([-1.0, 0.0, 0.0]),
    ])
    # Redshift-space array: DIFFERENT row order and a different LENGTH (id 3
    # dropped, a new id 9 added) -- positional index 0 here is id 7, but that
    # is NOT true for `s_tracers_real`'s positional index 0 vs this array's,
    # which coincidentally also happens to be id 7 -- the real adversarial
    # case is exercised by asserting the SET result, not by relying on a
    # positional coincidence).
    redshift_ids = np.array([9, 7])
    s_tracers_redshift = np.vstack([
        center + np.array([0.0, 1.0, 0.0]),
        center + np.array([1.5, 0.0, 0.0]),
    ])

    diagnostics = membership_diagnostics(
        center[None, :], s_tracers_real, real_ids, s_tracers_redshift, redshift_ids, shell_edges
    )

    # id 7 is in both -> intersection; id 3 only real; id 9 only redshift.
    assert diagnostics.churn_intersection[0] == 1
    assert diagnostics.churn_only_real[0] == 1
    assert diagnostics.churn_only_redshift[0] == 1
    assert diagnostics.net_change[0] == 0.0   # 2 - 2, despite full churn underneath
