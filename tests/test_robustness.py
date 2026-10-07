"""
test_robustness.py
------------------
Tests for `dvcorr.pipeline.robustness` -- the five "is this bin real?" tests.

The load-bearing property of the whole module is that a subset curve is the
SAME arithmetic as the published curve, so the first test here pins
`restack(stack)` against `normalize_stacked_dipole`'s `zeta_hat` / `sem` /
`monopole_norm` directly. Every other test is downstream of that: if the
re-sum drifts from the pipeline's normalization, a robustness panel would be
auditing a curve that is not the one on the pipeline figure.

The tests use small synthetic populations built with numpy -- the module's
arithmetic is what is under test, not any catalog's physics -- except where a
KNOWN answer is constructed on purpose (the mechanism split, whose whole point
is that it must recover an ordering that was put in by hand).

Backend discipline: `matplotlib.use("Agg")` here, before importing the
pipeline module, exactly as `tests/test_plot_wiring.py` documents.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.config import SPACING_LOG, ShellConfig
from dvcorr.estimators.shell_dipole import velocity_centered_shell_dipole
from dvcorr.pipeline.robustness import (
    RobustnessRunConfig,
    bootstrap_confidence,
    crowding_shell_index,
    crowding_split,
    make_robustness_figure,
    merge_shells,
    per_center_stack,
    precision_ladder,
    property_split,
    restack,
    seed_repeats,
    speed_split,
    trim_ladder,
)
from dvcorr.pipeline.velocity_centered import (
    normalize_result,
    shell_dipole_norm_scale,
)

_OBSERVER = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
_SUB_VOLUME_RADIUS = 60.0
_N_BAR = 1.0e-2  # tracers per (h^-1 Mpc)^3; arbitrary but fixed


def _config(**overrides) -> RobustnessRunConfig:
    """A small, fast robustness config on a shallow shell ladder."""
    defaults = dict(
        sub_volume_radius=_SUB_VOLUME_RADIUS,
        shells=ShellConfig(
            min_radius=2.0,
            max_radius=10.0,
            spacing=SPACING_LOG,
            n_bins=4,
            include_zero_bin=True,
        ),
        n_candidate_centers=200,
        n_subset_draws=4,
        subset_fractions=(0.25, 0.5, 1.0),
        n_bootstrap=64,
        trim_counts=(0, 1, 5),
        n_seed_repeats=2,
    )
    defaults.update(overrides)
    return RobustnessRunConfig(**defaults)


def _population(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform positions in a ball around the observer, plus random velocities."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.0, _SUB_VOLUME_RADIUS**3, size=n) ** (1.0 / 3.0)
    return _OBSERVER + radii[:, None] * directions, rng.normal(scale=300.0, size=(n, 3))


def _stack_and_result(cfg: RobustnessRunConfig, seed: int = 7):
    """Run the estimator on a synthetic population; return (result, stack)."""
    s_centers, v_centers = _population(120, seed)
    s_tracers, _ = _population(3000, seed + 1)
    result = velocity_centered_shell_dipole(
        s_centers=s_centers,
        v_centers=v_centers,
        s_tracers=s_tracers,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=_OBSERVER,
    )
    return result, per_center_stack(result, _N_BAR)


# ---------------------------------------------------------------------------
# The contract everything else rests on
# ---------------------------------------------------------------------------


def test_restack_all_centers_reproduces_the_pipeline_normalization():
    """`restack(stack)` == the published curve, to floating point.

    This is the module's licence to plot subset curves on the same axes as
    the pipeline figure. If it ever fails, every panel in
    `make_robustness_figure` is auditing a different statistic than the one
    the pipeline reports, silently and with no shape change to reveal it.
    """
    cfg = _config()
    result, stack = _stack_and_result(cfg)
    published = normalize_result(
        result,
        _N_BAR,
        cfg.shuffle_seed,
        cfg.gaussian_null_seed,
        cfg.n_null_realizations,
    )
    curve = restack(stack)

    np.testing.assert_allclose(curve.zeta_hat, published.zeta_hat, rtol=1e-12)
    np.testing.assert_allclose(curve.sem, published.sem, rtol=1e-12)
    np.testing.assert_allclose(curve.monopole_norm, published.monopole_norm, rtol=1e-12)
    assert curve.n_centers == result.n_centers


def test_restack_divides_by_the_subset_size_not_the_parent_size():
    """A subset's curve is the mean over the centers it KEPT.

    The failure mode this pins is dividing by the parent `n_centers`, which
    would make every trimmed or split curve smaller in exact proportion to
    how many centers it dropped -- a monotone "trend" manufactured entirely
    by the divisor, and one that looks exactly like a real trim effect.
    """
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    rows = np.arange(stack.n_centers // 2)
    curve = restack(stack, rows)

    expected = (
        stack.per_center_dipole[rows].sum(axis=0) / rows.size
    ) * stack.norm_scale
    np.testing.assert_allclose(curve.zeta_hat, expected, rtol=1e-12)
    assert curve.n_centers == rows.size


def test_restack_rejects_an_empty_subset():
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    with pytest.raises(ValueError, match="zero centers"):
        restack(stack, np.array([], dtype=int))


def test_per_center_stack_caches_the_pipeline_norm_scale():
    """The cached scale is the pipeline's own, not a re-derivation."""
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    np.testing.assert_allclose(
        stack.norm_scale,
        shell_dipole_norm_scale(cfg.shells.shell_edges, _N_BAR),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# merge_shells -- the fine-to-coarse bridge
# ---------------------------------------------------------------------------


def test_merge_shells_identity_boundaries_change_nothing():
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    merged = merge_shells(stack, range(stack.shell_centers.size + 1))

    np.testing.assert_allclose(merged.shell_edges, stack.shell_edges)
    np.testing.assert_allclose(merged.per_center_dipole, stack.per_center_dipole)
    np.testing.assert_allclose(restack(merged).zeta_hat, restack(stack).zeta_hat, rtol=1e-12)


def test_merge_shells_equals_a_coarse_run_on_the_same_centers():
    """A merged bin reproduces what the estimator would have returned had it
    been given the coarse edges directly.

    This is the property the whole helper exists for -- without it a resolved
    ladder cannot be compared to sec.A / sec.B's single wide bin at all -- and
    it is checked against the ESTIMATOR, not against a re-derivation of the
    same arithmetic, so a shared mistake cannot pass it.
    """
    cfg = _config()
    fine_result, fine_stack = _stack_and_result(cfg)
    edges = cfg.shells.shell_edges
    boundaries = [0, 2, edges.size - 1]

    coarse_result = velocity_centered_shell_dipole(
        s_centers=_population(120, 7)[0],
        v_centers=_population(120, 7)[1],
        s_tracers=_population(3000, 8)[0],
        shell_edges=edges[boundaries],
        sub_volume_radius=cfg.sub_volume_radius,
        observer=_OBSERVER,
    )
    merged = merge_shells(fine_stack, boundaries)
    np.testing.assert_allclose(
        restack(merged).zeta_hat,
        restack(per_center_stack(coarse_result, _N_BAR)).zeta_hat,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        restack(merged).monopole_norm,
        restack(per_center_stack(coarse_result, _N_BAR)).monopole_norm,
        rtol=1e-10,
    )


def test_merge_shells_is_not_a_mean_of_the_fine_bins():
    """Pins that merging is volume-weighted, not an average of `zeta_hat`.

    Averaging the fine bins' normalized values is the obvious wrong answer --
    each is divided by its own nbar*V_b -- and it is close enough to look
    plausible, so the difference is asserted rather than assumed.
    """
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    merged = restack(merge_shells(stack, [0, 3])).zeta_hat[0]
    naive = restack(stack).zeta_hat[:3].mean()
    assert not np.isclose(merged, naive, rtol=1e-6)


@pytest.mark.parametrize(
    "boundaries, match",
    [
        ([1], "at least two"),
        ([0, 2, 2], "increasing"),
        ([2, 0], "increasing"),
        ([-1, 3], r"\[0, "),
        ([0, 99], r"\[0, "),
    ],
)
def test_merge_shells_rejects_bad_boundaries(boundaries, match):
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    with pytest.raises(ValueError, match=match):
        merge_shells(stack, boundaries)


# ---------------------------------------------------------------------------
# Test 1 -- precision ladder
# ---------------------------------------------------------------------------


def test_precision_ladder_is_unbiased_across_rungs_and_closes_at_the_full_set():
    """Subset means are unbiased; the full rung has zero across-draw spread.

    Unbiasedness is the ladder's own control: a subset mean of a mean is the
    same mean in expectation, so any TREND in `zeta_hat_mean` with N_c would
    mean the subsets are not exchangeable with the parent -- a selection bug,
    not noise. The full rung's exactly-zero spread is the without-replacement
    contract, pinned so a switch to sampling with replacement cannot slip in.
    """
    cfg = _config(n_subset_draws=32)
    _, stack = _stack_and_result(cfg)
    ladder = precision_ladder(cfg, stack)

    assert ladder.n_centers[-1] == stack.n_centers
    np.testing.assert_allclose(ladder.zeta_hat_spread[-1], 0.0, atol=1e-9)
    np.testing.assert_allclose(
        ladder.zeta_hat_mean[-1], restack(stack).zeta_hat, rtol=1e-12
    )
    # Every rung's mean sits within its own across-draw spread of the full
    # answer -- the unbiasedness statement, stated as a tolerance rather than
    # an equality because a finite number of draws is itself noisy.
    for i in range(ladder.n_centers.size - 1):
        deviation = np.abs(ladder.zeta_hat_mean[i] - ladder.zeta_hat_mean[-1])
        assert np.all(deviation <= ladder.zeta_hat_spread[i] + 1e-9)


def test_precision_ladder_spread_falls_with_n_centers():
    """More centers, less scatter -- the whole reason to raise `n_candidate_centers`."""
    cfg = _config(n_subset_draws=32, subset_fractions=(0.1, 0.5, 1.0))
    _, stack = _stack_and_result(cfg)
    ladder = precision_ladder(cfg, stack)

    # Internal SEM is the monotone quantity (the empirical spread of 32 draws
    # is itself noisy); it must fall as the subsets grow, in every shell.
    assert np.all(np.diff(ladder.sem_mean, axis=0) <= 0.0)


# ---------------------------------------------------------------------------
# Test 2 -- bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_interval_brackets_the_point_estimate():
    cfg = _config(n_bootstrap=400)
    _, stack = _stack_and_result(cfg)
    interval = bootstrap_confidence(cfg, stack)

    assert np.all(interval.low <= interval.zeta_hat)
    assert np.all(interval.zeta_hat <= interval.high)
    assert interval.n_resamples == cfg.n_bootstrap
    np.testing.assert_allclose(interval.zeta_hat, restack(stack).zeta_hat, rtol=1e-12)


def test_bootstrap_std_tracks_the_gaussian_sem_on_a_well_behaved_stack():
    """On many centers with no heavy tail the two error statements agree.

    Stated as a factor-of-two band rather than an equality: they are two
    different estimators of the same width, and the POINT of carrying both is
    that they separate when the per-center contributions are heavy-tailed.
    This test pins the easy regime, so that a separation on real data is
    information about the data rather than about a bug here.
    """
    cfg = _config(n_bootstrap=400)
    _, stack = _stack_and_result(cfg)
    interval = bootstrap_confidence(cfg, stack)

    finite = np.isfinite(interval.sem) & (interval.sem > 0.0)
    ratio = interval.bootstrap_std[finite] / interval.sem[finite]
    assert np.all((ratio > 0.5) & (ratio < 2.0))


# ---------------------------------------------------------------------------
# Test 3 -- trim ladder
# ---------------------------------------------------------------------------


def test_trim_ladder_drops_the_largest_absolute_contributors():
    """Ranking is on |contribution|, both signs -- not on the positive tail.

    Trimming only the positive tail would be a biased estimator wearing a
    robustness check's clothes: it would drag every stack downward by
    construction and make any feature look fragile.
    """
    cfg = _config(trim_counts=(0, 1))
    _, stack = _stack_and_result(cfg)
    trim = trim_ladder(cfg, stack)

    contribution = (
        stack.per_center_dipole[:, cfg.trim_rank_shell]
        * stack.norm_scale[cfg.trim_rank_shell]
    )
    dropped = int(np.argmax(np.abs(contribution)))
    kept = np.array([i for i in range(stack.n_centers) if i != dropped])
    np.testing.assert_allclose(trim.zeta_hat[1], restack(stack, kept).zeta_hat, rtol=1e-12)
    assert trim.n_centers[0] == stack.n_centers
    assert trim.n_centers[1] == stack.n_centers - 1


def test_trim_ladder_baseline_rung_is_the_untrimmed_curve():
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    trim = trim_ladder(cfg, stack)

    assert trim.trim_counts[0] == 0
    np.testing.assert_allclose(trim.zeta_hat[0], restack(stack).zeta_hat, rtol=1e-12)
    np.testing.assert_allclose(
        trim.monopole_norm[0], restack(stack).monopole_norm, rtol=1e-12
    )


def test_trim_ladder_reports_the_contributing_count_on_a_sparse_shell():
    """A shell most centers are empty in has median 0 -- and must say so.

    This is the regime a resolved r -> 0 ladder lives in, and the failure it
    replaced was a ZeroDivisionError in a caller forming mean/median. The
    count is a field so the caller can branch on the measurement instead of
    catching an exception.
    """
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    # Exactly two contributing centers in the ranked shell. Both ends are set
    # explicitly: the synthetic field's innermost shell is already mostly
    # empty, so zeroing the tail alone would leave a column that happens to be
    # all-zero and would pass this test for the wrong reason.
    sparse = stack.per_center_dipole.copy()
    sparse[:, cfg.trim_rank_shell] = 0.0
    sparse[:2, cfg.trim_rank_shell] = [17.0, -42.0]
    trim = trim_ladder(cfg, replace(stack, per_center_dipole=sparse))

    assert trim.n_contributing == 2
    assert trim.median_contribution == 0.0
    assert trim.n_centers[0] == stack.n_centers


def test_trim_ladder_contributing_count_equals_n_centers_when_all_contribute():
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    dense = np.abs(stack.per_center_dipole) + 1.0
    trim = trim_ladder(cfg, replace(stack, per_center_dipole=dense))
    assert trim.n_contributing == stack.n_centers
    assert trim.median_contribution > 0.0


def test_trim_ladder_rejects_an_out_of_range_rank_shell():
    """Clamping would silently rank on the wrong bin; this raises instead."""
    cfg = _config(trim_rank_shell=99)
    _, stack = _stack_and_result(cfg)
    with pytest.raises(IndexError, match="trim_rank_shell"):
        trim_ladder(cfg, stack)


# ---------------------------------------------------------------------------
# Test 5 -- mechanism split
# ---------------------------------------------------------------------------


def test_property_split_shares_sum_to_one_and_means_reproduce_the_stack():
    """The split is a partition: shares add to 1, and the size-weighted means
    add back to the full curve. Both are stated because they fail
    differently -- a dropped group breaks the first, a wrong divisor the
    second."""
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    split = crowding_split(cfg, stack)

    assert split.n_centers.sum() == stack.n_centers
    np.testing.assert_allclose(split.share_of_stack.sum(axis=0), 1.0, rtol=1e-10)
    weighted = (
        split.mean_contribution * split.n_centers[:, None]
    ).sum(axis=0) / stack.n_centers
    np.testing.assert_allclose(weighted, restack(stack).zeta_hat, rtol=1e-10)


def test_property_split_recovers_an_ordering_that_was_put_in_by_hand():
    """The mechanism test must detect a mechanism when one is really there.

    Centers are labelled with a synthetic property and their dipole rows are
    scaled by it, so group means MUST come out ordered. Without this, a
    split that returned a flat result for every input would pass every other
    test in this file while being useless for its actual purpose.
    """
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    values = np.arange(stack.n_centers, dtype=float)
    # A CONSTANT row scaled by the property, not the estimator's own rows:
    # those are mostly zero in the innermost shell of a sparse synthetic field
    # and carry both signs, so an ordering in the property need not survive
    # into them. Here every center contributes, so the group means MUST order
    # -- which is what makes a failure a bug in the split rather than a
    # property of the toy.
    scaled = np.ones_like(stack.per_center_dipole) * (1.0 + values)[:, None]
    stack = replace(stack, per_center_dipole=scaled)

    edges = np.array([0.0, stack.n_centers / 3.0, 2.0 * stack.n_centers / 3.0, np.inf])
    split = property_split(stack, values, edges, label="synthetic")

    inner = split.mean_contribution[:, cfg.trim_rank_shell]
    assert np.all(np.diff(inner) > 0.0)


def test_property_split_rejects_misaligned_values():
    cfg = _config()
    _, stack = _stack_and_result(cfg)
    with pytest.raises(ValueError, match="row-aligned"):
        property_split(
            stack, np.zeros(stack.n_centers + 1), np.array([0.0, np.inf]), label="bad"
        )


def test_crowding_shell_index_does_not_treat_shell_zero_as_unset():
    """The falsy-zero trap, pinned.

    `cfg.crowding_shell or cfg.trim_rank_shell` reads correctly for every
    value except 0 -- the one an inner-bin study is most likely to set. An
    explicit `crowding_shell = 0` must stay 0 even when `trim_rank_shell` is
    something else.
    """
    assert crowding_shell_index(_config(trim_rank_shell=2)) == 2
    assert crowding_shell_index(_config(trim_rank_shell=2, crowding_shell=0)) == 0
    assert crowding_shell_index(_config(trim_rank_shell=0, crowding_shell=3)) == 3


def test_crowding_split_reads_the_shell_it_was_pointed_at():
    """The split groups by occupancy in `crowding_shell`, not `trim_rank_shell`.

    Decoupling the two is what keeps the mechanism test non-circular, so this
    pins that the decoupling actually reaches the grouping variable rather
    than only the label.
    """
    cfg = _config(trim_rank_shell=0, crowding_shell=2)
    _, stack = _stack_and_result(cfg)
    split = crowding_split(cfg, stack)

    expected = property_split(
        stack,
        stack.per_center_count[:, 2],
        np.asarray(cfg.crowding_edges, dtype=float),
        label="whatever",
    )
    np.testing.assert_array_equal(split.n_centers, expected.n_centers)
    np.testing.assert_allclose(split.mean_contribution, expected.mean_contribution)
    # And the label names that shell's radii, not the ranked shell's.
    assert f"{stack.shell_edges[2]:.3g}" in split.label


def test_figure_max_radius_trims_the_view_not_the_measurement():
    """The x-limit is cosmetic: every curve still carries every shell."""
    cfg = _config(n_bootstrap=32, n_subset_draws=3, n_seed_repeats=2,
                  n_candidate_centers=80, figure_max_radius=3.0)
    _, stack = _stack_and_result(cfg)
    s_pool, v_pool = _population(500, 11)
    s_tracers, _ = _population(3000, 12)
    mvir = np.full(s_pool.shape[0], conventions.PARTICLE_MASS)
    is_distinct = np.ones(s_pool.shape[0], dtype=bool)

    interval = bootstrap_confidence(cfg, stack)
    figures = make_robustness_figure(
        cfg, stack, precision_ladder(cfg, stack), interval, trim_ladder(cfg, stack),
        seed_repeats(cfg, s_pool, v_pool, mvir, is_distinct, s_tracers, _OBSERVER, _N_BAR),
        crowding_split(cfg, stack),
    )
    ax_boot = figures.dipole.get_axes()[1]
    assert ax_boot.get_xlim() == (0.0, 3.0)
    # the measurement is untouched -- all B shells still present
    assert interval.zeta_hat.shape == stack.shell_centers.shape
    assert stack.shell_centers.max() > cfg.figure_max_radius


def test_speed_split_group_sizes_follow_the_percentile_edges():
    """Percentile edges fix the group SIZES; only the contributions vary."""
    cfg = _config(speed_percentiles=(0.0, 50.0, 100.0))
    _, stack = _stack_and_result(cfg)
    split = speed_split(cfg, stack)

    assert split.n_centers.sum() == stack.n_centers
    # Two halves, to within the one center a median can put on either side.
    assert abs(split.n_centers[0] - split.n_centers[1]) <= 1


# ---------------------------------------------------------------------------
# Test 4 -- seed repeats
# ---------------------------------------------------------------------------


def test_seed_repeats_uses_distinct_streams_and_holds_the_tracers_fixed():
    """Independent draws, one estimator pass each, same tracer field.

    Distinctness of the seeds is asserted because `cfg.seed + i` was the
    obvious wrong implementation, and adjacent integer seeds are not
    guaranteed independent -- which would make the across-seed spread an
    understatement of exactly the thing it exists to measure.
    """
    cfg = _config(n_seed_repeats=3, n_candidate_centers=80)
    s_pool, v_pool = _population(500, 11)
    s_tracers, _ = _population(3000, 12)
    mvir = np.full(s_pool.shape[0], conventions.PARTICLE_MASS)
    is_distinct = np.ones(s_pool.shape[0], dtype=bool)

    repeats = seed_repeats(
        cfg, s_pool, v_pool, mvir, is_distinct, s_tracers, _OBSERVER, _N_BAR
    )

    assert repeats.seeds.size == cfg.n_seed_repeats
    assert np.unique(repeats.seeds).size == cfg.n_seed_repeats
    assert repeats.zeta_hat.shape == (cfg.n_seed_repeats, cfg.shells.shell_edges.size - 1)
    np.testing.assert_allclose(repeats.mean, repeats.zeta_hat.mean(axis=0), rtol=1e-12)
    assert np.all(repeats.spread >= 0.0)


def test_seed_repeats_is_reproducible_from_the_run_seed():
    cfg = _config(n_seed_repeats=2, n_candidate_centers=80)
    s_pool, v_pool = _population(500, 11)
    s_tracers, _ = _population(3000, 12)
    mvir = np.full(s_pool.shape[0], conventions.PARTICLE_MASS)
    is_distinct = np.ones(s_pool.shape[0], dtype=bool)

    args = (s_pool, v_pool, mvir, is_distinct, s_tracers, _OBSERVER, _N_BAR)
    first = seed_repeats(cfg, *args)
    second = seed_repeats(cfg, *args)
    np.testing.assert_array_equal(first.seeds, second.seeds)
    np.testing.assert_allclose(first.zeta_hat, second.zeta_hat, rtol=1e-12)


# ---------------------------------------------------------------------------
# Config guards and the figure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"subset_fractions": (0.0, 1.0)}, "subset_fraction"),
        ({"subset_fractions": (1.5,)}, "subset_fraction"),
        ({"n_subset_draws": 1}, "n_subset_draws"),
        ({"n_bootstrap": 1}, "n_bootstrap"),
        ({"confidence": 1.0}, "confidence"),
        ({"trim_counts": (-1,)}, "trim_counts"),
        ({"crowding_edges": (10.0, 1.0)}, "crowding_edges"),
        ({"n_seed_repeats": 1}, "n_seed_repeats"),
        ({"crowding_shell": -1}, "crowding_shell"),
        ({"figure_max_radius": 0.0}, "figure_max_radius"),
    ],
)
def test_config_rejects_bad_knobs(overrides, match):
    with pytest.raises(ValueError, match=match):
        _config(**overrides)


def test_robustness_figure_builds_with_cartesian_shell_axes():
    """The figure runs end to end, and every RADIAL axis stays cartesian.

    Log binning never implies log axes in this package
    (`tests/test_plot_wiring.py`). The precision-ladder panel is the
    deliberate exception -- its x-axis is N_c, not r -- so it is checked to
    be log while the shell panels (on `.dipole`) and the monopole figure
    (`.monopole`, CLAUDE.md hard rule 6 -- a separate Figure, not a sixth
    panel) are checked to be linear.
    """
    cfg = _config(n_bootstrap=32, n_subset_draws=3, n_seed_repeats=2, n_candidate_centers=80)
    _, stack = _stack_and_result(cfg)
    s_pool, v_pool = _population(500, 11)
    s_tracers, _ = _population(3000, 12)
    mvir = np.full(s_pool.shape[0], conventions.PARTICLE_MASS)
    is_distinct = np.ones(s_pool.shape[0], dtype=bool)

    figures = make_robustness_figure(
        cfg,
        stack,
        precision_ladder(cfg, stack),
        bootstrap_confidence(cfg, stack),
        trim_ladder(cfg, stack),
        seed_repeats(
            cfg, s_pool, v_pool, mvir, is_distinct, s_tracers, _OBSERVER, _N_BAR
        ),
        crowding_split(cfg, stack),
    )

    assert figures.dipole is not figures.monopole
    axes = figures.dipole.get_axes()
    assert len(axes) == 5
    ax_ladder = axes[0]
    assert ax_ladder.get_xscale() == "log"
    assert ax_ladder.get_yscale() == "log"
    for ax in (axes[1], axes[2], axes[3]):
        assert ax.get_xscale() == "linear"
        assert ax.get_yscale() == "linear"

    ax_mono = figures.monopole.get_axes()[0]
    assert ax_mono.get_xscale() == "linear"
    assert ax_mono.get_yscale() == "linear"


def test_robustness_figure_labels_negative_split_bars_below_the_axis():
    """A group that contributes negatively must keep a readable label.

    The signed split is the informative case near r = 0 -- virialized centers
    cancelling the shell under test -- and a fixed upward annotation offset
    buries exactly those labels under the zero line.
    """
    cfg = _config(n_bootstrap=32, n_subset_draws=3, n_seed_repeats=2, n_candidate_centers=80)
    _, stack = _stack_and_result(cfg)
    # Force a negative group: flip the sign of the most crowded centers' rows.
    crowded = np.argsort(-stack.per_center_count[:, cfg.trim_rank_shell])[:10]
    flipped = stack.per_center_dipole.copy()
    flipped[crowded] = -np.abs(flipped[crowded]) - 1.0
    stack = replace(stack, per_center_dipole=flipped)

    s_pool, v_pool = _population(500, 11)
    s_tracers, _ = _population(3000, 12)
    mvir = np.full(s_pool.shape[0], conventions.PARTICLE_MASS)
    is_distinct = np.ones(s_pool.shape[0], dtype=bool)
    split = crowding_split(cfg, stack)
    figures = make_robustness_figure(
        cfg, stack, precision_ladder(cfg, stack), bootstrap_confidence(cfg, stack),
        trim_ladder(cfg, stack),
        seed_repeats(cfg, s_pool, v_pool, mvir, is_distinct, s_tracers, _OBSERVER, _N_BAR),
        split,
    )
    ax_split = figures.dipole.get_axes()[4]
    values = split.mean_contribution[:, cfg.trim_rank_shell]
    assert values.min() < 0.0, "test setup failed to produce a negative group"
    # Zero is inside the view, and both extremes have room for their labels.
    low, high = ax_split.get_ylim()
    assert low < 0.0 < high
    assert low < values.min() and high > values.max()


# ---------------------------------------------------------------------------
# The binned companion (CLAUDE.md hard rule 10)
# ---------------------------------------------------------------------------


def _figure_args(cfg: RobustnessRunConfig):
    """Every positional argument `make_robustness_figure` needs, from one stack.

    Mirrors the setup already repeated across the figure tests above, so the
    binned-companion tests below can build both a line figure and a binned
    figure from the IDENTICAL inputs -- the load-bearing property the binned
    view is checked against.
    """
    _, stack = _stack_and_result(cfg)
    s_pool, v_pool = _population(500, 11)
    s_tracers, _ = _population(3000, 12)
    mvir = np.full(s_pool.shape[0], conventions.PARTICLE_MASS)
    is_distinct = np.ones(s_pool.shape[0], dtype=bool)
    args = (
        cfg,
        stack,
        precision_ladder(cfg, stack),
        bootstrap_confidence(cfg, stack),
        trim_ladder(cfg, stack),
        seed_repeats(
            cfg, s_pool, v_pool, mvir, is_distinct, s_tracers, _OBSERVER, _N_BAR
        ),
        crowding_split(cfg, stack),
    )
    return stack, args


def test_robustness_figure_binned_is_a_single_axis_with_no_monopole():
    """The binned companion is NOT the 5-panel sheet: none of the robustness
    diagnostics is the main data, so it is one cartesian axis pinned to r = 0
    carrying the full-stack zeta_hat_1 and its Gaussian SEM band only, and no
    monopole figure exists (hard rule 10). The line view keeps its 5 panels.
    """
    cfg = _config(n_bootstrap=32, n_subset_draws=3, n_seed_repeats=2, n_candidate_centers=80)
    _, args = _figure_args(cfg)

    line = make_robustness_figure(*args)
    figures = make_robustness_figure(*args, binned=True)

    assert len(line.dipole.get_axes()) == 5
    assert figures.monopole is None
    (ax,) = figures.dipole.get_axes()
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"
    assert ax.get_xlim()[0] == 0.0
    # Main signal and its Gaussian SEM band only: no bootstrap band, no trim
    # ladder, no seed repeats.
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [r"$\hat\zeta_1$"]
    plt.close("all")


def test_robustness_figure_binned_segments_match_the_line_views_values():
    """The binned companion is the line figure's panel 2 main curve with the
    geometry changed (`dvcorr.pipeline.shell_plotting`), never a second
    computation: its segments equal the line view's full-stack zeta_hat_1
    y-values, shell by shell.
    """
    cfg = _config(n_bootstrap=32, n_subset_draws=3, n_seed_repeats=2, n_candidate_centers=80)
    stack, args = _figure_args(cfg)
    n_shells = stack.shell_centers.size

    lines = make_robustness_figure(*args)
    bins = make_robustness_figure(*args, binned=True)

    line_ax = lines.dipole.get_axes()[1]  # 2: bootstrap interval
    (bin_ax,) = bins.dipole.get_axes()
    (curve,) = [l for l in line_ax.get_lines() if len(l.get_xdata()) == n_shells]
    segment_groups = [
        [seg for seg in coll.get_segments() if np.size(seg)]
        for coll in bin_ax.collections
        if isinstance(coll, LineCollection)
    ]
    assert len(segment_groups) == 1
    (segs,) = segment_groups
    finite = np.isfinite(curve.get_ydata())
    assert len(segs) == int(finite.sum())
    np.testing.assert_allclose(
        [seg[0, 1] for seg in segs], np.asarray(curve.get_ydata())[finite]
    )
    assert bin_ax.get_xlim()[0] == 0.0
    plt.close("all")


def test_robustness_figure_binned_draws_a_guide_at_every_shell_edge():
    """The binned axis marks every shell edge as a faint vertical guide
    (`shell_plotting.finish_shell_axis`); the line view draws none.
    """
    cfg = _config(n_bootstrap=32, n_subset_draws=3, n_seed_repeats=2, n_candidate_centers=80)
    stack, args = _figure_args(cfg)
    n_edges = stack.shell_edges.size

    lines = make_robustness_figure(*args)
    bins = make_robustness_figure(*args, binned=True)

    def _n_edge_guides(ax) -> int:
        return sum(
            1 for line in ax.get_lines()
            if len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) == 0
        )

    assert _n_edge_guides(bins.dipole.get_axes()[0]) == n_edges

    for ax in (lines.dipole.get_axes()[1], lines.dipole.get_axes()[2], lines.dipole.get_axes()[3]):
        assert _n_edge_guides(ax) == 0
    assert _n_edge_guides(lines.monopole.get_axes()[0]) == 0
    plt.close("all")
