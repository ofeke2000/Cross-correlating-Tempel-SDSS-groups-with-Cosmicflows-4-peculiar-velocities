"""
test_frame_ratios.py
--------------------
Tests for `dvcorr.pipeline.frame_ratios` -- the shell-by-shell ratios between
the 3D (velocity), observer and redshift-space curves.

The module is arithmetic on curves that are already measured, so the tests are
arithmetic too: synthetic `NormalizedDipole`s with hand-chosen values, and
assertions on the numbers that come out.

Two properties carry the module. First, the CHAIN IDENTITY: because all three
ratios are built from the same three curves, ``(3D/obs) * (obs/z)`` must equal
``3D/z`` bin by bin -- if it did not, the figure would be showing three
mutually inconsistent statements about one box. Second, the POLE GUARD: the
redshift-space curve crosses zero in this box, and a shell average that lets
the crossing bin in is reporting the binning, not the physics.

Backend discipline: `matplotlib.use("Agg")` here, before importing the
pipeline module, exactly as `tests/test_plot_wiring.py` documents.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dvcorr.config import volume_weighted_shell_radii
from dvcorr.pipeline.frame_ratios import (
    FrameRatioConfig,
    format_frame_ratio_table,
    frame_curve,
    frame_ratio,
    frame_ratios,
    make_frame_ratio_figure,
)
from dvcorr.pipeline.velocity_centered import NormalizedDipole

EDGES = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
N_BINS = EDGES.shape[0] - 1


def _normalized(zeta_hat, sem=None, monopole=None) -> NormalizedDipole:
    """A NormalizedDipole carrying only the fields this module reads."""
    zeta_hat = np.asarray(zeta_hat, dtype=float)
    zeros = np.zeros_like(zeta_hat)
    return NormalizedDipole(
        zeta_hat=zeta_hat,
        # Tiny by default, so the pole guard only fires where a test asks for it
        # by passing an explicit sem.
        sem=np.full_like(zeta_hat, 0.01) if sem is None else np.asarray(sem, dtype=float),
        zeta_hat_shuffle=zeros,
        null_spread_shuffle=zeros,
        zeta_hat_gaussian=zeros,
        null_spread_gaussian=zeros,
        monopole_norm=(
            np.full_like(zeta_hat, 100.0) if monopole is None
            else np.asarray(monopole, dtype=float)
        ),
    )


def _curve(label, zeta_hat, sem=None, monopole=None, n_centers=1000, edges=EDGES):
    return frame_curve(label, edges, _normalized(zeta_hat, sem, monopole), n_centers)


# ---------------------------------------------------------------------------
# frame_curve
# ---------------------------------------------------------------------------


def test_frame_curve_rejects_a_curve_whose_length_disagrees_with_the_edges():
    """A (B,) curve against (B'+1,) edges is a config mismatch, not a ratio."""
    with pytest.raises(ValueError, match="bins but"):
        frame_curve("obs", EDGES, _normalized([1.0, 2.0]), n_centers=10)


def test_frame_curve_carries_the_edges_and_the_center_count():
    curve = _curve("obs", [10.0, 8.0, 6.0, 4.0], n_centers=5514)
    assert curve.n_centers == 5514
    np.testing.assert_allclose(curve.shell_edges, EDGES)


# ---------------------------------------------------------------------------
# frame_ratio -- the division itself
# ---------------------------------------------------------------------------


def test_frame_ratio_is_elementwise_division():
    num = _curve("3D", [900.0, 600.0, 300.0, 100.0])
    den = _curve("obs", [300.0, 300.0, 100.0, 50.0])
    ratio = frame_ratio(FrameRatioConfig(), num, den)
    np.testing.assert_allclose(ratio.ratio, [3.0, 2.0, 3.0, 2.0])


def test_frame_ratio_divides_the_monopoles_too():
    """Hard rule 6: the dipole never travels without its monopole."""
    num = _curve("3D", [900.0] * N_BINS, monopole=[200.0] * N_BINS)
    den = _curve("obs", [300.0] * N_BINS, monopole=[100.0] * N_BINS)
    ratio = frame_ratio(FrameRatioConfig(), num, den)
    np.testing.assert_allclose(ratio.monopole_ratio, 2.0)
    assert ratio.mean_monopole_ratio == pytest.approx(2.0)


def test_frame_ratio_plots_against_the_volume_weighted_radii():
    """Same abscissa as every other figure in the package, not the midpoint."""
    ratio = frame_ratio(
        FrameRatioConfig(), _curve("3D", [1.0] * N_BINS), _curve("obs", [1.0] * N_BINS)
    )
    np.testing.assert_allclose(ratio.shell_centers, volume_weighted_shell_radii(EDGES))


def test_frame_ratio_rejects_curves_measured_on_different_shells():
    """Two ShellConfigs would divide unrelated radii and look perfectly smooth."""
    other = np.array([0.0, 3.0, 6.0, 12.0, 24.0])
    num = _curve("3D", [1.0] * N_BINS)
    den = _curve("obs", [1.0] * N_BINS, edges=other)
    with pytest.raises(ValueError, match="different shell edges"):
        frame_ratio(FrameRatioConfig(), num, den)


def test_frame_ratio_labels_itself_numerator_over_denominator():
    ratio = frame_ratio(
        FrameRatioConfig(), _curve("3D", [1.0] * N_BINS), _curve("z", [1.0] * N_BINS)
    )
    assert ratio.label == "3D/z"
    assert (ratio.numerator_label, ratio.denominator_label) == ("3D", "z")


# ---------------------------------------------------------------------------
# The pole guard
# ---------------------------------------------------------------------------


def test_a_shell_where_the_denominator_is_consistent_with_zero_is_not_averaged():
    """The redshift curve crosses zero; that bin is a pole, not a measurement."""
    num = _curve("obs", [40.0, 40.0, 40.0, 40.0])
    # Third shell: |zeta_hat| = 0.5 against sem = 1.0, far below 3 sigma.
    den = _curve("z", [20.0, 20.0, 0.5, 20.0], sem=[1.0, 1.0, 1.0, 1.0])
    ratio = frame_ratio(FrameRatioConfig(min_denominator_sigma=3.0), num, den)

    assert ratio.usable.tolist() == [True, True, False, True]
    assert ratio.ratio[2] == pytest.approx(80.0)     # the pole is still reported
    assert ratio.mean_ratio == pytest.approx(2.0)    # ... and still excluded
    assert ratio.n_used == 3


def test_the_unguarded_mean_is_kept_so_the_guard_can_be_checked_not_trusted():
    num = _curve("obs", [40.0, 40.0, 40.0, 40.0])
    den = _curve("z", [20.0, 20.0, 0.5, 20.0], sem=[1.0] * N_BINS)
    ratio = frame_ratio(FrameRatioConfig(min_denominator_sigma=3.0), num, den)

    assert ratio.mean_ratio == pytest.approx(2.0)
    assert ratio.naive_mean_ratio == pytest.approx((2.0 + 2.0 + 80.0 + 2.0) / 4.0)
    assert ratio.naive_mean_ratio > 10 * ratio.mean_ratio


def test_a_zero_sigma_threshold_disables_the_guard():
    """min_denominator_sigma = 0 must reproduce the unguarded mean exactly."""
    num = _curve("obs", [40.0, 40.0, 40.0, 40.0])
    den = _curve("z", [20.0, 20.0, 0.5, 20.0], sem=[1.0] * N_BINS)
    ratio = frame_ratio(FrameRatioConfig(min_denominator_sigma=0.0), num, den)
    assert ratio.mean_ratio == pytest.approx(ratio.naive_mean_ratio)
    assert ratio.n_used == N_BINS


def test_an_exactly_zero_denominator_is_excluded_rather_than_producing_inf():
    num = _curve("obs", [40.0] * N_BINS)
    den = _curve("z", [20.0, 0.0, 20.0, 20.0])
    ratio = frame_ratio(FrameRatioConfig(), num, den)
    assert not ratio.usable[1]
    assert ratio.mean_ratio == pytest.approx(2.0)
    assert np.isfinite(ratio.mean_ratio)


def test_mean_is_nan_when_no_shell_survives_the_guard():
    """An empty average is NaN, never a silent 0.0."""
    num = _curve("obs", [40.0] * N_BINS)
    den = _curve("z", [0.1] * N_BINS, sem=[1.0] * N_BINS)
    ratio = frame_ratio(FrameRatioConfig(min_denominator_sigma=3.0), num, den)
    assert ratio.n_used == 0
    assert np.isnan(ratio.mean_ratio)
    assert np.isnan(ratio.mean_monopole_ratio)


def test_resolved_min_radius_drops_the_zero_bin_from_the_mean_only():
    """The unresolved inner bin is plotted; it just does not set the number."""
    num = _curve("3D", [1000.0, 40.0, 40.0, 40.0])
    den = _curve("obs", [10.0, 20.0, 20.0, 20.0])

    kept = frame_ratio(FrameRatioConfig(), num, den)
    dropped = frame_ratio(FrameRatioConfig(resolved_min_radius=2.0), num, den)

    assert kept.usable[0] and not dropped.usable[0]
    assert kept.mean_ratio == pytest.approx((100.0 + 2.0 + 2.0 + 2.0) / 4.0)
    assert dropped.mean_ratio == pytest.approx(2.0)
    # The pole/unresolved shell is still present in the returned curve.
    assert dropped.ratio[0] == pytest.approx(100.0)
    assert dropped.n_shells == N_BINS


def test_resolved_min_radius_keeps_a_shell_that_straddles_the_threshold():
    """The cut is on the OUTER edge: [0, 2) is out, [2, 5) is in at 2.0."""
    ratio = frame_ratio(
        FrameRatioConfig(resolved_min_radius=2.0),
        _curve("3D", [1.0] * N_BINS),
        _curve("obs", [1.0] * N_BINS),
    )
    assert ratio.usable.tolist() == [False, True, True, True]


# ---------------------------------------------------------------------------
# frame_ratios -- the set of pairs, and the identity that ties them together
# ---------------------------------------------------------------------------


def test_frame_ratios_yields_the_three_ordered_pairs_in_information_order():
    curves = (_curve("3D", [9.0] * N_BINS), _curve("obs", [3.0] * N_BINS),
              _curve("z", [1.0] * N_BINS))
    labels = [r.label for r in frame_ratios(FrameRatioConfig(), curves)]
    assert labels == ["3D/obs", "3D/z", "obs/z"]


def test_the_three_ratios_satisfy_the_chain_identity():
    """(3D/obs) * (obs/z) == 3D/z, bin by bin -- the figure cannot self-contradict."""
    rng = np.random.default_rng(0)
    curves = (
        _curve("3D", rng.normal(500.0, 100.0, N_BINS)),
        _curve("obs", rng.normal(200.0, 40.0, N_BINS)),
        _curve("z", rng.normal(-80.0, 20.0, N_BINS)),
    )
    r_3d_obs, r_3d_z, r_obs_z = frame_ratios(FrameRatioConfig(), curves)
    np.testing.assert_allclose(r_3d_obs.ratio * r_obs_z.ratio, r_3d_z.ratio)


def test_frame_ratios_rejects_duplicate_labels():
    curves = (_curve("obs", [1.0] * N_BINS), _curve("obs", [1.0] * N_BINS))
    with pytest.raises(ValueError, match="distinct"):
        frame_ratios(FrameRatioConfig(), curves)


def test_frame_ratios_needs_at_least_two_curves():
    with pytest.raises(ValueError, match="at least two"):
        frame_ratios(FrameRatioConfig(), (_curve("obs", [1.0] * N_BINS),))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_denominator_sigma": -1.0}, "min_denominator_sigma"),
        ({"resolved_min_radius": -2.0}, "resolved_min_radius"),
    ],
)
def test_config_rejects_negative_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FrameRatioConfig(**kwargs)


def test_config_accepts_none_for_the_resolution_floor():
    assert FrameRatioConfig(resolved_min_radius=None).resolved_min_radius is None


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_table_reports_the_mean_the_used_count_and_the_unguarded_mean():
    num = _curve("obs", [40.0] * N_BINS)
    den = _curve("z", [20.0, 20.0, 0.5, 20.0], sem=[1.0] * N_BINS)
    text = format_frame_ratio_table((frame_ratio(FrameRatioConfig(), num, den),))
    assert "obs/z" in text
    assert "mean = 2.000" in text
    assert "3/4 shells" in text
    assert "unguarded mean 21.500" in text
    assert "NO (denominator ~ 0 or unresolved)" in text


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def _figure_curves():
    return (
        _curve("3D", [900.0, 600.0, 300.0, 100.0], n_centers=5514),
        _curve("obs", [300.0, 250.0, 120.0, 40.0], n_centers=5514),
        _curve("z", [-180.0, -100.0, 0.5, 12.0], sem=[1.0] * N_BINS, n_centers=3194),
    )


def test_the_figure_uses_a_linear_radial_axis_despite_logarithmic_binning():
    """House style, pinned repo-wide: log BINNING never implies a log AXIS."""
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    figures = make_frame_ratio_figure(cfg, curves, frame_ratios(cfg, curves))
    ax_ratio, ax_mono = figures.dipole.axes[0], figures.monopole.axes[0]
    assert ax_ratio.get_xscale() == "linear"
    assert ax_mono.get_xscale() == "linear"
    assert ax_ratio.get_yscale() == "linear"
    assert ax_mono.get_yscale() == "linear"
    plt.close(figures.dipole)
    plt.close(figures.monopole)


def test_the_figure_prints_each_mean_in_the_legend():
    """The number the plot exists to produce has to be readable off the plot."""
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    figures = make_frame_ratio_figure(cfg, curves, ratios)
    labels = [t.get_text() for t in figures.dipole.axes[0].get_legend().get_texts()]

    assert len(labels) == len(ratios)
    for ratio, label in zip(ratios, labels):
        assert ratio.label in label
        assert f"{ratio.mean_ratio:.2f}" in label
    plt.close(figures.dipole)
    plt.close(figures.monopole)


def test_a_pole_is_drawn_but_does_not_set_the_y_limits():
    """One bin next to a zero crossing must not flatten every real feature."""
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    figures = make_frame_ratio_figure(cfg, curves, ratios)

    pole = max(abs(r.ratio[2]) for r in ratios)   # obs/z at the crossing: ~240
    lo, hi = figures.dipole.axes[0].get_ylim()
    assert pole > 10 * max(abs(lo), abs(hi))

    usable = np.concatenate([r.ratio[r.usable] for r in ratios])
    assert lo < usable.min() and usable.max() < hi
    plt.close(figures.dipole)
    plt.close(figures.monopole)


def test_the_figure_draws_unusable_shells_with_open_markers():
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    figures = make_frame_ratio_figure(cfg, curves, ratios)

    fills = {line.get_fillstyle() for line in figures.dipole.axes[0].get_lines()
             if line.get_linestyle() == "None" and len(line.get_xdata()) > 0}
    assert "none" in fills and "full" in fills
    plt.close(figures.dipole)
    plt.close(figures.monopole)


def test_the_figure_and_its_monopole_companion_are_two_separate_figures():
    """CLAUDE.md hard rule 6: dipole and monopole are two Figures, not two
    panels of one -- a dipole PNG must be sendable on its own.
    """
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    figures = make_frame_ratio_figure(cfg, curves, frame_ratios(cfg, curves))
    assert figures.dipole is not figures.monopole
    ax_ratio, ax_mono = figures.dipole.axes[0], figures.monopole.axes[0]
    assert r"\hat\zeta_1" in ax_ratio.get_ylabel()
    assert r"\hat\zeta_0" in ax_mono.get_ylabel()
    plt.close(figures.dipole)
    plt.close(figures.monopole)


# ---------------------------------------------------------------------------
# The sign-change split -- a mean across a zero crossing is a cancellation
# ---------------------------------------------------------------------------


def test_a_ratio_that_changes_sign_is_flagged_and_split_into_branches():
    """obs/z runs negative inside the RSD crossing and positive outside it."""
    num = _curve("obs", [40.0, 40.0, 40.0, 40.0])
    den = _curve("z", [-20.0, -20.0, 20.0, 20.0])
    ratio = frame_ratio(FrameRatioConfig(), num, den)

    assert ratio.changes_sign
    assert ratio.mean_where_negative == pytest.approx(-2.0)
    assert ratio.mean_where_positive == pytest.approx(2.0)
    # The plain mean cancels to zero and says nothing -- which is the point.
    assert ratio.mean_ratio == pytest.approx(0.0)


def test_a_single_signed_ratio_puts_the_whole_mean_on_one_branch():
    num = _curve("3D", [40.0] * N_BINS)
    den = _curve("obs", [20.0] * N_BINS)
    ratio = frame_ratio(FrameRatioConfig(), num, den)

    assert not ratio.changes_sign
    assert ratio.mean_where_positive == pytest.approx(ratio.mean_ratio)
    assert np.isnan(ratio.mean_where_negative)


def test_the_branch_split_ignores_shells_the_pole_guard_excluded():
    """A pole must not create a spurious branch out of an excluded shell."""
    num = _curve("obs", [40.0, 40.0, 40.0, 40.0])
    den = _curve("z", [20.0, 20.0, -0.5, 20.0], sem=[1.0] * N_BINS)
    ratio = frame_ratio(FrameRatioConfig(min_denominator_sigma=3.0), num, den)

    assert not ratio.changes_sign          # the only negative shell is the pole
    assert np.isnan(ratio.mean_where_negative)
    assert ratio.mean_where_positive == pytest.approx(2.0)


def test_the_legend_shows_both_branches_when_a_ratio_changes_sign():
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    figures = make_frame_ratio_figure(cfg, curves, ratios)
    labels = [t.get_text() for t in figures.dipole.axes[0].get_legend().get_texts()]

    for ratio, label in zip(ratios, labels):
        if ratio.changes_sign:
            assert "across the sign change" in label
            assert f"{ratio.mean_where_negative:.2f}" in label
            assert f"{ratio.mean_where_positive:.2f}" in label
        else:
            assert "sign change" not in label
    assert any(r.changes_sign for r in ratios)   # the fixture must exercise it
    plt.close(figures.dipole)
    plt.close(figures.monopole)


def test_the_table_names_the_cancellation_when_a_ratio_changes_sign():
    num = _curve("obs", [40.0, 40.0, 40.0, 40.0])
    den = _curve("z", [-20.0, -20.0, 20.0, 20.0])
    text = format_frame_ratio_table((frame_ratio(FrameRatioConfig(), num, den),))
    assert "SIGN CHANGE" in text
    assert "-2.000 (negative)" in text
    assert "2.000 (positive)" in text


def _connecting_lines(ax, n_points: int) -> list:
    """The solid point-to-point `Line2D` per ratio, in `ratios` order --
    never a `StepPatch` (CLAUDE.md hard rule 10: lines, not steps)."""
    assert not ax.patches, "a stairs rendering is the regression this pins"
    return [line for line in ax.get_lines()
            if line.get_linestyle() == "-" and len(line.get_xdata()) == n_points]


def test_the_connecting_line_breaks_at_a_pole_instead_of_spiking_through_it():
    """A segment drawn to an off-scale point is a stripe that reads as structure."""
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    figures = make_frame_ratio_figure(cfg, curves, ratios)

    joined = _connecting_lines(figures.dipole.axes[0], N_BINS)
    assert len(joined) == len(ratios), "expected one connecting line per ratio"
    for ratio, line in zip(ratios, joined):
        ydata = np.asarray(line.get_ydata(), dtype=float)
        np.testing.assert_array_equal(np.isnan(ydata), ~ratio.usable)
    plt.close(figures.dipole)
    plt.close(figures.monopole)


# ---------------------------------------------------------------------------
# Per-ratio abscissa (each FrameRatio carries its own, never a borrowed one)
# ---------------------------------------------------------------------------


def test_the_figure_draws_each_ratio_on_its_own_radii_not_the_first_curves():
    """A ratio not involving `curves[0]` must be drawn on ITS OWN `shell_centers`.

    `frame_ratio` only checks a match WITHIN one numerator/denominator pair,
    never across different ratios, so two `FrameRatio`s handed to the figure
    together may legitimately sit on different ladders (e.g. a redshift-space
    curve built from a different `ShellConfig`). The figure must read each
    ratio's abscissa off the ratio itself, never off `curves[0]`.
    """
    edges_a = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
    edges_b = np.array([0.0, 3.0, 6.0, 12.0, 24.0])
    num_a = _curve("3D", [10.0, 8.0, 6.0, 4.0], edges=edges_a)
    den_a = _curve("obs", [5.0, 4.0, 3.0, 2.0], edges=edges_a)
    num_b = _curve("obs2", [5.0, 4.0, 3.0, 2.0], edges=edges_b)
    den_b = _curve("z", [1.0, 1.0, 1.0, 1.0], edges=edges_b)
    ratio_a = frame_ratio(FrameRatioConfig(), num_a, den_a)
    ratio_b = frame_ratio(FrameRatioConfig(), num_b, den_b)

    figures = make_frame_ratio_figure(
        FrameRatioConfig(), (num_a, den_a, num_b, den_b), (ratio_a, ratio_b)
    )

    lines = _connecting_lines(figures.dipole.axes[0], edges_a.size - 1)
    assert len(lines) == 2
    np.testing.assert_allclose(lines[0].get_xdata(), volume_weighted_shell_radii(edges_a))
    np.testing.assert_allclose(lines[1].get_xdata(), volume_weighted_shell_radii(edges_b))
    plt.close(figures.dipole)
    plt.close(figures.monopole)


def test_the_figure_builds_with_no_ratios_at_all():
    """Empty `curves` / `ratios` must not crash on a `curves[0]` index.

    The pre-`DipoleFigures` version of this figure never indexed into
    `curves` directly (only into `ratios`, inside the loop), so an empty
    pair was never a special case; `edges = curves[0].shell_edges` at the
    top of the function was a regression against that.
    """
    figures = make_frame_ratio_figure(FrameRatioConfig(), (), ())
    assert figures.dipole is not figures.monopole
    # Only the unity reference line (an unlabeled axhline) is drawn.
    assert all(line.get_label().startswith("_") for line in figures.dipole.axes[0].get_lines())
    plt.close(figures.dipole)
    plt.close(figures.monopole)


# ---------------------------------------------------------------------------
# Binned view (CLAUDE.md hard rule 10's binned companion)
# ---------------------------------------------------------------------------


def test_binned_figure_builds_with_the_same_figure_and_axes_shape_as_the_line_view():
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    line = make_frame_ratio_figure(cfg, curves, ratios, binned=False)
    binned = make_frame_ratio_figure(cfg, curves, ratios, binned=True)

    # The binned companion is dipole-only (hard rule 10).
    assert binned.monopole is None
    assert len(binned.dipole.axes) == len(line.dipole.axes) == 1
    assert len(line.monopole.axes) == 1
    plt.close(line.dipole)
    plt.close(line.monopole)
    plt.close(binned.dipole)


def test_binned_figure_segments_span_each_shells_edges_at_the_line_views_value():
    """Each shell's binned segment must be a flat horizontal line at exactly
    the value the line view plots for that shell, spanning that shell's own
    `[r_lo, r_hi)` -- the binned view is the line view's data with the
    geometry changed, never a second computation (`shell_plotting`'s module
    docstring).

    The ratio panel draws each ratio through TWO `draw_shell_series` calls
    (usable shells masked to NaN off the pole shells, and vice versa -- the
    binned counterpart of the line view's filled-vs-open marker split), so
    each ratio now owns two `hlines` collections rather than one; combined,
    every shell must still be carried by exactly one of the two at its exact
    edges and value.
    """
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    figures = make_frame_ratio_figure(cfg, curves, ratios, binned=True)
    ax = figures.dipole.axes[0]

    segments = [
        c for c in ax.collections
        if getattr(c, "get_segments", None) is not None and len(c.get_segments()) == N_BINS
    ]
    assert len(segments) == 2 * len(ratios), "expected two hlines collections per ratio"

    for i, ratio in enumerate(ratios):
        usable_segs = segments[2 * i].get_segments()
        pole_segs = segments[2 * i + 1].get_segments()
        for shell in range(N_BINS):
            seg = usable_segs[shell] if len(usable_segs[shell]) else pole_segs[shell]
            assert len(seg) == 2, f"shell {shell} of {ratio.label!r} drawn by neither call"
            (x0, y0), (x1, y1) = seg
            assert x0 == pytest.approx(ratio.shell_edges[shell])
            assert x1 == pytest.approx(ratio.shell_edges[shell + 1])
            assert y0 == pytest.approx(ratio.ratio[shell])
            assert y1 == pytest.approx(ratio.ratio[shell])
    plt.close(figures.dipole)


def test_binned_figure_still_distinguishes_usable_from_pole_markers():
    """The binned companion keeps the same filled-vs-open marker split the
    line view uses for the pole guard (module docstring: "each shell is its
    own segment, so the pole break is automatic", but the usable/unusable
    distinction on the markers is preserved regardless).

    The binned view draws this split through `draw_shell_series`'s
    `markerfacecolor` (open = ``"none"``) rather than the line view's
    `fillstyle`, since that is the knob the shared primitive exposes.
    """
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    figures = make_frame_ratio_figure(cfg, curves, ratios, binned=True)

    facecolors = {
        line.get_markerfacecolor() for line in figures.dipole.axes[0].get_lines()
        if line.get_linestyle() == "None" and len(line.get_xdata()) > 0
    }
    assert "none" in facecolors
    assert any(fc != "none" for fc in facecolors)
    assert any(not r.usable.all() for r in ratios), "fixture must exercise a pole"
    plt.close(figures.dipole)


def test_binned_figure_x_axis_starts_at_zero():
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    figures = make_frame_ratio_figure(cfg, curves, ratios, binned=True)

    assert figures.dipole.axes[0].get_xlim()[0] == pytest.approx(0.0)
    plt.close(figures.dipole)


def test_binned_figure_draws_shell_edge_guides():
    """The binned view marks every shell edge as a faint vertical guide
    (`shell_plotting.finish_shell_axis`); the line view carries only its
    own dotted unity reference line (also styled ":"), none of these guides.
    """
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    line = make_frame_ratio_figure(cfg, curves, ratios, binned=False)
    binned = make_frame_ratio_figure(cfg, curves, ratios, binned=True)

    def _dotted_lines(fig):
        return [ln for ln in fig.axes[0].get_lines() if ln.get_linestyle() == ":"]

    n_line = len(_dotted_lines(line.dipole))
    n_binned = len(_dotted_lines(binned.dipole))
    # The line view has exactly the one dotted unity-reference axhline; the
    # binned view has that SAME line plus one axvline guide per shell edge,
    # per ratio (edges.size = N_BINS + 1).
    assert n_line == 1
    assert n_binned == n_line + len(ratios) * (N_BINS + 1)
    plt.close(line.dipole)
    plt.close(line.monopole)
    plt.close(binned.dipole)


def test_binned_figure_with_no_ratios_does_not_crash():
    figures = make_frame_ratio_figure(FrameRatioConfig(), (), (), binned=True)
    assert figures.monopole is None
    plt.close(figures.dipole)


def test_line_view_output_is_unchanged_by_the_binned_parameter_existing():
    """`binned=False` (the default) must reproduce the pre-existing line-view
    figure exactly: same connecting-line data, same legend text, same y-limits.
    """
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    explicit = make_frame_ratio_figure(cfg, curves, ratios, binned=False)
    default = make_frame_ratio_figure(cfg, curves, ratios)

    joined_explicit = _connecting_lines(explicit.dipole.axes[0], N_BINS)
    joined_default = _connecting_lines(default.dipole.axes[0], N_BINS)
    assert len(joined_explicit) == len(joined_default) == len(ratios)
    for a, b in zip(joined_explicit, joined_default):
        np.testing.assert_array_equal(a.get_ydata(), b.get_ydata())
    assert explicit.dipole.axes[0].get_ylim() == default.dipole.axes[0].get_ylim()
    plt.close(explicit.dipole)
    plt.close(explicit.monopole)
    plt.close(default.dipole)
    plt.close(default.monopole)


def test_markers_sit_at_the_volume_weighted_radius_not_the_bins_lower_edge():
    """One abscissa per panel (CLAUDE.md hard rule 10): the usable/pole
    markers must sit on the connecting line's own abscissa,
    `shell_centers`'s volume-weighted radius -- not at the bin's lower edge,
    where the step rendering this replaces put them.
    """
    num = _curve("obs", [40.0, 40.0, 40.0, 40.0])
    den = _curve("z", [20.0, 20.0, -0.5, 20.0], sem=[1.0] * N_BINS)
    ratio = frame_ratio(FrameRatioConfig(min_denominator_sigma=3.0), num, den)
    lower_edges = ratio.shell_edges[:-1]
    # The fixture's lower edges and volume-weighted centers must actually
    # differ, or this test would pass by accident.
    assert not np.allclose(lower_edges, ratio.shell_centers)

    figures = make_frame_ratio_figure(FrameRatioConfig(min_denominator_sigma=3.0), (num, den), (ratio,))

    marker_lines = [
        line for line in figures.dipole.axes[0].get_lines()
        if line.get_linestyle() == "None" and len(line.get_xdata()) > 0
    ]
    assert marker_lines, "expected at least one usable/pole marker series"
    all_marker_x = np.sort(np.concatenate([line.get_xdata() for line in marker_lines]))
    np.testing.assert_allclose(all_marker_x, np.sort(ratio.shell_centers))
    plt.close(figures.dipole)
    plt.close(figures.monopole)
