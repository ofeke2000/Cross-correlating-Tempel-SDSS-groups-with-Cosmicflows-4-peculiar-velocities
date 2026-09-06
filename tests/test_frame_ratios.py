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
    fig = make_frame_ratio_figure(cfg, curves, frame_ratios(cfg, curves))
    ax_ratio, ax_mono = fig.axes
    assert ax_ratio.get_xscale() == "linear"
    assert ax_mono.get_xscale() == "linear"
    assert ax_ratio.get_yscale() == "linear"
    assert ax_mono.get_yscale() == "linear"
    plt.close(fig)


def test_the_figure_prints_each_mean_in_the_legend():
    """The number the plot exists to produce has to be readable off the plot."""
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    fig = make_frame_ratio_figure(cfg, curves, ratios)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

    assert len(labels) == len(ratios)
    for ratio, label in zip(ratios, labels):
        assert ratio.label in label
        assert f"{ratio.mean_ratio:.2f}" in label
    plt.close(fig)


def test_a_pole_is_drawn_but_does_not_set_the_y_limits():
    """One bin next to a zero crossing must not flatten every real feature."""
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    fig = make_frame_ratio_figure(cfg, curves, ratios)

    pole = max(abs(r.ratio[2]) for r in ratios)   # obs/z at the crossing: ~240
    lo, hi = fig.axes[0].get_ylim()
    assert pole > 10 * max(abs(lo), abs(hi))

    usable = np.concatenate([r.ratio[r.usable] for r in ratios])
    assert lo < usable.min() and usable.max() < hi
    plt.close(fig)


def test_the_figure_draws_unusable_shells_with_open_markers():
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    fig = make_frame_ratio_figure(cfg, curves, ratios)

    fills = {line.get_fillstyle() for line in fig.axes[0].get_lines()
             if line.get_linestyle() == "None" and len(line.get_xdata()) > 0}
    assert "none" in fills and "full" in fills
    plt.close(fig)


def test_the_figure_has_a_monopole_panel_under_the_dipole_panel():
    """Hard rule 6, at the figure level."""
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    fig = make_frame_ratio_figure(cfg, curves, frame_ratios(cfg, curves))
    ax_ratio, ax_mono = fig.axes
    assert r"\hat\zeta_1" in ax_ratio.get_ylabel()
    assert r"\hat\zeta_0" in ax_mono.get_ylabel()
    assert ax_mono.get_position().y0 < ax_ratio.get_position().y0
    plt.close(fig)


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
    fig = make_frame_ratio_figure(cfg, curves, ratios)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

    for ratio, label in zip(ratios, labels):
        if ratio.changes_sign:
            assert "across the sign change" in label
            assert f"{ratio.mean_where_negative:.2f}" in label
            assert f"{ratio.mean_where_positive:.2f}" in label
        else:
            assert "sign change" not in label
    assert any(r.changes_sign for r in ratios)   # the fixture must exercise it
    plt.close(fig)


def test_the_table_names_the_cancellation_when_a_ratio_changes_sign():
    num = _curve("obs", [40.0, 40.0, 40.0, 40.0])
    den = _curve("z", [-20.0, -20.0, 20.0, 20.0])
    text = format_frame_ratio_table((frame_ratio(FrameRatioConfig(), num, den),))
    assert "SIGN CHANGE" in text
    assert "-2.000 (negative)" in text
    assert "2.000 (positive)" in text


def test_the_connecting_line_breaks_at_a_pole_instead_of_spiking_through_it():
    """A segment drawn to an off-scale point is a stripe that reads as structure."""
    curves = _figure_curves()
    cfg = FrameRatioConfig()
    ratios = frame_ratios(cfg, curves)
    fig = make_frame_ratio_figure(cfg, curves, ratios)

    joined = [line.get_ydata() for line in fig.axes[0].get_lines()
              if line.get_linestyle() == "-" and len(line.get_xdata()) == N_BINS]
    assert joined, "expected one connecting line per ratio"
    for ratio, ydata in zip(ratios, joined):
        np.testing.assert_array_equal(np.isnan(np.asarray(ydata, dtype=float)), ~ratio.usable)
    plt.close(fig)
