"""
test_shell_plotting.py
-----------------------
Pins `dvcorr.pipeline.shell_plotting` -- the one drawing primitive behind
both renderings of a per-shell curve (CLAUDE.md hard rule 10) -- and its
first consumer, `velocity_centered.make_figure(..., binned=True)`.

What the binned view must guarantee, and what these tests check:

- each shell is ONE flat segment spanning exactly its own edges
  [edges[i], edges[i+1]), at exactly the value the line view plots there;
- the r_eff marker sits on that segment, so a point on the line figure
  maps to one segment;
- a NaN shell is a gap in both views, never a zero (cf. hard rule 5);
- the line view (`binned=False`) is unchanged: a `Line2D` through r_eff, no
  segments, no step artists;
- the x-axis starts at r = 0 in both views, and the binned view marks every
  shell edge.

Backend discipline: `matplotlib.use("Agg")` here, before any pipeline import
(see `test_plot_wiring.py`).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.patches import StepPatch

from dvcorr import conventions
from dvcorr.config import SPACING_LOG, ShellConfig, volume_weighted_shell_radii
from dvcorr.pipeline.shell_plotting import draw_shell_series, finish_shell_axis
from dvcorr.pipeline.velocity_centered import RunConfig, make_figure

from tests.test_plot_wiring import _tiny_result_and_normalized

_BAND_ALPHA = 0.2
_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
_VALUES = np.array([3.0, -1.0, np.nan, 2.5])


def _segments(ax: plt.Axes) -> list[np.ndarray]:
    """Every drawn horizontal segment on `ax`, as (2, 2) arrays. `hlines`
    keeps an EMPTY entry for a NaN value -- drawn as nothing -- so those are
    dropped here: a gap is the absence of a segment.
    """
    return [
        seg
        for coll in ax.collections
        if isinstance(coll, LineCollection)
        for seg in coll.get_segments()
        if np.size(seg)
    ]


def test_binned_view_draws_one_flat_segment_per_shell_over_its_own_edges():
    fig, ax = plt.subplots()
    draw_shell_series(ax, _EDGES, _VALUES, binned=True, color="k", band_alpha=_BAND_ALPHA)

    segments = _segments(ax)
    finite = np.isfinite(_VALUES)
    assert len(segments) == finite.sum()  # the NaN shell is a gap, not a zero
    for seg, lo, hi, value in zip(
        segments, _EDGES[:-1][finite], _EDGES[1:][finite], _VALUES[finite]
    ):
        np.testing.assert_allclose(seg[:, 0], [lo, hi])
        np.testing.assert_allclose(seg[:, 1], [value, value])
    plt.close(fig)


def test_binned_view_marks_r_eff_on_each_segment():
    fig, ax = plt.subplots()
    draw_shell_series(ax, _EDGES, _VALUES, binned=True, color="k", band_alpha=_BAND_ALPHA)

    markers = [line for line in ax.get_lines() if len(line.get_xdata()) == _VALUES.size]
    assert len(markers) == 1
    np.testing.assert_allclose(markers[0].get_xdata(), volume_weighted_shell_radii(_EDGES))
    np.testing.assert_array_equal(markers[0].get_ydata(), _VALUES)
    assert markers[0].get_linestyle() == "None"
    plt.close(fig)


def test_binned_band_is_one_box_per_shell_over_its_own_edges():
    lower, upper = _VALUES - 1.0, _VALUES + 1.0
    fig, ax = plt.subplots()
    draw_shell_series(
        ax, _EDGES, _VALUES, binned=True, lower=lower, upper=upper,
        color="k", band_alpha=_BAND_ALPHA,
    )

    boxes = ax.patches
    assert len(boxes) == _VALUES.size
    for box, lo, hi, bottom in zip(boxes, _EDGES[:-1], _EDGES[1:], lower):
        assert box.get_x() == lo
        assert box.get_width() == hi - lo
        if np.isfinite(bottom):
            assert box.get_y() == bottom
            assert box.get_height() == 2.0
    plt.close(fig)


def test_line_view_is_a_line_through_r_eff_with_no_segments_or_steps():
    fig, ax = plt.subplots()
    draw_shell_series(ax, _EDGES, _VALUES, color="k", band_alpha=_BAND_ALPHA)

    assert _segments(ax) == []
    assert not any(isinstance(p, StepPatch) for p in ax.patches)
    (line,) = ax.get_lines()
    np.testing.assert_allclose(line.get_xdata(), volume_weighted_shell_radii(_EDGES))
    assert line.get_linestyle() == "-"
    plt.close(fig)


def test_finish_shell_axis_starts_at_zero_and_guides_every_edge_only_when_binned():
    edges = _EDGES + 1.0  # innermost edge above zero: the axis must still start at 0
    for binned in (False, True):
        fig, ax = plt.subplots()
        draw_shell_series(ax, edges, _VALUES, binned=binned, color="k", band_alpha=_BAND_ALPHA)
        finish_shell_axis(ax, edges, binned=binned)

        assert ax.get_xlim()[0] == 0.0
        guides = [
            line for line in ax.get_lines()
            if len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) == 0
        ]
        assert len(guides) == (edges.size if binned else 0)
        plt.close(fig)


def test_make_figure_binned_draws_the_same_values_as_the_line_view():
    """The binned companion is the line figure with the geometry changed:
    every series' segment heights equal the line's y-values, shell by shell.
    Null curves and the monopole figure exist only in the line view.
    """
    shells = ShellConfig(
        min_radius=1.0, max_radius=64.0, spacing=SPACING_LOG, n_bins=12, include_zero_bin=True,
    )
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    lines = make_figure(cfg, result, normalized)
    bins = make_figure(cfg, result, normalized, binned=True)

    # The binned companion is dipole-only and main-signal-only (hard rule 10).
    assert bins.monopole is None
    line_ax, bin_ax = lines.dipole.axes[0], bins.dipole.axes[0]
    assert not any("null" in t.get_text() for t in bin_ax.get_legend().get_texts())
    curves = [
        l for l in line_ax.get_lines()
        if len(l.get_xdata()) == result.shell_edges.size - 1 and "null" not in l.get_label()
    ]
    heights = [
        [seg for seg in coll.get_segments() if np.size(seg)]
        for coll in bin_ax.collections
        if isinstance(coll, LineCollection)
    ]
    assert len(heights) == len(curves) == 1
    for curve, segs in zip(curves, heights):
        finite = np.isfinite(curve.get_ydata())
        np.testing.assert_allclose(
            [seg[0, 1] for seg in segs], np.asarray(curve.get_ydata())[finite]
        )
    assert bin_ax.get_xlim()[0] == 0.0
    plt.close("all")


def test_points_override_moves_the_marker_but_segments_still_span_the_edges():
    """A statistic with its own per-bin radius (e.g. a pair-weighted mean
    separation) marks that radius; the binned segment is still the bin.
    """
    points = np.array([0.1, 0.9, 1.1, 3.9])
    fig, ax = plt.subplots()
    draw_shell_series(ax, _EDGES, _VALUES, binned=True, points=points, color="k")

    (markers,) = [line for line in ax.get_lines() if len(line.get_xdata()) == _VALUES.size]
    np.testing.assert_allclose(markers.get_xdata(), points)
    for seg, lo, hi in zip(_segments(ax), _EDGES[:-1][np.isfinite(_VALUES)], _EDGES[1:][np.isfinite(_VALUES)]):
        np.testing.assert_allclose(seg[:, 0], [lo, hi])
    plt.close(fig)


def test_band_only_draws_no_curve_in_either_view_and_labels_the_band():
    lower, upper = _VALUES - 1.0, _VALUES + 1.0
    for binned in (False, True):
        fig, ax = plt.subplots()
        draw_shell_series(
            ax, _EDGES, None, binned=binned, lower=lower, upper=upper,
            color="k", band_alpha=_BAND_ALPHA, label="interval",
        )
        assert ax.get_lines() == []
        assert _segments(ax) == []
        assert [h.get_label() for h in ax.get_legend_handles_labels()[0]] == ["interval"]
        plt.close(fig)


def test_style_knobs_reach_the_curve_in_both_views():
    for binned in (False, True):
        fig, ax = plt.subplots()
        draw_shell_series(
            ax, _EDGES, _VALUES, binned=binned, color="k", label="s",
            linewidth=0.7, alpha=0.4, markersize=3.0, markerfacecolor="none",
        )
        curve = [line for line in ax.get_lines() if len(line.get_xdata()) == _VALUES.size][0]
        assert curve.get_alpha() == 0.4
        assert curve.get_markersize() == 3.0
        assert curve.get_markerfacecolor() == "none"
        if binned:
            (coll,) = [c for c in ax.collections if isinstance(c, LineCollection)]
            assert coll.get_linewidth()[0] == 0.7
        else:
            assert curve.get_linewidth() == 0.7
        plt.close(fig)


def test_band_without_band_alpha_or_nothing_to_draw_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        draw_shell_series(ax, _EDGES, _VALUES, lower=_VALUES, upper=_VALUES, color="k")
    with pytest.raises(ValueError):
        draw_shell_series(ax, _EDGES, None, color="k", band_alpha=_BAND_ALPHA)
    plt.close(fig)


def test_log_axis_keeps_its_limits_and_skips_the_zero_edge_guide():
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    draw_shell_series(ax, _EDGES, _VALUES, binned=True, color="k", band_alpha=_BAND_ALPHA)
    finish_shell_axis(ax, _EDGES, binned=True, start_at_zero=False)

    assert ax.get_xlim()[0] > 0.0
    guides = [
        line for line in ax.get_lines()
        if len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) == 0
    ]
    assert len(guides) == int((_EDGES > 0).sum())
    plt.close(fig)
