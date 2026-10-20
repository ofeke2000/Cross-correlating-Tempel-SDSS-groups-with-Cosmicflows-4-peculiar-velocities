"""
vpwf_error_figures.py
---------------------
Publication-style figures for `notebooks/13_cf4_vpwf_error.ipynb`, the
standalone write-up of the CF4 `Vpwf` log10 error for the advisor.

Every number is computed by `dvcorr.pipeline.real_catalog_audit` (the
peculiar-velocities test); this module only draws its tables, in a style that
departs from the project's audit figures on purpose (user's request for this
notebook):

- Okabe-Ito colors, fixed per velocity, with redundant encoding (marker and
  line style) so every figure survives color-blind viewing and grayscale;
- sans-serif type, no top/right spines, no grid, units in parentheses;
- bold panel letters (A, B, ...) on multi-panel figures;
- a pale blue-grey (`bone_r`) for point density under colored curves;
- legends outside the data area, so a figure sent on its own needs no
  explanation of what a line is;
- constrained layout, a white figure background.

The rcParams live in `PUBLICATION_RC` and are applied with `plt.rc_context`
inside each builder, so importing this module, or drawing one of its figures,
leaves every other figure in the session as it was.

Builders, one per notebook figure:

- `make_vpwf_comparison_figure` -- published Vpwf against Vpds and the two
  rebuilds, three scatter panels at print resolution.
- `make_noise_histogram_figure` -- v / sigma_u of the noise-dominated groups
  against the unit and the width-1/ln 10 Gaussians.
- `make_noise_depth_figure` -- std(v / sigma_u) per cz bin.
- `make_velocity_scatter_figure` -- each velocity against cz or measured
  distance with binned median, observed spread and predicted sigma_u.
"""

from __future__ import annotations

from string import ascii_uppercase

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from dvcorr.pipeline.real_catalog_audit import (
    VELOCITY_TEST_X_DISTANCE,
    VELOCITY_TEST_X_REDSHIFT,
    VELOCITY_VPDS_PUBLISHED,
    VELOCITY_VPWF_LN,
    VELOCITY_VPWF_LOG10,
    VELOCITY_VPWF_PUBLISHED,
    VPWF_ERROR_LABELS,
    AuditConfig,
)

# Okabe & Ito (2008) color-blind-safe palette, the entries used here.
_OKABE_BLUISH_GREEN = "#009E73"
_OKABE_VERMILLION = "#D55E00"
_OKABE_BLUE = "#0072B2"
_OKABE_REDDISH_PURPLE = "#CC79A7"
_INK = "#000000"
_GUIDE_GREY = "0.35"

#: matplotlib rcParams for every figure of this module (the
#: scientific-visualization publication preset, sized for on-screen reading:
#: 7.2 in is a two-column journal width, drawn at `figure.dpi`).
PUBLICATION_RC: dict[str, object] = {
    "figure.dpi": 130,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": _INK,
    "axes.labelcolor": _INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "legend.fontsize": 8,
    "legend.frameon": False,
}

#: Per velocity: color, marker, line style and whether the marker is open.
#: The published Vpwf -- the column in error -- takes vermillion; its log10
#: rebuild lies on it by construction, so it is dashed with open markers to
#: stay visible on top.
_COLORS: dict[str, str] = {
    VELOCITY_VPDS_PUBLISHED: _OKABE_BLUISH_GREEN,
    VELOCITY_VPWF_PUBLISHED: _OKABE_VERMILLION,
    VELOCITY_VPWF_LN: _OKABE_BLUE,
    VELOCITY_VPWF_LOG10: _OKABE_REDDISH_PURPLE,
}
_MARKERS: dict[str, str] = {
    VELOCITY_VPDS_PUBLISHED: "^",
    VELOCITY_VPWF_PUBLISHED: "o",
    VELOCITY_VPWF_LN: "s",
    VELOCITY_VPWF_LOG10: "D",
}
_DASHED = frozenset({VELOCITY_VPWF_LOG10})
#: Math-text names: short ones for axes, long ones for titles and legends.
_SHORT_NAMES: dict[str, str] = {
    VELOCITY_VPDS_PUBLISHED: r"$V_{\rm pds}$",
    VELOCITY_VPWF_PUBLISHED: r"$V_{\rm pwf}$",
    VELOCITY_VPWF_LN: r"$V_{\rm pwf}^{\,\ln}$",
    VELOCITY_VPWF_LOG10: r"$V_{\rm pwf}^{\,\log_{10}}$",
}
_LONG_NAMES: dict[str, str] = {
    VELOCITY_VPDS_PUBLISHED: r"$V_{\rm pds}$ (published)",
    VELOCITY_VPWF_PUBLISHED: r"$V_{\rm pwf}$ (published)",
    VELOCITY_VPWF_LN: r"$V_{\rm pwf}$ rebuilt with ln",
    VELOCITY_VPWF_LOG10: r"$V_{\rm pwf}$ rebuilt with log$_{10}$",
}
_VELOCITY_UNIT = r"(km s$^{-1}$)"
_X_AXIS_LABELS: dict[str, str] = {
    VELOCITY_TEST_X_REDSHIFT: r"$cz_{\rm CMB}$ " + _VELOCITY_UNIT,
    VELOCITY_TEST_X_DISTANCE: r"Measured distance $D$ (Mpc, $H_0$ = 74.6)",
}
_X_NAMES: dict[str, str] = {
    VELOCITY_TEST_X_REDSHIFT: "redshift",
    VELOCITY_TEST_X_DISTANCE: "measured distance",
}

_WIDTH = 7.2
_COMPARISON_FIGSIZE = (_WIDTH, 2.75)
#: Print resolution for the comparison figure, so its points stay sharp when zoomed.
_COMPARISON_DPI = 300
#: One point per group (area in pt^2), translucent so dense regions read darker.
_POINT_SIZE = 1.5
_POINT_ALPHA = 0.35
_GRID_FIGSIZE = (_WIDTH, 5.4)
_DEPTH_FIGSIZE = (_WIDTH * 0.75, 3.4)
_HEXBIN_GRID = 110
#: Neutral blue-grey, one map for every panel, so no panel's cloud takes a
#: velocity's color.
_BACKGROUND_CMAP = "bone_r"
#: Background density under colored curves: light enough to read them on.
_BACKGROUND_ALPHA = 0.55
_HIST_ALPHA = 0.45
_HIST_EDGE_WIDTH = 1.0
_GUIDE_WIDTH = 0.9
_GUIDE_ZORDER = 0.5
_MEDIAN_WIDTH = 1.8
_BAND_WIDTH = 1.2
_ZERO_WIDTH = 0.6
_GAUSSIAN_SAMPLES = 401
_PANEL_LABEL_SIZE = 11
_PANEL_LABEL_OFFSET = (-0.16, 1.04)
_ANNOTATION_POSITION = (0.04, 0.96)
_ANNOTATION_BOX = {"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none",
                   "alpha": 0.85}
_SUPTITLE_SIZE = 9.5


def _panel_labels(axes: np.ndarray) -> None:
    """Bold A, B, ... at the top left of each axis, outside the frame."""
    for letter, ax in zip(ascii_uppercase, np.ravel(axes)):
        ax.text(*_PANEL_LABEL_OFFSET, letter, transform=ax.transAxes,
                fontsize=_PANEL_LABEL_SIZE, fontweight="bold", va="bottom", ha="right")


def _annotate(ax: plt.Axes, text: str) -> None:
    """A short result printed in the top-left corner of the data area."""
    ax.text(*_ANNOTATION_POSITION, text, transform=ax.transAxes, va="top", ha="left",
            bbox=_ANNOTATION_BOX)


def _line_style(label: str) -> dict[str, object]:
    """Color, marker and dash of one velocity, open marker when dashed."""
    color = _COLORS[label]
    dashed = label in _DASHED
    return {"color": color, "marker": _MARKERS[label], "ls": "--" if dashed else "-",
            "mfc": "white" if dashed else color, "mec": color}


def make_vpwf_comparison_figure(
    table: pd.DataFrame, config: AuditConfig | None = None
) -> plt.Figure:
    """Published Vpwf against Vpds, the ln rebuild and the log10 rebuild.

    Three square panels on equal axes of half-width `scatter_velocity_limit`:
    every group as one small point in the published Vpwf's color, over two
    guides -- slope 1 (solid, the same velocity) and slope 1/ln 10 (dashed, a
    log10 velocity against a right one). Each panel prints the median ratio
    Vpwf / x over groups with |x| > `linear_regime_floor`. Drawn at
    `_COMPARISON_DPI`, so the inline image holds detail when zoomed.

    Parameters
    ----------
    table : pandas.DataFrame
        `real_catalog_audit.velocity_test_table` output.
    config : AuditConfig, optional
    """
    config = config or AuditConfig()
    lim = config.scatter_velocity_limit
    line = np.array([-lim, lim])
    published = table[VELOCITY_VPWF_PUBLISHED].to_numpy()
    references = (VELOCITY_VPDS_PUBLISHED, VELOCITY_VPWF_LN, VELOCITY_VPWF_LOG10)
    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(1, len(references), figsize=_COMPARISON_FIGSIZE,
                                 dpi=_COMPARISON_DPI, sharex=True, sharey=True,
                                 layout="constrained")
        for ax, label in zip(axes, references):
            x = table[label].to_numpy()
            # Guides under the points, so a cloud lying on one stays visible.
            ax.plot(line, line, color=_INK, lw=_GUIDE_WIDTH, label="Slope 1", zorder=_GUIDE_ZORDER)
            ax.plot(line, line / np.log(10.0), color=_INK, lw=_GUIDE_WIDTH, ls="--",
                    label=r"Slope 1/ln 10 = 0.434", zorder=_GUIDE_ZORDER)
            ax.scatter(x, published, s=_POINT_SIZE, color=_COLORS[VELOCITY_VPWF_PUBLISHED],
                       alpha=_POINT_ALPHA, linewidths=0)
            rows = np.abs(x) > config.linear_regime_floor
            ratio = float(np.median(published[rows] / x[rows]))
            _annotate(ax, f"median ratio\n{ratio:.4f}")
            ax.set_xlabel(f"{_SHORT_NAMES[label]} {_VELOCITY_UNIT}")
            ax.set_title(_LONG_NAMES[label])
            ax.set_aspect("equal")
        axes[0].set_ylabel(f"Published {_SHORT_NAMES[VELOCITY_VPWF_PUBLISHED]} {_VELOCITY_UNIT}")
        axes[0].set_xlim(-lim, lim)
        axes[0].set_ylim(-lim, lim)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
        _panel_labels(axes)
    return fig


def make_noise_histogram_figure(
    table: pd.DataFrame,
    config: AuditConfig | None = None,
    labels: tuple[str, ...] = VPWF_ERROR_LABELS,
) -> plt.Figure:
    """Watkins & Feldman's Fig. 1 test, one panel per velocity of `labels`.

    v / sigma_u over noise-dominated groups (sigma_u > `noise_dominated_sigma`)
    as a unit-area histogram, against the unit Gaussian (solid) the right
    velocity must follow and the width-1/ln 10 Gaussian (dashed) a log10
    velocity follows. Each panel prints its standard deviation.

    Parameters
    ----------
    table : pandas.DataFrame
        `real_catalog_audit.velocity_test_table` output.
    config : AuditConfig, optional
    labels : tuple of four str, optional
        Filled into a 2 x 2 grid. Default `VPWF_ERROR_LABELS`.
    """
    config = config or AuditConfig()
    sigma = table["sigma_u"].to_numpy()
    noisy = sigma > config.noise_dominated_sigma
    lim = config.normalized_velocity_limit
    bins = np.linspace(-lim, lim, config.normalized_velocity_bins + 1)
    x = np.linspace(-lim, lim, _GAUSSIAN_SAMPLES)
    narrow = 1.0 / np.log(10.0)
    # Unit-area Gaussians: exp(-x^2 / 2 s^2) / (s sqrt(2 pi)).
    unit = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    log10_scale = np.exp(-0.5 * (x / narrow) ** 2) / (narrow * np.sqrt(2.0 * np.pi))
    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(2, 2, figsize=_GRID_FIGSIZE, sharex=True, sharey=True,
                                 layout="constrained")
        for ax, label in zip(axes.flat, labels, strict=True):
            z = table[label].to_numpy()[noisy] / sigma[noisy]
            color = _COLORS[label]
            ax.hist(z, bins=bins, density=True, histtype="stepfilled", color=color,
                    alpha=_HIST_ALPHA)
            ax.hist(z, bins=bins, density=True, histtype="step", color=color,
                    lw=_HIST_EDGE_WIDTH)
            ax.plot(x, unit, color=_INK, lw=_MEDIAN_WIDTH, label="Unit Gaussian")
            ax.plot(x, log10_scale, color=_GUIDE_GREY, lw=_MEDIAN_WIDTH, ls="--",
                    label="Gaussian of width 1/ln 10")
            ax.set_title(_LONG_NAMES[label])
            _annotate(ax, f"std = {np.std(z):.3f}")
        for ax in axes[-1]:
            ax.set_xlabel(r"$v\,/\,\sigma_u$")
        for ax in axes[:, 0]:
            ax.set_ylabel("Probability density")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
        axes[0, 0].set_xlim(-lim, lim)
        axes[0, 0].set_ylim(bottom=0.0)
        fig.suptitle(rf"CF4 groups with $\sigma_u$ > {config.noise_dominated_sigma:,.0f} "
                     rf"km s$^{{-1}}$ (N = {int(noisy.sum()):,})", fontsize=_SUPTITLE_SIZE)
        _panel_labels(axes)
    return fig


def make_noise_depth_figure(
    by_depth: pd.DataFrame, labels: tuple[str, ...] = VPWF_ERROR_LABELS
) -> plt.Figure:
    """std(v / sigma_u) per cz bin for each velocity of `labels`, with guides
    at 1 (eDM explains the scatter) and 1/ln 10 (a column ln 10 too small).

    Parameters
    ----------
    by_depth : pandas.DataFrame
        `real_catalog_audit.velocity_test_by_depth(..., labels=labels)` output.
    labels : tuple of str, optional
        Default `VPWF_ERROR_LABELS`.
    """
    x = by_depth["cz_median"]
    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=_DEPTH_FIGSIZE, layout="constrained")
        ax.axhline(1.0, color=_GUIDE_GREY, lw=_GUIDE_WIDTH, label="1: eDM explains the scatter")
        ax.axhline(1.0 / np.log(10.0), color=_GUIDE_GREY, lw=_GUIDE_WIDTH, ls="--",
                   label="1/ln 10: column ln 10 too small")
        for label in labels:
            ax.plot(x, by_depth[label], label=_LONG_NAMES[label], **_line_style(label))
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"Median $cz_{\rm CMB}$ of bin " + _VELOCITY_UNIT)
        ax.set_ylabel(r"std($v\,/\,\sigma_u$)")
        ax.legend(loc="upper right", ncols=2)
    return fig


def make_velocity_scatter_figure(
    table: pd.DataFrame,
    binned: pd.DataFrame,
    x_column: str,
    config: AuditConfig | None = None,
    labels: tuple[str, ...] = VPWF_ERROR_LABELS,
) -> plt.Figure:
    """Each velocity of `labels` against cz or measured distance, one panel each.

    Pale blue-grey point density of every group; in the velocity's color the
    binned median (solid) and median ± observed std (dashed); in black
    (dotted) median ± the rms sigma_u that eDM predicts for the ln velocity.
    Both bands are centered on the median, so their widths compare whatever
    the median does. Shared axes.

    Parameters
    ----------
    table : pandas.DataFrame
        `real_catalog_audit.velocity_test_table` output.
    binned : pandas.DataFrame
        `real_catalog_audit.velocity_test_binned(table, x_column, config, labels)`.
    x_column : str
        `VELOCITY_TEST_X_REDSHIFT` or `VELOCITY_TEST_X_DISTANCE`.
    config : AuditConfig, optional
    labels : tuple of four str, optional
        Filled into a 2 x 2 grid. Default `VPWF_ERROR_LABELS`.
    """
    config = config or AuditConfig()
    lim = config.scatter_velocity_limit
    x_max = {
        VELOCITY_TEST_X_DISTANCE: config.distance_plot_max,
        VELOCITY_TEST_X_REDSHIFT: config.redshift_plot_max,
    }[x_column]
    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(2, 2, figsize=_GRID_FIGSIZE, sharex=True, sharey=True,
                                 layout="constrained")
        for ax, label in zip(axes.flat, labels, strict=True):
            color = _COLORS[label]
            ax.hexbin(table[x_column], table[label], gridsize=_HEXBIN_GRID,
                      cmap=_BACKGROUND_CMAP, norm=mcolors.LogNorm(), mincnt=1,
                      extent=(0.0, x_max, -lim, lim), alpha=_BACKGROUND_ALPHA, linewidths=0)
            ax.axhline(0.0, color=_GUIDE_GREY, lw=_ZERO_WIDTH)
            rows = binned[binned["velocity"] == label]
            x, median = rows["x_median"], rows["median"]
            for sign in (1, -1):
                ax.plot(x, median + sign * rows["sigma_u_rms"], color=_INK, lw=_BAND_WIDTH,
                        ls=":")
                ax.plot(x, median + sign * rows["std"], color=color, lw=_BAND_WIDTH, ls="--")
            ax.plot(x, median, color=color, lw=_MEDIAN_WIDTH)
            ax.set_title(_LONG_NAMES[label])
        for ax in axes[-1]:
            ax.set_xlabel(_X_AXIS_LABELS[x_column])
        for ax in axes[:, 0]:
            ax.set_ylabel(f"Peculiar velocity {_VELOCITY_UNIT}")
        axes[0, 0].set_xlim(0.0, x_max)
        axes[0, 0].set_ylim(-lim, lim)
        # Black and grey stand-ins: each panel draws its median and std in its
        # own color, so the legend describes the line style, not the color.
        handles = [
            Line2D([], [], color=_GUIDE_GREY, lw=_MEDIAN_WIDTH, label="Median"),
            Line2D([], [], color=_GUIDE_GREY, lw=_BAND_WIDTH, ls="--",
                   label=r"Median $\pm$ observed std"),
            Line2D([], [], color=_INK, lw=_BAND_WIDTH, ls=":",
                   label=r"Median $\pm\,\sigma_u$ predicted (eDM)"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
        fig.suptitle(f"CF4 groups against {_X_NAMES[x_column]} (N = {len(table):,})",
                     fontsize=_SUPTITLE_SIZE)
        _panel_labels(axes)
    return fig

