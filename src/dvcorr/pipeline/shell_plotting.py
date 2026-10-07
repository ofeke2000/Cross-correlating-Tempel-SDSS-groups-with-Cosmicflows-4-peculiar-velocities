"""
shell_plotting.py
------------------
The one drawing primitive for per-shell curves, in its two renderings
(CLAUDE.md hard rule 10).

Every figure that plots a per-shell statistic against separation r is built
twice from the SAME arrays:

- **line view** (`binned=False`, the default) -- one point per shell at the
  volume-weighted radius r_eff (`volume_weighted_shell_radii`), the points
  joined by a line, bands via `fill_between`. This is the figure that is read
  and sent.
- **binned view** (`binned=True`) -- each shell drawn as a flat horizontal
  segment spanning its own edges [r_lo, r_hi), bands as boxes over the same
  span, the r_eff point marked on the segment, and every shell edge drawn as
  a faint vertical line across the axis. This is the lookup companion: it
  answers "which shell does this point belong to, and what value was
  measured over it", which the line view cannot, because r_eff hides the
  bin's width.

Both renderings go through `draw_shell_series`, so a builder cannot draw its
two views from different data -- the binned figure is the line figure with
the geometry changed, never a second computation. A figure builder exposes
the choice as a `binned: bool = False` keyword and routes every per-shell
series through this function; `finish_shell_axis` then applies the shared
axis rules (x from r = 0; in the binned view, the shell-edge guides).

The binned view is a COMPANION, never a replacement: the line view stays the
primary figure. The binned companion is DIPOLE-ONLY and MAIN-SIGNAL-ONLY: a
builder called with `binned=True` draws just the measured signal curve(s)
with their own uncertainty bands -- no null curves or null bands -- and
creates no monopole figure at all (`DipoleFigures.monopole` is None). The
monopole and the nulls live in the line view, which carries hard rule 6's
dipole + monopole pair.

Pure functions on a caller-supplied `Axes`; no state, and no
`matplotlib.use` (backend discipline, see `velocity_centered`'s docstring).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dvcorr.config import volume_weighted_shell_radii

#: Shell-edge guide lines in the binned view: faint enough to sit behind the
#: data, visible enough to read a bin boundary off the axis.
_EDGE_GUIDE_COLOR = "0.6"
_EDGE_GUIDE_ALPHA = 0.5
_EDGE_GUIDE_WIDTH = 0.6
_EDGE_GUIDE_STYLE = ":"
#: Line width of a binned segment. Matches matplotlib's default line width so
#: the two views carry the same visual weight.
_SEGMENT_WIDTH = 1.5


def draw_shell_series(
    ax: plt.Axes,
    shell_edges: np.ndarray,
    values: np.ndarray | None,
    *,
    binned: bool = False,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    points: np.ndarray | None = None,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
    label: str | None = None,
    band_alpha: float | None = None,
    linewidth: float | None = None,
    alpha: float | None = None,
    markersize: float | None = None,
    markerfacecolor: str | None = None,
) -> None:
    """Draw one per-shell series on `ax`, as a line or as bins.

    Parameters
    ----------
    ax : matplotlib Axes
    shell_edges : ndarray, shape (B + 1,)
        Strictly increasing shell edges, h^-1 Mpc.
    values : ndarray, shape (B,), or None
        The statistic in each shell. NaN entries are left undrawn in both
        views (a gap, never a zero -- cf. hard rule 5). None draws the band
        alone (`lower`/`upper` required), e.g. a second interval around a
        curve that is drawn, and labelled, by its own call.
    binned : bool, default False
        False: a line through (points, values) with `marker`/`linestyle`.
        True: a horizontal segment per shell over [edges[i], edges[i+1]) in
        `linestyle`, with `marker` at `points` on it.
    lower, upper : ndarray, shape (B,), optional
        Band limits, drawn only when both are given: `fill_between` over
        `points` in the line view, one box per shell in the binned view.
    points : ndarray, shape (B,), optional
        The abscissa of each shell's point. Default: the volume-weighted
        radius `volume_weighted_shell_radii(shell_edges)`, which is right
        for every estimator shell (hard rule 10). Pass it only when a
        statistic defines its own per-bin radius -- e.g. a pair-weighted
        mean separation. The binned view's segments still span the edges.
    color, marker, linestyle, label :
        Matplotlib styling. `label` lands on exactly one legend entry, which
        shows the marker and the line style in both views (on the band when
        `values` is None).
    band_alpha : float, optional
        Band opacity; required when a band is drawn.
    linewidth, alpha, markersize, markerfacecolor : optional
        Passed through to the curve in both views when given; left at
        matplotlib's defaults otherwise, so a call that does not set them
        draws exactly what it drew before they existed. `markerfacecolor`
        of "none" gives hollow markers. The binned segments default to
        `_SEGMENT_WIDTH` when `linewidth` is not given.

    Returns
    -------
    None -- draws on `ax`.
    """
    shell_edges = np.asarray(shell_edges, dtype=float)
    r_points = (
        volume_weighted_shell_radii(shell_edges) if points is None
        else np.asarray(points, dtype=float)
    )
    has_band = lower is not None and upper is not None
    if values is None and not has_band:
        raise ValueError("draw_shell_series: values=None needs a band (lower and upper).")
    if has_band and band_alpha is None:
        raise ValueError("draw_shell_series: a band needs band_alpha.")
    # Only the knobs the caller set, so an unset one is matplotlib's default.
    line_style = {
        key: val
        for key, val in (
            ("linewidth", linewidth), ("alpha", alpha),
            ("markersize", markersize), ("markerfacecolor", markerfacecolor),
        )
        if val is not None
    }
    marker_style = {k: v for k, v in line_style.items() if k != "linewidth"}
    band_label = label if values is None else None

    if not binned:
        if has_band:
            ax.fill_between(
                r_points, lower, upper, color=color, alpha=band_alpha, label=band_label,
            )
        if values is not None:
            ax.plot(
                r_points, np.asarray(values, dtype=float), marker=marker,
                linestyle=linestyle, color=color, label=label, **line_style,
            )
        return

    left, right = shell_edges[:-1], shell_edges[1:]
    if has_band:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        # `bar` rather than `stairs`: a NaN band limit leaves that one box
        # undrawn instead of breaking the whole outline.
        ax.bar(
            left, upper - lower, width=right - left, bottom=lower, align="edge",
            color=color, alpha=band_alpha, linewidth=0, label=band_label,
        )
    if values is None:
        return
    values = np.asarray(values, dtype=float)
    ax.hlines(
        values, left, right, colors=color, linestyles=linestyle,
        linewidth=_SEGMENT_WIDTH if linewidth is None else linewidth,
        **({"alpha": alpha} if alpha is not None else {}),
    )
    ax.plot(r_points, values, marker=marker, linestyle="none", color=color, **marker_style)
    # Legend proxy: the segments are a LineCollection and the markers carry
    # no line, so neither alone shows both halves of the style.
    if label is not None:
        ax.plot([], [], marker=marker, linestyle=linestyle, color=color, label=label, **line_style)


def finish_shell_axis(
    ax: plt.Axes,
    shell_edges: np.ndarray,
    *,
    binned: bool = False,
    start_at_zero: bool = True,
) -> None:
    """Apply the shared radial-axis rules to `ax` after its series are drawn.

    Both views: the x-axis starts at r = 0 (hard rule 10), whatever the
    innermost edge is. Binned view only: a faint vertical guide at every
    shell edge, so a bin boundary can be read straight off the axis.

    Parameters
    ----------
    ax : matplotlib Axes
    shell_edges : ndarray, shape (B + 1,)
        Non-positive edges are skipped for the guides when the x-axis is
        logarithmic (they have no position there).
    binned : bool, default False
    start_at_zero : bool, default True
        False leaves the x-limits alone. Only for an axis on which r = 0
        does not exist -- a log x-axis -- never to hide the inner region
        of a cartesian one.

    Returns
    -------
    None -- modifies `ax`.
    """
    if binned:
        edges = np.asarray(shell_edges, dtype=float)
        if ax.get_xscale() == "log":
            edges = edges[edges > 0]
        for edge in edges:
            ax.axvline(
                edge, color=_EDGE_GUIDE_COLOR, alpha=_EDGE_GUIDE_ALPHA,
                linewidth=_EDGE_GUIDE_WIDTH, linestyle=_EDGE_GUIDE_STYLE, zorder=0,
            )
    if start_at_zero:
        ax.set_xlim(left=0.0)
