"""Ratios between the three frames the DM box is measured in, shell by shell.

Three curves exist for the same box, the same binning and the same estimator,
differing only in what the shell expansion is oriented on and what the tracer
positions are:

  ``3D``    the observer-free velocity frame -- axis ``z_hat = v_hat``, weight
            ``|v|`` (`dvcorr.pipeline.velocity_frame_comparison`).
  ``obs``   the observer frame -- axis ``z_hat = sign(u) * n_hat_V``, weight
            ``|u|``, i.e. the one line-of-sight component a real survey has
            (`dvcorr.pipeline.velocity_centered`).
  ``z``     the observer frame again, but with the tracer field displaced into
            redshift space (`dvcorr.pipeline.redshift_space_comparison`).

Each step down that list throws information away, and the point of this module
is to price each loss as a single number: the shell-averaged ratio. ``3D/obs``
is the projection cost, ``obs/z`` the redshift-space cost, ``3D/z`` the two of
them together -- and because the three ratios are built from the same three
curves, the identity ``(3D/obs) * (obs/z) = 3D/z`` holds bin by bin, so the
figure cannot silently disagree with itself.

WHY A RATIO NEEDS A GUARD.  ``z`` changes sign: in this box the redshift-space
dipole is negative inside r ~ 8 h^-1 Mpc and positive outside it (the
fingers-of-God reversal, `redshift_space_comparison`'s module docstring), so it
passes through zero, and any shell where it does is a POLE, not a measurement.
An unguarded shell average is then dominated by whichever bin happened to land
nearest the crossing -- a number set by the binning rather than by the physics.
`frame_ratio` therefore marks a shell unusable when the DENOMINATOR is not
itself detected, ``|zeta_hat| < min_denominator_sigma * sem``, and reports the
mean over the surviving shells together with the count. Unusable shells are
still returned and still plotted (open markers): the reader sees the pole, it
just does not enter the average. `FrameRatio.naive_mean_ratio` keeps the
unguarded number so the guard's effect is visible rather than assumed.

AND A MEAN CAN CANCEL.  Excluding the pole is not enough: a curve that crosses
zero is NEGATIVE on one side of the crossing and POSITIVE on the other, so its
shell average is a cancellation between two branches that say opposite things,
and a ratio averaging to ~0 can mean "the two frames agree nowhere" quite as
easily as "they disagree everywhere". `FrameRatio.changes_sign` flags it and
`mean_where_negative` / `mean_where_positive` report the two branches
separately, which is the number worth quoting when the flag is set. The figure
marks such a ratio in its legend rather than letting one averaged number stand
for two regimes.

WHAT THE RATIO IS NOT.  It is not a correction factor. The three curves are not
independent measurements of one quantity; ``obs`` is a projection of ``3D`` and
carries a geometric factor before any physics, so a ``3D/obs`` DIPOLE ratio
near 3 is the geometry, not a signal. Three, not two: for an isotropic velocity
field the observer frame keeps ``<mu^2> = 1/3`` of the velocity frame's dipole
-- one cosine from the scalar (``|u| = |v| |cos theta|``), one from projecting
the axis -- at any observer distance and with no distant-observer limit taken
(the derivation is in `dvcorr.estimators.velocity_frame_dipole`'s module
docstring; the number is pinned on one shared center set by
tests/test_velocity_frame_dipole.py::test_isotropic_frame_ratio_is_three). The
familiar ``<|u|>/<|v|> = 1/2`` belongs to the MONOPOLE panel below, where both
curves are sums of a positive-definite speed; reading its 2 as the dipole's
baseline understates the projection loss by 3/2 and turns a
correctly-behaving pair of estimators into a spurious scale-dependent drift.
What is worth reading is the ratio's RUN WITH SCALE relative to 3: a flat ratio
means the loss is a constant factor that a model can divide out, a sloped one
means it is scale-dependent and cannot.

MONOPOLES TRAVEL TOO (CLAUDE.md hard rule 6).  Every ratio carries the matching
ratio of monopoles, drawn in the lower panel. The monopole ratio is the part of
the dipole ratio explained by the weight and the shell occupancy alone; where
the two panels agree, the frames differ only in how much velocity they see, and
where they part company, the difference is genuinely in the dipole's ANISOTROPY.
Monopoles are sums of ``|u|`` and never cross zero, so that panel needs no pole
guard.

CENTER SETS.  Nothing here requires the curves to share a center set, and in the
DM notebook they do not -- ``3D`` and ``obs`` come from one draw, ``z`` from the
buffered carve's smaller one. Shell edges MUST match (checked); center sets need
not, and where they differ the ratio inherits both curves' sampling noise. The
honest like-for-like check is to compare a cross-section ratio against the
same-center one measured inside the redshift-space run itself
(``real / redshift``), which is the same pair of frames on one center set;
`frame_ratio` is happy to compute either, and the DM notebook prints both.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from dvcorr.config import volume_weighted_shell_radii
from dvcorr.pipeline.shell_plotting import draw_shell_series, finish_shell_axis
from dvcorr.pipeline.velocity_centered import (
    NormalizedDipole,
    _COLOR_ZERO_LINE,
    _DIPOLE_FIGSIZE,
    _GRID_ALPHA,
    _LABEL_COLOR,
    _MONOPOLE_FIGSIZE,
    _ZERO_LINE_WIDTH,
    DipoleFigures,
)

# One color per RATIO (not per frame): the figure's lines are ratios, and there
# are exactly as many of them as ordered pairs of the frames handed in.
_RATIO_COLORS = ("#7b2d8e", "#d95f02", "#1b9e77")
_MARKERS = ("o", "s", "^")
_USABLE_FILL = "full"       # marker style for shells that enter the mean
_UNUSABLE_FILL = "none"     # ... and for the poles, drawn but not averaged
_UNITY_LINE_STYLE = ":"     # ratio = 1 reference: the two frames agree there
_UNITY = 1.0                # ... the value of that reference, and the y-range's anchor
_LINE_ALPHA = 0.7           # connecting lines, lighter than their markers
_YLIM_PAD = 0.15            # fractional headroom around the usable-shell range
_MARKER_SIZE = 5.0


@dataclass(frozen=True)
class FrameRatioConfig:
    """Knobs for the shell-averaged frame ratios.

    Attributes
    ----------
    min_denominator_sigma : float
        A shell enters the average only if the DENOMINATOR curve is detected
        there at this many times its own across-center SEM. Guards the pole
        where a curve crosses zero (module docstring). The SEM understates the
        true uncertainty (centers share tracers,
        `dvcorr.estimators.shell_dipole.center_standard_error`), so this
        threshold is permissive by construction -- it removes poles, it does
        not certify the survivors.
    resolved_min_radius : float or None
        Shells whose OUTER edge is at or below this radius are excluded from
        the average as unresolved. Pass `ShellConfig.min_radius` to drop the
        ``[0, min_radius)`` bin. None keeps every shell.
    """

    min_denominator_sigma: float = 3.0
    resolved_min_radius: float | None = None

    def __post_init__(self) -> None:
        if self.min_denominator_sigma < 0.0:
            raise ValueError(
                f"min_denominator_sigma must be >= 0, got {self.min_denominator_sigma}"
            )
        if self.resolved_min_radius is not None and self.resolved_min_radius < 0.0:
            raise ValueError(
                f"resolved_min_radius must be >= 0 or None, got {self.resolved_min_radius}"
            )


@dataclass(frozen=True)
class FrameCurve:
    """One frame's normalized curve, ready to be divided by another's.

    A thin, frame-agnostic view of a `NormalizedDipole` plus the shell edges it
    was measured on -- built by `frame_curve` from any of the three pipelines,
    which is what lets this module compare across sections without importing
    any of them.

    Attributes
    ----------
    label : str
        Short name used in ratio labels and legends, e.g. ``"3D"``, ``"obs"``.
    shell_edges : ndarray, shape (B + 1,)
    zeta_hat : ndarray, shape (B,)
    sem : ndarray, shape (B,)
    monopole_norm : ndarray, shape (B,)
    n_centers : int
    """

    label: str
    shell_edges: np.ndarray
    zeta_hat: np.ndarray
    sem: np.ndarray
    monopole_norm: np.ndarray
    n_centers: int


def frame_curve(
    label: str,
    shell_edges: np.ndarray,
    normalized: NormalizedDipole,
    n_centers: int,
) -> FrameCurve:
    """Adapt a `NormalizedDipole` into a `FrameCurve`.

    Parameters
    ----------
    label : str
    shell_edges : ndarray, shape (B + 1,)
    normalized : NormalizedDipole
        Any pipeline's normalized result -- observer frame, velocity frame or
        redshift space; this module does not care which.
    n_centers : int
        Carried for the legend only; the ratio never uses it.

    Returns
    -------
    FrameCurve
    """
    edges = np.asarray(shell_edges, dtype=float)
    if normalized.zeta_hat.shape[0] != edges.shape[0] - 1:
        raise ValueError(
            f"curve {label!r}: zeta_hat has {normalized.zeta_hat.shape[0]} bins but "
            f"shell_edges implies {edges.shape[0] - 1}"
        )
    return FrameCurve(
        label=label,
        shell_edges=edges,
        zeta_hat=np.asarray(normalized.zeta_hat, dtype=float),
        sem=np.asarray(normalized.sem, dtype=float),
        monopole_norm=np.asarray(normalized.monopole_norm, dtype=float),
        n_centers=int(n_centers),
    )


@dataclass(frozen=True)
class FrameRatio:
    """One frame's curve divided by another's, shell by shell.

    Attributes
    ----------
    label : str
        ``"<numerator>/<denominator>"``.
    numerator_label, denominator_label : str
    shell_edges : ndarray, shape (B + 1,)
        The numerator's and denominator's shared edges (`frame_ratio` checks
        they match before this is built), never assumed to be any other
        ratio's or curve's edges: different `FrameRatio`s in one
        `frame_ratios()` tuple can legitimately sit on different ladders
        (e.g. a redshift-space curve built from a different `ShellConfig`
        than the real-space ones it is not paired against).
    shell_centers : ndarray, shape (B,)
        `volume_weighted_shell_radii(shell_edges)` -- the plotting abscissa
        (CLAUDE.md hard rule 10), the same one every other figure in the
        package uses.
    ratio : ndarray, shape (B,)
        ``zeta_hat_num / zeta_hat_den``, every shell, poles included.
    monopole_ratio : ndarray, shape (B,)
        ``monopole_norm_num / monopole_norm_den``, every shell.
    usable : ndarray of bool, shape (B,)
        Shell is resolved AND the denominator is detected there
        (`FrameRatioConfig`). Only these enter the means.
    mean_ratio, mean_monopole_ratio : float
        Arithmetic mean over `usable` shells; NaN if none survive.
    naive_mean_ratio : float
        The same mean over resolved shells with the detection guard OFF --
        kept so the guard's effect can be read rather than trusted.
    changes_sign : bool
        The usable shells carry BOTH signs, so `mean_ratio` is an average
        across a zero crossing and is a cancellation between two regimes.
        When this is True, quote the two branch means instead.
    mean_where_negative, mean_where_positive : float
        `mean_ratio` restricted to the usable shells of one sign; NaN when
        that branch is empty. With `changes_sign` False, one of the two IS
        `mean_ratio` and the other is NaN.
    n_used, n_shells : int
    """

    label: str
    numerator_label: str
    denominator_label: str
    shell_edges: np.ndarray
    shell_centers: np.ndarray
    ratio: np.ndarray
    monopole_ratio: np.ndarray
    usable: np.ndarray
    mean_ratio: float
    mean_monopole_ratio: float
    naive_mean_ratio: float
    changes_sign: bool
    mean_where_negative: float
    mean_where_positive: float
    n_used: int
    n_shells: int


def _resolved_mask(cfg: FrameRatioConfig, shell_edges: np.ndarray) -> np.ndarray:
    """Shells whose OUTER edge clears `cfg.resolved_min_radius`, shape (B,)."""
    if cfg.resolved_min_radius is None:
        return np.ones(shell_edges.shape[0] - 1, dtype=bool)
    return shell_edges[1:] > cfg.resolved_min_radius


def frame_ratio(
    cfg: FrameRatioConfig,
    numerator: FrameCurve,
    denominator: FrameCurve,
) -> FrameRatio:
    """Divide one frame's curve by another's and average over usable shells.

    The division is bin-by-bin on curves that MUST share `shell_edges` -- this
    is checked, not assumed, because two runs configured with different
    `ShellConfig`s would otherwise divide unrelated radii and produce a
    perfectly smooth, perfectly meaningless curve.

    Shells where the denominator is consistent with zero are computed and
    returned (so the pole is visible) but excluded from `mean_ratio`; see the
    module docstring for why an unguarded mean is set by the binning rather
    than by the physics.

    Parameters
    ----------
    cfg : FrameRatioConfig
    numerator, denominator : FrameCurve

    Returns
    -------
    FrameRatio
    """
    if not np.allclose(numerator.shell_edges, denominator.shell_edges):
        raise ValueError(
            f"{numerator.label!r} and {denominator.label!r} were measured on different "
            "shell edges; a bin-by-bin ratio between them is not defined"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator.zeta_hat / denominator.zeta_hat
        monopole_ratio = numerator.monopole_norm / denominator.monopole_norm

    resolved = _resolved_mask(cfg, denominator.shell_edges)
    detected = np.abs(denominator.zeta_hat) >= cfg.min_denominator_sigma * denominator.sem
    usable = resolved & detected & np.isfinite(ratio)

    naive = resolved & np.isfinite(ratio)

    negative = usable & (ratio < 0.0)
    positive = usable & (ratio > 0.0)

    def _branch_mean(mask: np.ndarray) -> float:
        return float(ratio[mask].mean()) if mask.any() else float("nan")

    return FrameRatio(
        label=f"{numerator.label}/{denominator.label}",
        numerator_label=numerator.label,
        denominator_label=denominator.label,
        shell_edges=denominator.shell_edges,
        shell_centers=volume_weighted_shell_radii(denominator.shell_edges),
        ratio=ratio,
        monopole_ratio=monopole_ratio,
        usable=usable,
        mean_ratio=float(ratio[usable].mean()) if usable.any() else float("nan"),
        mean_monopole_ratio=(
            float(monopole_ratio[usable].mean()) if usable.any() else float("nan")
        ),
        naive_mean_ratio=float(ratio[naive].mean()) if naive.any() else float("nan"),
        changes_sign=bool(negative.any() and positive.any()),
        mean_where_negative=_branch_mean(negative),
        mean_where_positive=_branch_mean(positive),
        n_used=int(usable.sum()),
        n_shells=int(usable.shape[0]),
    )


def frame_ratios(
    cfg: FrameRatioConfig,
    curves: tuple[FrameCurve, ...],
) -> tuple[FrameRatio, ...]:
    """Every ordered pair ``curves[i] / curves[j]`` with ``i < j``.

    Ordering the curves from most to least information -- ``(3D, obs, z)`` --
    makes the three pairs come out as ``3D/obs``, ``3D/z``, ``obs/z``: each
    ratio is then "what the lower frame loses", and each is >= 1 wherever the
    losses are genuine losses.

    Parameters
    ----------
    cfg : FrameRatioConfig
    curves : tuple of FrameCurve
        At least two, with distinct labels.

    Returns
    -------
    tuple of FrameRatio, length ``n * (n - 1) / 2``
    """
    if len(curves) < 2:
        raise ValueError(f"need at least two curves to form a ratio, got {len(curves)}")
    labels = [c.label for c in curves]
    if len(set(labels)) != len(labels):
        raise ValueError(f"curve labels must be distinct, got {labels}")
    return tuple(
        frame_ratio(cfg, curves[i], curves[j])
        for i in range(len(curves))
        for j in range(i + 1, len(curves))
    )


def format_frame_ratio_table(ratios: tuple[FrameRatio, ...]) -> str:
    """Shell-by-shell ratios as a printable table, one block per ratio.

    Parameters
    ----------
    ratios : tuple of FrameRatio

    Returns
    -------
    str
    """
    lines: list[str] = []
    for ratio in ratios:
        lines.append(
            f"{ratio.label}   mean = {ratio.mean_ratio:.3f} over {ratio.n_used}/"
            f"{ratio.n_shells} shells   (unguarded mean {ratio.naive_mean_ratio:.3f})"
        )
        if ratio.changes_sign:
            lines.append(
                f"{'':>10}   SIGN CHANGE -- that mean is a cancellation; the branches are "
                f"{ratio.mean_where_negative:.3f} (negative) and "
                f"{ratio.mean_where_positive:.3f} (positive)"
            )
        lines.append(f"{'r_eff':>10}{'ratio':>12}{'monopole':>12}   used")
        for r, value, mono, used in zip(
            ratio.shell_centers, ratio.ratio, ratio.monopole_ratio, ratio.usable
        ):
            flag = "yes" if used else "NO (denominator ~ 0 or unresolved)"
            lines.append(f"{r:10.2f}{value:12.3f}{mono:12.3f}   {flag}")
        lines.append("")
    return "\n".join(lines)


def _usable_ylim(ratios: tuple[FrameRatio, ...], key: str) -> tuple[float, float]:
    """Padded y-limits spanning only the usable shells of every ratio.

    The poles are plotted but must not set the scale: one bin next to a zero
    crossing can be two orders of magnitude off and would flatten every real
    feature onto the zero line. Off-scale points clip, which is the honest
    rendering of "this shell has no ratio".
    """
    values = np.concatenate(
        [np.asarray(getattr(r, key))[r.usable] for r in ratios] + [np.array([_UNITY])]
    )
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    pad = _YLIM_PAD * max(hi - lo, abs(hi), 1.0)
    return lo - pad, hi + pad


def make_frame_ratio_figure(
    cfg: FrameRatioConfig,
    curves: tuple[FrameCurve, ...],
    ratios: tuple[FrameRatio, ...],
    *,
    binned: bool = False,
) -> DipoleFigures:
    """Two separate figures: dipole ratios, and their monopole companion.

    Returned together as a `DipoleFigures` (CLAUDE.md hard rule 6 -- the
    dipole ratio figure and its monopole companion are produced by this one
    call but are two separate images, never one merged two-panel figure).
    The x-axis is LINEAR in r even though the binning is logarithmic: log
    binning is a binning choice and never implies a log axis (house style,
    pinned by `tests/test_plot_wiring.py`).

    Two renderings, chosen by `binned` (CLAUDE.md hard rule 10):

    - **line view** (`binned=False`, the default): every ratio curve is a
      LINE joining one point per shell at its OWN `ratio.shell_centers` (the
      volume-weighted radius), never a step held flat across the bin. The
      connecting line BREAKS at a pole rather than running through it (a NaN
      in the y array leaves that segment undrawn): a segment drawn to an
      off-scale value is a stripe across the panel that reads as structure
      and is nothing but the excluded bin.
    - **binned view** (`binned=True`): each shell is drawn as its own flat
      segment over `[ratio.shell_edges[i], ratio.shell_edges[i+1])`, so the
      pole break is automatic -- there is no adjacent segment for an
      excluded shell to bleed into, and no NaN-masking is needed the way the
      line view's `joined` array needs it. Every shell edge is drawn as a
      faint guide line (`dvcorr.pipeline.shell_plotting.finish_shell_axis`),
      once per ratio's own ladder since different `FrameRatio`s in `ratios`
      are not assumed to share one (see below). Both panels' binned segments
      go through `dvcorr.pipeline.shell_plotting.draw_shell_series`: the
      ratio panel calls it twice per ratio, once on the usable shells
      (`ratio.ratio` with the pole shells masked to NaN) with filled
      markers, once on the pole shells (the complementary mask) with
      `markerfacecolor="none"`, both at `markersize=_MARKER_SIZE` -- the
      binned counterpart of the line view's filled-vs-open split. The
      monopole panel needs no such split (monopoles never cross zero) and
      goes through the primitive once.

    Different `FrameRatio`s in `ratios` are NOT assumed to share one ladder
    (`frame_ratio` only checks a match within one numerator/denominator
    pair, never across ratios), so each is drawn against its own
    `shell_centers` / `shell_edges`, never a borrowed one. Markers sit on
    the same abscissa and carry only the categorical usable-vs-pole
    distinction (filled vs open), in BOTH views.

    DIPOLE figure: one curve per ratio, with its own filled markers on
    shells that enter the mean and OPEN markers on shells excluded by the
    pole guard, the mean printed in the legend. The dotted line at 1 is
    where the two frames agree; the y-limits span the usable shells only, so
    points at a pole clip off-scale rather than compressing everything else
    (`_usable_ylim`).

    MONOPOLE figure: the matching monopole ratios, on the same colors. This
    is the "how much of the dipole figure is just the weight?" figure: the
    monopole is ``<|u|>`` times the shell occupancy, so where a dipole ratio
    tracks its monopole ratio the frames differ only in how much velocity
    they see, and where it departs, the difference is in the anisotropy
    itself.

    Parameters
    ----------
    cfg : FrameRatioConfig
    curves : tuple of FrameCurve
        Only for the title's center counts; every plotted value and its
        abscissa come from `ratios` (each carrying its own `shell_centers`).
        May be empty (paired with an empty `ratios`) -- this function never
        indexes into `curves` directly.
    ratios : tuple of FrameRatio
    binned : bool, default False
        False: the line view. True: the binned lookup companion.

    Returns
    -------
    DipoleFigures
    """
    view = " -- binned" if binned else ""
    fig_ratio, ax_ratio = plt.subplots(figsize=_DIPOLE_FIGSIZE)
    fig_mono, ax_mono = plt.subplots(figsize=_MONOPOLE_FIGSIZE)

    for i, ratio in enumerate(ratios):
        color = _RATIO_COLORS[i % len(_RATIO_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        r = ratio.shell_centers

        # The legend entry carries the number the ratio exists to produce --
        # and, where the curve crosses zero, refuses to let one averaged number
        # stand for two regimes that disagree in sign.
        if ratio.changes_sign:
            text = (f"{ratio.label}   mean = {ratio.mean_ratio:.2f} = "
                    f"{ratio.mean_where_negative:.2f} | {ratio.mean_where_positive:.2f} "
                    f"across the sign change  ({ratio.n_used}/{ratio.n_shells} shells)")
        else:
            text = (f"{ratio.label}   mean = {ratio.mean_ratio:.2f}  "
                    f"({ratio.n_used}/{ratio.n_shells} shells)")

        if binned:
            # Each shell is its own segment, so the pole break is automatic
            # -- unlike the line view, there is no NaN-masked `joined` array
            # to build. Usable and pole shells are two masked series through
            # the shared primitive -- filled vs. `markerfacecolor="none"` --
            # the binned counterpart of the line view's `fillstyle` split
            # below. Only the usable call carries the legend label, so the
            # ratio gets exactly one legend entry.
            usable_ratio = np.where(ratio.usable, ratio.ratio, np.nan)
            pole_ratio = np.where(ratio.usable, np.nan, ratio.ratio)
            draw_shell_series(
                ax_ratio, ratio.shell_edges, usable_ratio, binned=True,
                color=color, marker=marker, markersize=_MARKER_SIZE, label=text,
            )
            draw_shell_series(
                ax_ratio, ratio.shell_edges, pole_ratio, binned=True,
                color=color, marker=marker, markersize=_MARKER_SIZE,
                markerfacecolor="none",
            )
            finish_shell_axis(ax_ratio, ratio.shell_edges, binned=True)
            finish_shell_axis(ax_mono, ratio.shell_edges, binned=True)
        else:
            # The connecting line BREAKS at a pole rather than spiking through
            # it: a segment drawn to an off-scale point is a stripe across the
            # panel that reads as structure and is nothing but the excluded bin.
            joined = np.where(ratio.usable, ratio.ratio, np.nan)
            ax_ratio.plot(r, joined, "-", color=color, alpha=_LINE_ALPHA, zorder=2)
            for mask, fill in ((ratio.usable, _USABLE_FILL), (~ratio.usable, _UNUSABLE_FILL)):
                if mask.any():
                    ax_ratio.plot(
                        r[mask], ratio.ratio[mask], marker, color=color,
                        fillstyle=fill, ms=_MARKER_SIZE, ls="none", zorder=3,
                    )
            ax_ratio.plot([], [], marker + "-", color=color, ms=_MARKER_SIZE, label=text)

        # Monopole panel: no pole guard here (monopoles never cross zero, see
        # module docstring), so this is a plain series through the shared
        # primitive, carrying its own legend label.
        if binned:
            draw_shell_series(
                ax_mono, ratio.shell_edges, ratio.monopole_ratio, binned=True,
                color=color, marker=marker, markersize=_MARKER_SIZE, label=ratio.label,
            )
        else:
            ax_mono.plot(r, ratio.monopole_ratio, marker + "-", color=color,
                         ms=_MARKER_SIZE, label=ratio.label)

    for ax in (ax_ratio, ax_mono):
        ax.axhline(_UNITY, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH, ls=_UNITY_LINE_STYLE)
        # finish_shell_axis (called per-ratio above, binned view only) already
        # pins x=0; this is the explicit fallback for the empty-`ratios` case
        # and reproduces the line view's original unconditional call exactly.
        ax.set_xlim(left=0.0)
        ax.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
        ax.grid(alpha=_GRID_ALPHA)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    title_info = (
        "(" + ", ".join(f"{c.label}: N$_c$={c.n_centers:,}" for c in curves) + ")"
    )

    ax_ratio.set_ylim(*_usable_ylim(ratios, "ratio"))
    ax_ratio.set_ylabel(r"$\hat\zeta_1$ ratio", color=_LABEL_COLOR)
    ax_ratio.set_title(
        f"frame-to-frame ratios of the velocity-centered dipole{view}   {title_info}\n"
        f"open markers: denominator below {cfg.min_denominator_sigma:g}$\\sigma$ "
        "-- a pole, plotted but not averaged",
        fontsize=9,
    )
    ax_mono.set_ylabel(r"$\hat\zeta_0$ ratio", color=_LABEL_COLOR)
    ax_mono.set_title(
        f"frame-to-frame ratios of the monopole{view}   {title_info}", fontsize=9,
    )
    # An empty `ratios` (paired with empty `curves`) has nothing to put in a
    # legend -- `ncol=len(ratios)` would be `ncol=0`, which matplotlib
    # refuses unconditionally. Skip both legends rather than pass a
    # nonsensical column count for a figure with no curves on it.
    if ratios:
        ax_ratio.legend(fontsize="small")
        ax_mono.legend(fontsize="small", ncol=len(ratios))
    fig_ratio.tight_layout()
    fig_mono.tight_layout()

    return DipoleFigures(dipole=fig_ratio, monopole=fig_mono)
