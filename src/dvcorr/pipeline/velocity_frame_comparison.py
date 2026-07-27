"""
velocity_frame_comparison.py
------------------------------
Reusable pipeline stage functions for the two-frame comparison: the
observer-frame velocity-centered dipole zeta_1
(`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`) against the
observer-free velocity-frame dipole
(`dvcorr.estimators.velocity_frame_dipole.velocity_frame_shell_dipole`), run
on the IDENTICAL shared center set so the only difference between the two
results is the axis and the velocity scalar (see
`dvcorr.estimators.velocity_frame_dipole`'s module docstring for the physics
motivation).

This is the single source of truth consumed by
`scripts/plot_velocity_frame_comparison.py` (CLAUDE.md's "one library, two
thin consumers" model). This module deliberately does NOT reimplement
`dvcorr.pipeline.velocity_centered`'s earlier pipeline stages -- `RunConfig`
(subclassed below, not copied), `NormalizedDipole`, `normalize_result`, and
`normalize_stacked_dipole` are all IMPORTED from there. `load_and_carve`,
`draw_candidates`, and `global_number_density` are likewise that module's,
consumed directly by the SCRIPT (this module's stage functions start one
step later, from already-loaded candidate arrays, since carving and
subsampling do not differ between the two frames).

Matplotlib backend discipline
------------------------------
Importing this module does NOT select a matplotlib backend: `matplotlib.use`
is never called here, only in the thin script's own module body (before it
imports this module) -- identical discipline to
`dvcorr.pipeline.velocity_centered`.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from dvcorr.estimators.shell_dipole import (
    VelocityCenteredShellDipoleResult,
    core_center_mask,
    velocity_centered_shell_dipole,
)
from dvcorr.estimators.velocity_frame_dipole import (
    VelocityFrameShellDipoleResult,
    velocity_frame_shell_dipole,
)
from dvcorr.geometry import unit_vector

# Stage functions and the styling constants reused verbatim from the
# observer-frame pipeline. _COLOR_SIGNAL is imported as `_COLOR_OBS` (below)
# so the obs-frame curve in this comparison figure is the visually identical
# blue used for zeta_hat in the single-frame figure; the neutral styling
# constants (_BAND_ALPHA, _COLOR_ZERO_LINE, _GRID_ALPHA, _HEIGHT_RATIOS,
# _LABEL_COLOR, _ZERO_LINE_WIDTH, _FIGSIZE) are reused the same way. Only the
# velocity-frame color and the two "null" tints (defined below) are new to
# this module.
from dvcorr.pipeline.velocity_centered import (
    _BAND_ALPHA,
    _COLOR_ZERO_LINE,
    _FIGSIZE,
    _GRID_ALPHA,
    _HEIGHT_RATIOS,
    _LABEL_COLOR,
    _ZERO_LINE_WIDTH,
    NormalizedDipole,
    RunConfig,
    normalize_result,
    normalize_stacked_dipole,
    shell_dipole_norm_scale,
)
from dvcorr.pipeline.velocity_centered import _COLOR_SIGNAL as _COLOR_OBS

_COLOR_OBS_NULL = "#a9c9ee"   # lighter tint of _COLOR_OBS: the obs-frame's own null (u-shuffle)
_COLOR_VEL = "#7b2d8e"        # distinct hue for the velocity frame's curves
_COLOR_VEL_NULL = "#cdb0d8"   # lighter tint of _COLOR_VEL: the vel-frame's own null (random-axis)
_SCATTER_ALPHA = 0.35         # low-alpha individual-center scatter, angle diagnostic top panel
_FIGSIZE_ANGLE = (8.0, 8.0)   # angle-diagnostic figure size, matches the comparison figure
_N_ANGLE_BINS = 12            # equal-width bins over [0, pi] for both angle-diagnostic panels


@dataclass
class ComparisonRunConfig(RunConfig):
    """`RunConfig`, plus the knobs the two-frame comparison itself needs.

    Everything shared between the two frames -- `sub_volume_radius`,
    `shells`, `n_candidate_centers`, `seed`, `shuffle_seed`, `min_center_speed`
    -- stays on the inherited `RunConfig`; this subclass adds only the
    comparison-specific outputs and the axis-null's own seed. The split is
    deliberate: `min_center_speed` lives on `RunConfig` because it filters
    the SHARED center set consumed by both estimators (moving it here would
    make it look like a comparison-only knob, when in fact
    `dvcorr.estimators.velocity_frame_dipole.velocity_frame_shell_dipole`
    alone already depends on it, with or without a comparison ever running).
    The three fields below, by contrast, have no meaning outside this
    comparison pipeline.

    Attributes
    ----------
    comparison_output_name : str
        Output filename for `make_comparison_figure`'s PNG, written under
        `PathsConfig().output_dir`.
    angle_diagnostic_output_name : str
        Output filename for `make_angle_diagnostic_figure`'s PNG.
    axis_null_seed : int
        Seed for `run_random_axis_null`'s isotropic random directions, kept
        DISTINCT from both `seed` (candidate subsampling) and `shuffle_seed`
        (the observer frame's velocity-shuffle null) so the random-axis
        null's randomness never silently reuses either of those streams.
    """

    comparison_output_name: str = "velocity_frame_comparison.png"
    angle_diagnostic_output_name: str = "velocity_frame_angle_diagnostic.png"
    axis_null_seed: int = 44


@dataclass(frozen=True)
class SharedCenterSet:
    """The single center set both frames are run on.

    Built by `select_shared_centers` in two cuts, funnel-style:
    `n_candidates` (input) -> core cut (`core_center_mask`) -> `n_core` ->
    speed floor (`cfg.min_center_speed`) -> `n_centers` (final). Selecting
    ONCE here and handing the identical `s_centers` / `v_centers` arrays to
    both `velocity_centered_shell_dipole` and `velocity_frame_shell_dipole`
    (see `run_both_frames`) is what guarantees the only difference between
    the two frames' results is the axis and the scalar -- not a difference
    in which centers were measured.

    Attributes
    ----------
    s_centers : ndarray, shape (N_c, 3)
        Surviving center positions, comoving h^-1 Mpc.
    v_centers : ndarray, shape (N_c, 3)
        Surviving center velocities, km/s, same row order as `s_centers`.
    n_candidates : int
        Number of candidates handed in, before either cut.
    n_core : int
        Number surviving `core_center_mask` alone (before the speed floor).
    n_dropped_slow : int
        `n_core - n_centers`: the number of otherwise-valid centers dropped
        by the speed floor alone. RETURNED as a field, not merely printed,
        so a caller (or a test) can assert on it directly.
    n_centers : int
        Final surviving count, `s_centers.shape[0]`.
    """

    s_centers: np.ndarray
    v_centers: np.ndarray
    n_candidates: int
    n_core: int
    n_dropped_slow: int
    n_centers: int


def select_shared_centers(
    cfg: ComparisonRunConfig,
    s_candidates: np.ndarray,
    v_candidates: np.ndarray,
    observer: np.ndarray,
) -> SharedCenterSet:
    """Apply the core cut, then the speed floor, ONCE, for both frames to share.

    Two sequential cuts:

    1. `core_center_mask(s_candidates, cfg.sub_volume_radius,
       cfg.shells.max_radius, observer=observer)` -- the IDENTICAL function
       both `velocity_centered_shell_dipole` and `velocity_frame_shell_dipole`
       apply internally (see `run_both_frames`'s docstring on why re-applying
       it there is idempotent). `core_margin = cfg.shells.max_radius` matches
       `velocity_centered_shell_dipole`'s own default (`core_margin=None` ->
       `shell_edges[-1]` = r_max), so a surviving center's full shell fits
       inside the carved sub-volume.
    2. `speeds = |v_core|; keep = (speeds > 0.0) & (speeds >= cfg.min_center_speed)`
       -- the floor that gives every surviving center a well-defined flow
       direction (see
       `dvcorr.estimators.velocity_frame_dipole.velocity_frame_shell_dipole`'s
       zero-speed `ValueError`, which this floor exists to prevent a caller
       from ever triggering). The explicit `speeds > 0.0` conjunct matters
       specifically for `cfg.min_center_speed == 0.0`: `speeds >=
       cfg.min_center_speed` is trivially true for every speed (a norm can
       never be negative), so without it the documented "0.0 means drop only
       exactly-zero-speed centers" contract (`RunConfig.min_center_speed`'s
       docstring) would be false -- a floor of 0.0 would drop nothing at
       all, including the exact-zero centers the floor exists to guard
       against. For any `min_center_speed > 0.0` the extra conjunct is a
       no-op (a positive floor already implies `speeds > 0.0`).

    Parameters
    ----------
    cfg : ComparisonRunConfig
    s_candidates, v_candidates : ndarray, shape (N_cand, 3)
        Candidate positions/velocities, e.g. from
        `dvcorr.pipeline.velocity_centered.draw_candidates`.
    observer : ndarray, shape (3,)

    Returns
    -------
    SharedCenterSet

    Raises
    ------
    RuntimeError
        If `s_candidates` is empty, or if zero centers survive both cuts.
        The empty-input case is checked BEFORE the core-cut survival
        percentage is printed, so a zero-candidate call raises this
        `RuntimeError` rather than a bare `n / 0` `ZeroDivisionError` with a
        misleading traceback (the same hazard `load_and_carve`'s docstring
        calls out and guards against).
    """
    n_candidates = s_candidates.shape[0]
    if n_candidates == 0:
        raise RuntimeError("select_shared_centers: s_candidates is empty.")

    core_mask = core_center_mask(
        s_candidates, cfg.sub_volume_radius, cfg.shells.max_radius, observer=observer
    )
    s_core = s_candidates[core_mask]
    v_core = v_candidates[core_mask]
    n_core = s_core.shape[0]

    speeds = np.linalg.norm(v_core, axis=1)
    keep = (speeds > 0.0) & (speeds >= cfg.min_center_speed)
    s_centers = s_core[keep]
    v_centers = v_core[keep]
    n_centers = s_centers.shape[0]
    n_dropped_slow = n_core - n_centers

    print(
        f"n_candidates = {n_candidates} -> core cut -> n_core = {n_core} "
        f"({100.0 * n_core / n_candidates:.1f}% survive) -> speed floor "
        f"(min_center_speed = {cfg.min_center_speed} km/s) -> n_centers = "
        f"{n_centers} ({n_dropped_slow} dropped as too slow)"
    )
    if n_centers == 0:
        raise RuntimeError(
            "select_shared_centers: zero centers survived the core cut and "
            "speed floor combined; widen sub_volume_radius, shrink the "
            "shell range, or lower min_center_speed."
        )

    return SharedCenterSet(
        s_centers=s_centers,
        v_centers=v_centers,
        n_candidates=n_candidates,
        n_core=n_core,
        n_dropped_slow=n_dropped_slow,
        n_centers=n_centers,
    )


def run_both_frames(
    cfg: ComparisonRunConfig,
    centers: SharedCenterSet,
    s_tracers: np.ndarray,
    observer: np.ndarray,
) -> FrameRunResults:
    """Run both estimators, plus the velocity frame's random-axis null.

    Calls `velocity_centered_shell_dipole` and `velocity_frame_shell_dipole`
    on the IDENTICAL `centers.s_centers` / `centers.v_centers` arrays, with
    identical `shell_edges`, `sub_volume_radius`, `core_margin` (left at its
    default `None` -> r_max, matching `select_shared_centers`'s own cut) and
    `observer`. Both estimators re-apply `core_center_mask` internally; since
    `centers` was already core-cut by `select_shared_centers`, that
    re-application is IDEMPOTENT and ORDER-PRESERVING (the same boolean mask,
    computed the same way, on an input that already satisfies it, keeps
    every row and reorders nothing) -- which is precisely what makes the two
    results ROW-ALIGNED per center. This is asserted explicitly below rather
    than assumed.

    Also runs `run_random_axis_null` on the same `centers`, so a single call
    to this function returns everything `normalize_comparison` needs.

    Parameters
    ----------
    cfg : ComparisonRunConfig
    centers : SharedCenterSet
        From `select_shared_centers`.
    s_tracers : ndarray, shape (N_t, 3)
        Density tracers -- ALL carved halos.
    observer : ndarray, shape (3,)

    Returns
    -------
    FrameRunResults

    Raises
    ------
    RuntimeError
        If either estimator's `n_centers` does not equal `centers.n_centers`
        -- the row-alignment invariant this whole comparison depends on.
    """
    obs_result = velocity_centered_shell_dipole(
        s_centers=centers.s_centers,
        v_centers=centers.v_centers,
        s_tracers=s_tracers,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )
    vel_result = velocity_frame_shell_dipole(
        s_centers=centers.s_centers,
        v_centers=centers.v_centers,
        s_tracers=s_tracers,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )

    if obs_result.n_centers != centers.n_centers or vel_result.n_centers != centers.n_centers:
        raise RuntimeError(
            "run_both_frames: row-alignment invariant violated -- "
            f"centers.n_centers = {centers.n_centers}, "
            f"obs_result.n_centers = {obs_result.n_centers}, "
            f"vel_result.n_centers = {vel_result.n_centers}. `centers` was "
            "already core-cut by select_shared_centers, so re-applying "
            "core_center_mask inside each estimator should be idempotent "
            "and keep every row; a mismatch means that assumption broke "
            "(e.g. cfg.sub_volume_radius or cfg.shells changed between the "
            "two calls) and the two frames' per-center arrays can no "
            "longer be compared row-for-row."
        )

    vel_null_result = run_random_axis_null(cfg, centers, s_tracers, observer)

    return FrameRunResults(
        obs_result=obs_result,
        vel_result=vel_result,
        vel_null_result=vel_null_result,
        centers=centers,
    )


def run_random_axis_null(
    cfg: ComparisonRunConfig,
    centers: SharedCenterSet,
    s_tracers: np.ndarray,
    observer: np.ndarray,
) -> VelocityFrameShellDipoleResult:
    """The velocity frame's null: randomize the AXIS, keep the speed and position.

    v_null_alpha = |v_alpha| * random_unit_direction_alpha, with the random
    directions drawn isotropically via
    `unit_vector(rng.normal(size=(N_c, 3)))` and
    `rng = np.random.default_rng(cfg.axis_null_seed)`. Reruns
    `velocity_frame_shell_dipole` on `centers.s_centers` with this replacement
    velocity array -- a second estimator pass, unlike the observer frame's
    shuffle null, deliberately (see below).

    WHY THIS NULL, AND NOT A SCALAR SHUFFLE -- important, read before
    "simplifying" this to match `normalize_result`'s pattern
    --------------------------------------------------------------------------
    `normalize_result`'s velocity-shuffle null permutes `per_center_u` among
    centers and recombines with the UNCHANGED `per_center_amplitude` -- no
    second estimator pass needed, because in the observer frame the axis
    (n_hat_V,alpha, fixed by each center's POSITION) and the scalar (u_alpha,
    the center's OWN radial velocity) are logically separable: the amplitude
    A_alpha,b depends only on the axis, so permuting only the scalar breaks
    the alpha <-> A_alpha,b pairing while leaving every A_alpha,b intact.
    That works as a null specifically because u_alpha averages to ~0 across
    an isotropic sample of centers (the sign of a radial projection is as
    likely to be positive as negative), so a random recombination collapses
    the stacked sum toward zero.

    The velocity frame has NO such separation available. Its axis
    z_hat_alpha = v_hat_alpha is built from the SAME velocity vector that
    supplies the scalar |v_alpha| -- the statistic is SELF-ALIGNED by
    construction. Permuting only |v_alpha| among centers would leave every
    center's z_hat_alpha (and therefore every A_alpha,b) exactly as it was;
    the recombined sum would be Sigma_alpha |v_permuted,alpha| * A_alpha,b,
    which is essentially the SAME quantity as the real signal, because
    |v_alpha| is positive-definite and cannot cancel the way u_alpha does --
    permuting a set of positive numbers among a set of amplitudes that never
    changed does not scramble the alignment that produced the signal in the
    first place. A scalar permutation is therefore not a null here at all;
    it would retain N_c * <|v|> * <A>, essentially the signal itself.

    Randomizing the AXIS instead is the correct guard against
    align-then-measure bias: it breaks the one thing that makes the
    statistic self-aligned (the axis's dependence on the SAME vector as the
    scalar) while leaving the speed and the tracer geometry untouched, and it
    genuinely does collapse to zero for an uncorrelated axis (see
    tests/test_velocity_frame_dipole.py, item 5's clustered-configuration
    assertion). It costs one extra estimator pass; that cost is deliberate,
    not an oversight -- there is no cheaper construction that is still a
    valid null for this frame.

    Parameters
    ----------
    cfg : ComparisonRunConfig
    centers : SharedCenterSet
        From `select_shared_centers` (or, within `run_both_frames`, already
        confirmed row-aligned).
    s_tracers : ndarray, shape (N_t, 3)
    observer : ndarray, shape (3,)

    Returns
    -------
    VelocityFrameShellDipoleResult
    """
    rng = np.random.default_rng(cfg.axis_null_seed)
    speeds = np.linalg.norm(centers.v_centers, axis=1)
    random_directions = unit_vector(rng.normal(size=centers.v_centers.shape))
    v_null = speeds[:, None] * random_directions

    return velocity_frame_shell_dipole(
        s_centers=centers.s_centers,
        v_centers=v_null,
        s_tracers=s_tracers,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )


@dataclass(frozen=True)
class FrameRunResults:
    """Both estimators' raw results, plus the velocity frame's null, row-aligned.

    Attributes
    ----------
    obs_result : VelocityCenteredShellDipoleResult
        Observer-frame zeta_1 result.
    vel_result : VelocityFrameShellDipoleResult
        Velocity-frame result, on the SAME `centers`.
    vel_null_result : VelocityFrameShellDipoleResult
        Velocity-frame result on the same centers with axes replaced by
        isotropic random directions (`run_random_axis_null`) -- the null
        this frame requires; see that function's docstring for why a scalar
        permutation would not be a valid null here.
    centers : SharedCenterSet
        The shared center set both frames (and the null) were run on.
    """

    obs_result: VelocityCenteredShellDipoleResult
    vel_result: VelocityFrameShellDipoleResult
    vel_null_result: VelocityFrameShellDipoleResult
    centers: SharedCenterSet


def normalize_velocity_frame_result(
    result: VelocityFrameShellDipoleResult,
    null_result: VelocityFrameShellDipoleResult,
    n_bar: float,
) -> NormalizedDipole:
    """Thin delegate to `normalize_stacked_dipole` for the velocity frame.

    For this frame, the returned `NormalizedDipole.zeta_hat_shuffle` (and
    `.sem_shuffle`) hold the RANDOM-AXIS null (`run_random_axis_null`'s
    output), not a scalar permutation -- the field name is inherited
    unchanged from the observer-frame path (`normalize_stacked_dipole`'s
    docstring already documents that it names "whatever null the caller
    built"), so it is repeated here explicitly to avoid the field name being
    misread as a promise about construction.

    Parameters
    ----------
    result : VelocityFrameShellDipoleResult
        The signal run, e.g. `FrameRunResults.vel_result`.
    null_result : VelocityFrameShellDipoleResult
        The random-axis null run, e.g. `FrameRunResults.vel_null_result`.
    n_bar : float
        Global tracer number density over the sub-volume.

    Returns
    -------
    NormalizedDipole
    """
    return normalize_stacked_dipole(
        shell_edges=result.shell_edges,
        dipole=result.dipole,
        monopole=result.monopole,
        per_center_dipole=result.per_center_dipole,
        null_dipole=null_result.dipole,
        null_per_center_dipole=null_result.per_center_dipole,
        n_centers=result.n_centers,
        n_bar=n_bar,
    )


@dataclass(frozen=True)
class FrameComparison:
    """Normalized curves for both frames, plus the per-center angle breakdown.

    Attributes
    ----------
    obs : NormalizedDipole
        Observer-frame normalized dipole/monopole/null, from `normalize_result`.
    vel : NormalizedDipole
        Velocity-frame normalized dipole/monopole/null, from
        `normalize_velocity_frame_result`.
    shell_centers : ndarray, shape (B,)
        Shell midpoints, h^-1 Mpc, shared by both frames (they were run on
        identical `shell_edges`).
    per_center_delta : ndarray, shape (N_c,)
        `vel_result.per_center_axis_angle`, RADIANS, in [0, pi] -- the angle
        between each center's own flow direction and its observer line of
        sight (see
        `dvcorr.estimators.velocity_frame_dipole.VelocityFrameShellDipoleResult
        .per_center_axis_angle`).
    per_center_dipole_difference : ndarray, shape (N_c,)
        Each center's OWN normalized dipole curve, averaged over shells, then
        differenced between frames (vel minus obs) -- see
        `normalize_comparison`'s docstring for the exact construction and why
        a shell-average summary, rather than a single shell, is used.
    """

    obs: NormalizedDipole
    vel: NormalizedDipole
    shell_centers: np.ndarray
    per_center_delta: np.ndarray
    per_center_dipole_difference: np.ndarray


def normalize_comparison(
    cfg: ComparisonRunConfig,
    results: FrameRunResults,
    n_bar: float,
) -> FrameComparison:
    """Normalize both frames and build the per-center frame-gap breakdown.

    The stacked curves: `results.obs_result` normalizes via `normalize_result`
    (the UNCHANGED path, its own velocity-shuffle null seeded by
    `cfg.shuffle_seed`); `results.vel_result` normalizes via
    `normalize_velocity_frame_result` (the random-axis null,
    `results.vel_null_result`).

    The per-center breakdown: for each surviving center alpha, its own
    normalized dipole curve -- `result.per_center_dipole[alpha, :] *
    norm_scale_b`, computed via
    `dvcorr.pipeline.velocity_centered.shell_dipole_norm_scale` -- the exact
    SAME per-shell scale, from its ONE home, that `normalize_stacked_dipole`
    uses for the stack (not a re-derived copy: the two used to be
    independent copies of the same three lines, a drift risk this shared
    helper removes) -- but deliberately WITHOUT the `/ n_centers` stacking
    average, averaged over shells to a single number, then differenced
    between frames:

        norm_scale_b      = shell_dipole_norm_scale(shell_edges, n_bar)  # (B,)
        per_center_curve  = per_center_dipole * norm_scale_b             # (N_c, B)
        summary_alpha     = per_center_curve.mean(axis=1)                # (N_c,)
        difference        = summary_vel - summary_obs                   # (N_c,)

    Why a shell-average SUMMARY per center, rather than picking one shell or
    keeping the full (N_c, B) breakdown: it collapses cleanly back to the
    plotted stack. Because `zeta_hat_b = (dipole_b / n_centers) *
    norm_scale_b = mean_alpha(per_center_dipole[alpha, b]) * norm_scale_b`,
    averaging `summary_alpha` over centers recovers the shell-average of the
    plotted `zeta_hat` curve:

        mean_alpha(summary_alpha) == mean_b(zeta_hat_b)

    i.e. the per-center decomposition SUMS BACK to the plotted curve -- a
    center with a large `|difference|` is a center whose OWN contribution to
    the stack disagrees between frames, not an artifact of how the summary
    was built.

    Parameters
    ----------
    cfg : ComparisonRunConfig
    results : FrameRunResults
    n_bar : float
        Global tracer number density over the sub-volume.

    Returns
    -------
    FrameComparison
    """
    obs = normalize_result(results.obs_result, n_bar, cfg.shuffle_seed)
    vel = normalize_velocity_frame_result(results.vel_result, results.vel_null_result, n_bar)

    # Same per-shell scale normalize_stacked_dipole uses for the stack, from
    # its one home (see shell_dipole_norm_scale's docstring) -- not re-derived.
    norm_scale_b = shell_dipole_norm_scale(results.obs_result.shell_edges, n_bar)  # (B,)

    obs_per_center_curve = results.obs_result.per_center_dipole * norm_scale_b  # (N_c, B)
    vel_per_center_curve = results.vel_result.per_center_dipole * norm_scale_b  # (N_c, B)

    summary_obs = obs_per_center_curve.mean(axis=1)  # (N_c,)
    summary_vel = vel_per_center_curve.mean(axis=1)  # (N_c,)

    return FrameComparison(
        obs=obs,
        vel=vel,
        shell_centers=results.obs_result.shell_centers,
        per_center_delta=results.vel_result.per_center_axis_angle,
        per_center_dipole_difference=summary_vel - summary_obs,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_comparison_figure(
    cfg: ComparisonRunConfig,
    results: FrameRunResults,
    comparison: FrameComparison,
) -> plt.Figure:
    """The main deliverable: two panels comparing the observer and velocity frames.

    Two panels sharing the x-axis (separation r), height ratios (3, 1) --
    identical layout convention to
    `dvcorr.pipeline.velocity_centered.make_figure`. Builder only: never
    saves, never calls `plt.show`.

    TOP panel: `zeta_hat_obs` and `zeta_hat_vel` overlaid, each with its
    across-center standard-error band (`fill_between`) and each with its OWN
    null, dashed, in a lighter tint of its own color -- the obs-frame null is
    the velocity shuffle (`normalize_result`'s permutation of per_center_u),
    the vel-frame null is the random-axis re-run (`run_random_axis_null`);
    the legend spells out "u-shuffle" vs. "random-axis" so the two nulls are
    never confused with each other.

    Standard-error caveat, restated here because it is easy to miss on a
    figure (see `center_standard_error`'s own docstring for the full
    argument): centers are NOT independent samples -- two centers closer
    together than a shell's radius share tracers, and on large scales the
    per-center measurements are correlated through shared large-scale
    velocity coherence -- so the plotted bands UNDERSTATE the true
    uncertainty for BOTH frames. Mock catalogs (the not-yet-built
    `dvcorr.mocks` arm) are the eventual, calibrated fix; this band is a
    diagnostic ("how noisy is the stack"), not a confidence interval.

    BOTTOM panel (monopole, ell=0, for both frames): a TWIN y-axis, the
    obs-frame monopole on the LEFT axis, the velocity-frame monopole on the
    RIGHT axis, each y-label colored to match its curve. This is a
    necessity, not a stylistic choice: the obs-frame monopole carries the
    finite-distance 2r/3R leakage documented in `geometry.mu_cosine` and
    `shell_dipole.py` -- an r-DEPENDENT trend, growing with shell radius r at
    fixed observer distance R -- while the velocity-frame monopole has no
    observer distance R anywhere in its construction (see
    `dvcorr.estimators.velocity_frame_dipole`'s module docstring, Monopole
    section) and so has NO such leakage term. It sits at roughly the mean
    halo speed <|v|> (hundreds of km/s), not at zero, because |v| is
    positive-definite and cannot cancel across centers the way the
    observer-frame's signed u_alpha does. A single shared y-axis would either
    crush the near-zero obs-frame curve flat or clip the vel-frame curve
    off-scale -- either way hiding the PRESENCE-VS-ABSENCE of the r-dependent
    trend, which is exactly the core finite-distance diagnostic this
    comparison exists to show.

    What BOTH plotted curves decline with, and what that does NOT mean: on a
    clustered tracer field each monopole is its mean scalar times the
    occupancy ratio 1 + xi_hh(r), so both curves fall steeply at small r from
    halo clustering alone -- the velocity frame is NOT flat here, and its
    decline is not a systematic. The frames separate only once that shared
    factor is divided out; on the first MDPL2 run
    (notebooks/06_velocity_frame_comparison.ipynb) that leaves the obs-frame
    residual trending -13.6 -> -2.7 km/s (the finite-distance leakage) while
    the vel-frame residual is flat at 546.7 -> 504.0 km/s against
    <|v_alpha|> = 504.5 km/s. Read the panel as "which curve keeps a trend
    after clustering is accounted for", not "which curve is flat on the page".

    Reading the TOP panel's amplitude gap: a larger vel-frame amplitude is
    NOT automatically a physically larger signal. Besides axis rotation and
    the |v| vs. u effect (see
    `dvcorr.estimators.velocity_frame_dipole`'s module docstring), the
    velocity frame's amplitude is inflated relative to the observer frame's
    by construction -- it is SELF-ALIGNED (the axis comes from the same
    vector as the scalar), so it does not suffer the observer frame's
    <cos_theta^2> ~= 1/3-type dilution from a position-fixed axis that only
    loosely tracks the true flow direction. `zeta_hat_vel_null`
    (random-axis) shows where the NULL floor sits, but a vel-frame curve
    well above its own null is not, by itself, evidence the excess is
    finite-distance or projection related rather than this construction
    effect -- see the module docstring's third paragraph on this.

    Parameters
    ----------
    cfg : ComparisonRunConfig
    results : FrameRunResults
    comparison : FrameComparison

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_dipole, ax_mono_obs) = plt.subplots(
        2, 1, figsize=_FIGSIZE, sharex=True, gridspec_kw={"height_ratios": _HEIGHT_RATIOS}
    )

    r = comparison.shell_centers

    # --- top panel: the two dipoles, each with its SEM band and its own null
    ax_dipole.fill_between(
        r, comparison.obs.zeta_hat - comparison.obs.sem, comparison.obs.zeta_hat + comparison.obs.sem,
        color=_COLOR_OBS, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.obs.zeta_hat, "o-", color=_COLOR_OBS,
        label=fr"$\hat\zeta_1^{{obs}}$  (N$_c$={results.obs_result.n_centers})",
    )
    ax_dipole.plot(
        r, comparison.obs.zeta_hat_shuffle, "o--", color=_COLOR_OBS_NULL,
        label="obs null (u-shuffle)",
    )

    ax_dipole.fill_between(
        r, comparison.vel.zeta_hat - comparison.vel.sem, comparison.vel.zeta_hat + comparison.vel.sem,
        color=_COLOR_VEL, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.vel.zeta_hat, "s-", color=_COLOR_VEL,
        label=fr"$\hat\zeta_1^{{vel}}$  (N$_c$={results.vel_result.n_centers})",
    )
    ax_dipole.plot(
        r, comparison.vel.zeta_hat_shuffle, "s--", color=_COLOR_VEL_NULL,
        label="vel null (random-axis)",
    )

    ax_dipole.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_dipole.set_ylabel(r"$\hat\zeta_1$  [km/s]", color=_LABEL_COLOR)
    ax_dipole.legend(fontsize="small")
    ax_dipole.grid(alpha=_GRID_ALPHA)
    ax_dipole.spines["top"].set_visible(False)
    ax_dipole.spines["right"].set_visible(False)
    ax_dipole.set_title(
        "observer frame vs. velocity frame  "
        f"(R_sub={cfg.sub_volume_radius:.0f} h$^{{-1}}$Mpc, "
        f"r_max={cfg.shells.max_radius:.0f} h$^{{-1}}$Mpc, "
        f"N$_c$={results.centers.n_centers})\n"
        "SEM bands treat centers as independent and UNDERSTATE the true "
        "uncertainty (center_standard_error)",
        fontsize=9,
    )

    # --- bottom panel: the two monopoles, twin y-axis (see docstring) ------
    ax_mono_vel = ax_mono_obs.twinx()

    ax_mono_obs.plot(r, comparison.obs.monopole_norm, "o-", color=_COLOR_OBS)
    ax_mono_obs.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_mono_obs.set_ylabel(r"$\hat\zeta_0^{obs}$  [km/s]", color=_COLOR_OBS)
    ax_mono_obs.tick_params(axis="y", labelcolor=_COLOR_OBS)
    ax_mono_obs.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
    ax_mono_obs.grid(alpha=_GRID_ALPHA)
    ax_mono_obs.spines["top"].set_visible(False)

    ax_mono_vel.plot(r, comparison.vel.monopole_norm, "s-", color=_COLOR_VEL)
    ax_mono_vel.set_ylabel(r"$\hat\zeta_0^{vel}$  [km/s]", color=_COLOR_VEL)
    ax_mono_vel.tick_params(axis="y", labelcolor=_COLOR_VEL)
    ax_mono_vel.spines["top"].set_visible(False)

    fig.tight_layout()
    return fig


def make_angle_diagnostic_figure(
    cfg: ComparisonRunConfig,
    comparison: FrameComparison,
) -> plt.Figure:
    """Two panels diagnosing how the frame gap depends on the axis-rotation angle.

    x-axis for both panels: `comparison.per_center_delta` (radians), plotted
    in DEGREES for readability -- this is
    `VelocityFrameShellDipoleResult.per_center_axis_angle`, the angle between
    each center's own flow direction and its observer line of sight (see
    that field's docstring; it is a pure diagnostic that never fed into
    either frame's statistic).

    TOP panel: `comparison.per_center_dipole_difference` (vel minus obs, see
    `normalize_comparison`) binned into `_N_ANGLE_BINS` equal-width bins over
    [0, pi]; each bin plots the mean difference with an SEM errorbar, over a
    light, low-alpha scatter of the individual centers so outliers stay
    visible even though the bin mean absorbs them. A bin with zero centers
    is left as NaN and simply produces a gap in the errorbar plot -- never a
    crash. A zero line marks "the two frames agree for this center".

    Interpretation: if a FEW high-delta centers are doing most of the work
    (a handful of large-|difference| points concentrated at large delta,
    with the bulk of centers near zero), that points to BULK-FLOW
    CONTAMINATION -- a small number of centers whose own peculiar velocity
    has a large component unrelated to the local infall they are meant to be
    tracing. A smooth, broad spread of differences across the whole delta
    range instead points to genuine PROJECTION GEOMETRY -- every center's
    transverse velocity component contributing a little, exactly as expected
    from the frame-agreement derivation in
    `dvcorr.estimators.velocity_frame_dipole`'s module docstring.

    BOTTOM panel: a histogram of delta (degrees), with the ISOTROPIC
    expectation overlaid as a dashed reference curve. For directions
    distributed isotropically relative to the line of sight, the probability
    density in the polar angle delta is P(delta) ~ sin(delta) -- the
    solid-angle Jacobian of d(cos(delta)), pure mathematics kept inline with
    a comment (CLAUDE.md hard rule 4's exemption for constants that are part
    of a derivation) rather than promoted to `dvcorr.conventions` -- scaled
    to match the histogram's total count. An EXCESS of centers at LOW delta
    relative to that reference means the flow directions are preferentially
    ALIGNED with the lines of sight, i.e. residual BULK MOTION shared with
    the observer frame's own axis choice, not a random projection effect.

    Builder only: never saves, never calls `plt.show`.

    Parameters
    ----------
    cfg : ComparisonRunConfig
    comparison : FrameComparison

    Returns
    -------
    matplotlib.figure.Figure
    """
    delta_deg = np.degrees(comparison.per_center_delta)

    fig, (ax_diff, ax_hist) = plt.subplots(2, 1, figsize=_FIGSIZE_ANGLE, sharex=True)

    # --- top panel: per-center dipole difference, binned by rotation angle -
    bin_edges_deg = np.linspace(0.0, 180.0, _N_ANGLE_BINS + 1)  # [0, pi] in degrees
    bin_centers_deg = 0.5 * (bin_edges_deg[:-1] + bin_edges_deg[1:])
    bin_index = np.clip(np.digitize(delta_deg, bin_edges_deg) - 1, 0, _N_ANGLE_BINS - 1)

    bin_mean = np.full(_N_ANGLE_BINS, np.nan)
    bin_sem = np.full(_N_ANGLE_BINS, np.nan)
    for k in range(_N_ANGLE_BINS):
        in_bin = comparison.per_center_dipole_difference[bin_index == k]
        if in_bin.size == 0:
            continue  # empty bin -> NaN, skipped by errorbar, never a crash
        bin_mean[k] = in_bin.mean()
        if in_bin.size >= 2:
            bin_sem[k] = in_bin.std(ddof=1) / np.sqrt(in_bin.size)

    ax_diff.scatter(
        delta_deg, comparison.per_center_dipole_difference,
        s=10, color=_COLOR_VEL, alpha=_SCATTER_ALPHA, label="individual centers",
    )
    ax_diff.errorbar(
        bin_centers_deg, bin_mean, yerr=bin_sem, fmt="o-", color=_COLOR_OBS,
        label=f"binned mean ({_N_ANGLE_BINS} bins)",
    )
    ax_diff.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_diff.set_ylabel(r"$\hat\zeta_1^{vel} - \hat\zeta_1^{obs}$ (per-center summary)  [km/s]")
    ax_diff.legend(fontsize="small")
    ax_diff.grid(alpha=_GRID_ALPHA)
    ax_diff.spines["top"].set_visible(False)
    ax_diff.spines["right"].set_visible(False)
    ax_diff.set_title(
        "frame gap vs. axis-rotation angle  (few high-delta outliers => "
        "bulk-flow contamination; smooth spread => projection geometry)",
        fontsize=9,
    )

    # --- bottom panel: delta histogram vs. the isotropic sin(delta) reference
    counts, edges_deg = np.histogram(delta_deg, bins=_N_ANGLE_BINS, range=(0.0, 180.0))
    centers_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    ax_hist.bar(
        centers_deg, counts, width=np.diff(edges_deg), color=_COLOR_VEL,
        alpha=_BAND_ALPHA + 0.2, edgecolor=_COLOR_VEL, label="measured",
    )

    delta_rad_centers = np.radians(centers_deg)
    isotropic_shape = np.sin(delta_rad_centers)  # sin(delta): solid-angle Jacobian, pure math
    isotropic_expected = isotropic_shape * (counts.sum() / isotropic_shape.sum())
    ax_hist.plot(
        centers_deg, isotropic_expected, "--", color=_COLOR_ZERO_LINE,
        label=r"isotropic $\propto \sin(\delta)$",
    )

    ax_hist.set_xlabel(r"$\delta_\alpha$ = axis rotation angle  [degrees]")
    ax_hist.set_ylabel("centers per bin")
    ax_hist.legend(fontsize="small")
    ax_hist.grid(alpha=_GRID_ALPHA)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
