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
(subclassed below, not copied), `NormalizedDipole`, `normalize_result`,
`normalize_stacked_dipole`, `matched_gaussian_sample`, `SharedCenterSet`, and
`select_shared_centers` are all IMPORTED from there. `load_and_carve`, `draw_candidates`, and
`box_number_density` are likewise that module's, consumed directly by the
SCRIPT (this module's stage functions start one step later, from
already-loaded candidate arrays, since carving and subsampling do not differ
between the two frames).

`SharedCenterSet` / `select_shared_centers` used to be DEFINED in this
module; they moved to `velocity_centered.py` (still the single source of
truth, just relocated) once `dvcorr.pipeline.redshift_space_comparison`
needed them too -- leaving them here would have made that pipeline depend on
this one, when both are equally built on top of `velocity_centered.py`. Both
names are re-exported by this import, so existing code importing them FROM
`velocity_frame_comparison` is unaffected.

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

from dvcorr.config import volume_weighted_shell_radii
from dvcorr.estimators.shell_dipole import (
    VelocityCenteredShellDipoleResult,
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
    SharedCenterSet,
    _binning_description,
    matched_gaussian_sample,
    null_realization_seeds,
    normalize_result,
    normalize_stacked_dipole,
    select_shared_centers,
    shell_dipole_norm_scale,
)
from dvcorr.pipeline.velocity_centered import _COLOR_SIGNAL as _COLOR_OBS

_COLOR_OBS_NULL = "#a9c9ee"   # lighter tint of _COLOR_OBS: the obs-frame's own null (u-shuffle)
_COLOR_VEL = "#7b2d8e"        # distinct hue for the velocity frame's curves
_COLOR_VEL_NULL = "#cdb0d8"   # lighter tint of _COLOR_VEL: the vel-frame's own null (random-axis)
# Each frame's SECOND null, the matched-Gaussian one: a darker, desaturated
# companion to that frame's null tint, so a reader groups it with its own
# frame first and reads it against that frame's other null second.
_COLOR_OBS_GAUSSIAN = "#5b7fa6"   # obs frame, matched-Gaussian null
_COLOR_VEL_GAUSSIAN = "#9b7aa8"   # vel frame, matched-Gaussian null
_SCATTER_ALPHA = 0.35         # low-alpha individual-center scatter, angle diagnostic top panel
_FIGSIZE_ANGLE = (8.0, 8.0)   # angle-diagnostic figure size, matches the comparison figure
_N_ANGLE_BINS = 12            # equal-width bins over [0, pi/2] for both angle-diagnostic panels
_MAX_ANGLE_DEG = 90.0         # delta's ceiling: cos(delta) = |u|/|v| >= 0 (per_center_axis_angle)


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
    The four fields below, by contrast, have no meaning outside this
    comparison pipeline.

    Attributes
    ----------
    comparison_output_name : str
        Output filename for `make_comparison_figure`'s PNG, written under
        `PathsConfig().output_dir`.
    angle_diagnostic_output_name : str
        Output filename for `make_angle_diagnostic_figure`'s PNG.
    axis_null_seed : int
        Base seed for `random_axis_null_dipoles`' isotropic directions, kept
        DISTINCT from both `seed` (candidate subsampling) and `shuffle_seed`
        (the observer frame's velocity-shuffle null) so the random-axis
        null's randomness never silently reuses either of those streams.
    velocity_gaussian_null_seed : int
        Base seed for `gaussian_velocity_null_dipoles`' matched-Gaussian velocity
        draw -- the velocity frame's SECOND null. Distinct from
        `axis_null_seed` (this frame's other null) and from the inherited
        `gaussian_null_seed` (the OBSERVER frame's matched-Gaussian null,
        which this pipeline also builds, via `normalize_result`): the two
        frames' Gaussian nulls are independent draws, not one draw viewed
        twice, so a coincidence between their curves means something. See
        `dvcorr.pipeline.velocity_centered.RunConfig.gaussian_null_seed` for
        the full ladder.
    """

    comparison_output_name: str = "velocity_frame_comparison.png"
    angle_diagnostic_output_name: str = "velocity_frame_angle_diagnostic.png"
    axis_null_seed: int = 44
    velocity_gaussian_null_seed: int = 47


# `SharedCenterSet` and `select_shared_centers` used to be defined here; they
# now live in `dvcorr.pipeline.velocity_centered` (imported above) -- see the
# module docstring's note on why they moved. `select_shared_centers` is
# called below (`run_both_frames`'s caller,
# `scripts/plot_velocity_frame_comparison.py`, and
# `notebooks/06_velocity_frame_comparison.ipynb`) with its `core_margin`
# argument left at its default (`None` -> `cfg.shells.max_radius`), reproducing
# this pipeline's original behavior exactly.


def run_both_frames(
    cfg: ComparisonRunConfig,
    centers: SharedCenterSet,
    s_tracers: np.ndarray,
    observer: np.ndarray,
) -> FrameRunResults:
    """Run both estimators, plus the velocity frame's two nulls.

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

    Also builds BOTH of the velocity frame's nulls on the same `centers` --
    `random_axis_null_dipoles` and `gaussian_velocity_null_dipoles` -- so a
    single call to this function returns everything `normalize_comparison`
    needs. TWO estimator passes in total, one per frame: every null in either
    frame is a recombination of arrays its estimator already returned, which
    is what makes `cfg.n_null_realizations` realizations of each affordable
    (the observer frame recombines `per_center_amplitude`, the velocity frame
    the cached `per_center_direction_sum`).

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

    return FrameRunResults(
        obs_result=obs_result,
        vel_result=vel_result,
        vel_null_dipoles=random_axis_null_dipoles(cfg, centers, vel_result),
        vel_gaussian_null_dipoles=gaussian_velocity_null_dipoles(cfg, centers, vel_result),
        centers=centers,
    )


def _null_dipoles_from_velocities(
    result: VelocityFrameShellDipoleResult,
    v_null: np.ndarray,
) -> np.ndarray:
    """One null realization: restack the velocity-frame dipole on replacement v.

    The recombination both of this frame's nulls are built from. Because
    Y_10 is linear in the direction cosine, a center's shell amplitude for
    ANY axis is a projection of the cached direction sum
    (`VelocityFrameShellDipoleResult.per_center_direction_sum`),

        |v| * A_alpha,b(v_hat) = sqrt(3/4pi) * (v_alpha . S_alpha,b)

    -- the speed and the unit vector recombine into the raw replacement
    vector, so the whole realization is one einsum over arrays the estimator
    already returned. No KDTree query, no second estimator pass.

    Parameters
    ----------
    result : VelocityFrameShellDipoleResult
        The SIGNAL run, only for its geometry: `per_center_direction_sum`.
        Nothing velocity-derived from it is used.
    v_null : ndarray, shape (N_c, 3)
        Replacement velocities, row-aligned with `result`'s centers.

    Returns
    -------
    ndarray, shape (B,)
        Raw stacked dipole for this realization, same units and convention as
        `result.dipole`.
    """
    return np.sqrt(3.0 / (4.0 * np.pi)) * np.einsum(  # sqrt(3/4pi): Y_10 normalization
        "ac,abc->b", v_null, result.per_center_direction_sum
    )


def random_axis_null_dipoles(
    cfg: ComparisonRunConfig,
    centers: SharedCenterSet,
    result: VelocityFrameShellDipoleResult,
) -> np.ndarray:
    """The velocity frame's null: randomize the AXIS, keep the speed and position.

    Per realization, v_null_alpha = |v_alpha| * random_unit_direction_alpha,
    with the directions drawn isotropically via
    `unit_vector(rng.normal(size=(N_c, 3)))`. `cfg.axis_null_seed` names the
    STREAM: it is spawned into `cfg.n_null_realizations` child seeds
    (`dvcorr.pipeline.velocity_centered.null_realization_seeds`), one
    realization each, and the stack of them is what gets normalized into a
    mean and a band.

    WHY THIS NULL, AND NOT A SCALAR SHUFFLE -- important, read before
    "simplifying" this to match `normalize_result`'s pattern
    --------------------------------------------------------------------------
    `normalize_result`'s velocity-shuffle null undoes the observer frame's
    axis flip (recovering the amplitude against the fixed +n_hat_V,alpha) and
    then permutes the SIGNED `per_center_u` against it. That works because the
    observer frame's axis is only PARTIALLY velocity-derived: it takes a sign
    from u_alpha and its line from the center's POSITION, so once the sign is
    divided back out, what remains is a pure function of position and can be
    held fixed while the velocity is permuted. And the permuted quantity is
    SIGNED, so it averages to ~0 across an isotropic sample of centers and the
    recombined stack collapses.

    The velocity frame has NO such separation available. Its axis
    z_hat_alpha = v_hat_alpha is built ENTIRELY from the velocity vector that
    also supplies the scalar |v_alpha| -- there is no position-derived
    remainder to hold fixed, and no sign to divide out. Permuting |v_alpha|
    among centers would leave every center's z_hat_alpha (and therefore every
    A_alpha,b) exactly as it was; the recombined sum would be Sigma_alpha
    |v_permuted,alpha| * A_alpha,b, which is essentially the SAME quantity as
    the real signal, because |v_alpha| is positive-definite and cannot cancel
    the way a signed scalar does. A scalar permutation is therefore not a null
    here at all; it would retain N_c * <|v|> * <A>. (This is also precisely
    why the observer frame's null must NOT be written as a permutation of its
    `per_center_speed`: that version would fall into the same trap.)

    Randomizing the AXIS instead is the correct guard against
    align-then-measure bias: it breaks the one thing that makes the statistic
    self-aligned (the axis's dependence on the SAME vector as the scalar)
    while leaving the speed and the tracer geometry untouched, and it
    genuinely does collapse to zero for an uncorrelated axis (see
    tests/test_velocity_frame_dipole.py, item 5's clustered-configuration
    assertion).

    What it no longer costs
    -------------------------
    This used to re-run `velocity_frame_shell_dipole` per null, which capped
    the pipeline at one realization per curve -- and one realization of a null
    is a sample of size one, which on the inner shells is O(100 km/s) from
    zero with a seed-determined sign (`RunConfig.n_null_realizations`).
    Caching the per-shell DIRECTION SUMS turned every axis draw into an
    einsum, so realizations are now essentially free and the null is reported
    as a mean with a band. The expensive half -- the neighbor query and the
    per-tracer unit vectors -- depends only on positions, which no null
    touches, so nothing about the construction changed; only its cost did.

    Parameters
    ----------
    cfg : ComparisonRunConfig
    centers : SharedCenterSet
        From `select_shared_centers` (or, within `run_both_frames`, already
        confirmed row-aligned with `result`).
    result : VelocityFrameShellDipoleResult
        The signal run, for its cached geometry (`per_center_direction_sum`).

    Returns
    -------
    ndarray, shape (R, B)
        R = `cfg.n_null_realizations` raw stacked dipoles, one row per
        realization, ready for `normalize_stacked_dipole`.
    """
    speeds = np.linalg.norm(centers.v_centers, axis=1)

    return np.array([
        _null_dipoles_from_velocities(
            result,
            speeds[:, None] * unit_vector(
                np.random.default_rng(child).normal(size=centers.v_centers.shape)
            ),
        )
        for child in null_realization_seeds(cfg.axis_null_seed, cfg.n_null_realizations)
    ])


def gaussian_velocity_null_dipoles(
    cfg: ComparisonRunConfig,
    centers: SharedCenterSet,
    result: VelocityFrameShellDipoleResult,
) -> np.ndarray:
    """The velocity frame's SECOND null: replace v with a matched Gaussian draw.

    Per realization, v_null_alpha ~ N(<v>, sigma_v^2) column-wise, from
    `dvcorr.pipeline.velocity_centered.matched_gaussian_sample` on
    `centers.v_centers`. `cfg.velocity_gaussian_null_seed` names the stream
    and is spawned into `cfg.n_null_realizations` children, exactly as
    `random_axis_null_dipoles` does with its own seed.

    HOW THIS DIFFERS FROM `random_axis_null_dipoles`, WHICH IS THE POINT
    -----------------------------------------------------------------------
    The two nulls are NOT redundant, and the difference is narrow enough to
    be worth stating exactly. Both produce a replacement velocity vector;
    they differ in what they preserve:

        random-axis :  v_null = |v_alpha| * e_hat_alpha, e_hat isotropic
        gaussian    :  v_null ~ N(<v>, sigma_v^2), independently per center

    The random-axis null keeps each center's OWN speed attached to that
    center, so the speed <-> environment correlation survives -- a center in
    a dense region keeps both its large |v| and its large shell occupancy
    N_alpha,b, and since Var(D_b^null) ~ Sum_alpha |v_alpha|^2 Var(A_alpha,b)
    with those two factors positively correlated, its band is the noise floor
    OF THIS SAMPLE. It also destroys the sample's bulk motion, because an
    isotropic axis has no preferred direction left.

    The Gaussian null does the opposite on both counts: the drawn speed is
    detached from its center (so the speed <-> environment correlation is
    gone, and the band is a model's floor rather than this sample's), while
    <v> is matched, so the bulk-flow direction SURVIVES. It is therefore
    mildly anisotropic by construction (`matched_gaussian_sample` documents
    why the mean is matched rather than zeroed).

    Read them together: random-axis is the conservative floor -- change one
    thing, the direction -- and Gaussian is the parametric comparison whose
    distance from it measures what the halo velocity distribution's shape
    (non-Gaussian speed tail, speed-environment coupling, retained bulk
    flow) is worth. Neither replaces the other; `random_axis_null_dipoles`
    remains this frame's PRIMARY null.

    Parameters
    ----------
    cfg : ComparisonRunConfig
    centers : SharedCenterSet
        From `select_shared_centers` (or, within `run_both_frames`, already
        confirmed row-aligned with `result`).
    result : VelocityFrameShellDipoleResult
        The signal run, for its cached geometry (`per_center_direction_sum`).

    Returns
    -------
    ndarray, shape (R, B)
        R = `cfg.n_null_realizations` raw stacked dipoles, one row per
        realization.

    Notes
    -----
    A drawn v_null_alpha of exactly zero length would have an undefined axis.
    It cannot bite here the way it would in an estimator pass: the
    recombination consumes the raw VECTOR (speed and direction together, see
    `_null_dipoles_from_velocities`), so a zero vector contributes zero to
    the stack rather than needing a direction at all -- and three independent
    normals landing within float-epsilon of -<v> simultaneously is not
    something a continuous draw does in practice regardless.
    """
    return np.array([
        _null_dipoles_from_velocities(
            result, matched_gaussian_sample(centers.v_centers, int(child))
        )
        for child in null_realization_seeds(
            cfg.velocity_gaussian_null_seed, cfg.n_null_realizations
        )
    ])


@dataclass(frozen=True)
class FrameRunResults:
    """Both estimators' raw results, plus the velocity frame's nulls, row-aligned.

    Attributes
    ----------
    obs_result : VelocityCenteredShellDipoleResult
        Observer-frame zeta_1 result. Its OWN two nulls are recombinations
        built later, inside `normalize_result` -- nothing to store here.
    vel_result : VelocityFrameShellDipoleResult
        Velocity-frame result, on the SAME `centers`.
    vel_null_dipoles : ndarray, shape (R, B)
        R realizations of the velocity frame's PRIMARY null -- the same
        centers with axes replaced by isotropic random directions
        (`random_axis_null_dipoles`); see that function for why a scalar
        permutation would not be a valid null here. Raw stacked dipoles, one
        row per realization, not estimator results: each is a recombination
        of `vel_result`'s cached direction sums.
    vel_gaussian_null_dipoles : ndarray, shape (R, B)
        R realizations of this frame's second null, the velocities replaced
        by a matched Gaussian draw (`gaussian_velocity_null_dipoles`).
    centers : SharedCenterSet
        The shared center set both frames (and both nulls) were run on.
    """

    obs_result: VelocityCenteredShellDipoleResult
    vel_result: VelocityFrameShellDipoleResult
    vel_null_dipoles: np.ndarray
    vel_gaussian_null_dipoles: np.ndarray
    centers: SharedCenterSet


def normalize_velocity_frame_result(
    result: VelocityFrameShellDipoleResult,
    null_dipoles: np.ndarray,
    gaussian_null_dipoles: np.ndarray,
    n_bar: float,
) -> NormalizedDipole:
    """Thin delegate to `normalize_stacked_dipole` for the velocity frame.

    For this frame, the returned `NormalizedDipole.zeta_hat_shuffle` (and
    `.null_spread_shuffle`) hold the RANDOM-AXIS null
    (`random_axis_null_dipoles`' output), not a scalar permutation -- the field name is inherited
    unchanged from the observer-frame path (`normalize_stacked_dipole`'s
    docstring already documents that it names "whatever null the caller
    built"), so it is repeated here explicitly to avoid the field name being
    misread as a promise about construction. `.zeta_hat_gaussian` /
    `.null_spread_gaussian` need no such caveat: they hold a matched-Gaussian
    null in BOTH frames, differing only in whether the draw is the (N_c,) u or
    the (N_c, 3) v (`gaussian_velocity_null_dipoles`).

    Parameters
    ----------
    result : VelocityFrameShellDipoleResult
        The signal run, e.g. `FrameRunResults.vel_result`.
    null_dipoles : ndarray, shape (R, B)
        Random-axis null realizations, e.g. `FrameRunResults.vel_null_dipoles`.
    gaussian_null_dipoles : ndarray, shape (R, B)
        Matched-Gaussian null realizations, e.g.
        `FrameRunResults.vel_gaussian_null_dipoles`. Required here, unlike on
        `normalize_stacked_dipole` where it is optional: this pipeline always
        builds both nulls, so there is no "arithmetic-only" call pattern for
        it to accommodate.
    n_bar : float
        Mean halo number density over the whole box, e.g. from
        `dvcorr.pipeline.velocity_centered.box_number_density`.

    Returns
    -------
    NormalizedDipole
    """
    return normalize_stacked_dipole(
        shell_edges=result.shell_edges,
        dipole=result.dipole,
        monopole=result.monopole,
        per_center_dipole=result.per_center_dipole,
        null_dipoles=null_dipoles,
        n_centers=result.n_centers,
        n_bar=n_bar,
        gaussian_null_dipoles=gaussian_null_dipoles,
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
        `vel_result.per_center_axis_angle`, RADIANS, in [0, pi/2] -- the
        angle between each center's own flow direction and the observer
        frame's axis sign(u_alpha) * n_hat_V,alpha for that same center (see
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
    (its two recombination nulls, seeded by `cfg.shuffle_seed` and
    `cfg.gaussian_null_seed`); `results.vel_result` normalizes via
    `normalize_velocity_frame_result` (its two recombination nulls,
    `results.vel_null_dipoles` and `results.vel_gaussian_null_dipoles`). Four
    null curves reach the figure, two per frame, and the pairing is what they
    are for -- within a frame the two nulls share their first two moments and
    differ only in distribution shape.

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

    This identity is pure index exchange (`.mean(axis=1)` over shells versus
    `.mean(axis=0)` over centers commute on the same 2-D array) and holds for
    ANY `shell_edges` binning -- it does NOT break when the shells switch from
    linear to log spacing. What DOES change is the INTERPRETATION: an
    unweighted mean over the shell INDEX b is a mean "uniform in r" under the
    old linear spacing, and becomes a mean "uniform in log r" once
    `cfg.shells.spacing == dvcorr.config.SPACING_LOG` -- five of twelve bins
    then sit below 5.66 h^-1 Mpc, so this per-center frame-gap summary becomes
    noticeably more sensitive to the one-halo/exclusion regime than it was
    under the old linear default.

    A `n_bar*V_b`-weighted mean (rather than the current unweighted
    `.mean(axis=1)`) would be the natural fix if that sensitivity becomes a
    problem in practice -- but it is deliberately NOT done here, because it
    would BREAK the very identity this docstring promises: the plotted
    `mean_b(zeta_hat_b)` on the comparison figure is itself an UNWEIGHTED mean
    over shells, so only an unweighted `summary_alpha` can sum back to it. A
    `summary_radial_window` knob (restricting the mean to a chosen r-range,
    rather than reweighting it) is the right follow-up if the diagnostic
    degrades under log spacing in practice.

    Parameters
    ----------
    cfg : ComparisonRunConfig
    results : FrameRunResults
    n_bar : float
        Mean halo number density over the whole box, e.g. from
        `dvcorr.pipeline.velocity_centered.box_number_density`.

    Returns
    -------
    FrameComparison
    """
    obs = normalize_result(
        results.obs_result,
        n_bar,
        cfg.shuffle_seed,
        cfg.gaussian_null_seed,
        cfg.n_null_realizations,
    )
    vel = normalize_velocity_frame_result(
        results.vel_result,
        results.vel_null_dipoles,
        results.vel_gaussian_null_dipoles,
        n_bar,
    )

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
    TWO nulls in tints of its own color -- the frame's primary null dashed in
    a lighter tint, its matched-Gaussian null dotted in a darker one. So six
    curves, grouped by frame rather than by null type, and the legend names
    every construction ("u-shuffle", "random-axis", "gaussian") so no two are
    ever confused:

        obs : u-shuffle (`normalize_result`'s permutation of per_center_u)
              gaussian  (`normalize_result`'s matched draw of per_center_u)
        vel : random-axis (`random_axis_null_dipoles`)
              gaussian    (`gaussian_velocity_null_dipoles`)

    How to read four nulls without drowning in them: WITHIN a frame, the two
    nulls are matched on their first two moments by construction, so their
    SEPARATION is the distribution-shape term and nothing else (see
    `dvcorr.pipeline.velocity_centered.matched_gaussian_sample`). ACROSS
    frames they are not comparable curve-for-curve -- the obs pair is two
    recombinations of one estimator pass, the vel pair is two independent
    constructions measuring different things (`gaussian_velocity_null_dipoles`
    documents exactly what each of the vel-frame pair preserves). The primary
    null of each frame -- dashed -- is the one to read the signal against;
    the dotted one is read against its own dashed partner.

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
    section) and so has NO such leakage term.

    Neither curve sits at zero. Both frames weight by a SPEED -- |u_alpha|
    for the observer frame since the axis carries the sign
    (`dvcorr.conventions.VELOCITY_AXIS_CONVENTION`), |v_alpha| for the
    velocity frame -- and a positive-definite weight has no near/far
    cancellation available to it, so the two curves sit near <|u|> and <|v|>
    respectively. They are the same kind of quantity now, but still a
    projection factor apart in magnitude, so a single shared y-axis would
    compress the smaller one and hide the PRESENCE-VS-ABSENCE of the
    r-dependent trend -- exactly the core finite-distance diagnostic this
    comparison exists to show.

    What BOTH plotted curves decline with, and what that does NOT mean: on a
    clustered tracer field each monopole is its mean speed times the
    occupancy ratio 1 + xi_hh(r), so both curves fall steeply at small r from
    halo clustering alone -- neither frame is flat here, and neither decline
    is a systematic. Dividing that ratio out leaves, in BOTH frames, the same
    genuine speed-density correlation (-7.9% and -7.8% over the shells on the
    first MDPL2 run with the signed axis).

    The finite-distance (2r/3R) leakage is NOT visible in this panel. It is a
    SIGNED effect, and both frames now weight by a speed -- see
    `dvcorr.estimators.velocity_frame_dipole`'s Monopole section for the full
    argument and for the signed monopole, `(per_center_u[:, None] *
    per_center_count).sum(axis=0)`, which does still carry it and is what to
    reach for when that is the question being asked.

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

    # comparison.shell_centers (the plain midpoint) is deliberately left
    # alone -- r_eff, the volume-weighted radius, is the plotting abscissa.
    r = volume_weighted_shell_radii(results.obs_result.shell_edges)

    # --- top panel: the two dipoles, each with its SEM band and its two nulls
    ax_dipole.fill_between(
        r, comparison.obs.zeta_hat - comparison.obs.sem, comparison.obs.zeta_hat + comparison.obs.sem,
        color=_COLOR_OBS, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.obs.zeta_hat, "o-", color=_COLOR_OBS,
        label=fr"$\hat\zeta_1^{{obs}}$  (N$_c$={results.obs_result.n_centers})",
    )
    ax_dipole.fill_between(
        r,
        comparison.obs.zeta_hat_shuffle - comparison.obs.null_spread_shuffle,
        comparison.obs.zeta_hat_shuffle + comparison.obs.null_spread_shuffle,
        color=_COLOR_OBS_NULL, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.obs.zeta_hat_shuffle, "o--", color=_COLOR_OBS_NULL,
        label="obs null (u-shuffle)",
    )
    ax_dipole.fill_between(
        r,
        comparison.obs.zeta_hat_gaussian - comparison.obs.null_spread_gaussian,
        comparison.obs.zeta_hat_gaussian + comparison.obs.null_spread_gaussian,
        color=_COLOR_OBS_GAUSSIAN, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.obs.zeta_hat_gaussian, "o:", color=_COLOR_OBS_GAUSSIAN,
        label="obs null (gaussian)",
    )

    ax_dipole.fill_between(
        r, comparison.vel.zeta_hat - comparison.vel.sem, comparison.vel.zeta_hat + comparison.vel.sem,
        color=_COLOR_VEL, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.vel.zeta_hat, "s-", color=_COLOR_VEL,
        label=fr"$\hat\zeta_1^{{vel}}$  (N$_c$={results.vel_result.n_centers})",
    )
    ax_dipole.fill_between(
        r,
        comparison.vel.zeta_hat_shuffle - comparison.vel.null_spread_shuffle,
        comparison.vel.zeta_hat_shuffle + comparison.vel.null_spread_shuffle,
        color=_COLOR_VEL_NULL, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.vel.zeta_hat_shuffle, "s--", color=_COLOR_VEL_NULL,
        label="vel null (random-axis)",
    )
    ax_dipole.fill_between(
        r,
        comparison.vel.zeta_hat_gaussian - comparison.vel.null_spread_gaussian,
        comparison.vel.zeta_hat_gaussian + comparison.vel.null_spread_gaussian,
        color=_COLOR_VEL_GAUSSIAN, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.vel.zeta_hat_gaussian, "s:", color=_COLOR_VEL_GAUSSIAN,
        label="vel null (gaussian)",
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
        f"{_binning_description(cfg.shells)}, "
        f"N$_c$={results.centers.n_centers})\n"
        "SEM bands treat centers as independent and UNDERSTATE the true "
        "uncertainty (center_standard_error)",
        fontsize=9,
    )

    # --- bottom panel: the two monopoles, twin y-axis (see docstring) ------
    ax_mono_vel = ax_mono_obs.twinx()

    # No zero reference line: with the |u_alpha| weight the obs-frame
    # monopole sits near <|u|>, not near zero (see this function's docstring).
    ax_mono_obs.plot(r, comparison.obs.monopole_norm, "o-", color=_COLOR_OBS)
    ax_mono_obs.set_ylabel(r"$\hat\zeta_0^{obs} \simeq \langle|u|\rangle$  [km/s]", color=_COLOR_OBS)
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
    each center's own flow direction and the OBSERVER frame's axis for that
    same center, sign(u_alpha) * n_hat_V,alpha (see that field's docstring;
    it is a pure diagnostic that never fed into either frame's statistic).
    Both panels run over [0, 90] degrees, not [0, 180]: cos(delta) =
    |u_alpha| / |v_alpha| is non-negative, so 90 degrees is a hard ceiling,
    enforced by `velocity_frame_shell_dipole` rather than assumed here.

    TOP panel: `comparison.per_center_dipole_difference` (vel minus obs, see
    `normalize_comparison`) binned into `_N_ANGLE_BINS` equal-width bins over
    [0, pi/2]; each bin plots the mean difference with an SEM errorbar, over a
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
    expectation overlaid as a dashed reference curve. For flow directions
    distributed isotropically relative to the line of sight, cos(delta) =
    |c| with c uniform on [-1, 1], so the probability density in delta is
    P(delta) ~ sin(delta) on [0, pi/2] -- the solid-angle Jacobian of
    d(cos(delta)), folded onto the half-range by the absolute value; pure
    mathematics kept inline with a comment (CLAUDE.md hard rule 4's exemption
    for constants that are part of a derivation) rather than promoted to
    `dvcorr.conventions` -- scaled to match the histogram's total count. An
    EXCESS of centers at LOW delta relative to that reference means the flow
    directions are preferentially ALIGNED with the lines of sight, i.e.
    residual BULK MOTION shared with the observer frame's own axis choice,
    not a random projection effect.

    Sensitivity to log-spaced shells -- read before trusting this figure under
    the current default binning
    ----------------------------------------------------------------------------
    Both panels' y-quantity is `comparison.per_center_dipole_difference`,
    `normalize_comparison`'s per-center summary, an UNWEIGHTED mean over the
    shell INDEX b (`per_center_curve.mean(axis=1)`, not an occupancy-weighted
    one -- see that function's docstring for why the weighting cannot change
    without breaking the `mean_alpha(summary_alpha) == mean_b(zeta_hat_b)`
    identity this figure's top panel implicitly relies on). Under the log
    default, five of twelve shells sit below 5.66 h^-1 Mpc, where a center's
    expected occupancy is tiny (~0.02 pairs per center in the innermost
    bins), so the unweighted mean is now dominated by shot noise from those
    few bins rather than by the bulk of each center's shell range. Measured
    on an unclustered synthetic: std(summary_alpha) across centers goes
    14.9 -> 234.3 switching from linear to log spacing, and the three
    innermost log bins alone supply 75% of that variance (bin 0 alone 33%).
    Consequence: this figure's stated purpose is "do a few high-delta
    centers drive the frame gap", but under log spacing it is at risk of
    instead becoming a plot of which centers happened to catch a tracer
    inside ~2 h^-1 Mpc -- a one-halo/exclusion-regime artifact, not evidence
    of bulk-flow contamination or projection geometry. Read a scattered,
    high-|difference| outlier here with that caveat in mind, especially if
    it also has a small `per_center_count` in the innermost shells.

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
    bin_edges_deg = np.linspace(0.0, _MAX_ANGLE_DEG, _N_ANGLE_BINS + 1)  # [0, pi/2] in degrees
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
    counts, edges_deg = np.histogram(
        delta_deg, bins=_N_ANGLE_BINS, range=(0.0, _MAX_ANGLE_DEG)
    )
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
