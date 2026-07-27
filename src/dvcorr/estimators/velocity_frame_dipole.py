"""
velocity_frame_dipole.py
--------------------------
Velocity-frame dipole: an OBSERVER-FREE variant of the velocity-centered
zeta_1 statistic (`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`).

CONTEXT / MOTIVATION
---------------------
The observer-frame estimator (zeta, Nusser 2017 eq. 23-24, implemented in
`shell_dipole.py`) puts a velocity object alpha at the center of each shell
and axes it on n_hat_V,alpha = observer -> s_alpha, the line of sight FROM THE
OBSERVER. That axis choice is forced by what a peculiar-velocity survey
actually measures: a line-of-sight (radial) speed
u_alpha = v_alpha . n_hat_V,alpha, never the full 3-vector. The axis and the
scalar it multiplies are therefore tied to the SAME finite-distance
direction by construction -- an artifact of having an observer at all, not a
property of the flow.

This module removes the observer from the statistic entirely. For each center
alpha it defines the axis as the center's OWN flow direction,

    z_hat_alpha = v_alpha / |v_alpha|          (v_hat_alpha)

and uses the FULL speed |v_alpha| as the scalar, in place of the observer-frame
radial projection u_alpha. Comparing the two frames quantifies exactly what is
lost by having a finite-distance observer restricted to radial velocities: in
the ideal spherical-infall, distant-observer limit the two coincide, because a
purely radial flow has v_hat_alpha -> n_hat_V,alpha and |v_alpha| -> u_alpha
(see tests/test_velocity_frame_dipole.py's frame-agreement test, item 2, which
pins this on real numbers). Away from that limit, the divergence between the
frames measures projection geometry -- the TRANSVERSE component of v_alpha,
invisible to the observer frame -- plus the center's own bulk motion.

Read the two frames' quantities literally, because it matters for interpreting
any amplitude difference between them: zeta_velframe correlates the center's
FULL SPEED with density-along-its-own-flow-direction, while zeta_obsframe
correlates the center's RADIAL speed with density-along-the-line-of-sight. An
amplitude difference is therefore partly the |v| vs. v . n_hat_V effect (a
positive-definite scalar vs. a signed, generally smaller one) and not purely a
rotation of the axis -- do not attribute the whole gap to axis geometry
without checking the speed-vs-radial-speed piece first.

A THIRD contributor, often the largest of the three, and easy to miss because
neither of the two above names it: this estimator is SELF-ALIGNED. The axis
z_hat_alpha is built from the SAME velocity vector v_alpha that also supplies
the scalar |v_alpha|, whereas the observer frame's axis n_hat_V,alpha is fixed
by each center's POSITION and only coincides with the true flow direction in
the distant-observer limit. Concretely: stack the amplitude A_alpha,b over
many centers whose true flow direction is uncorrelated with n_hat_V,alpha, and
the observer frame's per-center cos_theta values are effectively drawn from a
distribution with <cos_theta^2> ~= 1/3 (the same shell-averaging dilution
`geometry.mu_cosine`'s docstring describes for mu), while the velocity frame's
cos_theta is 1 for EVERY center, by construction -- no averaging, no dilution,
because the tracer that sits at a given angle relative to z_hat_alpha is being
measured with the SAME vector that defines z_hat_alpha. On
tests/test_velocity_frame_dipole.py's sign-gate toy (where this effect is not
diluted by anything else) this alone is worth a factor of 3 in the raw
per-center amplitude -- see that test's docstring for the full derivation.
This is NOT the frame-agreement limit (which concerns whether the two axes
literally coincide) and NOT the |v| vs. u effect (which concerns the scalar,
not the amplitude) -- it is a THIRD, independent enhancement of the velocity
frame's amplitude relative to the observer frame's, present even when the two
axes are only loosely correlated. `run_random_axis_null`
(`dvcorr.pipeline.velocity_frame_comparison`) measures the NULL floor of this
statistic (what the amplitude looks like with no axis-density correlation at
all) but does not, by itself, isolate this enhancement from a genuine
finite-distance effect -- when comparing the two frames' normalized dipole
amplitudes side by side (`dvcorr.pipeline.velocity_frame_comparison
.make_comparison_figure`), do not read a larger velocity-frame amplitude as
automatically meaning a physically larger signal; check whether it survives
after accounting for this construction-level difference.

HARD RULE 0 AUDIT (performed before writing this module; recorded here for
the next time it is touched)
--------------------------------------------------------------------------
`Imports from old repo/vector3d.py` provides only `Vector3D.norm()` (a thin
wrapper over `np.linalg.norm`) and `Vector3D.periodic_delta` (a PBC
minimum-image subtraction). Neither is relevant: `np.linalg.norm` is used
directly below for `|v_alpha|`, and no periodic reduction belongs in this
module for the same reason none belongs in `geometry.py` -- this module
inherits `geometry.py`'s PBC contract of already-unwrapped coordinates,
identical to `velocity_centered_shell_dipole`.

`Imports from old repo/specific_utils.py::radial_velocity_and_error_pbc`
builds a line-of-sight unit vector as `r_hat = disp / r_norm[:, None]` after a
hand-rolled zero-guard: when `r_norm == 0` it sets `r_norm = ORIGIN_EPS` and
adds `ORIGIN_EPS` to every component of `disp`, i.e. it INVENTS a direction
(the constant vector `(1, 1, 1) / sqrt(3)`) for an object with no
displacement, rather than leaving the direction undefined.
`dvcorr.geometry.unit_vector` (masked divide, zero rows returned as zeros, no
epsilon, no invented direction) together with `mu_cosine` already cover the
normalize-and-project pattern this module needs, and are the safer choice
here specifically: inventing a direction for a zero-length vector is exactly
the failure mode this module's zero-speed guard (see
`velocity_frame_shell_dipole`'s Raises section) exists to prevent for the
AXIS, `z_hat_alpha = unit_vector(v_alpha)`. Nothing was ported from either
file.

CONSTRUCTION
------------
For each surviving center alpha (see Observer role below):

    z_hat_alpha = unit_vector(v_alpha)                     (the flow axis)
    speed_alpha = |v_alpha|                                 (the scalar)

and, for each shell b, the real l=1, m=0 spherical harmonic summed over the
tracers in that shell, exactly as in the observer frame but with z_hat_alpha
in place of n_hat_V,alpha:

    A_alpha,b = sum_{i in b} Y_10(n_hat_i)
              = sqrt(3/4pi) * sum_{i in b} (z_hat_alpha . n_hat_i)

with n_hat_i the unit vector FROM the center alpha TO tracer i -- the same
REVERSED orientation `velocity_centered_shell_dipole` uses (center -> tracer,
the exact negative of `dvcorr.conventions`' frozen r = s_V - s_T for the same
physical pair), built the identical way: pass the velocity center in the
"center" slot of `pair_separation`, then `mu_cosine(unit_vector(r_vec),
z_hat_alpha)`. This reuses `real_y10` and the reversed-construction machinery
unchanged; the ONLY departures from `velocity_centered_shell_dipole` are the
axis and the scalar (and the diagnostic angle below).

The per-center statistic is speed_alpha * A_alpha,b, stacked over centers:

    per_center_dipole = per_center_speed[:, None] * per_center_amplitude
    dipole            = per_center_dipole.sum(axis=0)
    monopole          = (per_center_speed[:, None] * per_center_count).sum(axis=0)

By construction (mirroring `VelocityCenteredShellDipoleResult`'s invariants):

    dipole      == per_center_dipole.sum(axis=0)
    pair_count  == per_center_count.sum(axis=0)
    monopole    == (per_center_speed[:, None] * per_center_count).sum(axis=0)

OBSERVER ROLE -- read before assuming this is "observer-free" everywhere
--------------------------------------------------------------------------
The statistic itself never touches the observer position. It enters the
per-center bookkeeping in exactly two places, both deliberate:

  (a) `core_center_mask` -- pure geometry, IDENTICAL for both frames (it is
      imported unchanged from `shell_dipole.py` and called the same way here
      as in `velocity_centered_shell_dipole`), so the two frames see the
      identical candidate set before either estimator ever runs. This is not
      a compromise on the "observer-free" claim: the core cut decides which
      centers have a COMPLETE shell inside the carved sub-volume, a data-
      availability question that has nothing to do with which axis a center
      uses once selected.
  (b) `per_center_axis_angle`, a pure DIAGNOSTIC (see the dataclass docstring
      below) -- the angle between the center's own flow direction and its
      line of sight to the observer. It never touches `dipole`, `monopole`,
      `per_center_dipole`, or `per_center_amplitude`; it exists so a caller
      can ask "how far did this center's axis rotate away from the
      observer-frame axis" without re-deriving it, but the statistic would be
      numerically identical if this field were deleted.

That is the entire point of the variant: outside of (a) and (b), there is no
observer in this module at all.

SIGN -- derive and pin, this is load-bearing
-----------------------------------------------
Under coherent infall the velocity-frame dipole is POSITIVE (the SAME sign as
the observer-frame zeta_1, unlike the (-1)^ell flip between zeta and the
group-centered xi_Tu -- see below for why that is not a contradiction).

Derivation: a center alpha falling toward an overdensity has its own velocity
v_alpha pointing AT the excess density, so z_hat_alpha = v_hat_alpha satisfies
z_hat_alpha . n_hat_i > 0 for the tracers i that constitute that excess,
giving A_alpha,b > 0 for those tracers' shells. The scalar |v_alpha| is
positive-definite by construction (a norm, never negative), so the product
speed_alpha * A_alpha,b > 0 for every infalling center, and the sum over
centers stays positive. Equivalently, in one sentence: mass flows toward
overdensities, so the density excess lies AHEAD of the flow, along +z_hat_alpha
-- the defining property of a positive dipole moment.

This agrees in SIGN with the observer-frame zeta_1, which is also positive for
infall (`shell_dipole.py`'s documented zeta_ell = (-1)**ell * xi_Tu,ell
relation, combined with CLAUDE.md hard rule 2's negative group-centered
xi_Tu,1: zeta_1 = -xi_Tu,1 > 0). This is not a coincidence needing its own
separate derivation -- it is required by the frame-agreement limit above: the
two frames coincide for a purely radial flow, so whichever sign the
observer-frame dipole has under infall, the velocity-frame dipole must share
it, or the two would disagree exactly where they are constructed to agree.

A NEGATIVE velocity-frame dipole from an infall mock is therefore an
ORIENTATION BUG, not a result -- see the sign gate in
tests/test_velocity_frame_dipole.py, which also checks agreement with the
observer-frame zeta_1 on the identical toy configuration, guarding
specifically against a silent flip introduced by the rotated axis.

MONOPOLE -- read precisely, do not overstate this
----------------------------------------------------
monopole(r_b) = sum_alpha |v_alpha| * N_alpha,b, i.e. the speed-weighted
occupancy. Normalized (dividing by n_centers and the expected occupancy,
mirroring `dvcorr.pipeline.velocity_centered.normalize_stacked_dipole`) it is

    ~ <|v|> * (realized occupancy / expected occupancy)

and it sits at the MEAN HALO SPEED (hundreds of km/s), NOT at zero. That
offset is a real, documented difference from the observer-frame monopole:
|v| is positive-definite and cannot cancel the way the observer-frame's
u_alpha does (u_alpha averages to ~0 across centers on an isotropic sample of
lines of sight, since the sign of the radial projection is as likely to be
positive as negative). |v| has no such cancellation available to it.

Do NOT expect this monopole to be flat in r on a clustered tracer field. The
occupancy ratio above IS 1 + xi_hh(r), the tracer autocorrelation, so BOTH
frames' monopoles inherit its shape; on the first MDPL2 run
(notebooks/06_velocity_frame_comparison.ipynb) that ratio falls 1.46 -> 1.005
over r = 7.5 -> 57.5 h^-1 Mpc, dragging the raw velocity-frame monopole from
799 down to 506 km/s. Flatness is what the UNCLUSTERED (Poisson) case gives,
not this one.

The diagnostic content of THIS monopole is therefore neither "is it near
zero" (it will not be) nor "is it flat" (it will not be, on a clustered
field), but what survives after the occupancy ratio is divided out. There the
two frames genuinely part company, and this is the finite-distance story:

  - the observer-frame monopole carries the finite-distance (2r/3R) leakage
    documented in `geometry.mu_cosine` and `shell_dipole.py`, a trend IN r
    (it grows with shell radius r at fixed R) that SURVIVES the division --
    measured -13.6 -> -2.7 km/s over the same shells, decaying toward its
    <u_alpha> = -2.67 km/s asymptote;
  - the velocity frame has no R at all -- there is no observer distance
    anywhere in its construction -- so nothing survives: the same division
    leaves 546.7 -> 504.0 km/s against <|v_alpha|> = 504.5 km/s, i.e. flat to
    ~8% with no leakage term to produce a trend.

That residual ~8% rise at small r is not leakage either; it is a genuine
speed-density correlation (fast halos preferentially living in denser
environments), which -- along with shell incompleteness, the same
boundary-truncation mechanism `core_center_mask` already guards against -- is
what a velocity-frame monopole residual can legitimately mean. See
`dvcorr.pipeline.velocity_frame_comparison.make_comparison_figure`'s monopole
panel, which plots both frames' monopoles on separate y-axes for exactly this
reason -- the presence-vs-absence of the r-dependent trend is the comparison's
core finite-distance diagnostic, and the two curves sit at incomparable
offsets (near zero vs. near <|v|>) so a shared axis would visually hide it.

Periodic boundary conditions
------------------------------
Identical contract to `shell_dipole.py` and `geometry.py`: this function does
NOT wrap coordinates. The minimum-image reduction is resolved one level up,
where the neighbor sub-volume around each center is carved and unwrapped into
that center's local continuous frame. The only periodicity obligation
enforced here is that the outermost shell edge must not exceed
`conventions.MAX_ANALYSIS_RADIUS`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from dvcorr import conventions
from dvcorr.geometry import mu_cosine, pair_separation, unit_vector
from dvcorr.estimators.shell_dipole import core_center_mask, real_y10


@dataclass(frozen=True)
class VelocityFrameShellDipoleResult:
    """Per-shell raw sums of the velocity-frame (observer-free) statistic.

    Mirrors `dvcorr.estimators.shell_dipole.VelocityCenteredShellDipoleResult`
    field-for-field where the two frames share a quantity (`shell_edges`,
    `shell_centers`, `pair_count`, `monopole`, `dipole`, the `per_center_*`
    breakdown, `n_candidates`, `n_centers`), and adds two velocity-frame-only
    fields (`per_center_speed` in place of `per_center_u`, and the diagnostic
    `per_center_axis_angle`, which has no observer-frame analogue). Everything
    is a RAW sum -- nothing here is normalized; see
    `dvcorr.pipeline.velocity_centered.normalize_stacked_dipole` for the
    downstream normalization, shared with the observer frame.

    Attributes
    ----------
    shell_edges : ndarray, shape (B + 1,)
        The radial bin boundaries passed in, h^-1 Mpc.
    shell_centers : ndarray, shape (B,)
        Midpoints 0.5 * (edge_low + edge_high) of each shell, h^-1 Mpc.
    pair_count : ndarray, shape (B,)
        Realised Sigma_alpha N_alpha,b, occupancy summed over centers. Pure
        geometry -- identical to what `velocity_centered_shell_dipole` would
        report for the SAME center/tracer configuration, since occupancy
        depends only on radius, never on the axis (see
        tests/test_velocity_frame_dipole.py's row-alignment test, which pins
        this invariant across frames explicitly). Empty shells are 0.0.
    monopole : ndarray, shape (B,)
        Sigma_alpha |v_alpha| * N_alpha,b, the l=0 companion (CLAUDE.md hard
        rule 6). See the module docstring's Monopole section: offset to
        ~<|v|> rather than to zero, and NOT flat on a clustered field (it
        inherits the 1 + xi_hh(r) occupancy ratio, as the observer-frame
        monopole does) -- what distinguishes the frames is that no
        finite-distance trend survives dividing that ratio out. Both are real
        differences from the observer-frame monopole, not bugs. Empty shells
        are 0.0.
    dipole : ndarray, shape (B,)
        Sigma_alpha |v_alpha| * A_alpha,b, the raw stacked l=1 numerator.
        Empty shells are 0.0, never NaN.
    per_center_dipole : ndarray, shape (N_c, B)
        |v_alpha| * A_alpha,b, retained per surviving center per shell.
        `dipole == per_center_dipole.sum(axis=0)` by construction.
    per_center_amplitude : ndarray, shape (N_c, B)
        A_alpha,b = Sigma_i Y_10(n_hat_i) per surviving center per shell --
        the pure-geometry accumulator, before the |v_alpha| weight is
        applied. `per_center_speed[:, None] * per_center_amplitude` and
        summing over centers reproduces `dipole`.
    per_center_count : ndarray, shape (N_c, B)
        N_alpha,b, the realized tracer occupancy per surviving center per
        shell. `pair_count == per_center_count.sum(axis=0)` by construction,
        and this array is IDENTICAL to
        `VelocityCenteredShellDipoleResult.per_center_count` when both
        estimators are run on the same center/tracer arrays -- occupancy is
        pure geometry and cannot depend on which axis is used.
    per_center_speed : ndarray, shape (N_c,)
        |v_alpha| for each surviving center, km/s. Positive-definite by
        construction (a norm); the zero-speed case is excluded upstream, see
        `velocity_frame_shell_dipole`'s Raises section.
    per_center_axis_angle : ndarray, shape (N_c,)
        delta_alpha = arccos(clip(z_hat_alpha . n_hat_V,alpha, -1, 1)), the
        angle in RADIANS, range [0, pi], between the center's own flow
        direction and its observer-frame line of sight. This is the ONLY
        place the observer enters the per-center bookkeeping besides the
        (frame-shared) core cut -- see the module docstring's Observer role
        section -- and it is a pure DIAGNOSTIC: it never feeds back into
        `dipole`, `monopole`, `per_center_dipole`, or `per_center_amplitude`.
        delta_alpha ~ 0 means this center's flow points straight along its
        line of sight (the frame-agreement limit); delta_alpha ~ pi/2 means
        the flow is purely transverse, the case the observer frame is blind
        to entirely.
    n_candidates : int
        Number of candidate centers passed in, before the core cut.
    n_centers : int
        Number of surviving centers, N_c, after `core_center_mask` -- the
        SAME cut, applied the same way, as
        `VelocityCenteredShellDipoleResult.n_centers` would report for an
        identical candidate set (see
        dvcorr.pipeline.velocity_frame_comparison.select_shared_centers,
        which selects once and hands the identical arrays to both frames so
        this equality is guaranteed rather than coincidental).

    Notes
    -----
    Cross-check invariants that hold by construction:
    `dipole == per_center_dipole.sum(axis=0)`,
    `pair_count == per_center_count.sum(axis=0)`,
    `monopole == (per_center_speed[:, None] * per_center_count).sum(axis=0)`.
    """

    shell_edges: np.ndarray
    shell_centers: np.ndarray
    pair_count: np.ndarray
    monopole: np.ndarray
    dipole: np.ndarray
    per_center_dipole: np.ndarray
    per_center_amplitude: np.ndarray
    per_center_count: np.ndarray
    per_center_speed: np.ndarray
    per_center_axis_angle: np.ndarray
    n_candidates: int
    n_centers: int


def velocity_frame_shell_dipole(
    s_centers: np.ndarray,
    v_centers: np.ndarray,
    s_tracers: np.ndarray,
    shell_edges: np.ndarray,
    sub_volume_radius: float,
    core_margin: float | None = None,
    observer: np.ndarray | None = None,
) -> VelocityFrameShellDipoleResult:
    """Stack the observer-free velocity-frame dipole over many centers.

    For each surviving center alpha: z_hat_alpha = unit_vector(v_alpha) (the
    center's own flow direction), speed_alpha = |v_alpha|. For each shell b,
    sum the real l=1, m=0 spherical harmonic over the tracers in that shell,

        A_alpha,b = sum_{i in b} Y_10(n_hat_i)
                  = sqrt(3/4pi) * sum_{i in b} (z_hat_alpha . n_hat_i)

    with n_hat_i the unit vector FROM alpha TO tracer i (the same REVERSED
    orientation `velocity_centered_shell_dipole` uses -- see that function's
    Orientation section, unchanged and reused here), then accumulate
    speed_alpha * A_alpha,b, stacked over centers. See the module docstring
    for the full construction, the sign derivation, and the monopole
    interpretation -- all load-bearing, read it before changing anything
    here.

    Structure: this function copies `velocity_centered_shell_dipole`'s
    validation block, KDTree loop, and binning semantics EXACTLY (left-
    closed/right-open shells, the outermost edge folded into the last shell,
    below-innermost-edge tracers excluded, `np.digitize` + `np.bincount`, one
    `cKDTree` built on `s_tracers` and reused across all surviving centers).
    The ONLY differences are the axis (`z_hat_alpha` instead of
    `n_hat_V,alpha`), the scalar (`|v_alpha|` instead of `u_alpha`), and the
    additional `per_center_axis_angle` diagnostic -- everything else,
    including `core_center_mask` itself, is shared unchanged with the
    observer frame so the two frames see the identical candidate set.

    Parameters
    ----------
    s_centers : ndarray, shape (3,) or (N_cand, 3)
        Candidate velocity-object positions, comoving h^-1 Mpc.
    v_centers : ndarray, shape (3,) or (N_cand, 3)
        Candidate 3-D peculiar velocities, km/s, same row order as
        `s_centers`. Must be finite everywhere: a missing peculiar velocity
        is missing data (CLAUDE.md hard rule 5) and the candidate must be
        dropped by the CALLER before this function is reached, never entered
        as v = 0 -- a non-finite entry is a ValueError here, not a silent
        drop.
    s_tracers : ndarray, shape (N_t, 3)
        Density tracer positions, comoving h^-1 Mpc, in the same continuous
        (already carved, no wrapping needed) frame as `s_centers`.
    shell_edges : ndarray, shape (B + 1,)
        Strictly increasing shell boundaries, h^-1 Mpc, defining B shells.
        Left-closed / right-open, [edge_low, edge_high), outermost edge
        folded into the last shell. Outermost edge must not exceed
        conventions.MAX_ANALYSIS_RADIUS.
    sub_volume_radius : float
        Radius of the spherical sub-volume carved around the observer,
        h^-1 Mpc. Must be > 0 and <= conventions.MAX_ANALYSIS_RADIUS. Used
        only by `core_center_mask` -- the observer-anchored candidate cut
        shared with the observer frame (see the module docstring's Observer
        role section); it never enters the statistic itself.
    core_margin : float, optional
        Minimum clearance to the sub-volume boundary required of a surviving
        center (see `core_center_mask`). None (the default) uses
        `shell_edges[-1]`, i.e. r_max. Must be >= 0.
    observer : ndarray, shape (3,), optional
        Observer position. Defaults to conventions.OBSERVER_POSITION. Used
        only for `core_center_mask` and for the `per_center_axis_angle`
        diagnostic -- see the module docstring's Observer role section.

    Returns
    -------
    VelocityFrameShellDipoleResult
        Struct of raw per-shell sums plus the per-center breakdown, the
        axis-angle diagnostic, and shell geometry. Monopole and dipole are
        always returned together (hard rule 6). Zero surviving centers is a
        VALID result, not an error: every per-shell sum is 0.0 (never NaN)
        and the per-center arrays have shape (0, B) / (0,).

    Raises
    ------
    ValueError
        If shell_edges is not 1-D with at least two entries, is not strictly
        increasing, or has an outermost edge exceeding
        conventions.MAX_ANALYSIS_RADIUS; if sub_volume_radius is not in
        (0, conventions.MAX_ANALYSIS_RADIUS]; if core_margin < 0; if
        v_centers.shape does not match s_centers.shape; if v_centers contains
        a non-finite value; or if, AFTER the core cut, any SURVIVING center
        has |v_alpha| == 0.0.

        The zero-speed case is deliberately a loud error rather than a
        silent zero: `unit_vector` returns a zero row for a zero-length
        vector, so a zero-speed center would otherwise silently contribute
        A = 0 and speed = 0 to its shells -- not a crash, but a row that
        still occupies a slot in every `per_center_*` array and therefore
        DILUTES the across-center mean and `center_standard_error`'s
        standard error, exactly as an entered-as-zero missing velocity would
        (CLAUDE.md hard rule 5's "missing velocity is missing data" is the
        same class of bug). Such centers must be dropped upstream, at the
        orchestration level, by
        `dvcorr.pipeline.velocity_centered.RunConfig.min_center_speed` (a
        TUNABLE floor that lives on that config, not here and not in
        `conventions.py`, since it is a data-quality knob rather than a
        frozen convention) -- see
        `dvcorr.pipeline.velocity_frame_comparison.select_shared_centers`,
        which applies it before either estimator ever runs.

    Notes
    -----
    A tracer coincident with its center (r_mag ~ 0) has no direction:
    `unit_vector` returns a zero row, so cos_theta = 0 and it contributes to
    `per_center_count` (occupancy) but nothing to `per_center_amplitude`.
    Identical documented contract to `shell_dipole` and
    `velocity_centered_shell_dipole`'s coincident-neighbor case.
    """
    observer = (
        conventions.OBSERVER_POSITION if observer is None
        else np.asarray(observer, dtype=float)
    )

    edges = np.asarray(shell_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("shell_edges must be 1-D with at least two entries.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("shell_edges must be strictly increasing.")
    if edges[-1] > conventions.MAX_ANALYSIS_RADIUS:
        raise ValueError(
            f"outermost shell edge {edges[-1]} exceeds "
            f"conventions.MAX_ANALYSIS_RADIUS = {conventions.MAX_ANALYSIS_RADIUS}."
        )
    n_bins = edges.size - 1

    if not (0.0 < sub_volume_radius <= conventions.MAX_ANALYSIS_RADIUS):
        raise ValueError(
            "sub_volume_radius must be > 0 and <= conventions.MAX_ANALYSIS_RADIUS "
            f"({conventions.MAX_ANALYSIS_RADIUS}), got {sub_volume_radius}."
        )

    if core_margin is None:
        core_margin = float(edges[-1])
    if core_margin < 0.0:
        raise ValueError(f"core_margin must be >= 0, got {core_margin}.")

    s_centers = np.atleast_2d(np.asarray(s_centers, dtype=float))
    v_centers = np.atleast_2d(np.asarray(v_centers, dtype=float))
    s_tracers = np.atleast_2d(np.asarray(s_tracers, dtype=float))
    n_candidates = s_centers.shape[0]

    if v_centers.shape != s_centers.shape:
        raise ValueError(
            f"v_centers shape {v_centers.shape} does not match "
            f"s_centers shape {s_centers.shape}."
        )
    if not np.all(np.isfinite(v_centers)):
        raise ValueError(
            "v_centers contains a non-finite value; a missing peculiar "
            "velocity is missing data and the candidate must be dropped by "
            "the caller, never entered as 0 (CLAUDE.md hard rule 5)."
        )

    survives = core_center_mask(s_centers, sub_volume_radius, core_margin, observer=observer)
    s_survivors = s_centers[survives]
    v_survivors = v_centers[survives]
    n_centers = s_survivors.shape[0]

    if n_centers > 0:
        survivor_speed = np.linalg.norm(v_survivors, axis=1)
        if np.any(survivor_speed == 0.0):
            raise ValueError(
                "a surviving center has |v_alpha| == 0.0, i.e. an undefined "
                "flow direction: unit_vector would return a zero row, so "
                "this center would silently contribute A = 0 and speed = 0 "
                "-- diluting the across-center mean and standard error "
                "rather than crashing (CLAUDE.md hard rule 5's 'missing "
                "velocity is missing data', the same class of bug). Drop "
                "such centers upstream via "
                "dvcorr.pipeline.velocity_centered.RunConfig.min_center_speed "
                "before calling this function, not here."
            )

    shell_centers = 0.5 * (edges[:-1] + edges[1:])

    per_center_count = np.zeros((n_centers, n_bins), dtype=float)
    per_center_amplitude = np.zeros((n_centers, n_bins), dtype=float)
    per_center_speed = np.zeros(n_centers, dtype=float)
    per_center_axis_angle = np.zeros(n_centers, dtype=float)

    if n_centers > 0:
        z_hat = unit_vector(v_survivors)                            # (N_c, 3), v_hat_alpha
        per_center_speed = np.linalg.norm(v_survivors, axis=1)      # (N_c,), |v_alpha|

        # Diagnostic only (module docstring, Observer role): the angle
        # between this center's own flow axis and its observer line of
        # sight. Never feeds back into the statistic below.
        n_hat_V = unit_vector(s_survivors - observer)                # (N_c, 3)
        axis_dot = np.clip(np.einsum("ij,ij->i", z_hat, n_hat_V), -1.0, 1.0)
        per_center_axis_angle = np.arccos(axis_dot)                  # (N_c,), radians in [0, pi]

        tree = cKDTree(s_tracers) if s_tracers.shape[0] > 0 else None

        for a in range(n_centers):
            if tree is None:
                continue
            s_alpha = s_survivors[a]
            idx = np.asarray(tree.query_ball_point(s_alpha, edges[-1]), dtype=int)
            if idx.size == 0:
                continue
            s_near = s_tracers[idx]

            # Reversed orientation, identical to velocity_centered_shell_dipole:
            # the center in the "center" slot of pair_separation gives
            # r_vec = s_tracer - s_alpha, center -> tracer.
            r_vec, r_mag = pair_separation(s_alpha, s_near)
            cos_theta = mu_cosine(unit_vector(r_vec), z_hat[a])

            in_range = (r_mag >= edges[0]) & (r_mag <= edges[-1])
            bin_index = np.digitize(r_mag, edges) - 1
            bin_index = np.where(bin_index == n_bins, n_bins - 1, bin_index)

            b = bin_index[in_range]
            y10_in = real_y10(cos_theta[in_range])

            per_center_count[a] = np.bincount(b, minlength=n_bins)[:n_bins].astype(float)
            per_center_amplitude[a] = np.bincount(b, weights=y10_in, minlength=n_bins)[:n_bins]

    per_center_dipole = per_center_speed[:, None] * per_center_amplitude

    pair_count = per_center_count.sum(axis=0)
    monopole = (per_center_speed[:, None] * per_center_count).sum(axis=0)
    dipole = per_center_dipole.sum(axis=0)

    return VelocityFrameShellDipoleResult(
        shell_edges=edges,
        shell_centers=shell_centers,
        pair_count=pair_count,
        monopole=monopole,
        dipole=dipole,
        per_center_dipole=per_center_dipole,
        per_center_amplitude=per_center_amplitude,
        per_center_count=per_center_count,
        per_center_speed=per_center_speed,
        per_center_axis_angle=per_center_axis_angle,
        n_candidates=n_candidates,
        n_centers=n_centers,
    )
