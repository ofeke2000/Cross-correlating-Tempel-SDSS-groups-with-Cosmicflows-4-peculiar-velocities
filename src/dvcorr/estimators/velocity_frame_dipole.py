"""
velocity_frame_dipole.py
--------------------------
Velocity-frame dipole: an OBSERVER-FREE variant of the velocity-centered
zeta_1 statistic (`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`).

CONTEXT / MOTIVATION
---------------------
The observer-frame estimator (zeta, Nusser 2017 eq. 23-24, implemented in
`shell_dipole.py`) puts a velocity object alpha at the center of each shell
and axes it on z_hat_obs,alpha = sign(u_alpha) * n_hat_V,alpha -- the
component of the center's motion that the observer can see, along the line of
sight n_hat_V,alpha = observer -> s_alpha
(`dvcorr.conventions.VELOCITY_AXIS_CONVENTION`). That axis choice is forced by
what a peculiar-velocity survey actually measures: a line-of-sight (radial)
speed u_alpha = v_alpha . n_hat_V,alpha, never the full 3-vector. The axis is
therefore the true flow direction PROJECTED onto, and collapsed into, a single
finite-distance line -- an artifact of having an observer at all, not a
property of the flow.

This module removes the observer from the statistic entirely. For each center
alpha it defines the axis as the center's OWN, FULL flow direction,

    z_hat_alpha = v_alpha / |v_alpha|          (v_hat_alpha)

and uses the FULL speed |v_alpha| as the scalar, in place of the observer
frame's radial speed |u_alpha|. The two frames are therefore the SAME
construction -- axis along the motion, weight the speed along that axis --
differing only in whether the motion is the full 3-vector or its radial
projection. Comparing them quantifies exactly what is lost by having a
finite-distance observer restricted to radial velocities: in the ideal
spherical-infall, distant-observer limit the two coincide, because a purely
radial flow has v_hat_alpha -> sign(u_alpha) n_hat_V,alpha and |v_alpha| ->
|u_alpha| (see tests/test_velocity_frame_dipole.py's frame-agreement test,
item 2, which pins this on real numbers -- INCLUDING for an inbound flow,
where both axes point back toward the observer together). Away from that
limit, the divergence between the frames measures projection geometry -- the
TRANSVERSE component of v_alpha, invisible to the observer frame -- plus the
center's own bulk motion.

Read the two frames' quantities literally, because it matters for interpreting
any amplitude difference between them: zeta_velframe correlates the center's
FULL SPEED with density-along-its-own-flow-direction, while zeta_obsframe
correlates its RADIAL SPEED with density-along-the-radial-part-of-that-flow.
Both scalars are positive-definite and |u_alpha| <= |v_alpha| always, so an
amplitude difference is partly that magnitude effect and not purely a rotation
of the axis -- do not attribute the whole gap to axis geometry without checking
the speed-vs-radial-speed piece first.

A THIRD contributor, often the largest of the three, and easy to miss because
neither of the two above names it: this estimator is FULLY SELF-ALIGNED, and
the observer frame is only PARTIALLY so. Both axes now derive something from
the center's own velocity -- but the velocity frame takes the whole direction
from it, while the observer frame takes only a SIGN (sign(u_alpha)) and gets
the rest, the line n_hat_V,alpha, from the center's POSITION. That remaining
positional dependence is the dilution: the observer-frame axis coincides with
the true flow direction only in the distant-observer limit. Concretely: stack
the amplitude A_alpha,b over many centers whose true flow direction is
uncorrelated with n_hat_V,alpha, and the observer frame's per-center cos_theta
values are effectively drawn from a distribution with <cos_theta^2> ~= 1/3
(the same shell-averaging dilution `geometry.mu_cosine`'s docstring describes
for mu), while the velocity frame's cos_theta is 1 for EVERY center, by
construction -- no averaging, no dilution, because the tracer that sits at a
given angle relative to z_hat_alpha is being measured with the SAME vector
that defines z_hat_alpha. On tests/test_velocity_frame_dipole.py's sign-gate
toy (where this effect is not diluted by anything else) this alone is worth a
factor of 3 in the raw per-center amplitude -- see that test's docstring for
the full derivation. This is NOT the frame-agreement limit (which concerns
whether the two axes literally coincide) and NOT the |v| vs. |u| effect (which
concerns the scalar, not the amplitude) -- it is a THIRD, independent
enhancement of the velocity frame's amplitude relative to the observer
frame's, present even when the two axes are only loosely correlated. `random_axis_null_dipoles`
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
      below) -- the angle between this frame's axis and the observer frame's
      axis for the same center. It never touches `dipole`, `monopole`,
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

and it sits at the MEAN HALO SPEED (hundreds of km/s), NOT at zero: |v| is
positive-definite and has no near/far cancellation available to it. Since the
signed axis landed (`dvcorr.conventions.VELOCITY_AXIS_CONVENTION`) the SAME
is true of the observer frame, whose weight is now the radial speed
|u_alpha| -- its monopole sits near <|u|> rather than near zero. The two
frames' monopoles are therefore now the same KIND of quantity, differing
only in |u| vs. |v|, which is what makes them worth reading side by side.

Do NOT expect either monopole to be flat in r on a clustered tracer field.
The occupancy ratio above IS 1 + xi_hh(r), the tracer autocorrelation, so
BOTH frames' monopoles inherit its shape; on the first MDPL2 run
(notebooks/06_velocity_frame_comparison.ipynb) that ratio falls steeply over
r = 7.5 -> 57.5 h^-1 Mpc, dragging the raw monopoles down with it. Flatness
is what the UNCLUSTERED (Poisson) case gives, not this one.

The diagnostic content of these monopoles is therefore neither "is it near
zero" (neither is) nor "is it flat" (neither is, on a clustered field), but
what survives after the occupancy ratio is divided out. What survives, in
BOTH frames, is a genuine speed-density correlation (fast halos
preferentially living in denser environments), plus shell incompleteness --
the boundary-truncation mechanism `core_center_mask` already guards against.
On the first MDPL2 run with the signed axis
(notebooks/06_velocity_frame_comparison.ipynb) the two residuals decline by
-7.9% and -7.8% over r = 7.5 -> 57.5 h^-1 Mpc: the SAME effect seen twice,
not a difference between the frames.

WHAT THIS MONOPOLE NO LONGER SHOWS -- read before using it as the
finite-distance diagnostic
------------------------------------------------------------------------
The observer frame's finite-distance (2r/3R) leakage documented in
`geometry.mu_cosine` and `shell_dipole.py` is a SIGNED effect. It lived in
Sigma_alpha u_alpha N_alpha,b, a quantity sitting near zero (<u> = -2.67 km/s
on that run), so an ~11 km/s trend in r stood out plainly against it. The
flow-signed axis moves that sign into z_hat and leaves |u_alpha| as the
weight, and |u| discards exactly the sign the leakage lived in -- so the
speed-weighted observer-frame monopole does NOT isolate it any more, and any
residual trend is buried under a ~250 km/s offset.

That is a real loss of diagnostic content in the ell=0 companion, not a
subtlety to gloss over, and it is why
`dvcorr.estimators.shell_dipole.VelocityCenteredShellDipoleResult
.per_center_u` is retained SIGNED. The leakage diagnostic is fully
recoverable from the public fields:

    signed_monopole = (per_center_u[:, None] * per_center_count).sum(axis=0)

normalized exactly as `monopole` is. notebooks/06's summary cell computes and
prints this alongside the speed-weighted pair, and the trend still survives
the occupancy division there. If that check becomes a routine pipeline step
rather than a notebook read-out, promote the line to a library function
rather than copying it a second time.

See `dvcorr.pipeline.velocity_frame_comparison.make_comparison_figure`'s
monopole panel, which plots the two frames' monopoles on separate y-axes
because they sit a projection factor apart (<|u|> vs. <|v|>) and a shared
axis would compress the smaller one.

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
from dvcorr.geometry import mu_cosine, pair_separation, radial_flow_axis, unit_vector
from dvcorr.estimators.shell_dipole import _neighbors_by_center, core_center_mask, real_y10

#: Ceiling on `per_center_axis_angle`: cos(delta) = |u_alpha| / |v_alpha| is
#: non-negative by construction, so delta cannot exceed pi/2. Checked, not
#: assumed -- see `velocity_frame_shell_dipole`.
_MAX_AXIS_ANGLE = np.pi / 2.0
#: Float slack on that ceiling: arccos near its endpoint amplifies the ~1e-16
#: rounding in a unit-vector dot product, so the check is not exact.
_AXIS_ANGLE_TOL = 1e-9


@dataclass(frozen=True)
class VelocityFrameShellDipoleResult:
    """Per-shell raw sums of the velocity-frame (observer-free) statistic.

    Mirrors `dvcorr.estimators.shell_dipole.VelocityCenteredShellDipoleResult`
    field-for-field where the two frames share a quantity (`shell_edges`,
    `shell_centers`, `pair_count`, `monopole`, `dipole`, the `per_center_*`
    breakdown including `per_center_speed`, `n_candidates`, `n_centers`). The
    frames differ in what `per_center_speed` HOLDS -- the full |v_alpha| here,
    the radial |u_alpha| there -- and this result carries one extra field with
    no observer-frame analogue, the diagnostic `per_center_axis_angle`.
    There is no `per_center_u` here: the observer frame retains the signed
    u_alpha because its axis sign is derived from it, and this frame has no
    such sign to record. Everything
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
        monopole does). Both properties are now shared with the observer
        frame, whose weight is likewise a speed; what distinguishes the two
        is that no finite-distance trend survives dividing the occupancy
        ratio out here. Empty shells are 0.0.
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
    per_center_direction_sum : ndarray, shape (N_c, B, 3)
        S_alpha,b = Sigma_{i in b} n_hat_i, the raw VECTOR sum of the tracer
        directions in each shell -- `per_center_amplitude` before it is
        projected onto an axis. The two are related exactly, not
        approximately:

            A_alpha,b(z_hat) = sqrt(3/4pi) * (z_hat . S_alpha,b)

        because Y_10 is LINEAR in the direction cosine, so the sum over
        tracers commutes with the projection. Hence
        `per_center_amplitude == sqrt(3/4pi) * einsum(z_hat, S)` for the
        axis this run used (pinned in tests/test_velocity_frame_dipole.py).

        This is what makes an axis-randomized null a RECOMBINATION rather
        than a second estimator pass: the KDTree query and the per-tracer
        unit vectors -- all of the cost -- depend only on the positions,
        which no null touches. Re-projecting S onto a fresh set of axes
        costs one einsum per realization, so the many realizations a null
        band needs are affordable at all (see
        `dvcorr.pipeline.velocity_frame_comparison.random_axis_null_dipoles`).
        Cheap to carry: (N_c, B, 3) floats, ~2 MB for the production
        N_c ~ 7700, B = 13.
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
        delta_alpha = arccos(clip(z_hat_alpha . z_hat_obs,alpha, -1, 1)), the
        angle in RADIANS between this frame's axis and the OBSERVER frame's
        axis for the same center, z_hat_obs,alpha = sign(u_alpha) *
        n_hat_V,alpha (`dvcorr.conventions.VELOCITY_AXIS_CONVENTION`). This
        is the ONLY place the observer enters the per-center bookkeeping
        besides the (frame-shared) core cut -- see the module docstring's
        Observer role section -- and it is a pure DIAGNOSTIC: it never feeds
        back into `dipole`, `monopole`, `per_center_dipole`, or
        `per_center_amplitude`.

        Range is [0, pi/2], NOT [0, pi]: cos(delta_alpha) =
        v_hat_alpha . sign(u_alpha) n_hat_V,alpha = |u_alpha| / |v_alpha|,
        which is non-negative by construction. That ceiling is the point of
        measuring against the signed axis rather than the bare line of
        sight -- an INBOUND center used to report delta ~ pi (axes
        "opposite") when in fact the two frames agreed perfectly about which
        way it was moving; the sign was an artifact of the unsigned
        reference direction, not a disagreement. delta_alpha ~ 0 now means
        the two frames' axes coincide (the frame-agreement limit, reached by
        any purely radial flow in either direction); delta_alpha ~ pi/2 means
        the flow is almost purely transverse, the case the observer frame is
        blind to entirely.

        For flow directions distributed isotropically about the line of
        sight, cos(delta) = |c| with c uniform on [-1, 1], so P(delta) ~
        sin(delta) on [0, pi/2] -- the reference curve
        `dvcorr.pipeline.velocity_frame_comparison.make_angle_diagnostic_figure`
        overlays.
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
    `monopole == (per_center_speed[:, None] * per_center_count).sum(axis=0)`,
    `per_center_amplitude == sqrt(3/4pi) * einsum("ac,abc->ab", z_hat,
    per_center_direction_sum)` with z_hat = unit_vector(v_centers).
    """

    shell_edges: np.ndarray
    shell_centers: np.ndarray
    pair_count: np.ndarray
    monopole: np.ndarray
    dipole: np.ndarray
    per_center_dipole: np.ndarray
    per_center_amplitude: np.ndarray
    per_center_direction_sum: np.ndarray
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
    below-innermost-edge tracers excluded, the r = 0 self-pair excluded
    explicitly on `r_mag > 0`, `np.digitize` + `np.bincount`, one `cKDTree`
    built on `s_tracers` and reused across all surviving centers).
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
    RuntimeError
        If any `per_center_axis_angle` exceeds pi/2 (beyond float slack).
        That is an internal invariant, not user input: cos(delta_alpha) =
        |u_alpha| / |v_alpha| is non-negative by construction, so an angle
        above the ceiling means the observer-frame axis lost its
        sign(u_alpha) factor. It is checked because the failure is silent --
        a delta of ~pi looks like a perfectly plausible "axes point opposite
        ways" reading of the diagnostic figure.

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
    per_center_direction_sum = np.zeros((n_centers, n_bins, 3), dtype=float)
    per_center_speed = np.zeros(n_centers, dtype=float)
    per_center_axis_angle = np.zeros(n_centers, dtype=float)

    if n_centers > 0:
        z_hat = unit_vector(v_survivors)                            # (N_c, 3), v_hat_alpha
        per_center_speed = np.linalg.norm(v_survivors, axis=1)      # (N_c,), |v_alpha|

        # Diagnostic only (module docstring, Observer role): the angle
        # between this center's own flow axis and the OBSERVER FRAME'S axis
        # for the same center, z_hat_obs = sign(u) n_hat_V
        # (conventions.VELOCITY_AXIS_CONVENTION) -- not the bare line of
        # sight, which is only defined up to a sign. Never feeds back into
        # the statistic below.
        n_hat_V = unit_vector(s_survivors - observer)                # (N_c, 3)
        z_hat_obs, _ = radial_flow_axis(v_survivors, n_hat_V)        # (N_c, 3)
        axis_dot = np.clip(np.einsum("ij,ij->i", z_hat, z_hat_obs), -1.0, 1.0)
        per_center_axis_angle = np.arccos(axis_dot)                  # (N_c,), radians in [0, pi/2]

        # Invariant, checked rather than assumed: cos(delta) = v_hat . sign(u)
        # n_hat_V = |u| / |v| >= 0, so delta can never exceed pi/2. An angle
        # above the ceiling means the axis sign was dropped somewhere (the
        # pre-signed-axis behaviour, which reported delta ~ pi for inbound
        # centers) -- a silent misreading of the angle diagnostic, not a
        # crash, so it is caught here at the one place delta is produced.
        if np.any(per_center_axis_angle > _MAX_AXIS_ANGLE + _AXIS_ANGLE_TOL):
            raise RuntimeError(
                "per_center_axis_angle exceeded pi/2 "
                f"(max {per_center_axis_angle.max():.6f} rad); cos(delta) = "
                "|u_alpha| / |v_alpha| is non-negative by construction, so "
                "this means z_hat_obs lost its sign(u_alpha) factor "
                "(conventions.VELOCITY_AXIS_CONVENTION)."
            )

        tree = cKDTree(s_tracers) if s_tracers.shape[0] > 0 else None

        for a, idx in _neighbors_by_center(tree, s_survivors, edges[-1]):
            if idx.size == 0:
                continue
            s_alpha = s_survivors[a]
            s_near = s_tracers[idx]

            # Reversed orientation, identical to velocity_centered_shell_dipole:
            # the center in the "center" slot of pair_separation gives
            # r_vec = s_tracer - s_alpha, center -> tracer.
            r_vec, r_mag = pair_separation(s_alpha, s_near)
            n_hat_i = unit_vector(r_vec)
            cos_theta = mu_cosine(n_hat_i, z_hat[a])

            # r_mag > 0 drops the center's own self-pair (centers are
            # subsampled from the tracer array, so there is exactly one per
            # center at zero separation, with no direction and hence a
            # spurious cos_theta = 0). Identical term and identical rationale
            # to `dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`;
            # a no-op unless edges[0] == 0
            # (`dvcorr.config.ShellConfig.include_zero_bin`).
            in_range = (r_mag > 0.0) & (r_mag >= edges[0]) & (r_mag <= edges[-1])
            bin_index = np.digitize(r_mag, edges) - 1
            bin_index = np.where(bin_index == n_bins, n_bins - 1, bin_index)

            b = bin_index[in_range]
            y10_in = real_y10(cos_theta[in_range])

            per_center_count[a] = np.bincount(b, minlength=n_bins)[:n_bins].astype(float)
            per_center_amplitude[a] = np.bincount(b, weights=y10_in, minlength=n_bins)[:n_bins]

            # S_alpha,b = Sigma_i n_hat_i, the same shell binning applied to
            # the direction vectors themselves rather than to their projection
            # on this run's axis. Three more bincounts on an index array that
            # is already computed; it buys every axis-randomized null for free
            # downstream (see the dataclass docstring).
            n_hat_in = n_hat_i[in_range]
            for component in range(3):  # 3: the spatial dimensions of n_hat_i
                per_center_direction_sum[a, :, component] = np.bincount(
                    b, weights=n_hat_in[:, component], minlength=n_bins
                )[:n_bins]

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
        per_center_direction_sum=per_center_direction_sum,
        per_center_count=per_center_count,
        per_center_speed=per_center_speed,
        per_center_axis_angle=per_center_axis_angle,
        n_candidates=n_candidates,
        n_centers=n_centers,
    )
