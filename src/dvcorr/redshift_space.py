"""
redshift_space.py
------------------
The redshift-space coordinate transform: displace a halo's position along its
own observer line of sight by its own radial peculiar velocity, so the
velocity-centered dipole (`dvcorr.estimators.shell_dipole
.velocity_centered_shell_dipole`) can be run a second time on the resulting
apparent (redshift-space) positions, alongside the existing real-space run
(`dvcorr.pipeline.velocity_centered`), for a like-for-like comparison built
by `dvcorr.pipeline.redshift_space_comparison`.

This is a coordinate transform ONLY. It is deliberately not an estimator and
not a pipeline stage: it has no notion of centers, tracers, shells, or
n_bar -- those live one level up, in the redshift-space pipeline -- and it is
not wired into `velocity_centered_shell_dipole`, which is UNCHANGED by this
module and stays agnostic to which positions it is handed (it already
accepted `s_centers`/`s_tracers` as plain arrays; that agnosticism is the
design property this module is built to preserve, not to work around).

The transform
-------------
The observer sits at the sub-volume center, AT REST -- there is no observer
peculiar velocity to subtract inside the box (unlike a real CF4-style
Local-Group-frame transform, `dvcorr.conventions`'s Reference frame section).
Every halo i displaces along its own observer-to-halo line of sight by its
own radial velocity:

    s_i = r_i + (v_i . n_hat_i) / 100 * n_hat_i,    n_hat_i = unit_vector(r_i - observer)

with r_i the real (comoving, box-coordinate) position and v_i the 3-D
peculiar velocity. The division by 100 is the SAME definitional literal
documented at `dvcorr.config.cosmology.CosmologyConfig.h` (cosmology.py:92-100):
H0 = 100 h km/s/Mpc, so v [km/s] / 100 = v / (100 h) * h is a distance in
h^-1 Mpc regardless of the numerical value of h. It is not a tunable
constant and is kept inline with this comment, exactly as that precedent
does, rather than promoted to `dvcorr.conventions` or a config field.

What this mapping does and does NOT approximate
------------------------------------------------
MDPL2 snapshot 125 (the catalog this project uses) is z ~= 0, so in the
box's comoving coordinates this displacement is EXACT, not an approximation
to some smaller effect: there is no redshift-to-comoving-distance conversion
anywhere in this simulation-validation pipeline (no cz/H0 linear approximation,
no int dz/E(z) integral) for this transform to be a shortcut for -- that
question belongs entirely to the future real-data (CF4/Tempel) pipeline,
where an object's OBSERVED redshift, not its simulation velocity, is what is
available, and the distance inferred from it is a genuinely different
computation. Do not quote a sub-percent accuracy figure for this function:
within the box, at z ~= 0, `s_i` above is not an approximation to anything
more exact.

n_hat is invariant under this transform -- read this before assuming otherwise
--------------------------------------------------------------------------------
The naive expectation is that displacing a halo changes the direction the
observer sees it in. It does not: the displacement is PURELY RADIAL, so

    s_i = (|r_i - observer| + v_r,i / 100) * n_hat_i + observer,   v_r,i = v_i . n_hat_i

is `observer + (scalar) * n_hat_i` for the SAME n_hat_i used to compute that
scalar -- moving a point along its own radial direction cannot change that
direction (except through the sign flip the through-observer guard below
exists to catch). Consequently

    u_alpha = v_alpha . n_hat_V,alpha

(`dvcorr.conventions.VELOCITY_AXIS_CONVENTION`'s scalar) is IDENTICAL whether
n_hat_V,alpha is computed from a center's real or redshift-space position --
there is no carry-over parameter to thread through `velocity_centered_shell_dipole`,
and no new argument for it to accept. Verified numerically on the project's
toy infall configuration (`tests.test_geometry._infall_shell_with_velocities`):
n_hat agrees between the two spaces to 3.3e-16, and u_alpha to 8.5e-14 --
floating-point round-off, not a residual physical effect. What changes
between the real-space and redshift-space runs is SHELL COMPOSITION only --
which tracers fall in which radial bin around a given center -- never the
center's own axis or scalar weight.

Through-observer flip guard -- a positive floor, not `< 0`
-------------------------------------------------------------
A halo close to the observer moving inward fast enough can have
`|r_i - observer| + v_r,i / 100 <= 0`: the displaced radial coordinate
crosses through the observer and the object's redshift-space position is on
the OPPOSITE side of the observer from its real position -- n_hat is
genuinely, physically reversed for that object, not merely noisy.

Guarding on `s_mag < 0` is NOT sufficient. At `s_mag ~= 0`,
`dvcorr.geometry.unit_vector` returns a ZERO ROW for a zero-length vector
(documented at geometry.py:74-84) -- no NaN, no warning. Such an object would
silently enter the estimator with mu = 0 (or cos_theta = 0, in the
velocity-centered construction) and u = 0 (or a zero amplitude weight),
contributing to `pair_count`/`per_center_count` but nothing to the dipole or
monopole -- a silent, wrong answer, not a crash. `FLIP_GUARD_FLOOR` below is
therefore a small but strictly POSITIVE floor: objects with
`s_mag <= FLIP_GUARD_FLOOR` are dropped entirely (never displaced, never
handed downstream), and the dropped count is returned on `RedshiftSpaceTransform`
rather than silently absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dvcorr import conventions
from dvcorr.geometry import radial_flow_axis, unit_vector

#: Minimum allowed |r| + v_r/100 (h^-1 Mpc) for a displaced position to retain
#: a well-defined observer line of sight (see the "Through-observer flip
#: guard" section above). `s_mag` is an OBSERVER-CENTERED RADIAL DISTANCE,
#: not a pair separation, so it has no relationship to the shell binning
#: (`dvcorr.pipeline.velocity_centered`'s `min_radius`/`radii_step`) at all --
#: those are two unrelated scales, and this floor is re-justified below
#: against the quantity it actually bounds. Chosen to be:
#:   - a negligible volume fraction of the carved sub-volume: this pipeline's
#:     sub-volume has observer-distance scale R_sub = 300 h^-1 Mpc, of which a
#:     1 h^-1 Mpc ball is a (1/300)**3 ~= 4e-8 volume fraction, expected
#:     occupancy n_bar * (4*pi/3) * 1**3 ~= 0.017 halos -- so this floor drops
#:     essentially nothing that is not a genuine through-observer flip,
#:     independently of whatever radial shell binning is in use;
#:   - well ABOVE floating-point round-off on box coordinates of order
#:     BOX_SIZE ~ 1e3 h^-1 Mpc, so it is a physical floor, not a numerical one.
#: A module constant rather than an inline literal (CLAUDE.md hard rule 4).
FLIP_GUARD_FLOOR: float = 1.0  # h^-1 Mpc


def radial_velocity(
    positions: np.ndarray,
    velocities: np.ndarray,
    observer: np.ndarray | None = None,
) -> np.ndarray:
    """Observer-centered signed radial velocity v_r = v . n_hat, per object.

    Thin wrapper around `dvcorr.geometry.radial_flow_axis`, which already
    computes this scalar as its `u` return value; this module never needs
    the accompanying flow-signed axis `z_hat` (that convention is specific to
    velocity-centered shell constructions, `dvcorr.conventions
    .VELOCITY_AXIS_CONVENTION`), so it is discarded here rather than
    re-derived by hand. Used to derive the `v_margin` statistic
    (`v_margin_from_statistic`) over a population BEFORE the through-observer
    flip guard applies -- the guard only matters for the actual position
    displacement, not for measuring how fast objects move radially.

    Parameters
    ----------
    positions : ndarray, shape (3,) or (N, 3)
        Real (comoving, box-coordinate) positions, h^-1 Mpc.
    velocities : ndarray, shape (3,) or (N, 3)
        3-D peculiar velocities, km/s, same row order as `positions`.
    observer : ndarray, shape (3,), optional
        Observer position. Defaults to conventions.OBSERVER_POSITION.

    Returns
    -------
    ndarray, shape (N,)
        v_r, km/s, signed (positive = receding from the observer).
    """
    observer = (
        conventions.OBSERVER_POSITION if observer is None
        else np.asarray(observer, dtype=float)
    )
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))

    n_hat = unit_vector(positions - observer)
    _, v_r = radial_flow_axis(velocities, n_hat)
    return v_r


@dataclass(frozen=True)
class RedshiftSpaceTransform:
    """Result of displacing a batch of objects into redshift space.

    Attributes
    ----------
    s_redshift : ndarray, shape (N_kept, 3)
        Displaced (redshift-space) positions for the objects that cleared the
        through-observer flip guard, comoving h^-1 Mpc.
    kept_mask : ndarray, shape (N,), bool
        True for input rows retained in `s_redshift`, aligned with the
        ORIGINAL `positions`/`velocities` arrays handed to `to_redshift_space`
        -- so `positions[kept_mask]` / `velocities[kept_mask]` are the
        real-space counterparts of `s_redshift`, row for row.
    n_input : int
        Number of objects handed in, before the flip guard.
    n_dropped : int
        Number of objects dropped by the flip guard,
        `n_input - kept_mask.sum()`. Exposed here rather than only printed,
        so a caller (or a test) can assert on it directly.
    flip_guard_floor : float
        The floor actually used, h^-1 Mpc (see `FLIP_GUARD_FLOOR`).
    """

    s_redshift: np.ndarray
    kept_mask: np.ndarray
    n_input: int
    n_dropped: int
    flip_guard_floor: float


def to_redshift_space(
    positions: np.ndarray,
    velocities: np.ndarray,
    observer: np.ndarray | None = None,
    flip_guard_floor: float = FLIP_GUARD_FLOOR,
) -> RedshiftSpaceTransform:
    """Displace positions along each object's own line of sight by v_r / 100.

    s_i = observer + (|r_i - observer| + v_r,i / 100) * n_hat_i, with
    n_hat_i = unit_vector(r_i - observer) and v_r,i = v_i . n_hat_i
    (`radial_velocity`). See the module docstring for the exactness claim (no
    redshift-to-distance conversion is involved at z ~= 0), the n_hat
    invariance this implies, and the through-observer flip guard.

    Parameters
    ----------
    positions : ndarray, shape (3,) or (N, 3)
        Real (comoving, box-coordinate) positions, h^-1 Mpc. Must already be
        in the same continuous frame as `observer` -- same PBC contract as
        `dvcorr.geometry`: this function does not wrap, and does not apply
        the minimum-image convention.
    velocities : ndarray, shape (3,) or (N, 3)
        3-D peculiar velocities, km/s, same row order as `positions`. Must be
        finite everywhere: a missing peculiar velocity is missing data
        (CLAUDE.md hard rule 5) and the object must be dropped by the
        CALLER before this function is reached, never entered as v = 0 -- a
        non-finite entry is a ValueError here, not a silent drop.
    observer : ndarray, shape (3,), optional
        Observer position. Defaults to conventions.OBSERVER_POSITION.
    flip_guard_floor : float, optional
        Minimum allowed `|r_i - observer| + v_r,i / 100` for object i to
        survive (see `FLIP_GUARD_FLOOR`). Must be > 0.

    Returns
    -------
    RedshiftSpaceTransform

    Raises
    ------
    ValueError
        If `velocities.shape` does not match `positions.shape`, if
        `velocities` contains a non-finite value, or if `flip_guard_floor`
        is not strictly positive.
    """
    observer = (
        conventions.OBSERVER_POSITION if observer is None
        else np.asarray(observer, dtype=float)
    )
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))

    if velocities.shape != positions.shape:
        raise ValueError(
            f"velocities shape {velocities.shape} does not match "
            f"positions shape {positions.shape}."
        )
    if not np.all(np.isfinite(velocities)):
        raise ValueError(
            "velocities contains a non-finite value; a missing peculiar "
            "velocity is missing data and the object must be dropped by "
            "the caller, never entered as 0 (CLAUDE.md hard rule 5)."
        )
    if flip_guard_floor <= 0.0:
        raise ValueError(f"flip_guard_floor must be > 0, got {flip_guard_floor}.")

    n_hat = unit_vector(positions - observer)
    r_mag = np.linalg.norm(positions - observer, axis=-1)
    v_r = radial_velocity(positions, velocities, observer=observer)

    s_mag = r_mag + v_r / 100.0  # H0 = 100 h km/s/Mpc -> h^-1 Mpc; see module docstring

    kept_mask = s_mag > flip_guard_floor
    n_input = positions.shape[0]
    n_dropped = int(n_input - np.count_nonzero(kept_mask))

    s_redshift = observer + s_mag[kept_mask, None] * n_hat[kept_mask]

    return RedshiftSpaceTransform(
        s_redshift=s_redshift,
        kept_mask=kept_mask,
        n_input=n_input,
        n_dropped=n_dropped,
        flip_guard_floor=flip_guard_floor,
    )


def v_margin_from_statistic(
    v_r: np.ndarray,
    statistic: str,
    percentile: float,
) -> float:
    """The v_margin statistic (km/s) driving the tracer buffer and center margin.

    Defined on |v_r| -- the RADIAL velocity magnitude, never the 3-D speed
    |v|: displacement (and hence how far the tracer/center field can leak
    across the R_sub boundary) is driven by v_r alone, and a percentile
    computed on |v| would systematically UNDER-cover the actual radial
    displacement (|v| >= |v_r| always), silently dropping fewer objects than
    the margin is meant to guarantee.

    Parameters
    ----------
    v_r : ndarray, shape (N,)
        Signed radial velocities, km/s, e.g. from `radial_velocity`.
    statistic : {"max", "percentile"}
        "max": v_margin = max(|v_r|) -- the TRUE worst case; every object's
        displacement is covered by construction, so no further velocity cut
        is needed anywhere downstream.
        "percentile": v_margin = the `percentile`-th percentile of |v_r|
        (e.g. 99.9) -- cheaper (a much smaller buffer/margin) but no longer a
        true bound, so `dvcorr.pipeline.redshift_space_comparison` additionally
        drops centers with |v_r| above this threshold GLOBALLY when this
        statistic is selected (see that module's docstring for why the
        global cut is legitimate there while a displaced-position cut is
        not).
    percentile : float
        Percentile in [0, 100], used only when `statistic == "percentile"`.

    Returns
    -------
    float
        v_margin, km/s.

    Raises
    ------
    ValueError
        If `statistic` is not one of the two supported values.
    """
    abs_v_r = np.abs(np.asarray(v_r, dtype=float))
    if statistic == "max":
        return float(np.max(abs_v_r))
    if statistic == "percentile":
        return float(np.percentile(abs_v_r, percentile))
    raise ValueError(
        f"unknown v_margin statistic {statistic!r}; expected 'max' or 'percentile'."
    )
