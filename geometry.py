"""
geometry.py
-----------
Geometry primitives for the group-centred pair construction.

Every function here is a pure, array-vectorised free function: no state, no
catalogue knowledge, no I/O. They are the layer the sign convention actually
lives in, so they are deliberately small enough to be fully covered by unit
tests (see tests/test_geometry.py).

Conventions are frozen in config.py and restated in each docstring below; do
not re-derive them locally.

Shape contract
--------------
Throughout this module:

    (3,)    one vector
    (N, 3)  N vectors, one per row
    (N,)    N scalars

Functions accept either the single-vector or the stacked form where the
docstring says so, and broadcast in the usual numpy way. A function documented
as returning (N,) never returns a scalar for N == 1; shapes are stable so that
downstream binning code does not need to special-case single-object input.

Periodic boundary conditions
----------------------------
The box is periodic with side config.BOX_SIZE, but **none of the functions in
this module apply the minimum-image convention.** They are pure Euclidean
geometry on whatever coordinates they are handed.

Periodicity is discharged one level up, at the point where neighbours are
selected: the pipeline carves a sub-volume around each centre and unwraps it
into that centre's local continuous frame, so coordinates reaching this module
may legitimately lie outside [0, BOX_SIZE) and are already the nearest image of
their centre. Re-applying a minimum-image reduction to such coordinates would
wrap correctly-unwrapped neighbours back across the box -- a silent corruption
of r_hat, since the result still looks like a short, plausible separation.

The contract this module depends on, and cannot check:

    every position handed in is the image nearest the relevant centre,
    expressed continuously with it.

Hard rule 3 (PBC everywhere) is therefore honoured by the carving step, not by
these primitives. Do not "fix" a minimum-image reduction into `pair_separation`.
Separations must still stay below config.MAX_ANALYSIS_RADIUS = BOX_SIZE / 2,
beyond which the nearest image is not unique and the carving is ill-defined.
"""

from __future__ import annotations

import numpy as np


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Normalise vectors to unit length.

    Parameters
    ----------
    vec : ndarray, shape (3,) or (N, 3)
        Vector, or stack of N vectors as rows. Not assumed to be a periodic
        displacement -- callers pass an already minimum-imaged difference if
        that is what they mean.

    Returns
    -------
    ndarray, shape (3,) or (N, 3)
        Same shape as the input, each row scaled to unit norm.

    Notes
    -----
    Zero-length vectors are returned as zeros. They are neither divided by nor
    raised on: the division is masked, so no NaN and no warning is produced.

    A zero-length vector has no direction, so this is a placeholder, not an
    answer. A row of zeros propagates into `mu_cosine` as mu = 0 -- a
    perfectly ordinary-looking cosine -- so coincident pairs must still be
    excluded upstream. The reason for returning zeros rather than raising is
    that self-pairs are a routine, expected product of a neighbour query and
    are filtered by the caller on r_mag; making the primitive raise would force
    every caller to pre-filter before it can normalise.
    """
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0.0)


def pair_separation(
    s_center: np.ndarray,
    s_others: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separation vectors from one central object to many neighbours.

    Implements the frozen orientation r = s_V - s_T: the vector points FROM
    the central density object TO each velocity object. Reversing it would flip
    the sign of every odd multipole downstream (see config.py).

    Parameters
    ----------
    s_center : ndarray, shape (3,)
        Position of the central (density/Tempel-like) object, s_T, in comoving
        coordinates, h^-1 Mpc. Must already be in the same unwrapped frame as
        `s_others` (see Notes).
    s_others : ndarray, shape (N, 3)
        Positions of the N neighbour (velocity) objects, s_V, same units and
        same frame.

    Returns
    -------
    r_vec : ndarray, shape (N, 3)
        Separation vectors s_V - s_T, one per row.
    r_mag : ndarray, shape (N,)
        Euclidean norms |r|, h^-1 Mpc. Always (N,), never scalar.

    Notes
    -----
    **This function does NOT apply the minimum-image convention, and adding it
    here would be a bug, not a fix.**

    Periodicity is resolved one level up: the pipeline carves a sub-volume
    around each centre and hands down neighbour coordinates already unwrapped
    into that centre's local frame, so images near a box face arrive with
    coordinates outside [0, BOX_SIZE). A minimum-image reduction applied to
    those coordinates would wrap a legitimately unwrapped neighbour back across
    the box and corrupt both r_mag and r_hat -- silently, since the result is
    still a plausible short separation.

    The invariant the caller owes this function is therefore stronger than
    "coordinates are in the box": every neighbour must be the image nearest
    `s_center`, expressed continuously with it. Given that, a plain difference
    is the correct and complete answer, and hard rule 3 is satisfied by the
    carving step rather than by this primitive.

    Shells must still be restricted to config.MAX_ANALYSIS_RADIUS, beyond which
    the nearest image is not unique and the carving itself is ill-defined.
    """
    s_center = np.asarray(s_center, dtype=float)
    s_others = np.atleast_2d(np.asarray(s_others, dtype=float))

    r_vec = s_others - s_center
    r_mag = np.linalg.norm(r_vec, axis=-1)

    return r_vec, r_mag


def mu_cosine(r_hat: np.ndarray, n_T_hat: np.ndarray) -> np.ndarray:
    """Group-centred angular cosine mu = n_hat_T . r_hat.

    Parameters
    ----------
    r_hat : ndarray, shape (3,) or (N, 3)
        Unit separation direction(s) from the central object to the
        neighbour(s), i.e. unit_vector of the output of `pair_separation`.
    n_T_hat : ndarray, shape (3,) or (N, 3)
        Unit line-of-sight direction observer -> CENTRAL object. This is the
        central object's line of sight, not the neighbour's; using the
        neighbour's is a different (and, at finite distance, unequal)
        statistic. Broadcasts against r_hat, so a single (3,) line of sight may
        be paired with an (N, 3) stack of separations.

    Returns
    -------
    ndarray, shape (N,)
        Cosines in [-1, 1]. mu > 0 places the neighbour beyond the central
        object along the line of sight (far side); mu < 0 places it between the
        observer and the central object (near side).

    Notes
    -----
    Both arguments must already be unit vectors -- this function does not
    normalise, so that an un-normalised input shows up as a wrong correlation
    amplitude at the point of use rather than being quietly rescaled here.

    The returned cosines are clipped to [-1, 1]. The clip is a guard against
    floating-point overshoot of order 1e-16 in the dot product of two unit
    vectors; it is not a normalisation, and it will not rescue genuinely
    un-normalised input, which simply saturates at +-1.

    In the distant-observer limit (r << R = |s_T|) the neighbour's own line of
    sight n_hat_V coincides with n_hat_T and the observed radial velocity of a
    pure infall reduces to u ~= v_inf(r) * mu. Since v_inf < 0 for infall, the
    product u * L_1(mu) = u * mu is negative on BOTH hemispheres, which is the
    origin of the negative dipole. At finite R this degrades as O(r/R): a
    perfectly spherical infall leaks a monopole U_0 = (2r / 3R) v_inf, which is
    why the monopole is measured alongside the dipole rather than assumed zero.
    """
    r_hat = np.atleast_2d(np.asarray(r_hat, dtype=float))
    n_T_hat = np.asarray(n_T_hat, dtype=float)

    mu = np.sum(r_hat * n_T_hat, axis=-1)

    return np.clip(mu, -1.0, 1.0)
