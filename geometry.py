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
Box coordinates are periodic with side config.BOX_SIZE. Any difference of two
positions must be reduced by the minimum-image convention

    d -= BOX_SIZE * round(d / BOX_SIZE)

before its norm or its direction is used. This applies to the observer -> object
direction as much as to the pair separation: the observer is a point in the same
periodic box, and the nearest image of a halo is the one it is physically near.
Minimum-image geometry is only unambiguous for separations below
config.MAX_ANALYSIS_RADIUS = BOX_SIZE / 2.
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

    Raises
    ------
    ValueError
        If any input vector has zero (or numerically negligible) norm. A
        zero-length separation has no direction, and returning NaN or 0 here
        would propagate into mu as a silently wrong cosine rather than an
        error. Coincident pairs must be excluded by the caller.
    """
    raise NotImplementedError


def pair_separation(
    s_center: np.ndarray,
    s_others: np.ndarray,
    box_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Separation vectors from one central object to many neighbours.

    Implements the frozen orientation r = s_V - s_T: the vector points FROM
    the central density object TO each velocity object. Reversing it would flip
    the sign of every odd multipole downstream (see config.py).

    Parameters
    ----------
    s_center : ndarray, shape (3,)
        Position of the central (density/Tempel-like) object, s_T, in comoving
        box coordinates, h^-1 Mpc.
    s_others : ndarray, shape (N, 3)
        Positions of the N neighbour (velocity) objects, s_V, same units.
    box_size : float, optional
        Periodic box side, h^-1 Mpc. Defaults to config.BOX_SIZE. Pass a value
        explicitly only to test non-default boxes; pass None for production.

    Returns
    -------
    r_vec : ndarray, shape (N, 3)
        Minimum-image separation vectors s_V - s_T, one per row.
    r_mag : ndarray, shape (N,)
        Euclidean norms |r|, h^-1 Mpc. Always (N,), never scalar.

    Notes
    -----
    The minimum-image reduction is applied to r_vec before its norm is taken,
    so r_mag is bounded by sqrt(3) * box_size / 2 and pairs never "wrap the
    long way round". Callers must restrict shells to
    config.MAX_ANALYSIS_RADIUS, beyond which the nearest image is not unique.
    """
    raise NotImplementedError


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
    normalise, so that an un-normalised input shows up as a |mu| > 1 test
    failure rather than as a quietly rescaled correlation amplitude.

    In the distant-observer limit (r << R = |s_T|) the neighbour's own line of
    sight n_hat_V coincides with n_hat_T and the observed radial velocity of a
    pure infall reduces to u ~= v_inf(r) * mu. Since v_inf < 0 for infall, the
    product u * L_1(mu) = u * mu is negative on BOTH hemispheres, which is the
    origin of the negative dipole. At finite R this degrades as O(r/R): a
    perfectly spherical infall leaks a monopole U_0 = (2r / 3R) v_inf, which is
    why the monopole is measured alongside the dipole rather than assumed zero.
    """
    raise NotImplementedError
