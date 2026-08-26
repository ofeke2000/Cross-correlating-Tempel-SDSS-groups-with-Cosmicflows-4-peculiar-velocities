"""
wide_angle.py
-------------
Exact pair geometry for a finite-distance observer, and the numerical Legendre
moments that need it -- everything `dipole.py`'s distant-observer-limit
formulae drop.

The geometry
============
Put the VELOCITY object at `r = r_obs z_hat` and the separation `y` at angle
`arccos(mu)` from `z_hat`. The density point is then at
`r_tilde = r + y`, and

    r_tilde  = sqrt(r_obs^2 + y^2 + 2 r_obs y mu)
    nu       = r_hat . r_tilde_hat = (r_obs + y mu) / r_tilde
    mu_tilde = r_tilde_hat . y_hat = (r_obs mu + y) / r_tilde

As `y / r_obs -> 0` these go to `nu -> 1` and `mu_tilde -> mu`, which is the
distant-observer limit (DOL) and is asserted in `tests/test_theory.py`.

What actually depends on distance
=================================
Not the real-space dipole. Nusser eq. (13)'s
`xi_u delta = -(1/f)(3 Psi'_perp + Psi'_-) mu` involves neither `nu` nor
`mu_tilde` nor `r_tilde`, so the real-space dipole is EXACTLY its DOL value at
any observer distance. Every finite-distance effect on this statistic is a
redshift-space effect, entering through two terms of eq. (15):

- the `xi_wu` pair `-Psi'_perp (2 mu_tilde nu + mu) - (Psi'_- - 2 Psi'_perp)
  mu_tilde^2 mu`, which reduces to `xi_wu` of `dipole.py` in the DOL; and
- the Kaiser-rocket term `-2 (nu Psi_perp + mu mu_tilde Psi_-) / r_tilde`,
  which is EVEN in `mu` in the DOL and therefore contributes only to
  `ell = 0` and `ell = 2` there. Its dipole is a pure finite-distance effect
  and must vanish for `y << r_obs`.

That last statement is the best available diagnostic that this module's
geometry is right: `xi_uu_over_r_tilde_moment(..., ell=1)` sitting non-zero at
`y << r_obs` is a geometry bug, not physics. `xi_uu_over_r_tilde_moment(...,
ell=0)` is meanwhile substantial -- comparable to the dipoles themselves at
`r_obs = 50 h^-1 Mpc`, as Nusser notes -- so the two moments together
distinguish "the term is small" from "the term is being computed wrongly".

Sign of the rocket term
=======================
`xi_uu_over_r_tilde_moment` returns moments of `+2 xi_uu / r_tilde` (Nusser's
Fig. 1 curves 4 and 5). Eq. (15) subtracts it, so the contribution to
`xi^s_u delta` is minus what this function returns. Its DOL monopole is
`-2 (Psi_perp + Psi_-/3) / r_tilde`, Nusser's eq. (16).

Numerical moments
=================
Because `r_tilde` depends on `mu`, none of this factorizes and the moments are
taken by Gauss-Legendre quadrature in `mu` (`legendre_moment`), in Nusser's
normalization `(1/2) Int_-1^1 X(mu) L_ell(mu) dmu`. The integrands are
polynomials in `mu` divided by `r_tilde(mu)`, smooth on [-1, 1], so
`DEFAULT_MU_NODES` is generous rather than marginal.

Shapes
======
`y` is (B,), the quadrature nodes are (M,), and every angular array here is
(B, M) -- separation along rows, angle along columns. The Gorski functions
depend on `y` alone and are broadcast in as (B, 1).

Units
=====
As elsewhere in `dvcorr.theory`: `y`, `r_obs` in h^-1 Mpc, results in h^-1 Mpc
until multiplied by one factor of `a H` (km/s per h^-1 Mpc).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dvcorr.theory.dipole import (
    SPACE_REDSHIFT,
    VALID_SPACES,
    CONVENTION_NUSSER,
    convention_factor,
)
from dvcorr.theory.gorski import GorskiFunctions

#: Gauss-Legendre nodes used for every `mu` moment here. Odd, so `mu = 0` is a
#: node and a `mu`-odd integrand's cancellation is exact rather than nearly so.
DEFAULT_MU_NODES: int = 513


def gauss_legendre_nodes(n_nodes: int = DEFAULT_MU_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [-1, 1], shapes (M,) and (M,).

    Parameters
    ----------
    n_nodes : int
        Number of nodes M.

    Returns
    -------
    mu : ndarray, shape (M,)
    weights : ndarray, shape (M,)
    """
    mu, weights = np.polynomial.legendre.leggauss(int(n_nodes))
    return mu, weights


def legendre_moment(
    values: np.ndarray, mu: np.ndarray, weights: np.ndarray, ell: int, *,
    convention: str = CONVENTION_NUSSER,
) -> np.ndarray:
    """`(1/2) Int_-1^1 X(mu) L_ell(mu) dmu` by Gauss-Legendre quadrature, shape (B,).

    Nusser's multipole normalization by default -- no `(2 ell + 1)`. See
    `dipole.convention_factor`.

    Parameters
    ----------
    values : ndarray, shape (B, M)
        The integrand evaluated at the quadrature nodes, one row per
        separation.
    mu : ndarray, shape (M,)
        Nodes, from `gauss_legendre_nodes`.
    weights : ndarray, shape (M,)
        Matching weights.
    ell : int
        Multipole order.
    convention : str
        `CONVENTION_NUSSER` or `CONVENTION_STANDARD`.

    Returns
    -------
    ndarray, shape (B,)
    """
    from scipy.special import eval_legendre

    # The 1/2 is Nusser's normalization (the module docstring's definition),
    # not a tunable.
    return (
        0.5
        * convention_factor(convention, ell)
        * np.sum(weights * np.asarray(values, dtype=float) * eval_legendre(int(ell), mu), axis=-1)
    )


@dataclass(frozen=True)
class WideAngleGeometry:
    """Exact pair geometry on a (separation, angle) grid -- a holder, not a calculator.

    Built by `wide_angle_geometry`. Every array is (B, M): `y` along rows,
    the `mu` quadrature nodes along columns.

    Attributes
    ----------
    y : ndarray, shape (B, 1)
        Pair separations, h^-1 Mpc, broadcast-shaped for the angular grid.
    mu : ndarray, shape (M,)
        `mu = r_hat . y_hat`, the angle cosine AT THE VELOCITY OBJECT.
    r_obs : float
        Distance from the observer to the velocity object, h^-1 Mpc.
    r_tilde : ndarray, shape (B, M)
        Distance from the observer to the DENSITY point, h^-1 Mpc.
    nu : ndarray, shape (B, M)
        `nu = r_hat . r_tilde_hat`; `-> 1` in the distant-observer limit.
    mu_tilde : ndarray, shape (B, M)
        `mu_tilde = r_tilde_hat . y_hat`; `-> mu` in the distant-observer
        limit.
    """

    y: np.ndarray
    mu: np.ndarray
    r_obs: float
    r_tilde: np.ndarray
    nu: np.ndarray
    mu_tilde: np.ndarray


def wide_angle_geometry(
    y: np.ndarray, mu: np.ndarray, r_obs: float
) -> WideAngleGeometry:
    """Exact `r_tilde`, `nu`, `mu_tilde` for the velocity object at distance `r_obs`.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Pair separations, h^-1 Mpc, strictly positive.
    mu : ndarray, shape (M,)
        Angle cosines at the velocity object, in [-1, 1].
    r_obs : float
        Observer-to-velocity-object distance, h^-1 Mpc, strictly positive.

    Returns
    -------
    WideAngleGeometry
        With arrays of shape (B, M).

    Raises
    ------
    ValueError
        If `r_obs` is not strictly positive. `r_tilde` may still reach zero if
        the density point lands exactly on the observer (`y = r_obs`,
        `mu = -1`), which is a genuine singularity of the rocket term rather
        than an input error; the Gauss-Legendre nodes never include
        `mu = -1` exactly, so the quadrature does not evaluate it.
    """
    if r_obs <= 0.0:
        raise ValueError(f"wide_angle_geometry: r_obs must be > 0, got {r_obs}.")

    y_col = np.atleast_1d(np.asarray(y, dtype=float))[:, None]
    mu = np.asarray(mu, dtype=float)
    r_tilde = np.sqrt(r_obs**2 + y_col**2 + 2.0 * r_obs * y_col * mu)
    return WideAngleGeometry(
        y=y_col,
        mu=mu,
        r_obs=float(r_obs),
        r_tilde=r_tilde,
        nu=(r_obs + y_col * mu) / r_tilde,
        mu_tilde=(r_obs * mu + y_col) / r_tilde,
    )


def xi_uu(geometry: WideAngleGeometry, functions: GorskiFunctions) -> np.ndarray:
    """`xi_uu = nu Psi_perp + mu mu_tilde Psi_-`, shape (B, M), (h^-1 Mpc)^2.

    Nusser eq. (9): the correlation of the two objects' OBSERVER-CENTERED
    radial velocities, in exact geometry. In the DOL it reduces to
    `Psi_perp + mu^2 Psi_-`, which is even in `mu`.

    Parameters
    ----------
    geometry : WideAngleGeometry
    functions : GorskiFunctions
        Evaluated on the same `y` grid, shape (B,).

    Returns
    -------
    ndarray, shape (B, M)
    """
    return geometry.nu * functions.perp[:, None] + (
        geometry.mu * geometry.mu_tilde * functions.minus[:, None]
    )


def xi_uu_over_r_tilde_moment(
    y: np.ndarray,
    r_obs: float,
    functions: GorskiFunctions,
    ell: int,
    *,
    n_mu_nodes: int = DEFAULT_MU_NODES,
    a_hubble: float = 1.0,
    convention: str = CONVENTION_NUSSER,
) -> np.ndarray:
    """`<2 xi_uu / r_tilde>_ell` in exact geometry, shape (B,).

    Curves 4 (`ell = 0`) and 5 (`ell = 1`) of Nusser's Fig. 1. Sign as
    written: eq. (15) SUBTRACTS this, so the contribution to
    `xi^s_u delta` is its negative.

    `ell = 1` is the diagnostic -- it vanishes identically in the DOL, so a
    non-zero value at `y << r_obs` means the geometry is wrong (see the module
    docstring). `ell = 0` reduces to Nusser eq. (16),
    `-2 (Psi_perp + Psi_-/3) / r_tilde`, up to that same overall sign.

    Bias-free: this is a velocity-velocity correlation, so `b` does not enter.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Pair separations, h^-1 Mpc.
    r_obs : float
        Observer-to-velocity-object distance, h^-1 Mpc.
    functions : GorskiFunctions
        Evaluated on the same `y`.
    ell : int
        Multipole order.
    n_mu_nodes : int
        Gauss-Legendre nodes in `mu`.
    a_hubble : float
        `a H` in km/s per h^-1 Mpc; 1.0 leaves the result in h^-1 Mpc.
    convention : str
        Multipole normalization.

    Returns
    -------
    ndarray, shape (B,)
    """
    mu, weights = gauss_legendre_nodes(n_mu_nodes)
    geometry = wide_angle_geometry(y, mu, r_obs)
    integrand = 2.0 * xi_uu(geometry, functions) / geometry.r_tilde
    return float(a_hubble) * legendre_moment(
        integrand, mu, weights, ell, convention=convention
    )


def xi_udelta_wide_angle(
    geometry: WideAngleGeometry,
    functions: GorskiFunctions,
    *,
    b: float = 1.0,
    space: str = SPACE_REDSHIFT,
) -> np.ndarray:
    """Nusser eq. (15) with exact geometry, shape (B, M), h^-1 Mpc.

    The full angular dependence of the redshift-space velocity-density
    correlation, with the duplicated `-3 Psi'_perp mu` of the printed eq. (15)
    line 2 removed (the correct `xi_wu` is eq. 11):

        xi^s_u delta = -(b/f) (3 Psi'_perp + Psi'_-) mu
                       - Psi'_perp (2 mu_tilde nu + mu)
                       - (Psi'_- - 2 Psi'_perp) mu_tilde^2 mu
                       - 2 (nu Psi_perp + mu mu_tilde Psi_-) / r_tilde

    Only the first term carries `b` -- the other three are velocity-velocity
    and bias-free. `SPACE_REAL` returns that first term alone, which has no
    `r_tilde`, `nu` or `mu_tilde` in it: the real-space correlation is
    distance-independent, and this function returns the same values for any
    `r_obs`.

    Parameters
    ----------
    geometry : WideAngleGeometry
    functions : GorskiFunctions
        Evaluated on the same `y` grid, carrying the `f` to use.
    b : float
        Linear bias of the density tracer.
    space : str
        `SPACE_REAL` or `SPACE_REDSHIFT`.

    Returns
    -------
    ndarray, shape (B, M)
        Multiply by `a H` for km/s.

    Raises
    ------
    ValueError
        On an unrecognized `space`.
    """
    if space not in VALID_SPACES:
        raise ValueError(
            f"xi_udelta_wide_angle: space must be one of {VALID_SPACES}, got {space!r}."
        )

    mu = geometry.mu
    perp_prime = functions.perp_prime[:, None]
    minus_prime = functions.minus_prime[:, None]

    out = -(float(b) / functions.f) * (3.0 * perp_prime + minus_prime) * mu
    if space == SPACE_REDSHIFT:
        # xi_wu, exact geometry (Nusser eq. 11 generalized): reduces to
        # -[3 Psi'_perp mu + (Psi'_- - 2 Psi'_perp) mu^3] when nu -> 1,
        # mu_tilde -> mu.
        out = out - perp_prime * (2.0 * geometry.mu_tilde * geometry.nu + mu)
        out = out - (minus_prime - 2.0 * perp_prime) * geometry.mu_tilde**2 * mu
        # The Kaiser-rocket term: even in mu in the DOL, so it leaves the
        # dipole alone there and returns only at finite distance.
        out = out - 2.0 * xi_uu(geometry, functions) / geometry.r_tilde
    return out


def xi_udelta_wide_angle_moment(
    y: np.ndarray,
    r_obs: float,
    functions: GorskiFunctions,
    ell: int,
    *,
    b: float = 1.0,
    space: str = SPACE_REDSHIFT,
    n_mu_nodes: int = DEFAULT_MU_NODES,
    a_hubble: float = 1.0,
    convention: str = CONVENTION_NUSSER,
) -> np.ndarray:
    """`<xi^s_u delta>_ell` at finite observer distance `r_obs`, shape (B,).

    Curve 3 of Nusser's Fig. 1 at `ell = 1`. It should track the DOL dipole
    (`dipole.xi_udelta_dipole`) closely at survey separations -- that
    agreement is Nusser's statement that the DOL is an excellent
    approximation, and it is a claim to CHECK rather than to assume, since
    both this and the DOL formula would look plausible if the geometry were
    wrong.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Pair separations, h^-1 Mpc.
    r_obs : float
        Observer-to-velocity-object distance, h^-1 Mpc.
    functions : GorskiFunctions
        Evaluated on the same `y`, carrying the `f` to use.
    ell : int
        Multipole order.
    b : float
        Linear bias of the density tracer.
    space : str
        `SPACE_REAL` or `SPACE_REDSHIFT`.
    n_mu_nodes : int
        Gauss-Legendre nodes in `mu`.
    a_hubble : float
        `a H` in km/s per h^-1 Mpc; 1.0 leaves the result in h^-1 Mpc.
    convention : str
        Multipole normalization.

    Returns
    -------
    ndarray, shape (B,)
        Same sign convention as `dipole.xi_udelta_dipole`: positive dipole for
        coherent infall, velocity-centered.
    """
    mu, weights = gauss_legendre_nodes(n_mu_nodes)
    geometry = wide_angle_geometry(y, mu, r_obs)
    integrand = xi_udelta_wide_angle(geometry, functions, b=b, space=space)
    return float(a_hubble) * legendre_moment(
        integrand, mu, weights, ell, convention=convention
    )
