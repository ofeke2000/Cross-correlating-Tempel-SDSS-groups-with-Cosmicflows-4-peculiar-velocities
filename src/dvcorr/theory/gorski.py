"""
gorski.py
---------
The Gorski (1988) velocity correlation functions and the two derivatives every
density-velocity multipole is built from.

Definitions (Nusser 2017 eq. 8, with `x = k y`)
===============================================
    Psi_{perp,par}(y) = (f^2 / 2 pi^2) Int_0^inf dk P(k) K_{perp,par}(k y)

    K_perp(x) = j_1(x) / x
    K_par(x)  = j_0(x) - 2 j_1(x) / x

`Psi_par` and `Psi_perp` are the two independent components of the
velocity-velocity correlation tensor of a potential flow: the correlation of
the velocity components along, and transverse to, the line joining the pair.
Everything below is algebra on those two.

The difference and the derivatives
==================================
Subtracting, and using the recurrence `j_0(x) - 3 j_1(x) / x = -j_2(x)`:

    Psi_-(y) = Psi_par - Psi_perp = -(f^2 / 2 pi^2) Int dk P(k) j_2(k y)

and, with `d/dx [j_1/x] = -j_2/x`, `j_2'(x) = j_1(x) - 3 j_2(x)/x`, and
`d/dy = k d/dx`:

    Psi'_perp(y) = -(f^2 / 2 pi^2) Int dk P(k) k j_2(k y) / (k y)  = -f^2 B(y)
    Psi'_-(y)    = -(f^2 / 2 pi^2) Int dk P(k) k [j_1 - 3 j_2/(k y)] dk
                 = -f^2 (A(y) - 3 B(y))

with `A` and `B` the master integrals of `transforms.py`.

The free consistency check
==========================
Gorski's relation `Psi_-(y) = y Psi'_perp(y)` follows identically from the
definitions, and this module is deliberately arranged so that checking it is
NOT circular. `psi_minus` is built as `psi_parallel - psi_perp`, i.e. from the
`j_0` (`ell = 0`) and `j_1/x` (`ell = 1`) transforms; `psi_perp_prime` is
built from the `j_2` (`ell = 2`) transform. Asserting `Psi_- == y Psi'_perp`
therefore compares three independent FFTLog orders against each other through
a Bessel recurrence, and would catch a phase or normalization error at any one
of them. `tests/test_theory.py` and
`intuition/nusser_dipole_real_and_redshift_space.ipynb` both run it.

Signs and magnitudes, for orientation
=====================================
Over the separations linear theory is used on, `A > 0` and `B > 0`, so
`Psi'_perp < 0` and `Psi_- < 0`, while `Psi_perp > 0` and `Psi_par > 0`. The
dipole nevertheless comes out POSITIVE in Nusser's velocity-centered
convention (`dipole.py`), which is `-1` times this project's group-centered
`xi_Tu,1`.

Units
=====
`Psi` in (h^-1 Mpc)^2, `Psi'` in h^-1 Mpc, for `y` in h^-1 Mpc, `k` in
h Mpc^-1 and `P(k)` in (h^-1 Mpc)^3. Nusser writes distances in km/s, which
hides one factor of `a H` per distance; the conversion is applied once, in
`dipole.py`, and never here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dvcorr.theory import transforms


def psi_perp(y: np.ndarray, k: np.ndarray, pk: np.ndarray, f: float) -> np.ndarray:
    """`Psi_perp(y) = (f^2 / 2 pi^2) Int dk P(k) j_1(k y) / (k y)`, shape (B,), (h^-1 Mpc)^2.

    The transverse velocity correlation function. Positive over the range
    linear theory is used on.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers (h Mpc^-1) and `P(k)` ((h^-1 Mpc)^3).
    f : float
        Growth rate. Enters only as the scalar `f^2`.

    Returns
    -------
    ndarray, shape (B,)
    """
    return float(f) ** 2 * transforms.psi_perp_integral(y, k, pk)


def psi_parallel(y: np.ndarray, k: np.ndarray, pk: np.ndarray, f: float) -> np.ndarray:
    """`Psi_par(y) = (f^2 / 2 pi^2) Int dk P(k) [j_0(k y) - 2 j_1(k y) / (k y)]`, (B,), (h^-1 Mpc)^2.

    The longitudinal velocity correlation function, assembled from the
    `ell = 0` transform and `psi_perp`'s `ell = 1` one -- the `2` is the
    kernel's own coefficient, part of Gorski's definition rather than a
    tunable (CLAUDE.md hard rule 4's pure-mathematics exemption).

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers and `P(k)`.
    f : float
        Growth rate.

    Returns
    -------
    ndarray, shape (B,)
    """
    j0_moment = transforms.bessel_moment(y, k, pk, ell=0, k_power=0) / (2.0 * np.pi**2)
    return float(f) ** 2 * (j0_moment - 2.0 * transforms.psi_perp_integral(y, k, pk))


def psi_minus(y: np.ndarray, k: np.ndarray, pk: np.ndarray, f: float) -> np.ndarray:
    """`Psi_-(y) = Psi_par - Psi_perp`, shape (B,), (h^-1 Mpc)^2.

    Computed as the DIFFERENCE of the two Gorski functions, not through its
    own closed `j_2` form, so that the Gorski identity `Psi_- = y Psi'_perp`
    stays an independent check on the `ell = 2` transform rather than a
    tautology -- see the module docstring. Negative over the range linear
    theory is used on.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers and `P(k)`.
    f : float
        Growth rate.

    Returns
    -------
    ndarray, shape (B,)
    """
    return psi_parallel(y, k, pk, f) - psi_perp(y, k, pk, f)


def psi_perp_prime(y: np.ndarray, k: np.ndarray, pk: np.ndarray, f: float) -> np.ndarray:
    """`Psi'_perp(y) = -f^2 B(y)`, shape (B,), h^-1 Mpc.

    `d Psi_perp / d y`, from the `ell = 2` transform via
    `d/dx [j_1/x] = -j_2/x`. Negative over the range linear theory is used on.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers and `P(k)`.
    f : float
        Growth rate.

    Returns
    -------
    ndarray, shape (B,)
    """
    return -(float(f) ** 2) * transforms.b_integral(y, k, pk)


def psi_minus_prime(y: np.ndarray, k: np.ndarray, pk: np.ndarray, f: float) -> np.ndarray:
    """`Psi'_-(y) = -f^2 (A(y) - 3 B(y))`, shape (B,), h^-1 Mpc.

    `d Psi_- / d y`, from `j_2'(x) = j_1(x) - 3 j_2(x) / x`. The `3` is that
    recurrence, not a parameter.

    This is the ONLY place a `j_1` and a `j_2` transform are combined with
    unequal weight, and it is what makes the exact `j_2` cancellation in
    `dipole.py` a non-trivial statement rather than an identity of the
    integrals.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers and `P(k)`.
    f : float
        Growth rate.

    Returns
    -------
    ndarray, shape (B,)
    """
    a = transforms.a_integral(y, k, pk)
    b = transforms.b_integral(y, k, pk)
    return -(float(f) ** 2) * (a - 3.0 * b)


@dataclass(frozen=True)
class GorskiFunctions:
    """The five Gorski quantities on one `y` grid, at one `f` and one `P(k)`.

    A holder, not a calculator (CLAUDE.md's "classes hold things"): every
    array is precomputed by `gorski_functions`. It exists because the
    wide-angle expressions in `wide_angle.py` need four of the five at once
    and passing them around individually is how a `Psi_perp` ends up where a
    `Psi'_perp` belongs.

    Attributes
    ----------
    y : ndarray, shape (B,)
        The separation grid the rest were evaluated on, h^-1 Mpc.
    f : float
        Growth rate they carry.
    perp : ndarray, shape (B,)
        `Psi_perp`, (h^-1 Mpc)^2.
    parallel : ndarray, shape (B,)
        `Psi_par`, (h^-1 Mpc)^2.
    minus : ndarray, shape (B,)
        `Psi_-  = Psi_par - Psi_perp`, (h^-1 Mpc)^2.
    perp_prime : ndarray, shape (B,)
        `Psi'_perp`, h^-1 Mpc.
    minus_prime : ndarray, shape (B,)
        `Psi'_-`, h^-1 Mpc.
    """

    y: np.ndarray
    f: float
    perp: np.ndarray
    parallel: np.ndarray
    minus: np.ndarray
    perp_prime: np.ndarray
    minus_prime: np.ndarray

    def gorski_identity_residual(self) -> np.ndarray:
        """Relative residual of `Psi_- = y Psi'_perp`, shape (B,), dimensionless.

        The free consistency check of the module docstring, as a number rather
        than an assertion, so a caller can print it. Zero to FFTLog accuracy
        if the `ell = 0, 1, 2` transforms all carry the right phase and
        normalization.

        Returns
        -------
        ndarray, shape (B,)
            `Psi_- / (y Psi'_perp) - 1`, and NaN at `y = 0`. The identity
            holds there (both sides are exactly zero) but as `0/0` it has no
            relative form, so NaN is the honest answer rather than a 0 that
            would read as perfect agreement. Use `np.nanmax` on the result.
        """
        denominator = self.y * self.perp_prime
        return np.where(
            denominator == 0.0, np.nan, self.minus / np.where(denominator == 0.0, 1.0, denominator) - 1.0
        )


def gorski_functions(
    y: np.ndarray, k: np.ndarray, pk: np.ndarray, f: float
) -> GorskiFunctions:
    """Evaluate all five Gorski quantities on `y`, as a `GorskiFunctions`.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers (h Mpc^-1) and `P(k)` ((h^-1 Mpc)^3).
    f : float
        Growth rate.

    Returns
    -------
    GorskiFunctions
        All arrays shape (B,), on the `y` passed in.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    perp = psi_perp(y, k, pk, f)
    parallel = psi_parallel(y, k, pk, f)
    return GorskiFunctions(
        y=y,
        f=float(f),
        perp=perp,
        parallel=parallel,
        minus=parallel - perp,
        perp_prime=psi_perp_prime(y, k, pk, f),
        minus_prime=psi_minus_prime(y, k, pk, f),
    )
