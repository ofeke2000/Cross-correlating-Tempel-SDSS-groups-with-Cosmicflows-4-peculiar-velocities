"""
dipole.py
---------
The first moment of the velocity-density cross-correlation, `<xi_u delta>_1`,
in real and in redshift space -- the public API of `dvcorr.theory`.

The result, first
=================
After an exact cancellation (below), both spaces collapse onto ONE shape:

    <xi_u delta_g>_1   = (a H) (f / 3)  b            A(y)         real space
    <xi^s_u delta_g>_1 = (a H) (f / 3) (b + 3 f / 5) A(y)         redshift space

with `A` the `j_1` master integral of `transforms.py`. Linear redshift-space
distortions do exactly one thing to the dipole: `b -> b + 3 f / 5`, a
SCALE-INDEPENDENT amplitude boost. The ratio between the two is
`1 + 3 f / (5 b) = 1 + 3 beta / 5` -- pure `beta`, no `sigma_8`, flat in `y`,
and a measured ratio that is NOT flat is a bug rather than a discovery.

How it is derived, and the check that pins it
=============================================
Nusser (2017) eq. (13) gives the real-space angular dependence,

    xi_u delta(y, mu) = -(1/f) (3 Psi'_perp + Psi'_-) mu

whose `ell = 1` moment (using `<mu^2> = 1/3`) is `-(1/f)(Psi'_perp +
Psi'_-/3)`. Redshift space adds the dipole of `xi_wu` (eq. 11 in the
distant-observer limit),

    xi_wu = -[3 Psi'_perp mu + (Psi'_- - 2 Psi'_perp) mu^3]
    <xi_wu>_1 = -(3/5 Psi'_perp + 1/5 Psi'_-)                  [<mu^4> = 1/5]

giving Nusser's eq. (17), which as PRINTED carries the wrong sign on the
`Psi'_-` coefficient. The corrected version is

    <xi^s_u delta>_1 = -(3/5 + 1/f) Psi'_perp - (1/5 + 1/(3f)) Psi'_-

and substituting `Psi'_perp = -f^2 B`, `Psi'_- = -f^2 (A - 3B)` makes the
`B` coefficient

    (3/5 + 1/f) - 3 (1/5 + 1/(3f)) = 0                EXACTLY, for any f

so only `j_1` survives. Carrying a bias on the density leg (the `xi_wu` term
is velocity-velocity and therefore bias-free) leaves the coefficient
`b/f + 3/5 - 3 (b/(3f) + 1/5) = 0` -- still exactly zero, for any `b` too.

That cancellation is the sharpest available check on the sign fix, and it is
the reason `dipole_from_gorski` exists alongside `xi_udelta_dipole`: the
former evaluates the UN-cancelled expression from `Psi'_perp` and `Psi'_-`,
the latter the cancelled one from `A` alone, and they must agree to machine
precision. With the printed `-1/(3f)` they do not: a residual
`-(2/(3f)) Psi'_-` survives, contaminating the dipole with `j_2` and making
the redshift/real ratio scale-DEPENDENT.

The third term of Nusser's eq. (15), `-2 (nu Psi_perp + mu mu_tilde
Psi_-)/r_tilde`, contributes only to `ell = 0` and `ell = 2` in the
distant-observer limit -- the Kaiser rocket is absent from the dipole by
construction. It returns at finite distance; see `wide_angle.py`.

Parameter dependence
====================
Writing `P = sigma_8^2 P_hat` and `A = sigma_8^2 A_hat`,

    <xi^s_u delta_g>_1(y) = (a H / 3) [ (f sigma_8)(b sigma_8)
                                        + (3/5) (f sigma_8)^2 ] A_hat(y)

which is the structure of Nusser's eq. (35) with `sigma_8^g = b sigma_8`. The
SHAPE `A_hat(y)` does not depend on `f`, `sigma_8` or `b` at all -- only the
scalar in front moves, which is what makes the collapse test in
`intuition/nusser_dipole_real_and_redshift_space.ipynb` meaningful.

Two conventions, kept apart by name
===================================
Nusser normalizes multipoles as `(1/2) Int xi L_ell dmu`, with NO
`(2 ell + 1)`. The community standard carries `(2 ell + 1)/2`, a factor of 3
for the dipole. `CONVENTION_NUSSER` is the default here because Nusser's
figures are what these curves are checked against;
`conventions.nusser_multipole_sign` is a SEPARATE matter (orientation, not
normalization) and is never applied inside this module.

Sign
====
`xi_udelta_dipole` is POSITIVE for coherent infall -- Nusser centers on the
velocity object, so the density excess sits at positive `mu`. This project's
group-centered `xi_Tu,1` is NEGATIVE for infall (CLAUDE.md hard rule 2). The
two are the same physics with the pair vector reversed; convert with
`conventions.nusser_multipole_sign(1)` at the call site, explicitly.
"""

from __future__ import annotations

import numpy as np

from dvcorr.config import CosmologyConfig, scale_factor_hubble
from dvcorr.theory import transforms
from dvcorr.theory.gorski import GorskiFunctions

#: Real space: no redshift-space distortion on the density leg.
SPACE_REAL: str = "real"

#: Redshift space: linear (Kaiser) distortion on the density leg only. The
#: velocity field itself is undistorted, which is why the boost is a pure
#: bias shift.
SPACE_REDSHIFT: str = "redshift"

VALID_SPACES: tuple[str, str] = (SPACE_REAL, SPACE_REDSHIFT)

#: Nusser (2017)'s multipole normalization: `(1/2) Int xi L_ell dmu`.
CONVENTION_NUSSER: str = "nusser"

#: The community-standard normalization: `((2 ell + 1)/2) Int xi L_ell dmu`.
CONVENTION_STANDARD: str = "standard"

VALID_CONVENTIONS: tuple[str, str] = (CONVENTION_NUSSER, CONVENTION_STANDARD)


def convention_factor(convention: str, ell: int) -> float:
    """Multipole-normalization factor relative to Nusser's, dimensionless.

    Parameters
    ----------
    convention : str
        `CONVENTION_NUSSER` or `CONVENTION_STANDARD`.
    ell : int
        Multipole order.

    Returns
    -------
    float
        1.0 for Nusser's convention, `2 ell + 1` for the standard one.

    Raises
    ------
    ValueError
        On an unrecognized `convention` -- deliberately, rather than falling
        back to a default: a typo here is a silent factor of 3.
    """
    if convention == CONVENTION_NUSSER:
        return 1.0
    if convention == CONVENTION_STANDARD:
        return float(2 * int(ell) + 1)
    raise ValueError(
        f"convention_factor: convention must be one of {VALID_CONVENTIONS}, got {convention!r}."
    )


def kaiser_dipole_bias(b: float, f: float, space: str = SPACE_REDSHIFT) -> float:
    """The effective bias the dipole sees: `b` in real space, `b + 3f/5` in redshift space.

    The entire content of linear RSD for this statistic. The `3/5` is
    `<mu_k^2 L_1(mu_k)> / <mu_k L_1(mu_k)>` from angle-averaging the Kaiser
    factor `b + f mu_k^2` against `mu_k` -- pure Legendre algebra, part of the
    derivation (CLAUDE.md hard rule 4's pure-mathematics exemption).

    Parameters
    ----------
    b : float
        Linear bias of the density tracer.
    f : float
        Growth rate.
    space : str
        `SPACE_REAL` or `SPACE_REDSHIFT`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        On an unrecognized `space`.
    """
    if space == SPACE_REAL:
        return float(b)
    if space == SPACE_REDSHIFT:
        return float(b) + 3.0 * float(f) / 5.0
    raise ValueError(f"kaiser_dipole_bias: space must be one of {VALID_SPACES}, got {space!r}.")


def dipole_amplitude(
    f: float,
    b: float = 1.0,
    space: str = SPACE_REDSHIFT,
    *,
    a_hubble: float = 1.0,
    convention: str = CONVENTION_NUSSER,
) -> float:
    """The scalar in front of `A(y)`: `a H (f/3) * kaiser_dipole_bias(b, f, space)`.

    Everything in the dipole prediction that is NOT a function of `y`. Split
    out so a parameter scan can multiply one cached `A(y)` rather than
    re-running the transform, and so the collapse test has something to divide
    by that was not itself computed from a curve.

    Parameters
    ----------
    f : float
        Growth rate.
    b : float
        Linear bias of the density tracer.
    space : str
        `SPACE_REAL` or `SPACE_REDSHIFT`.
    a_hubble : float
        `a H` in km/s per h^-1 Mpc (`dvcorr.config.scale_factor_hubble`).
        The default 1.0 leaves the result in h^-1 Mpc, i.e. Nusser's
        distance-as-km/s units divided by `a H`.
    convention : str
        `CONVENTION_NUSSER` or `CONVENTION_STANDARD`.

    Returns
    -------
    float
    """
    # The 1/3 is <mu^2> over a uniform shell, part of the derivation.
    return (
        float(a_hubble)
        * (float(f) / 3.0)
        * kaiser_dipole_bias(b, f, space)
        * convention_factor(convention, 1)
    )


def xi_udelta_dipole(
    y: np.ndarray,
    k: np.ndarray,
    pk: np.ndarray,
    *,
    f: float,
    b: float = 1.0,
    space: str = SPACE_REDSHIFT,
    convention: str = CONVENTION_NUSSER,
    a_hubble: float = 1.0,
    shape_integral: np.ndarray | None = None,
) -> np.ndarray:
    """First moment `<xi_u delta_g>_1(y)` of the velocity-density correlation, shape (B,).

    The cancelled form -- `amplitude * A(y)` -- which is exact, not an
    approximation; see the module docstring for the `j_2` cancellation that
    makes it so, and `dipole_from_gorski` for the un-cancelled expression it
    must agree with.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Pair separations, h^-1 Mpc, strictly positive. This is NUSSER's
        separation (velocity object -> density point); its magnitude equals
        this project's `r`, but the multipoles differ by
        `conventions.nusser_multipole_sign(ell)`. See the `dvcorr.theory`
        package docstring.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers (h Mpc^-1) and `P(k)` ((h^-1 Mpc)^3). Ignored
        when `shape_integral` is supplied.
    f : float
        Growth rate.
    b : float
        Linear bias of the density tracer. `1.0` is the matter field.
    space : str
        `SPACE_REAL` or `SPACE_REDSHIFT` (the default -- what a redshift
        survey actually measures).
    convention : str
        Multipole normalization, `CONVENTION_NUSSER` (default) or
        `CONVENTION_STANDARD`.
    a_hubble : float
        `a H` in km/s per h^-1 Mpc, from
        `dvcorr.config.scale_factor_hubble(cosmology, z)`. The default 1.0
        returns h^-1 Mpc; pass the real value to get km/s. This is the ONLY
        place the conversion is applied.
    shape_integral : ndarray, shape (B,), optional
        A precomputed `A(y)` on the SAME `y` grid, to skip the FFTLog. This is
        the right architecture for a parameter scan: `A` depends on `P(k)`
        alone, so recomputing it per `(f, b, sigma_8)` is both wasteful and
        makes a collapse test tautological.

    Returns
    -------
    ndarray, shape (B,)
        km/s if `a_hubble` was supplied, otherwise h^-1 Mpc. POSITIVE for
        coherent infall, in Nusser's velocity-centered orientation.
    """
    a_of_y = transforms.a_integral(y, k, pk) if shape_integral is None else np.asarray(
        shape_integral, dtype=float
    )
    return dipole_amplitude(
        f, b, space, a_hubble=a_hubble, convention=convention
    ) * a_of_y


def dipole_from_gorski(
    functions: GorskiFunctions,
    *,
    b: float = 1.0,
    space: str = SPACE_REDSHIFT,
    convention: str = CONVENTION_NUSSER,
    a_hubble: float = 1.0,
) -> np.ndarray:
    """`<xi_u delta_g>_1` from `Psi'_perp` and `Psi'_-` directly, shape (B,).

    The UN-cancelled expression, i.e. Nusser eq. (17) with the corrected
    `Psi'_-` sign and a bias carried on the density leg:

        real:     -(b/f) (Psi'_perp + Psi'_-/3)
        redshift: -(b/f) (Psi'_perp + Psi'_-/3) - (3/5) Psi'_perp - (1/5) Psi'_-

    Kept separate from `xi_udelta_dipole` on purpose. The two are equal
    identically -- the `j_2` pieces cancel for every `b` and every `f` -- so
    comparing them is a test with no free parameters, and it FAILS if the
    printed sign of eq. (17) is used instead of the corrected one. That is the
    single most valuable assertion in this subpackage.

    Note that only the density-leg term carries `b`: the `xi_wu` addition is a
    velocity-velocity correlation and is bias-free, which is exactly why the
    RSD boost is `b -> b + 3f/5` rather than `b (1 + 3f/5)`.

    Parameters
    ----------
    functions : GorskiFunctions
        Evaluated on the `y` grid wanted, carrying the `f` to use.
    b : float
        Linear bias of the density tracer.
    space : str
        `SPACE_REAL` or `SPACE_REDSHIFT`.
    convention : str
        `CONVENTION_NUSSER` or `CONVENTION_STANDARD`.
    a_hubble : float
        `a H` in km/s per h^-1 Mpc; 1.0 leaves the result in h^-1 Mpc.

    Returns
    -------
    ndarray, shape (B,)
        Same units, sign and convention as `xi_udelta_dipole`.

    Raises
    ------
    ValueError
        On an unrecognized `space`.
    """
    if space not in VALID_SPACES:
        raise ValueError(f"dipole_from_gorski: space must be one of {VALID_SPACES}, got {space!r}.")

    f = functions.f
    # Real-space density term: <mu^2> = 1/3 over a uniform shell.
    out = -(float(b) / f) * (functions.perp_prime + functions.minus_prime / 3.0)
    if space == SPACE_REDSHIFT:
        # <xi_wu>_1, from <mu^2> = 1/3 and <mu^4> = 1/5. Bias-free.
        out = out - (3.0 / 5.0) * functions.perp_prime - (1.0 / 5.0) * functions.minus_prime
    return float(a_hubble) * convention_factor(convention, 1) * out


def dipole_boost(b: float, f: float) -> float:
    """The redshift/real dipole ratio `1 + 3 f / (5 b) = 1 + 3 beta / 5`, dimensionless.

    Flat in `y` and independent of `sigma_8`: `b` and `f` enter only through
    `beta = f / b`. This is the acceptance target for any measured
    redshift-space/real-space amplitude ratio, and its flatness is what
    distinguishes linear RSD from a normalization mistake.

    Parameters
    ----------
    b : float
        Linear bias, non-zero.
    f : float
        Growth rate.

    Returns
    -------
    float
    """
    return 1.0 + 3.0 * float(f) / (5.0 * float(b))


def a_hubble_for(cosmology: CosmologyConfig | None = None, z: float = 0.0) -> float:
    """`a H(z)` in km/s per h^-1 Mpc for `cosmology` -- the `a_hubble` argument above.

    A one-line convenience over `dvcorr.config.scale_factor_hubble`, so a
    caller of this module does not have to reach into `dvcorr.config` for the
    units factor. Exactly 100 at z = 0.

    Parameters
    ----------
    cosmology : CosmologyConfig, optional
        Defaults to the MDPL2 cosmology.
    z : float
        Redshift.

    Returns
    -------
    float
    """
    return scale_factor_hubble(cosmology or CosmologyConfig(), z)
