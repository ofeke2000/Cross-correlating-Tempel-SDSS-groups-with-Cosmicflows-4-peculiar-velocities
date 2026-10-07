"""
test_theory.py
--------------
Tests for `dvcorr.theory`: the FFTLog transforms, the Gorski identity, the
exact `j_2` cancellation that pins the corrected sign of Nusser (2017)
eq. (17), the parameter scalings, and the wide-angle geometry.

Almost everything here runs on a SYNTHETIC power-law spectrum rather than on
CAMB: a power law has a closed-form Bessel moment, so the transform can be
checked against an exact answer instead of against another numerical method,
and the suite then costs nothing and needs no optional dependency. The two
tests that do need CAMB are marked `importorskip`.

The tolerances here are tight (1e-9 to 1e-12) on purpose. They started loose,
at the ~1e-4 the transforms first delivered, and that looseness turned out to
be a BUG rather than a fact about FFTLog: mcfit's default bias exponent is
only valid for its own `k^3 P(k)` integrand, and `transforms.fftlog_tilt` now
derives one per `(ell, k_power)`. A loose tolerance would have accepted the
broken version, so these are set at what the correct code actually reaches.
"""

from __future__ import annotations

import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.config import CosmologyConfig, growth_rate, scale_factor_hubble
from dvcorr.theory import dipole, gorski, transforms, wide_angle

# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

#: Separations the assertions are made at, h^-1 Mpc. Spans the range linear
#: theory is used on plus a bit either side.
Y_TEST = np.array([5.0, 10.0, 20.0, 50.0, 90.0])

#: Growth rates swept in the cancellation and scaling tests. The note's
#: f in [0.2, 1.0].
F_SWEEP = (0.2, 0.4, 0.5251, 0.7, 1.0)

#: Biases swept alongside them.
B_SWEEP = (0.5, 1.0, 1.6, 2.5)


def power_law_spectrum(index: float, n_k: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """A log-spaced `(k, k**index)` over a very wide range, shapes (n_k,), (n_k,).

    Wide enough that mcfit's power-law extrapolation is exact rather than
    approximate, so any deviation from the closed form is the transform's own.
    """
    k = np.geomspace(1e-6, 1e4, n_k)
    return k, k**index


def closed_form_bessel_moment(
    y: np.ndarray, index: float, ell: int, k_power: int
) -> np.ndarray:
    """`Int dk k^k_power k^index j_ell(k y)` in closed form, shape (B,).

    `Int_0^inf x^(s-1) j_ell(x) dx = 2^(s-2) sqrt(pi) Gamma((ell+s)/2) /
    Gamma((ell-s+3)/2)` with `s = index + k_power + 1`, converging for
    `-ell < s < 2`.
    """
    from scipy.special import gamma as gamma_fn

    s = index + k_power + 1.0
    return y ** (-s) * 2.0 ** (s - 2.0) * np.sqrt(np.pi) * (
        gamma_fn((ell + s) / 2.0) / gamma_fn((ell - s + 3.0) / 2.0)
    )


def smooth_spectrum(n_k: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """A CDM-like `(k, P(k))` with no CAMB: `k^ns / (1 + (k/k_eq)^2)^2`, (n_k,) each.

    Rises as `k^ns` in the IR and falls as `k^(ns-4)` in the UV, so every
    `(ell, k_power)` pair used in this package converges at both ends -- which
    is what the reference quadrature needs.
    """
    k = np.geomspace(1e-5, 50.0, n_k)
    # The 2e9 is a normalization only, chosen so P(k) and every quantity
    # derived from it land in the same numerical range as a real CDM spectrum
    # -- an `allclose` with a default `atol` on 1e-9-sized numbers passes for
    # the wrong reason.
    return k, 2e9 * k**0.96 / (1.0 + (k / 0.02) ** 2) ** 2


@pytest.fixture(scope="module")
def camb_spectrum():
    """`(cosmology, z, k, P_lin(k))` at the MDPL2 cosmology, or skip."""
    pytest.importorskip("camb")
    pytest.importorskip("mcfit")
    from dvcorr.pipeline import galaxy_bias as gb

    cosmology = CosmologyConfig()
    k, pk = gb.linear_power_spectrum(cosmology, gb.SNAPSHOT_REDSHIFT, nonlinear=False)
    return cosmology, gb.SNAPSHOT_REDSHIFT, k, pk


# --------------------------------------------------------------------------
# The transforms
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ell, k_power, index",
    [
        (0, 1, -1.0),   # the density monopole integral
        (1, 1, -1.0),   # A
        (2, 0, -1.0),   # B * y
        (1, -1, -0.5),  # Psi_perp * y / f^2   (index chosen so s > -ell)
        (3, 1, -1.0),   # the RSD octupole, for completeness
    ],
)
def test_bessel_moment_matches_the_closed_form_at_every_order_and_power(ell, k_power, index):
    """Each `(ell, k_power)` pair reproduces the exact power-law answer.

    The load-bearing test of the whole subpackage: mcfit's phase convention is
    `i**ell`, and taking `Im(...)` instead of `Re(.../i**ell)` would pass at
    `ell = 1` and fail here at every other order. Different orders have
    genuinely different closed forms, so an `ell = 1` answer wearing an
    `ell = 2` label cannot pass.
    """
    pytest.importorskip("mcfit")
    k, pk = power_law_spectrum(index)
    got = transforms.bessel_moment(Y_TEST, k, pk, ell=ell, k_power=k_power)
    expected = closed_form_bessel_moment(Y_TEST, index, ell, k_power)
    assert np.allclose(got, expected, rtol=1e-9), (
        f"ell={ell} k_power={k_power}: max frac dev "
        f"{np.max(np.abs(got / expected - 1.0)):.3e}"
    )


def test_bessel_moment_matches_brute_quadrature_for_the_j2_integral():
    """FFTLog `B(y)` against dense trapezoid quadrature on a CDM-like spectrum.

    The `j_2` transform is the one with no prior use anywhere in this project
    -- `multipole_shape_integral` only ever ran `ell = 1` and `ell = 3` -- so
    it gets an independent numerical answer rather than inheriting trust.
    """
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    y = np.array([10.0, 50.0])
    fftlog = transforms.bessel_moment(y, k, pk, ell=2, k_power=0)
    brute = transforms.bessel_moment_quadrature(y, k, pk, ell=2, k_power=0)
    assert np.allclose(fftlog, brute, rtol=1e-6), (
        f"j_2: FFTLog {fftlog} vs quadrature {brute}"
    )


def test_a_and_b_carry_the_two_pi_squared_that_bessel_moment_does_not():
    """`a_integral` and `b_integral` are the `1/(2 pi^2)`-normalized moments."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    two_pi_sq = 2.0 * np.pi**2
    assert np.allclose(
        transforms.a_integral(Y_TEST, k, pk),
        transforms.bessel_moment(Y_TEST, k, pk, ell=1, k_power=1) / two_pi_sq,
    )
    assert np.allclose(
        transforms.b_integral(Y_TEST, k, pk),
        transforms.bessel_moment(Y_TEST, k, pk, ell=2, k_power=0) / (two_pi_sq * Y_TEST),
    )


# --------------------------------------------------------------------------
# Gorski
# --------------------------------------------------------------------------


def test_gorski_identity_psi_minus_equals_y_psi_perp_prime():
    """`Psi_- = y Psi'_perp` on a power law, where FFTLog is essentially exact.

    Non-circular by construction: the left side comes from the `ell = 0` and
    `ell = 1` transforms (`Psi_par - Psi_perp`), the right side from the
    `ell = 2` one. Passing means the Bessel recurrence
    `j_0 - 3 j_1/x = -j_2` survives three independent FFTLog calls.
    """
    pytest.importorskip("mcfit")
    k, pk = power_law_spectrum(-0.5)
    functions = gorski.gorski_functions(Y_TEST, k, pk, f=0.5)
    residual = functions.gorski_identity_residual()
    assert np.max(np.abs(residual)) < 1e-10, f"residual {residual}"


def test_gorski_identity_holds_on_the_camb_spectrum(camb_spectrum):
    """The same identity on the real `P_lin`, with its BAO wiggles and its
    numerically junk (but utterly negligible) sub-1e-4 h/Mpc tail.

    Slightly looser than the power-law case because the spectrum is not
    scale-free anywhere, but still four orders tighter than what the
    default-bias transform managed.
    """
    _, _, k, pk = camb_spectrum
    functions = gorski.gorski_functions(Y_TEST, k, pk, f=0.5)
    residual = np.abs(functions.gorski_identity_residual())
    assert np.max(residual) < 1e-9, f"residual {residual}"


def test_psi_prime_signs_are_negative_and_psi_perp_positive():
    """Sign orientation of the Gorski functions over the linear regime."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    functions = gorski.gorski_functions(Y_TEST, k, pk, f=0.5)
    assert np.all(functions.perp > 0.0)
    assert np.all(functions.perp_prime < 0.0)
    assert np.all(functions.minus < 0.0)


# --------------------------------------------------------------------------
# The j_2 cancellation -- the sign fix on Nusser eq. (17)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("f", F_SWEEP)
@pytest.mark.parametrize("b", B_SWEEP)
@pytest.mark.parametrize("space", [dipole.SPACE_REAL, dipole.SPACE_REDSHIFT])
def test_the_j2_terms_cancel_exactly_for_every_f_and_b(f, b, space):
    """`dipole_from_gorski` == `xi_udelta_dipole`, to machine precision.

    The un-cancelled expression in `Psi'_perp` and `Psi'_-` and the cancelled
    one in `A` alone are the same number, for any `f` and any `b`. This is the
    test the CORRECTED sign of eq. (17) passes and the printed one does not.
    """
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    functions = gorski.gorski_functions(Y_TEST, k, pk, f)
    cancelled = dipole.xi_udelta_dipole(Y_TEST, k, pk, f=f, b=b, space=space)
    explicit = dipole.dipole_from_gorski(functions, b=b, space=space)
    assert np.allclose(cancelled, explicit, rtol=1e-12, atol=0.0), (
        f"f={f} b={b} {space}: max frac dev "
        f"{np.max(np.abs(cancelled / explicit - 1.0)):.3e}"
    )


def test_the_printed_sign_of_eq_17_leaves_a_scale_dependent_j2_residual():
    """The negative control: with `-1/(3f)` the cancellation fails, and fails in `y`.

    Without this, "the cancellation works" is only evidence that the algebra
    was done consistently, not that the sign is the right one. With the
    printed coefficient a residual `-(2/(3f)) Psi'_-` survives; because
    `Psi'_-` and `A` have different shapes, the resulting redshift/real ratio
    is no longer constant in `y` -- which is the observable symptom.
    """
    pytest.importorskip("mcfit")
    f, b = 0.5251, 1.0
    k, pk = smooth_spectrum()
    functions = gorski.gorski_functions(Y_TEST, k, pk, f)

    correct = dipole.dipole_from_gorski(functions, b=b, space=dipole.SPACE_REDSHIFT)
    # Nusser eq. (17) exactly as printed: +1/(3f) where the corrected version
    # has -1/(3f) on the Psi'_- coefficient.
    printed = (
        -(3.0 / 5.0 + b / f) * functions.perp_prime
        - (1.0 / 5.0 - b / (3.0 * f)) * functions.minus_prime
    )
    assert not np.allclose(correct, printed, rtol=1e-6, atol=0.0)

    real = dipole.dipole_from_gorski(functions, b=b, space=dipole.SPACE_REAL)
    printed_ratio = printed / real
    assert np.ptp(printed_ratio) > 1e-3, (
        "the printed-sign ratio should be visibly scale-dependent, got "
        f"spread {np.ptp(printed_ratio):.3e}"
    )


# --------------------------------------------------------------------------
# Parameter scalings
# --------------------------------------------------------------------------


@pytest.mark.parametrize("f", F_SWEEP)
@pytest.mark.parametrize("b", B_SWEEP)
def test_the_redshift_real_ratio_is_flat_in_y_and_equals_one_plus_three_beta_over_five(f, b):
    """`<xi^s>_1 / <xi>_1 = 1 + 3f/(5b)`, constant in `y`."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    shared = {"f": f, "b": b}
    ratio = dipole.xi_udelta_dipole(
        Y_TEST, k, pk, space=dipole.SPACE_REDSHIFT, **shared
    ) / dipole.xi_udelta_dipole(Y_TEST, k, pk, space=dipole.SPACE_REAL, **shared)
    assert np.ptp(ratio) < 1e-12, f"ratio not flat: {ratio}"
    assert np.allclose(ratio, dipole.dipole_boost(b, f), rtol=1e-12)


def test_the_ratio_depends_on_f_and_b_only_through_beta():
    """Two `(f, b)` pairs at the same `beta = f/b` give the same boost."""
    assert dipole.dipole_boost(b=1.0, f=0.5) == pytest.approx(
        dipole.dipole_boost(b=2.0, f=1.0), rel=1e-15
    )


def test_the_curves_collapse_when_divided_by_their_amplitude():
    """One shape for every `(f, sigma_8, b)`, to 1e-10.

    `A_hat` is cached and reused rather than recomputed per parameter, so this
    tests the factorization rather than restating it: `sigma_8` enters by
    rescaling `P` (exactly, since `P` is linear in `A_s`), which is a
    different code path from the scalar `f` and `b` prefactors.
    """
    pytest.importorskip("mcfit")
    k, pk_fid = smooth_spectrum()
    sigma8_fid = 0.8228

    reference = None
    for f in F_SWEEP:
        for b in B_SWEEP:
            for sigma8 in (0.7, 0.8228, 0.95):
                pk = pk_fid * (sigma8 / sigma8_fid) ** 2
                curve = dipole.xi_udelta_dipole(
                    Y_TEST, k, pk, f=f, b=b, space=dipole.SPACE_REDSHIFT
                )
                # The amplitude in sigma_8-factorized form: the bracket of
                # (f sigma_8)(b sigma_8) + (3/5)(f sigma_8)^2.
                amplitude = dipole.dipole_amplitude(f, b, dipole.SPACE_REDSHIFT) * (
                    sigma8 / sigma8_fid
                ) ** 2
                shape = curve / amplitude
                if reference is None:
                    reference = shape
                assert np.allclose(shape, reference, rtol=1e-10), f"{f=} {b=} {sigma8=}"


def test_real_space_is_strictly_proportional_to_b_and_redshift_space_is_affine():
    """`b`-scan: real `propto b`; redshift `= b * slope + f/3 * 3f/5 * A`."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    f = 0.5251
    a_of_y = transforms.a_integral(Y_TEST, k, pk)

    one = dipole.xi_udelta_dipole(Y_TEST, k, pk, f=f, b=1.0, space=dipole.SPACE_REAL)
    two = dipole.xi_udelta_dipole(Y_TEST, k, pk, f=f, b=2.0, space=dipole.SPACE_REAL)
    assert np.allclose(two, 2.0 * one, rtol=1e-13)

    intercept = dipole.xi_udelta_dipole(
        Y_TEST, k, pk, f=f, b=0.0, space=dipole.SPACE_REDSHIFT
    )
    assert np.allclose(intercept, (f / 3.0) * (3.0 * f / 5.0) * a_of_y, rtol=1e-13)


def test_standard_convention_is_exactly_three_times_nusser_for_the_dipole():
    """The `(2 ell + 1)` factor, applied at one place and nowhere else."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    shared = {"f": 0.5251, "b": 1.3, "space": dipole.SPACE_REDSHIFT}
    nusser = dipole.xi_udelta_dipole(
        Y_TEST, k, pk, convention=dipole.CONVENTION_NUSSER, **shared
    )
    standard = dipole.xi_udelta_dipole(
        Y_TEST, k, pk, convention=dipole.CONVENTION_STANDARD, **shared
    )
    assert np.allclose(standard, 3.0 * nusser, rtol=1e-15)
    assert dipole.convention_factor(dipole.CONVENTION_STANDARD, 3) == 7.0


def test_an_unrecognized_space_or_convention_raises_rather_than_defaulting():
    """A typo in either string is a silent factor of 3 or a wrong space."""
    with pytest.raises(ValueError, match="convention"):
        dipole.convention_factor("standrad", 1)
    with pytest.raises(ValueError, match="space"):
        dipole.kaiser_dipole_bias(1.0, 0.5, space="redshfit")


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_a_hubble_is_exactly_one_hundred_at_z_zero():
    """`a H = 100 km/s per h^-1 Mpc` at z = 0 -- the hand-computable value.

    The one check that catches `H * h` written where `H / h` was meant, which
    otherwise rescales every prediction by `h**2` with no change of shape.
    """
    cosmology = CosmologyConfig()
    assert scale_factor_hubble(cosmology, 0.0) == pytest.approx(100.0, abs=1e-12)
    assert dipole.a_hubble_for(cosmology, 0.0) == pytest.approx(100.0, abs=1e-12)
    # At z > 0 it is a * H(z) / h, which is BELOW 100 for LCDM at low z.
    assert scale_factor_hubble(cosmology, 1.0) < 100.0


def test_a_hubble_scales_the_dipole_linearly_and_nowhere_else():
    """The units factor enters exactly once."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    shared = {"f": 0.5251, "b": 1.0, "space": dipole.SPACE_REDSHIFT}
    unitless = dipole.xi_udelta_dipole(Y_TEST, k, pk, a_hubble=1.0, **shared)
    km_s = dipole.xi_udelta_dipole(Y_TEST, k, pk, a_hubble=100.0, **shared)
    assert np.allclose(km_s, 100.0 * unitless, rtol=1e-15)


def test_growth_rate_matches_the_omega_to_the_gamma_form():
    """`f = Om(z)^gamma`, and `Om(0) = Om0` for a flat cosmology."""
    cosmology = CosmologyConfig()
    assert growth_rate(cosmology, 0.0) == pytest.approx(
        cosmology.Om0**cosmology.growth_index, rel=1e-14
    )


# --------------------------------------------------------------------------
# Wide-angle geometry
# --------------------------------------------------------------------------


def test_the_wide_angle_cosines_reach_their_distant_observer_limits():
    """`nu -> 1` and `mu_tilde -> mu` as `y / r_obs -> 0`."""
    mu, _ = wide_angle.gauss_legendre_nodes(65)
    near = wide_angle.wide_angle_geometry(np.array([1.0]), mu, r_obs=1e8)
    assert np.allclose(near.nu, 1.0, atol=1e-7)
    assert np.allclose(near.mu_tilde, mu, atol=1e-7)
    assert np.allclose(near.r_tilde, 1e8, rtol=1e-7)


def test_the_wide_angle_geometry_is_the_law_of_cosines():
    """`r_tilde`, `nu`, `mu_tilde` against an explicit vector construction."""
    y_value, r_obs, mu_value = 30.0, 50.0, 0.3
    geometry = wide_angle.wide_angle_geometry(
        np.array([y_value]), np.array([mu_value]), r_obs
    )
    r_vec = np.array([0.0, 0.0, r_obs])
    y_vec = y_value * np.array([np.sqrt(1.0 - mu_value**2), 0.0, mu_value])
    r_tilde_vec = r_vec + y_vec
    r_tilde = np.linalg.norm(r_tilde_vec)
    assert geometry.r_tilde[0, 0] == pytest.approx(r_tilde, rel=1e-14)
    assert geometry.nu[0, 0] == pytest.approx(
        r_vec @ r_tilde_vec / (r_obs * r_tilde), rel=1e-14
    )
    assert geometry.mu_tilde[0, 0] == pytest.approx(
        r_tilde_vec @ y_vec / (r_tilde * y_value), rel=1e-14
    )


def test_r_obs_must_be_positive():
    with pytest.raises(ValueError, match="r_obs"):
        wide_angle.wide_angle_geometry(np.array([1.0]), np.array([0.0]), r_obs=0.0)


def test_legendre_moment_reproduces_the_orthogonality_normalization():
    """`(1/2) Int L_ell L_ell dmu = 1 / (2 ell + 1)` in Nusser's convention."""
    from scipy.special import eval_legendre

    mu, weights = wide_angle.gauss_legendre_nodes()
    for ell in (0, 1, 2, 3):
        values = eval_legendre(ell, mu)[None, :]
        got = wide_angle.legendre_moment(values, mu, weights, ell)
        assert got[0] == pytest.approx(1.0 / (2 * ell + 1), rel=1e-13)
        # The standard convention turns that into 1 exactly.
        standard = wide_angle.legendre_moment(
            values, mu, weights, ell, convention=dipole.CONVENTION_STANDARD
        )
        assert standard[0] == pytest.approx(1.0, rel=1e-13)


def test_the_real_space_dipole_does_not_depend_on_observer_distance():
    """Nusser eq. (13) has no `r_tilde`, `nu` or `mu_tilde` in it.

    So every finite-distance effect on this statistic is a REDSHIFT-space
    effect. Worth pinning, because it is the reason curves 3-5 of Nusser's
    Fig. 1 are all redshift-space objects.
    """
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    functions = gorski.gorski_functions(Y_TEST, k, pk, f=0.5251)
    near = wide_angle.xi_udelta_wide_angle_moment(
        Y_TEST, 30.0, functions, 1, space=dipole.SPACE_REAL
    )
    far = wide_angle.xi_udelta_wide_angle_moment(
        Y_TEST, 5000.0, functions, 1, space=dipole.SPACE_REAL
    )
    dol = dipole.xi_udelta_dipole(Y_TEST, k, pk, f=0.5251, space=dipole.SPACE_REAL)
    assert np.allclose(near, far, rtol=1e-14)
    assert np.allclose(near, dol, rtol=1e-12)


def test_the_rocket_dipole_vanishes_in_the_distant_observer_limit():
    """`<2 xi_uu / r_tilde>_1 -> 0`, and its monopole -> Nusser eq. (16).

    The module's own stated diagnostic: a non-zero `ell = 1` at `y << r_obs`
    means the geometry is wrong, not that the physics is interesting.
    """
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    functions = gorski.gorski_functions(Y_TEST, k, pk, f=0.5251)
    r_obs = 1e6

    dipole_moment = wide_angle.xi_uu_over_r_tilde_moment(Y_TEST, r_obs, functions, 1)
    monopole = wide_angle.xi_uu_over_r_tilde_moment(Y_TEST, r_obs, functions, 0)
    assert np.max(np.abs(dipole_moment)) < 1e-7
    # Eq. (16) up to the sign eq. (15) applies: 2 (Psi_perp + Psi_-/3) / r_obs.
    expected = 2.0 * (functions.perp + functions.minus / 3.0) / r_obs
    assert np.allclose(monopole, expected, rtol=1e-8)


def test_both_rocket_denominators_match_their_first_order_dipoles():
    """The fixed-`r_obs` and exact-`r_tilde` rocket dipoles, against eps = y/r_obs.

    Fixed (Nusser's Fig. 1): `+(4 y / 15 r^2) Psi_-`, from the `mu_tilde` tilt
    alone. Exact (eq. 6): `-(2 y / 3 r^2)(Psi_perp + Psi_-/5)`, dominated by
    `1/r_tilde` weighting the near side up. The residual is O(eps^2) relative.
    """
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    functions = gorski.gorski_functions(Y_TEST, k, pk, f=0.5251)
    r_obs = 1e4

    fixed = wide_angle.xi_uu_over_r_tilde_moment(
        Y_TEST, r_obs, functions, 1,
        rocket_denominator=wide_angle.ROCKET_DENOMINATOR_FIXED,
    )
    exact = wide_angle.xi_uu_over_r_tilde_moment(
        Y_TEST, r_obs, functions, 1,
        rocket_denominator=wide_angle.ROCKET_DENOMINATOR_EXACT,
    )
    scale = Y_TEST / r_obs**2
    assert np.allclose(fixed, 4.0 / 15.0 * scale * functions.minus, rtol=1e-3)
    assert np.allclose(
        exact, -2.0 / 3.0 * scale * (functions.perp + functions.minus / 5.0), rtol=1e-3
    )

    # The default is Nusser's.
    default = wide_angle.xi_uu_over_r_tilde_moment(Y_TEST, r_obs, functions, 1)
    assert np.array_equal(default, fixed)


def test_an_unknown_rocket_denominator_raises():
    mu, _ = wide_angle.gauss_legendre_nodes(5)
    geometry = wide_angle.wide_angle_geometry(np.array([1.0]), mu, r_obs=50.0)
    functions = gorski.GorskiFunctions.__new__(gorski.GorskiFunctions)
    with pytest.raises(ValueError, match="denominator must be one of"):
        wide_angle.rocket_term(geometry, functions, "r_tilde")


def test_the_finite_distance_dipole_converges_to_the_dol_as_one_over_r_squared():
    """Curve 3 -> curve 1, at the rate wide-angle corrections are supposed to.

    Agreement at one distance could be a coincidence of two wrong formulae;
    a clean `r_obs^-2` convergence over four decades cannot.
    """
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    f = 0.5251
    functions = gorski.gorski_functions(Y_TEST, k, pk, f)
    dol = dipole.xi_udelta_dipole(Y_TEST, k, pk, f=f, space=dipole.SPACE_REDSHIFT)

    deviations = []
    for r_obs in (1e3, 1e4, 1e5):
        wide = wide_angle.xi_udelta_wide_angle_moment(
            Y_TEST, r_obs, functions, 1, space=dipole.SPACE_REDSHIFT
        )
        deviations.append(np.max(np.abs(wide / dol - 1.0)))

    # Each decade in r_obs must buy about two decades in the deviation.
    for coarse, fine in zip(deviations[:-1], deviations[1:]):
        assert fine / coarse == pytest.approx(1e-2, rel=0.2), deviations


# --------------------------------------------------------------------------
# Agreement with the production pipeline
# --------------------------------------------------------------------------


def test_the_real_space_dipole_reproduces_zeta_one_linear_prediction(camb_spectrum):
    """`dvcorr.theory` and `pipeline.galaxy_bias` are the same number.

    `zeta_one_linear_prediction` is the production overlay and predates this
    subpackage; if the two disagreed, one of them would be wrong and the
    measurement it is compared against would inherit it.
    """
    from dvcorr.pipeline import galaxy_bias as gb

    cosmology, z, k, pk = camb_spectrum
    b = 1.3
    production = gb.zeta_one_linear_prediction(
        Y_TEST, b=b, cosmology=cosmology, z=z, power_spectrum=(k, pk)
    )
    theory = dipole.xi_udelta_dipole(
        Y_TEST,
        k,
        pk,
        f=growth_rate(cosmology, z),
        b=b,
        space=dipole.SPACE_REAL,
        a_hubble=scale_factor_hubble(cosmology, z),
    )
    assert np.allclose(production, theory, rtol=1e-13)


def test_the_group_centered_dipole_is_negative_for_infall(camb_spectrum):
    """Hard rule 2, through the Nusser -> dvcorr orientation flip.

    `xi_udelta_dipole` is velocity-centered and POSITIVE; multiplying by
    `conventions.nusser_multipole_sign(1)` gives the group-centered `xi_Tu,1`,
    which must be negative. The factor is applied here, at the call site,
    explicitly -- never inside an estimator or a theory function.
    """
    cosmology, z, k, pk = camb_spectrum
    velocity_centered = dipole.xi_udelta_dipole(
        Y_TEST,
        k,
        pk,
        f=growth_rate(cosmology, z),
        space=dipole.SPACE_REDSHIFT,
        a_hubble=scale_factor_hubble(cosmology, z),
    )
    assert np.all(velocity_centered > 0.0)

    group_centered = conventions.nusser_multipole_sign(1) * velocity_centered
    assert np.all(np.sign(group_centered) == conventions.INFALL_DIPOLE_SIGN)


# --------------------------------------------------------------------------
# Zero separation
# --------------------------------------------------------------------------
#
# y = 0 is never on the FFTLog output grid -- that grid is logarithmic, so it
# cannot contain zero at any resolution -- but every quantity has a closed form
# there. These pin the closed forms, because the failure mode they replaced was
# a SILENT NaN out of `log_interp(0)`.


def test_bessel_moment_at_zero_separation_is_the_closed_form():
    """`j_ell(0) = 0` for `ell >= 1`; at `ell = 0` what survives is `Int k^p P dk`."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    y = np.array([0.0, 10.0])
    for ell in (1, 2, 3):
        got = transforms.bessel_moment(y, k, pk, ell=ell, k_power=1)
        assert got[0] == 0.0, f"ell={ell} gave {got[0]!r} at y=0"
    monopole = transforms.bessel_moment(y, k, pk, ell=0, k_power=0)
    assert monopole[0] == pytest.approx(
        transforms.zero_separation_moment(k, pk, 0), rel=1e-14
    )


def test_zero_separation_moment_matches_dense_quadrature():
    """The trapezoid-in-ln-k spectrum integral against the reference quadrature."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    reference = transforms.bessel_moment_quadrature(
        np.array([1e-30]), k, pk, ell=0, k_power=0
    )[0]
    assert transforms.zero_separation_moment(k, pk, 0) == pytest.approx(reference, rel=1e-5)


def test_the_two_zero_over_zero_limits_are_taken_not_divided():
    """`B(0) = 0` and `psi_perp_integral(0) = Int P dk / (6 pi^2)`.

    Both are `0/0` as written; the second is the only quantity in the module
    that is finite and NON-zero at zero separation, so a shared "just return 0"
    shortcut would be wrong for exactly one of them.
    """
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    y = np.array([0.0, 10.0])
    assert transforms.a_integral(y, k, pk)[0] == 0.0
    assert transforms.b_integral(y, k, pk)[0] == 0.0
    assert transforms.psi_perp_integral(y, k, pk)[0] == pytest.approx(
        transforms.zero_separation_moment(k, pk, 0) / (6.0 * np.pi**2), rel=1e-14
    )
    assert np.all(np.isfinite(transforms.psi_perp_integral(y, k, pk)))


def test_the_gorski_functions_are_isotropic_at_zero_separation():
    """`Psi_perp(0) = Psi_par(0) = sigma_v^2(1D)`, `Psi_-(0) = 0`, both primes 0.

    Physically: at zero lag the velocity correlation tensor is `delta_ij
    sigma_v^2 / 3`, so there is no longitudinal/transverse distinction left to
    make and `Psi_-` must vanish.
    """
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    f = 0.5251
    functions = gorski.gorski_functions(np.array([0.0, 10.0]), k, pk, f)
    sigma_v_sq = f**2 * transforms.zero_separation_moment(k, pk, 0) / (6.0 * np.pi**2)
    assert functions.perp[0] == pytest.approx(sigma_v_sq, rel=1e-14)
    assert functions.parallel[0] == pytest.approx(sigma_v_sq, rel=1e-12)
    assert functions.minus[0] == pytest.approx(0.0, abs=1e-12 * sigma_v_sq)
    assert functions.perp_prime[0] == 0.0
    assert functions.minus_prime[0] == 0.0
    # The identity is 0 = 0 there, which has no relative form.
    assert np.isnan(functions.gorski_identity_residual()[0])


def test_the_dipole_vanishes_at_zero_separation_in_both_spaces():
    """No separation, no direction, no dipole -- exactly, not approximately."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    for space in (dipole.SPACE_REAL, dipole.SPACE_REDSHIFT):
        got = dipole.xi_udelta_dipole(
            np.array([0.0, 10.0]), k, pk, f=0.5251, b=1.3, space=space
        )
        assert got[0] == 0.0


def test_the_rocket_term_is_finite_and_dipole_free_at_zero_separation():
    """`<2 xi_uu/r~>_0 -> 2 Psi_perp(0)/r_obs`, and its dipole is exactly zero."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    y = np.array([0.0, 10.0])
    functions = gorski.gorski_functions(y, k, pk, 0.5251)
    r_obs = 50.0
    monopole = wide_angle.xi_uu_over_r_tilde_moment(y, r_obs, functions, 0)
    dipole_moment = wide_angle.xi_uu_over_r_tilde_moment(y, r_obs, functions, 1)
    assert monopole[0] == pytest.approx(2.0 * functions.perp[0] / r_obs, rel=1e-12)
    assert dipole_moment[0] == pytest.approx(0.0, abs=1e-12 * monopole[0])


def test_a_negative_separation_raises_rather_than_returning_nan():
    """The old failure mode was a silent NaN; anything outside [0, inf) is an error."""
    pytest.importorskip("mcfit")
    k, pk = smooth_spectrum()
    with pytest.raises(ValueError, match="y must be >= 0"):
        transforms.bessel_moment(np.array([-1.0]), k, pk, ell=1, k_power=1)
