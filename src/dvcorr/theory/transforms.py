"""
transforms.py
-------------
Spherical-Bessel moments of the linear power spectrum -- the k-integrals every
linear density-velocity prediction in this project is assembled from.

The one integral
================
Everything reduces to

    M_ell^p(y) = Int_0^inf dk k^p P(k) j_ell(k y)

`ell` selects the multipole kernel, `p` the extra power of `k` that the
particular correlation function carries. Four `(ell, p)` pairs cover the whole
of Nusser (2017)'s section 2:

    (1,  1)  -> A(y),  the dipole master integral
    (2,  0)  -> B(y) * y,  the j_2 master integral
    (1, -1)  -> Psi_perp(y) * y / f^2
    (0,  1)  -> the density monopole, if it is ever wanted

`bessel_moment` is the definition site for all of them and
`dvcorr.pipeline.galaxy_bias.multipole_shape_integral` is its `p = 1` special
case, delegating here rather than carrying a second FFTLog implementation.

Why `p` is a parameter and not four functions
=============================================
Because the two things most likely to be got wrong -- the mcfit phase
convention and the FFTLog bias -- must be got right ONCE.

The phase: mcfit's `P2xi(l=ell)` carries an `i^ell`, so a caller who takes
`Im(...)` gets the right answer at `ell = 1`, the NEGATED answer at `ell = 3`,
and zero at even `ell`, with no exception and no NaN. Passing mcfit's `n`
argument instead makes it drop the phase entirely, which removes the decision
rather than documenting it.

The bias: FFTLog transforms `k^-q F(k)`, and `q` must sit both inside the
Mellin kernel's convergence strip and between the integrand's two asymptotic
slopes. mcfit's default is chosen for its own `k^3 P(k)` integrand and is
valid here at `p = 1` alone -- at `p = 0` and `p = -1` it returns numbers
wrong by six orders of magnitude, silently. `fftlog_tilt` derives `q` from the
spectrum's own endpoint slopes; with it, every closed form in
`tests/test_theory.py` comes back to ~1e-12 instead of ~1e-5.

Master integrals
================
`A` and `B` below carry the `1 / (2 pi^2)` that `bessel_moment` does not, so
they are the quantities that appear literally in the boxed formulae of
Nusser's section 2 and in `gorski.py`:

    A(y) = (1 / 2 pi^2) Int dk P(k) k j_1(k y)                 [h^-1 Mpc]
    B(y) = (1 / 2 pi^2) Int dk P(k) k j_2(k y) / (k y)         [h^-1 Mpc]

Keeping the `2 pi^2` out of `bessel_moment` and in `A`/`B` is deliberate: it
leaves `bessel_moment` numerically identical to the existing
`multipole_shape_integral`, so the delegation cannot change a production
number.

Units
=====
`y` in h^-1 Mpc, `k` in h Mpc^-1, `P(k)` in (h^-1 Mpc)^3, so `M_ell^p` is in
(h^-1 Mpc)^(2 - p) and both `A` and `B` are in h^-1 Mpc.

Convergence
===========
IR: the `A` integrand goes as `k^(n_s + 2)`, so `k_min ~ 1e-5 h/Mpc` is far
more than enough. UV: it decays only as `k^-2` times an oscillation, so it
converges slowly, and the smallest `y` plotted is what sets how much `k_max`
matters. Check a `k_max` ladder rather than assuming; the notebook does.

Zero separation
===============
`y = 0` is supported and is NOT interpolated. A FFTLog output grid is
logarithmic -- its nodes have a fixed ratio, spanning `[1/k_max, 1/k_min]` --
so it can never contain zero at any resolution, and asking `log_interp` for it
used to return a silent NaN. But nothing needs the grid there, because every
value is a closed form:

    M_ell^p(0) = 0                    for ell >= 1   (j_ell(0) = 0)
    M_0^p(0)   = Int k^p P dk                        (`zero_separation_moment`)
    A(0) = B(0) = 0
    psi_perp_integral(0) = Int P dk / (6 pi^2)       <- finite and NON-zero

The last one is the exception worth remembering: it is the 1-D velocity
dispersion `sigma_v^2` divided by `f^2`, i.e. the zero-lag value of both
`Psi_perp` and `Psi_par`, which are equal there because the velocity
correlation tensor is isotropic at zero lag. `B` and `psi_perp_integral` are
each `0/0` as written and take DIFFERENT limits, so they are handled
separately (`_divide_by_y_with_limit`) rather than by one shared shortcut.

A negative `y` raises. Callers taking RATIOS of two predictions must exclude
`y = 0`: both dipoles vanish there, so the ratio is `0/0` even though each
side is exact.
"""

from __future__ import annotations

import numpy as np

#: Log-spaced quadrature nodes below `QUADRATURE_SPLIT_K` in
#: `bessel_moment_quadrature`, where the integrand is smooth and spans decades.
QUADRATURE_LOG_NODES: int = 200_000

#: Linearly spaced nodes above `QUADRATURE_SPLIT_K`, where `j_ell(k y)`
#: oscillates and uniform resolution is what the trapezoid rule needs.
QUADRATURE_LINEAR_NODES: int = 1_000_000

#: Wavenumber at which the reference quadrature switches from log to linear
#: spacing, h Mpc^-1. Roughly where the Bessel oscillation starts to dominate
#: the integrand's variation for the separations linear theory is used on.
QUADRATURE_SPLIT_K: float = 0.1


def log_interp(x_target: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cubic-spline interpolate `y(x)` onto `x_target` in log x, shape (len(x_target),).

    The FFTLog output lands on its own logarithmically spaced grid, never on
    the caller's; this is how every transform here is delivered onto a
    requested `y` array. Shared with
    `dvcorr.pipeline.galaxy_bias._log_interp`, which delegates here.

    Parameters
    ----------
    x_target : ndarray, shape (B,)
        Abscissae to evaluate at. All entries must be strictly positive and
        inside the span of `x`; outside it this is extrapolation, and a cubic
        spline extrapolates by taking an excursion of arbitrary size.
    x : ndarray, shape (G,)
        Source abscissae, strictly positive; sorted internally.
    y : ndarray, shape (G,)
        Source ordinates.

    Returns
    -------
    ndarray, shape (B,)
    """
    from scipy.interpolate import CubicSpline

    order = np.argsort(x)
    return CubicSpline(np.log(np.asarray(x, dtype=float)[order]), np.asarray(y, dtype=float)[order])(
        np.log(np.asarray(x_target, dtype=float))
    )


def _endpoint_log_slopes(k: np.ndarray, pk: np.ndarray) -> tuple[float, float]:
    """Log-log slopes of `P(k)` on its first and last segments, `(low, high)`.

    Two-point segments deliberately: `extrap=True` extends the input by a
    power law fitted to exactly those segments, so these describe the function
    FFTLog actually transforms, pads included. A wider fit would describe a
    different function.
    """
    log_k = np.log(k)
    log_pk = np.log(pk)
    return (
        float((log_pk[1] - log_pk[0]) / (log_k[1] - log_k[0])),
        float((log_pk[-1] - log_pk[-2]) / (log_k[-1] - log_k[-2])),
    )


#: Slack allowed when comparing the two ends of the admissible tilt window, so
#: an exactly scale-free input (where the window collapses to a point) is not
#: rejected by floating-point noise in the slope estimates.
TILT_WINDOW_TOLERANCE: float = 1e-9


def fftlog_tilt(k: np.ndarray, pk: np.ndarray, ell: int, k_power: int) -> float:
    """The FFTLog bias exponent `q` for `bessel_moment`, dimensionless.

    NOT a cosmetic tuning knob -- get this wrong and the transform returns
    numerical garbage several orders of magnitude off, with no warning. This
    is the one thing that stopped the generalization to arbitrary `k_power`
    from working when it was left at mcfit's fixed default.

    FFTLog transforms the biased function `k^-q F(k)`, where
    `F(k) = k^(k_power + 1) P(k)` is the integrand divided by the `dk/k`
    measure. Two conditions must hold at once:

    1. `k^-q F` must decay at BOTH ends of the (padded) grid, i.e.
       `s_high < q < s_low` with `s` the log-log slope of `F` there. mcfit's
       default `q` is chosen for its own `P2xi` integrand and satisfies this
       for a CDM spectrum at `k_power = 1` only. For a scale-free input
       (`s_low == s_high`) the window collapses to a point and `q` must equal
       the slope EXACTLY -- which is why a power-law test spectrum was the
       thing that exposed the problem.
    2. `q` must lie in the convergence strip of the Mellin kernel of
       `j_ell`, `-ell < q < 2`.

    The midpoint of the intersection is taken. Where the two conditions are
    incompatible the integral itself diverges, and this raises rather than
    returning a number.

    Parameters
    ----------
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers and `P(k)`.
    ell : int
        Bessel order.
    k_power : int
        Power of `k` in the integrand.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If no admissible `q` exists -- i.e. the integral does not converge for
        this `(ell, k_power)` against this spectrum's asymptotic slopes.
    """
    slope_low, slope_high = _endpoint_log_slopes(k, pk)
    integrand_slope_low = k_power + 1 + slope_low
    integrand_slope_high = k_power + 1 + slope_high

    lower = max(integrand_slope_high, -float(ell))
    upper = min(integrand_slope_low, 2.0)
    if lower > upper + TILT_WINDOW_TOLERANCE:
        raise ValueError(
            f"fftlog_tilt: no admissible FFTLog bias for ell={ell}, k_power={k_power}. "
            f"The integrand goes as k^{integrand_slope_low:.3f} at low k and "
            f"k^{integrand_slope_high:.3f} at high k, and the j_{ell} Mellin kernel "
            f"converges only for -{ell} < q < 2; the window ({lower:.3f}, {upper:.3f}) "
            "is empty, which means the integral itself diverges."
        )
    return 0.5 * (lower + upper)


def zero_separation_moment(k: np.ndarray, pk: np.ndarray, k_power: int) -> float:
    """`Int dk k^k_power P(k)`, the `y = 0` value of an `ell = 0` Bessel moment.

    At zero separation `j_0(0) = 1`, so the Bessel kernel drops out and what is
    left is a plain integral over the spectrum -- which FFTLog cannot deliver
    (its output grid is logarithmic and so never reaches zero) but ordinary
    quadrature can, cheaply and to full accuracy, because the integrand is
    smooth and convergent at both ends for a CDM spectrum.

    `k_power = 0` is the case that matters: `Int P dk` is `6 pi^2` times the
    1-D velocity dispersion `sigma_v^2` divided by `f^2`, i.e. the zero-lag
    value of `Psi_perp` and `Psi_par` (`gorski.psi_perp`).

    Parameters
    ----------
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers (h Mpc^-1) and `P(k)` ((h^-1 Mpc)^3).
    k_power : int
        Power of `k` in the integrand.

    Returns
    -------
    float
        (h^-1 Mpc)^(2 - k_power).

    Notes
    -----
    Trapezoid in `ln k` (the grid is log-spaced, so this is the natural
    measure): `Int k^p P dk = Int k^(p+1) P dln k`.
    """
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    return float(np.trapezoid(pk * k ** (int(k_power) + 1), np.log(k)))


def bessel_moment(
    y: np.ndarray,
    k: np.ndarray,
    pk: np.ndarray,
    ell: int,
    k_power: int = 1,
    *,
    tilt: float | None = None,
) -> np.ndarray:
    """`Int dk k^k_power P(k) j_ell(k y)`, shape (B,), (h^-1 Mpc)^(2 - k_power).

    The single FFTLog call behind every linear prediction in this project; see
    the module docstring for why `k_power` is an argument rather than four
    hardcoded variants, and `fftlog_tilt` for the bias exponent, which is
    chosen from the spectrum's own endpoint slopes rather than left at a
    default.

    mcfit's `P2xi(l=ell, n=k_power - 2)` is the underlying call. Passing `n`
    (rather than pre-multiplying `P` by `k**(k_power - 2)` and dividing the
    result by `i**ell`) is what makes this a single code path: mcfit drops the
    `i**ell` phase itself when `n` is given, so there is no `Re`-versus-`Im`
    decision left to get wrong at even orders.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k : ndarray, shape (K,)
        Wavenumbers, h Mpc^-1, LOG-SPACED and strictly increasing -- FFTLog
        requires a logarithmically uniform grid.
        `dvcorr.pipeline.galaxy_bias.linear_power_spectrum` supplies one.
    pk : ndarray, shape (K,)
        `P(k)` on that grid, (h^-1 Mpc)^3, strictly positive.
    ell : int
        Order of the spherical Bessel function, >= 0.
    k_power : int
        The power of `k` multiplying `P(k)` in the integrand. `1` is the
        dipole/octupole shape integral; `0` and `-1` are the `j_2` and
        `Psi_perp` integrals of `gorski.py`.
    tilt : float, optional
        Override for the FFTLog bias. None (the default) calls `fftlog_tilt`.
        Supply one only to demonstrate the sensitivity -- the automatic choice
        reproduces every closed form in `tests/test_theory.py` to ~1e-12.

    Returns
    -------
    ndarray, shape (B,)
        The integral at each `y`.
    """
    import mcfit

    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    if k.ndim != 1 or k.shape != pk.shape:
        raise ValueError(
            f"bessel_moment: k {k.shape} and pk {pk.shape} must be 1-D and the same shape."
        )
    ell = int(ell)
    if ell < 0:
        raise ValueError(f"bessel_moment: ell must be >= 0, got {ell}.")

    k_power = int(k_power)

    # y = 0 is exact, and is NOT on the FFTLog grid: a logarithmic grid cannot
    # contain zero at any resolution. j_ell(0) = 0 for every ell >= 1, so the
    # moment vanishes identically; at ell = 0, j_0(0) = 1 and what survives is
    # a plain spectrum integral. Handling it here rather than at the call site
    # is what stops `log_interp(0)` from returning a silent NaN.
    y_array = np.atleast_1d(np.asarray(y, dtype=float))
    if np.any(y_array < 0.0):
        raise ValueError("bessel_moment: y must be >= 0.")
    at_zero = y_array == 0.0
    if np.any(at_zero):
        out = np.empty(y_array.shape, dtype=float)
        out[at_zero] = (
            zero_separation_moment(k, pk, k_power) if ell == 0 else 0.0
        )
        if np.any(~at_zero):
            out[~at_zero] = bessel_moment(
                y_array[~at_zero], k, pk, ell, k_power, tilt=tilt
            )
        return out

    q_total = fftlog_tilt(k, pk, ell, k_power) if tilt is None else float(tilt)
    # mcfit's `n` shifts the tilt it was constructed with by n, so ask for
    # `q_total - n` to land the ACTUAL bias on q_total.
    order_shift = k_power - 2
    y_grid, transform = mcfit.P2xi(
        k, l=ell, n=order_shift, q=q_total - order_shift, lowring=True
    )(pk, extrap=True)
    return log_interp(y, y_grid, 2.0 * np.pi**2 * np.real(transform))


def bessel_moment_quadrature(
    y: np.ndarray,
    k: np.ndarray,
    pk: np.ndarray,
    ell: int,
    k_power: int = 1,
    *,
    k_min: float | None = None,
    k_max: float | None = None,
) -> np.ndarray:
    """`bessel_moment` by dense trapezoid quadrature instead of FFTLog, shape (B,).

    Slow (of order a second per `y`) and trustworthy: the independent answer
    the FFTLog normalization and phase are validated against, per `(ell,
    k_power)` pair. Not for production use.

    Unlike slicing the FFTLog input, the `k_min`/`k_max` here are a HARD
    truncation -- mcfit's `extrap=True` power-law-extends the spectrum past
    its endpoints, so only quadrature can answer a k-range question.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc.
    k, pk : ndarray, shape (K,)
        As `bessel_moment`. `P(k)` is cubic-spline interpolated in log-log
        onto the quadrature grid.
    ell : int
    k_power : int
    k_min, k_max : float, optional
        Truncation of the k integral, h Mpc^-1. Default to the endpoints of
        `k`.

    Returns
    -------
    ndarray, shape (B,)
    """
    from scipy.interpolate import CubicSpline
    from scipy.special import spherical_jn

    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    log_pk = CubicSpline(np.log(k), np.log(pk))

    k_min = float(k[0]) if k_min is None else float(k_min)
    k_max = float(k[-1]) if k_max is None else float(k_max)
    split = min(QUADRATURE_SPLIT_K, k_max)
    k_nodes = np.unique(
        np.concatenate(
            [
                np.geomspace(k_min, split, QUADRATURE_LOG_NODES),
                np.linspace(split, k_max, QUADRATURE_LINEAR_NODES),
            ]
        )
    )
    weighted_pk = k_nodes ** int(k_power) * np.exp(log_pk(np.log(k_nodes)))

    y = np.atleast_1d(np.asarray(y, dtype=float))
    out = np.empty(y.size)
    for i, y_i in enumerate(y):
        out[i] = np.trapezoid(weighted_pk * spherical_jn(ell, k_nodes * y_i), k_nodes)
    return out


def _divide_by_y_with_limit(
    y: np.ndarray, numerator: np.ndarray, zero_limit: float
) -> np.ndarray:
    """`numerator / y`, substituting `zero_limit` wherever `y == 0`, shape (B,).

    Both `b_integral` and `psi_perp_integral` are `0/0` at zero separation with
    a finite (and different) limit; this is the one place that is handled, so
    neither can be given a NaN by accident.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    numerator = np.atleast_1d(np.asarray(numerator, dtype=float))
    out = np.empty(y.shape, dtype=float)
    at_zero = y == 0.0
    out[at_zero] = zero_limit
    out[~at_zero] = numerator[~at_zero] / y[~at_zero]
    return out


def a_integral(y: np.ndarray, k: np.ndarray, pk: np.ndarray) -> np.ndarray:
    """`A(y) = (1 / 2 pi^2) Int dk P(k) k j_1(k y)`, shape (B,), h^-1 Mpc.

    The master integral of the whole subpackage: after the `j_2` terms cancel
    (see `dipole.xi_udelta_dipole`), BOTH the real-space and the
    redshift-space dipole are `A(y)` times a scalar. Positive over the range
    linear theory is used on.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers and `P(k)`; see `bessel_moment`.

    Returns
    -------
    ndarray, shape (B,)
    """
    return bessel_moment(y, k, pk, ell=1, k_power=1) / (2.0 * np.pi**2)


def b_integral(y: np.ndarray, k: np.ndarray, pk: np.ndarray) -> np.ndarray:
    """`B(y) = (1 / 2 pi^2) Int dk P(k) k j_2(k y) / (k y)`, shape (B,), h^-1 Mpc.

    The `j_2` master integral. It appears in `Psi'_perp` and `Psi'_-`
    individually but cancels EXACTLY out of the dipole, for any `f` and any
    `b` -- which is the sharpest available check on the corrected sign in
    Nusser's eq. (17), because the cancellation fails if the sign is the
    printed one.

    The `1 / (k y)` is pulled out of the integral as `1 / y`, so this is the
    `(ell, k_power) = (2, 0)` moment divided by `y`.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers and `P(k)`; see `bessel_moment`.

    Returns
    -------
    ndarray, shape (B,)
    """
    # B(0) = 0: j_2(x)/x ~ x/15, so the ratio vanishes linearly in y. The
    # naive expression is 0/0 there, hence the explicit limit.
    return _divide_by_y_with_limit(
        y, bessel_moment(y, k, pk, ell=2, k_power=0) / (2.0 * np.pi**2), zero_limit=0.0
    )


def psi_perp_integral(y: np.ndarray, k: np.ndarray, pk: np.ndarray) -> np.ndarray:
    """`(1 / 2 pi^2) Int dk P(k) j_1(k y) / (k y)`, shape (B,), (h^-1 Mpc)^2.

    The `f`-free part of Gorski's `Psi_perp` -- kept here with the other
    transforms so that `gorski.py` holds physics and no FFTLog plumbing.
    `Psi_perp = f^2 *` this.

    Parameters
    ----------
    y : ndarray, shape (B,)
        Separations, h^-1 Mpc, strictly positive.
    k, pk : ndarray, shape (K,)
        Log-spaced wavenumbers and `P(k)`; see `bessel_moment`.

    Returns
    -------
    ndarray, shape (B,)
    """
    # At y = 0, j_1(x)/x -> 1/3, so this tends to Int P dk / (6 pi^2) -- the
    # 1-D velocity dispersion divided by f^2, finite and NON-zero, unlike every
    # other quantity in this module. Again 0/0 in the naive expression.
    return _divide_by_y_with_limit(
        y,
        bessel_moment(y, k, pk, ell=1, k_power=-1) / (2.0 * np.pi**2),
        zero_limit=zero_separation_moment(k, pk, 0) / (6.0 * np.pi**2),
    )
