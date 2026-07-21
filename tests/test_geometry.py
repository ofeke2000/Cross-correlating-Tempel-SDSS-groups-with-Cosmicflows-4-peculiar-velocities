"""
test_geometry.py
----------------
The sign gate.

This is the test the whole project hangs on. The dipole's sign carries the
physics: infall must come out negative. Nothing in the code raises if the pair
vector is oriented the wrong way -- the analysis runs, produces a smooth curve,
and reports a growth rate with the wrong sign. This file is the only thing
standing between that mistake and a result.

Everything here is currently marked xfail: geometry.py and
estimators/shell_dipole.py are stubs that raise NotImplementedError. The
assertions are written out in full anyway, so that implementing the physics is
a matter of making an existing, already-agreed test pass rather than inventing
the expected answer afterwards. Remove each `xfail` marker as its dependency
lands; do not weaken an assertion to make it pass.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from geometry import mu_cosine, pair_separation, unit_vector
from estimators.shell_dipole import shell_dipole


# ---------------------------------------------------------------------------
# Toy setup shared by the sign tests
# ---------------------------------------------------------------------------
#
# The configuration is deliberately the simplest one in which the dipole is
# non-zero and its sign is known analytically:
#
#   - The observer sits at the box centre (config.OBSERVER_POSITION).
#
#   - One central density object T sits at a known distance R along +x from the
#     observer. R is large compared with the shell radius so that the
#     distant-observer limit holds and the finite-distance monopole leakage,
#     which is O(r/R), stays small. It is NOT infinite, so the leakage is real
#     and the monopole is not expected to be exactly zero.
#
#   - N neighbours are placed on a single thin shell of radius r around T,
#     spread as uniformly as possible over the sphere. Uniformity matters: an
#     anisotropic shell produces a dipole from geometry alone, which is
#     precisely the systematic the monopole diagnostic exists to catch.
#
#   - Pure radial infall is imposed by hand: every neighbour moves toward T
#     with the same speed, v_i = V_INFALL * r_hat_i with V_INFALL < 0. No
#     random component, no bulk flow, no Hubble term -- this is not a physical
#     velocity field, it is a controlled input whose answer is known.
#
# Expected outcome: u_i = v_i . n_hat_V,i is positive on the near side (mu < 0)
# and negative on the far side (mu > 0), so u_i * mu_i < 0 for every neighbour
# and the summed dipole is negative. In the distant-observer limit the
# normalised dipole recovers V_INFALL itself.

R_CENTER = 200.0     # observer -> central object distance, h^-1 Mpc
R_SHELL = 20.0       # shell radius around the central object, h^-1 Mpc
V_INFALL = -200.0    # infall speed, km/s; negative = toward the centre
N_SHELL = 2048       # neighbours on the shell


def _sphere_directions(n: int) -> np.ndarray:
    """Return (n, 3) near-uniform unit vectors via a Fibonacci sphere.

    Used instead of random directions so the test is deterministic and the
    residual shell anisotropy is a fixed, small, known quantity rather than a
    source of run-to-run scatter in the recovered dipole.
    """
    i = np.arange(n, dtype=float) + 0.5
    cos_theta = 1.0 - 2.0 * i / n
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, None))
    phi = np.pi * (1.0 + 5.0**0.5) * i
    return np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)
    )


def _infall_shell() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the toy setup.

    Returns
    -------
    s_center : (3,) central object position, box coordinates
    s_neighbors : (N_SHELL, 3) neighbour positions on the shell
    u : (N_SHELL,) observer-centred radial peculiar velocities, km/s
    """
    observer = np.asarray(config.OBSERVER_POSITION, dtype=float)
    s_center = observer + np.array([R_CENTER, 0.0, 0.0])

    r_hat = _sphere_directions(N_SHELL)
    s_neighbors = s_center + R_SHELL * r_hat

    # Pure radial infall toward the centre, then projected onto each
    # neighbour's OWN line of sight -- that projection is what a peculiar
    # velocity catalogue actually measures, and it is where the sign is set.
    v_vec = V_INFALL * r_hat
    n_hat_v = unit_vector(s_neighbors - observer)
    u = np.einsum("ij,ij->i", v_vec, n_hat_v)

    return s_center, s_neighbors, u


@pytest.mark.xfail(
    reason="geometry.py and shell_dipole are stubs; this is the target behaviour",
    raises=NotImplementedError,
    strict=True,
)
def test_spherical_infall_gives_negative_dipole():
    """Coherent infall must produce a negative dipole. This is the sign gate."""
    s_center, s_neighbors, u = _infall_shell()
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    monopole, dipole = shell_dipole(
        s_center=s_center,
        s_neighbors=s_neighbors,
        shell_edges=shell_edges,
        weights=u,
    )

    # Every neighbour is in the single shell.
    assert monopole.shape == (1,)
    assert dipole.shape == (1,)

    # The assertion the project rests on.
    assert dipole[0] < 0.0
    assert np.sign(dipole[0]) == config.INFALL_DIPOLE_SIGN

    # And it should recover the imposed infall speed, not merely its sign.
    # Normalising the L_1 accumulator by the count picks up the <mu^2> = 1/3
    # of a uniform shell, hence the factor of 3. Tolerance is loose because the
    # setup is at finite R, where the recovery is exact only to O(r/R).
    recovered = 3.0 * dipole[0] / monopole[0]
    assert recovered == pytest.approx(V_INFALL, rel=0.1)


@pytest.mark.xfail(
    reason="geometry.py is a stub; this is the target behaviour",
    raises=NotImplementedError,
    strict=True,
)
def test_reversing_the_pair_vector_flips_the_dipole():
    """r = s_T - s_V instead of s_V - s_T must flip the dipole's sign.

    Documents the failure mode rather than guarding against it: the flip is
    mathematically correct behaviour, which is exactly why it is dangerous. If
    someone swaps the arguments to `pair_separation`, no error appears anywhere
    -- only this sign changes.
    """
    s_center, s_neighbors, u = _infall_shell()
    observer = np.asarray(config.OBSERVER_POSITION, dtype=float)

    n_T_hat = unit_vector(s_center - observer)
    r_vec, _ = pair_separation(s_center, s_neighbors)

    mu = mu_cosine(unit_vector(r_vec), n_T_hat)
    mu_reversed = mu_cosine(unit_vector(-r_vec), n_T_hat)

    np.testing.assert_allclose(mu_reversed, -mu, atol=1e-12)
    assert np.sum(u * mu) < 0.0            # frozen convention: infall negative
    assert np.sum(u * mu_reversed) > 0.0   # reversed: the wrong answer


@pytest.mark.xfail(
    reason="geometry.py is a stub; this is the target behaviour",
    raises=NotImplementedError,
    strict=True,
)
def test_mu_is_a_cosine_and_spans_the_shell():
    """mu must lie in [-1, 1] and cover both hemispheres for a full shell."""
    s_center, s_neighbors, _ = _infall_shell()
    observer = np.asarray(config.OBSERVER_POSITION, dtype=float)

    n_T_hat = unit_vector(s_center - observer)
    r_vec, r_mag = pair_separation(s_center, s_neighbors)
    mu = mu_cosine(unit_vector(r_vec), n_T_hat)

    assert mu.shape == (N_SHELL,)
    assert np.all(np.abs(mu) <= 1.0 + 1e-12)
    assert mu.min() < -0.9 and mu.max() > 0.9      # both hemispheres populated
    assert np.mean(mu) == pytest.approx(0.0, abs=1e-2)  # uniform shell: no
    #                                                     geometric dipole
    np.testing.assert_allclose(r_mag, R_SHELL, rtol=1e-12)


# TODO(sign-gate): once the estimator is implemented, extend this file with
#   - a velocity-shuffle null test (permuting u across positions must kill the
#     cross-correlation),
#   - a periodic-wrap test placing the central object near a box face, where a
#     plain Euclidean difference and the minimum image disagree,
#   - a finite-distance monopole check against U_0 = (2r / 3R) v_inf, which
#     validates the leakage term the monopole diagnostic is meant to expose.
