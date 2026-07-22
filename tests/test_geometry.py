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


# ---------------------------------------------------------------------------
# Primitive-level tests
# ---------------------------------------------------------------------------
#
# These exercise the three geometry primitives in isolation, on hand-checkable
# inputs. They are deliberately separate from the sign gate above: the gate
# tests a physical claim about a velocity field, whereas these test the
# mechanical contracts -- shapes, norms, orientation, clipping -- that the gate
# silently assumes. A primitive failure here should be read before the gate is
# even consulted, since a broken norm makes the gate's verdict meaningless.


class TestUnitVector:
    def test_known_vector_normalises(self):
        np.testing.assert_allclose(
            unit_vector(np.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0.0]
        )

    def test_single_vector_keeps_shape(self):
        assert unit_vector(np.array([1.0, 2.0, 3.0])).shape == (3,)

    def test_batch_normalises_rowwise(self):
        vec = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0], [-1.0, 0.0, 0.0]])
        out = unit_vector(vec)

        assert out.shape == (3, 3)
        np.testing.assert_allclose(
            out, [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_already_unit_vector_is_unchanged(self):
        vec = _sphere_directions(64)
        np.testing.assert_allclose(unit_vector(vec), vec, atol=1e-15)

    def test_direction_is_preserved(self):
        """Normalisation must scale, never rotate or reflect."""
        vec = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        out = unit_vector(vec)
        # Each output is a positive multiple of its input.
        np.testing.assert_array_less(0.0, np.einsum("ij,ij->i", vec, out))

    def test_zero_vector_returns_zeros_not_nan(self):
        """Documented choice: no division by zero, no raise, no NaN."""
        out = unit_vector(np.array([0.0, 0.0, 0.0]))

        assert out.shape == (3,)
        np.testing.assert_array_equal(out, np.zeros(3))
        assert np.all(np.isfinite(out))

    def test_zero_row_in_batch_does_not_poison_neighbours(self):
        """A zero row must be contained -- the other rows are still correct."""
        vec = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        out = unit_vector(vec)

        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(
            out, [[0.6, 0.8, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

    def test_zero_vector_emits_no_warning(self):
        """The masked divide must not trip numpy's invalid-value warning."""
        with np.errstate(divide="raise", invalid="raise"):
            unit_vector(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


class TestPairSeparation:
    def test_hand_checked_case(self):
        s_center = np.array([1.0, 1.0, 1.0])
        s_others = np.array([[4.0, 5.0, 1.0], [1.0, 1.0, 3.0]])

        r_vec, r_mag = pair_separation(s_center, s_others)

        np.testing.assert_allclose(r_vec, [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
        np.testing.assert_allclose(r_mag, [5.0, 2.0])

    def test_points_from_center_to_others(self):
        """The frozen orientation r = s_V - s_T, asserted directionally.

        Displacing a neighbour further along +x must make r more positive in x.
        If the subtraction were reversed this test fails while every norm-based
        check above still passes -- which is the whole point of having it.
        """
        s_center = np.array([10.0, 0.0, 0.0])
        s_others = np.array([[15.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

        r_vec, _ = pair_separation(s_center, s_others)

        assert r_vec[0, 0] > 0.0   # neighbour beyond the centre
        assert r_vec[1, 0] < 0.0   # neighbour in front of the centre
        np.testing.assert_allclose(r_vec[:, 0], [5.0, -5.0])

    def test_magnitude_matches_norm_of_vector(self):
        rng = np.random.default_rng(20260721)
        s_center = rng.normal(size=3) * 50.0
        s_others = rng.normal(size=(128, 3)) * 50.0

        r_vec, r_mag = pair_separation(s_center, s_others)

        np.testing.assert_allclose(r_mag, np.linalg.norm(r_vec, axis=1))

    def test_shapes_are_stable(self):
        r_vec, r_mag = pair_separation(np.zeros(3), np.ones((7, 3)))
        assert r_vec.shape == (7, 3)
        assert r_mag.shape == (7,)

    def test_single_neighbour_still_returns_stacked_shapes(self):
        """N == 1 must not collapse to (3,) and (), per the shape contract."""
        r_vec, r_mag = pair_separation(np.zeros(3), np.array([[3.0, 4.0, 0.0]]))

        assert r_vec.shape == (1, 3)
        assert r_mag.shape == (1,)
        np.testing.assert_allclose(r_mag, [5.0])

    def test_coincident_pair_gives_zero_not_nan(self):
        """Self-pairs are a normal neighbour-query product; caller filters them."""
        r_vec, r_mag = pair_separation(np.array([2.0, 2.0, 2.0]),
                                       np.array([[2.0, 2.0, 2.0]]))

        np.testing.assert_array_equal(r_vec, np.zeros((1, 3)))
        np.testing.assert_array_equal(r_mag, [0.0])

    def test_no_periodic_wrapping_is_applied(self):
        """Coordinates are taken at face value -- periodicity is upstream.

        A neighbour unwrapped past the box face sits at a large positive
        offset. Minimum-imaging here would report a short separation with the
        opposite sign, so this pins the documented contract.
        """
        s_center = np.array([config.BOX_SIZE - 10.0, 0.0, 0.0])
        s_others = np.array([[config.BOX_SIZE + 10.0, 0.0, 0.0]])

        r_vec, r_mag = pair_separation(s_center, s_others)

        np.testing.assert_allclose(r_vec, [[20.0, 0.0, 0.0]])
        np.testing.assert_allclose(r_mag, [20.0])

    def test_inputs_are_not_mutated(self):
        s_center = np.array([1.0, 2.0, 3.0])
        s_others = np.array([[4.0, 5.0, 6.0]])
        before_center, before_others = s_center.copy(), s_others.copy()

        pair_separation(s_center, s_others)

        np.testing.assert_array_equal(s_center, before_center)
        np.testing.assert_array_equal(s_others, before_others)

    def test_accepts_readonly_observer_constant(self):
        """config.OBSERVER_POSITION is write-locked; it must still be usable."""
        r_vec, r_mag = pair_separation(
            config.OBSERVER_POSITION, config.OBSERVER_POSITION[None, :]
        )
        np.testing.assert_array_equal(r_mag, [0.0])


class TestMuCosine:
    N_T_HAT = np.array([1.0, 0.0, 0.0])

    def test_parallel_gives_plus_one(self):
        mu = mu_cosine(np.array([[1.0, 0.0, 0.0]]), self.N_T_HAT)
        np.testing.assert_allclose(mu, [1.0])

    def test_antiparallel_gives_minus_one(self):
        mu = mu_cosine(np.array([[-1.0, 0.0, 0.0]]), self.N_T_HAT)
        np.testing.assert_allclose(mu, [-1.0])

    def test_perpendicular_gives_zero(self):
        r_hat = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        np.testing.assert_allclose(mu_cosine(r_hat, self.N_T_HAT), [0.0, 0.0, 0.0])

    def test_known_intermediate_angle(self):
        r_hat = unit_vector(np.array([[1.0, 1.0, 0.0]]))
        np.testing.assert_allclose(mu_cosine(r_hat, self.N_T_HAT), [np.sqrt(0.5)])

    def test_shape_is_n(self):
        r_hat = _sphere_directions(37)
        assert mu_cosine(r_hat, self.N_T_HAT).shape == (37,)

    def test_single_separation_returns_length_one(self):
        assert mu_cosine(np.array([[0.0, 1.0, 0.0]]), self.N_T_HAT).shape == (1,)

    def test_clips_overshooting_input(self):
        """Guards float overshoot: |mu| must never exceed 1.

        The inputs here overshoot by far more than float error to make the clip
        observable at all; the real overshoot it exists for is ~1e-16.
        """
        r_hat = np.array([[1.0 + 1e-9, 0.0, 0.0], [-1.0 - 1e-9, 0.0, 0.0]])
        mu = mu_cosine(r_hat, self.N_T_HAT)

        np.testing.assert_array_equal(mu, [1.0, -1.0])
        assert np.all(np.abs(mu) <= 1.0)

    def test_clip_is_not_a_normalisation(self):
        """A short un-normalised input is passed through, not rescaled to 1.

        Documents that the clip only bounds; it does not repair bad input.
        """
        mu = mu_cosine(np.array([[0.5, 0.0, 0.0]]), self.N_T_HAT)
        np.testing.assert_allclose(mu, [0.5])

    def test_matches_explicit_dot_product(self):
        rng = np.random.default_rng(7)
        r_hat = unit_vector(rng.normal(size=(256, 3)))
        n_T_hat = unit_vector(rng.normal(size=3))

        np.testing.assert_allclose(
            mu_cosine(r_hat, n_T_hat), r_hat @ n_T_hat, atol=1e-15
        )

    def test_broadcasts_per_pair_lines_of_sight(self):
        """An (N, 3) n_T_hat pairs row-wise, not as an outer product."""
        r_hat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        n_T_hat = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        np.testing.assert_allclose(mu_cosine(r_hat, n_T_hat), [1.0, 0.0])

    def test_reversing_r_hat_negates_mu(self):
        """The primitive-level statement of the sign hazard in config.py."""
        r_hat = _sphere_directions(128)
        n_T_hat = unit_vector(np.array([0.3, -0.5, 0.8]))

        np.testing.assert_allclose(
            mu_cosine(-r_hat, n_T_hat), -mu_cosine(r_hat, n_T_hat), atol=1e-15
        )


class TestPrimitivesCompose:
    def test_mu_is_plus_one_directly_behind_the_centre(self):
        """End-to-end orientation check with no velocities involved.

        A neighbour placed further from the observer than the centre, along the
        same ray, must give mu = +1 ("far side"). This is the composition the
        estimator performs, and it fixes the sign of the geometry independently
        of the infall gate above.
        """
        observer = np.asarray(config.OBSERVER_POSITION, dtype=float)
        s_center = observer + np.array([R_CENTER, 0.0, 0.0])
        s_far = s_center + np.array([[R_SHELL, 0.0, 0.0]])
        s_near = s_center - np.array([[R_SHELL, 0.0, 0.0]])

        n_T_hat = unit_vector(s_center - observer)

        r_far, mag_far = pair_separation(s_center, s_far)
        r_near, mag_near = pair_separation(s_center, s_near)

        np.testing.assert_allclose(mag_far, [R_SHELL])
        np.testing.assert_allclose(mag_near, [R_SHELL])
        np.testing.assert_allclose(mu_cosine(unit_vector(r_far), n_T_hat), [1.0])
        np.testing.assert_allclose(mu_cosine(unit_vector(r_near), n_T_hat), [-1.0])


# TODO(sign-gate): once the estimator is implemented, extend this file with
#   - a velocity-shuffle null test (permuting u across positions must kill the
#     cross-correlation),
#   - a periodic-wrap test placing the central object near a box face, where a
#     plain Euclidean difference and the minimum image disagree,
#   - a finite-distance monopole check against U_0 = (2r / 3R) v_inf, which
#     validates the leakage term the monopole diagnostic is meant to expose.
