"""
test_shell_dipole.py
--------------------
Unit tests for the shell-binned monopole/dipole estimator.

Two kinds of test live here. The first is the estimator-level sign gate: it
reuses the EXACT infall toy from the geometry sign gate (test_geometry.py) and
pushes it through `shell_dipole`, confirming that the binning and accumulation
layer preserves the sign already verified at the primitive level. If this fails
while the geometry sign gate passes, the bug is in the estimator wiring -- the
binning, the weight handling, or the moment sums -- not in the convention.

The rest pin the mechanical contracts the sign gate silently relies on:
isotropy kills the dipole while leaving the count, neighbors land in the
shells their radii dictate, out-of-range neighbors are excluded, and empty
shells return zeros rather than NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from geometry import unit_vector
from estimators.shell_dipole import shell_dipole

# Reuse the SAME infall toy the geometry sign gate is built on, rather than
# rebuilding a parallel one that could drift out of agreement with it.
from tests.test_geometry import (
    _infall_shell,
    _sphere_directions,
    R_SHELL,
    V_INFALL,
    N_SHELL,
)


# ---------------------------------------------------------------------------
# Estimator-level sign gate
# ---------------------------------------------------------------------------


def test_estimator_preserves_the_infall_sign():
    """The negative-dipole convention must survive binning and accumulation.

    Same toy as the geometry sign gate, now run through `shell_dipole` with the
    neighbor radial velocities as weights. The dipole of the single occupied
    shell must be negative, and normalizing it by the pair count must recover
    the imposed infall speed (the factor 3 = 2*ell + 1 picks up <mu^2> = 1/3).
    """
    s_center, s_neighbors, u = _infall_shell()
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    result = shell_dipole(s_center, s_neighbors, shell_edges, weights=u)

    assert result.dipole.shape == (1,)
    assert result.pair_count[0] == N_SHELL          # every neighbor in one shell

    assert result.dipole[0] < 0.0
    assert np.sign(result.dipole[0]) == config.INFALL_DIPOLE_SIGN

    recovered = 3.0 * result.dipole[0] / result.pair_count[0]
    assert recovered == pytest.approx(V_INFALL, rel=0.1)


def test_reversed_pair_orientation_would_flip_the_estimator_dipole():
    """Guard the silent failure mode at the estimator level.

    Swapping the roles of center and neighbor reverses r, hence mu, hence the
    dipole. Nothing raises -- only the sign changes -- so this documents that
    the estimator inherits the primitive-level hazard rather than curing it.
    Here the single 'neighbor' is the former center and vice versa, so the
    weight cannot be per-neighbor in the same way; we instead check the sign
    flip on a symmetric two-point pair.
    """
    s_center, s_neighbors, u = _infall_shell()
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    forward = shell_dipole(s_center, s_neighbors, shell_edges, weights=u)

    # Mirror every neighbor through the center: s' = 2*s_center - s. This
    # sends r -> -r and mu -> -mu while keeping |r| (and thus the shell) fixed,
    # so the dipole must flip sign and the count must not change.
    s_mirrored = 2.0 * s_center - s_neighbors
    reversed_ = shell_dipole(s_center, s_mirrored, shell_edges, weights=u)

    assert reversed_.pair_count[0] == forward.pair_count[0]
    assert np.sign(reversed_.dipole[0]) == -np.sign(forward.dipole[0])
    assert reversed_.dipole[0] > 0.0            # the wrong (positive) answer


# ---------------------------------------------------------------------------
# Isotropic null: dipole vanishes, monopole does not
# ---------------------------------------------------------------------------


def test_isotropic_shell_has_vanishing_dipole_but_finite_monopole():
    """An isotropic shell with no infall structure has zero dipole.

    Neighbors are spread uniformly on a shell (a Fibonacci sphere, so the
    residual anisotropy is a fixed small quantity) and every neighbor is given
    the SAME nonzero weight -- a pure monopole with no angular structure. The
    monopole must be large (weight times count) while the dipole is consistent
    with zero to sampling noise. This is the monopole-vs-dipole contrast made
    concrete: the count survives, the dipole cancels.
    """
    observer = np.asarray(config.OBSERVER_POSITION, dtype=float)
    s_center = observer + np.array([200.0, 0.0, 0.0])

    r_hat = _sphere_directions(N_SHELL)
    s_neighbors = s_center + R_SHELL * r_hat

    weight_value = 300.0
    weights = np.full(N_SHELL, weight_value)
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    result = shell_dipole(s_center, s_neighbors, shell_edges, weights=weights)

    assert result.pair_count[0] == N_SHELL
    assert result.monopole[0] == pytest.approx(weight_value * N_SHELL)

    # Normalised dipole: 3 * <w mu> / <w> = 3 * <mu> since the weight is
    # uniform, and <mu> -> 0 for an isotropic shell.
    normalized_dipole = 3.0 * result.dipole[0] / result.monopole[0]
    assert normalized_dipole == pytest.approx(0.0, abs=1e-2)


def test_random_velocities_uncorrelated_with_position_give_zero_dipole():
    """A velocity-shuffle-style null: weights unrelated to geometry cancel.

    Isotropic positions, but now the weights are random (mean zero) and carry
    no angular information. Both the raw dipole and the normalized dipole must
    sit at zero within sampling noise, while the geometry (count) is unchanged.
    """
    observer = np.asarray(config.OBSERVER_POSITION, dtype=float)
    s_center = observer + np.array([200.0, 0.0, 0.0])

    r_hat = _sphere_directions(N_SHELL)
    s_neighbors = s_center + R_SHELL * r_hat

    rng = np.random.default_rng(20260722)
    weights = rng.normal(loc=0.0, scale=250.0, size=N_SHELL)
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    result = shell_dipole(s_center, s_neighbors, shell_edges, weights=weights)

    assert result.pair_count[0] == N_SHELL
    # <w mu> ~ 0 because w and mu are independent, each mean ~0.
    normalized_dipole = 3.0 * result.dipole[0] / result.pair_count[0]
    assert normalized_dipole == pytest.approx(0.0, abs=30.0)  # << 250 km/s


# ---------------------------------------------------------------------------
# Binning correctness
# ---------------------------------------------------------------------------


def _line_of_sight_center() -> tuple[np.ndarray, np.ndarray]:
    """Central object and the +x radial direction from the observer."""
    observer = np.asarray(config.OBSERVER_POSITION, dtype=float)
    s_center = observer + np.array([200.0, 0.0, 0.0])
    return observer, s_center


def test_neighbors_land_in_the_shell_their_radius_dictates():
    """Hand-placed neighbors at known radii fall in the expected shells."""
    _, s_center = _line_of_sight_center()
    # One neighbor at r = 5, 15, 25 along +y (perpendicular to the line of
    # sight, so mu = 0 and the placement is a pure binning check).
    radii = np.array([5.0, 15.0, 25.0])
    s_neighbors = s_center + np.column_stack(
        (np.zeros_like(radii), radii, np.zeros_like(radii))
    )
    shell_edges = np.array([0.0, 10.0, 20.0, 30.0])

    result = shell_dipole(s_center, s_neighbors, shell_edges)

    np.testing.assert_array_equal(result.pair_count, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(result.shell_centers, [5.0, 15.0, 25.0])


def test_neighbors_outside_the_edge_range_are_excluded():
    """Below the innermost edge and beyond the outermost edge are both dropped."""
    _, s_center = _line_of_sight_center()
    radii = np.array([5.0, 15.0, 25.0, 35.0])  # 5 below edges[0]=10, 35 above 30
    s_neighbors = s_center + np.column_stack(
        (np.zeros_like(radii), radii, np.zeros_like(radii))
    )
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = shell_dipole(s_center, s_neighbors, shell_edges)

    # Only r = 15 and r = 25 survive, one per shell.
    np.testing.assert_array_equal(result.pair_count, [1.0, 1.0])
    assert result.pair_count.sum() == 2.0


def test_outermost_edge_is_included_in_the_last_shell():
    """A neighbor sitting exactly on the outer edge is not silently dropped."""
    _, s_center = _line_of_sight_center()
    s_neighbors = s_center + np.array([[0.0, 30.0, 0.0]])  # r = 30 == edges[-1]
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = shell_dipole(s_center, s_neighbors, shell_edges)

    np.testing.assert_array_equal(result.pair_count, [0.0, 1.0])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_shell_returns_zero_not_nan():
    """A shell no neighbor reaches has count 0 and dipole 0.0, never NaN."""
    _, s_center = _line_of_sight_center()
    s_neighbors = s_center + np.array([[0.0, 5.0, 0.0]])  # only in the first shell
    shell_edges = np.array([0.0, 10.0, 20.0])

    result = shell_dipole(s_center, s_neighbors, shell_edges, weights=np.array([7.0]))

    np.testing.assert_array_equal(result.pair_count, [1.0, 0.0])
    assert result.dipole[1] == 0.0
    assert result.monopole[1] == 0.0
    assert np.all(np.isfinite(result.pair_count))
    assert np.all(np.isfinite(result.dipole))
    assert np.all(np.isfinite(result.monopole))


def test_no_neighbors_at_all_returns_all_zero_shells():
    """Zero neighbors is a valid, NaN-free result of all-empty shells."""
    _, s_center = _line_of_sight_center()
    s_neighbors = np.empty((0, 3))
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = shell_dipole(s_center, s_neighbors, shell_edges)

    np.testing.assert_array_equal(result.pair_count, [0.0, 0.0])
    np.testing.assert_array_equal(result.dipole, [0.0, 0.0])
    assert np.all(np.isfinite(result.dipole))


def test_uniform_weights_make_monopole_equal_count():
    """With w = 1 the L_0 moment is the pair count, by definition."""
    _, s_center = _line_of_sight_center()
    radii = np.array([5.0, 15.0, 15.0])
    s_neighbors = s_center + np.column_stack(
        (np.zeros_like(radii), radii, np.zeros_like(radii))
    )
    shell_edges = np.array([0.0, 10.0, 20.0])

    result = shell_dipole(s_center, s_neighbors, shell_edges)  # weights=None

    np.testing.assert_array_equal(result.monopole, result.pair_count)
    np.testing.assert_array_equal(result.pair_count, [1.0, 2.0])


def test_coincident_neighbor_contributes_to_count_but_not_dipole():
    """r = 0 has no direction: mu = 0, so it counts but adds nothing to Q."""
    _, s_center = _line_of_sight_center()
    # One neighbor on top of the center (r = 0), one at r = 5 with mu != 0.
    s_offset = unit_vector(s_center - np.asarray(config.OBSERVER_POSITION, float))
    s_neighbors = np.vstack([s_center, s_center + 5.0 * s_offset])
    shell_edges = np.array([0.0, 10.0])
    weights = np.array([100.0, 100.0])

    result = shell_dipole(s_center, s_neighbors, shell_edges, weights=weights)

    assert result.pair_count[0] == 2.0                    # both are in the shell
    assert result.monopole[0] == pytest.approx(200.0)
    # The coincident one carries mu = 0; only the r = 5 neighbor (mu = +1,
    # directly behind the center) contributes to the dipole.
    assert result.dipole[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_non_increasing_edges_raise():
    _, s_center = _line_of_sight_center()
    with pytest.raises(ValueError):
        shell_dipole(s_center, s_center[None, :] + 5.0, np.array([20.0, 10.0]))


def test_edge_beyond_max_analysis_radius_raises():
    _, s_center = _line_of_sight_center()
    too_far = np.array([0.0, config.MAX_ANALYSIS_RADIUS + 1.0])
    with pytest.raises(ValueError):
        shell_dipole(s_center, s_center[None, :] + 5.0, too_far)


def test_mismatched_weights_length_raises():
    _, s_center = _line_of_sight_center()
    s_neighbors = s_center + np.array([[0.0, 5.0, 0.0], [0.0, 6.0, 0.0]])
    with pytest.raises(ValueError):
        shell_dipole(s_center, s_neighbors, np.array([0.0, 10.0]),
                     weights=np.array([1.0]))


def test_non_finite_weight_raises_not_treated_as_zero():
    """Missing velocity is missing data (hard rule 5), not a silent zero."""
    _, s_center = _line_of_sight_center()
    s_neighbors = s_center + np.array([[0.0, 5.0, 0.0], [0.0, 6.0, 0.0]])
    with pytest.raises(ValueError):
        shell_dipole(s_center, s_neighbors, np.array([0.0, 10.0]),
                     weights=np.array([1.0, np.nan]))
