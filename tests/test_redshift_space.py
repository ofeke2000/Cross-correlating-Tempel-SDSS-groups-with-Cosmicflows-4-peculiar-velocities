"""
test_redshift_space.py
-----------------------
Unit tests for the redshift-space coordinate transform
(`dvcorr.redshift_space`): displacing a halo along its own observer line of
sight by its own radial peculiar velocity, and the through-observer flip
guard that keeps that displacement well-defined.

These are transform-LEVEL tests -- pure `to_redshift_space` / `radial_velocity`
/ `v_margin_from_statistic` mechanics, no estimator, no pipeline. The
pipeline-level tests (the redshift-space sign gate, epsilon-continuity, the
shared center set) live in `tests/test_velocity_centered_dipole.py` (the
former, since they extend that file's existing infall toy directly) and
`tests/test_redshift_space_comparison.py` (the latter).
"""

from __future__ import annotations

import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.geometry import radial_flow_axis, unit_vector
from dvcorr.redshift_space import (
    FLIP_GUARD_FLOOR,
    RedshiftSpaceTransform,
    radial_velocity,
    to_redshift_space,
    v_margin_from_statistic,
)

# Reuse the project's shared infall toy rather than building a parallel one
# that could drift out of agreement with it.
from tests.test_geometry import _infall_shell_with_velocities, _sphere_directions


# ---------------------------------------------------------------------------
# Zero-velocity limit
# ---------------------------------------------------------------------------


def test_zero_velocity_gives_identical_positions():
    """All velocities zero -> redshift-space positions equal real positions exactly.

    v_r = 0 everywhere, so s_mag = r_mag + 0/100 = r_mag: the transform is the
    identity. Nothing is dropped (r_mag > 0 for every non-degenerate input
    here, comfortably above FLIP_GUARD_FLOOR).
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    rng = np.random.default_rng(0)
    n = 500
    positions = observer + rng.uniform(20.0, 200.0, size=n)[:, None] * _sphere_directions(n)
    velocities = np.zeros_like(positions)

    transform = to_redshift_space(positions, velocities, observer=observer)

    assert transform.n_dropped == 0
    assert np.all(transform.kept_mask)
    np.testing.assert_allclose(transform.s_redshift, positions, atol=1e-10)


# ---------------------------------------------------------------------------
# Displacement direction
# ---------------------------------------------------------------------------


def test_outward_velocity_moves_the_halo_further_from_the_observer():
    """A halo with positive radial velocity moves AWAY from the observer.

    u = v . n_hat > 0 means s_mag = r_mag + u/100 > r_mag: the displaced
    distance from the observer strictly exceeds the real one.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    n_hat = np.array([1.0, 0.0, 0.0])
    position = observer + 100.0 * n_hat
    velocity = 500.0 * n_hat  # purely radial, outbound

    transform = to_redshift_space(position, velocity, observer=observer)

    assert transform.n_dropped == 0
    r_mag = np.linalg.norm(position - observer)
    s_mag = np.linalg.norm(transform.s_redshift[0] - observer)
    assert s_mag > r_mag
    assert s_mag == pytest.approx(r_mag + 500.0 / 100.0)


def test_inward_velocity_moves_the_halo_closer_to_the_observer():
    """A halo with negative radial velocity moves TOWARD the observer."""
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    n_hat = np.array([0.0, 1.0, 0.0])
    position = observer + 100.0 * n_hat
    velocity = -500.0 * n_hat  # purely radial, inbound

    transform = to_redshift_space(position, velocity, observer=observer)

    assert transform.n_dropped == 0
    r_mag = np.linalg.norm(position - observer)
    s_mag = np.linalg.norm(transform.s_redshift[0] - observer)
    assert s_mag < r_mag
    assert s_mag == pytest.approx(r_mag - 500.0 / 100.0)


# ---------------------------------------------------------------------------
# n_hat invariance
# ---------------------------------------------------------------------------


def test_n_hat_and_u_are_invariant_under_the_transform():
    """n_hat (and hence u = v . n_hat) is unchanged by the transform.

    The displacement is purely radial, so it cannot rotate the line of
    sight -- see `dvcorr.redshift_space`'s module docstring. Measured on the
    project's shared infall toy (`_infall_shell_with_velocities`): n_hat
    agrees to ~3.3e-16 and u to ~8.5e-14 (floating-point round-off), so an
    explicit, generous ATOL is used rather than exact equality.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_center, s_neighbors, u, v_vec = _infall_shell_with_velocities()

    transform = to_redshift_space(s_neighbors, v_vec, observer=observer)
    assert transform.n_dropped == 0

    n_hat_real = unit_vector(s_neighbors - observer)
    n_hat_redshift = unit_vector(transform.s_redshift - observer)
    np.testing.assert_allclose(n_hat_real, n_hat_redshift, atol=1e-10)

    _, u_real = radial_flow_axis(v_vec, n_hat_real)
    _, u_redshift = radial_flow_axis(v_vec, n_hat_redshift)
    np.testing.assert_allclose(u_real, u_redshift, atol=1e-8)


# ---------------------------------------------------------------------------
# Through-observer flip guard
# ---------------------------------------------------------------------------


def test_through_observer_guard_drops_the_flipped_halo():
    """A near-observer, fast-inward halo is detected and dropped, count reported.

    r_mag = 0.5 h^-1 Mpc, v_r = -100 km/s -> s_mag = 0.5 - 1.0 = -0.5, well
    below FLIP_GUARD_FLOOR (1.0 h^-1 Mpc): the object must be dropped, not
    silently handed a zero-row (or worse, reversed) direction.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    n_hat = np.array([1.0, 0.0, 0.0])

    flipping = observer + 0.5 * n_hat
    ordinary = observer + 100.0 * n_hat
    positions = np.vstack([flipping, ordinary])
    velocities = np.vstack([-100.0 * n_hat, 10.0 * n_hat])  # fast inbound, then a normal one

    transform = to_redshift_space(positions, velocities, observer=observer)

    assert transform.n_dropped == 1
    assert transform.n_input == 2
    np.testing.assert_array_equal(transform.kept_mask, [False, True])
    assert transform.s_redshift.shape == (1, 3)


def test_through_observer_guard_floor_is_a_strict_positive_boundary():
    """s_mag exactly at the floor is dropped; just above it survives.

    The guard is `s_mag > flip_guard_floor` (strict), matching
    `FLIP_GUARD_FLOOR`'s "positive floor, not `< 0`" contract.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    n_hat = np.array([0.0, 0.0, 1.0])
    floor = 2.0

    at_floor = observer + floor * n_hat  # v = 0 -> s_mag == floor exactly
    just_above = observer + (floor + 1e-6) * n_hat

    transform = to_redshift_space(
        np.vstack([at_floor, just_above]),
        np.zeros((2, 3)),
        observer=observer,
        flip_guard_floor=floor,
    )

    np.testing.assert_array_equal(transform.kept_mask, [False, True])
    assert transform.n_dropped == 1


def test_flip_guard_floor_default_is_documented_and_positive():
    """FLIP_GUARD_FLOOR is a strictly positive module constant (hard rule 4)."""
    assert FLIP_GUARD_FLOOR > 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_to_redshift_space_rejects_mismatched_shapes():
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    positions = np.zeros((5, 3))
    velocities = np.zeros((4, 3))
    with pytest.raises(ValueError):
        to_redshift_space(positions, velocities, observer=observer)


def test_to_redshift_space_rejects_non_finite_velocity():
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    positions = observer + np.array([[100.0, 0.0, 0.0]])
    velocities = np.array([[np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError):
        to_redshift_space(positions, velocities, observer=observer)


def test_to_redshift_space_rejects_non_positive_flip_guard_floor():
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    positions = observer + np.array([[100.0, 0.0, 0.0]])
    velocities = np.array([[10.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        to_redshift_space(positions, velocities, observer=observer, flip_guard_floor=0.0)


# ---------------------------------------------------------------------------
# radial_velocity / v_margin_from_statistic
# ---------------------------------------------------------------------------


def test_radial_velocity_matches_hand_computed_projection():
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    n_hat = unit_vector(np.array([[1.0, 1.0, 0.0]]))[0]
    position = observer + 50.0 * n_hat
    velocity = np.array([30.0, -10.0, 5.0])

    v_r = radial_velocity(position[None, :], velocity[None, :], observer=observer)
    expected = np.dot(velocity, n_hat)
    assert v_r[0] == pytest.approx(expected)


def test_v_margin_max_statistic_is_the_true_maximum():
    v_r = np.array([-500.0, 10.0, 300.0, -1200.0, 42.0])
    assert v_margin_from_statistic(v_r, "max", percentile=99.9) == pytest.approx(1200.0)


def test_v_margin_percentile_statistic_matches_numpy_percentile():
    rng = np.random.default_rng(1)
    v_r = rng.normal(scale=300.0, size=10000)
    expected = np.percentile(np.abs(v_r), 99.9)
    assert v_margin_from_statistic(v_r, "percentile", percentile=99.9) == pytest.approx(expected)


def test_v_margin_unknown_statistic_raises():
    with pytest.raises(ValueError):
        v_margin_from_statistic(np.array([1.0, 2.0]), "mean", percentile=99.9)


def test_redshift_space_transform_is_a_frozen_dataclass():
    """Cheap contract check: the result type is immutable, matching every
    other result dataclass in this project."""
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    transform = to_redshift_space(
        observer + np.array([[10.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), observer=observer
    )
    assert isinstance(transform, RedshiftSpaceTransform)
    with pytest.raises(Exception):
        transform.n_dropped = 99  # frozen dataclass: attribute assignment must fail
