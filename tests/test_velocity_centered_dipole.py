"""
test_velocity_centered_dipole.py
---------------------------------
Unit tests for the velocity-centered (Nusser 2017 eq. 23-24, "zeta") estimator.

Mirrors tests/test_shell_dipole.py in tone and structure. The centerpiece is
the joint sign gate: the group-centered xi_Tu and the velocity-centered zeta
run on the IDENTICAL toy infall configuration
(tests.test_geometry._infall_shell_with_velocities) and must come out with
OPPOSITE signs -- the zeta_ell = (-1)**ell * xi_Tu,ell relation
(conventions.nusser_multipole_sign) made concrete on real numbers, not just
asserted in a docstring. If this fails while the group-centered gate
(tests/test_shell_dipole.py) still passes, the bug is in the new estimator's
orientation bookkeeping, not in the shared convention.

The rest pin the mechanical contracts the joint gate silently relies on:
`real_y10` against scipy's own normalized spherical harmonic, the reversed
construction against an independent hand-assembled path, isotropy killing
the dipole while leaving the count, binning correctness, empty/degenerate
inputs, the core-cut's effect on the boundary-truncation bias, a shuffle
null, and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import sph_harm_y

from dvcorr import conventions
from dvcorr.geometry import mu_cosine, pair_separation, unit_vector
from dvcorr.estimators.shell_dipole import (
    center_standard_error,
    core_center_mask,
    expected_shell_occupancy,
    real_y10,
    shell_dipole,
    velocity_centered_shell_dipole,
)

# Reuse the SAME infall toy the geometry/group-centered sign gates are built
# on, rather than rebuilding a parallel one that could drift out of agreement.
from tests.test_geometry import (
    _infall_shell_with_velocities,
    _sphere_directions,
    R_CENTER,
    R_SHELL,
    V_INFALL,
    N_SHELL,
)


# ---------------------------------------------------------------------------
# Joint sign gate
# ---------------------------------------------------------------------------
#
# Geometry: every shell member sits at most R_CENTER + R_SHELL = 220 h^-1 Mpc
# from the observer (triangle inequality: |s_center - observer| = R_CENTER,
# |s_neighbor - s_center| = R_SHELL). `sub_volume_radius` and the DEFAULT
# `core_margin` (None -> shell_edges[-1] = 1.5 * R_SHELL = 30.0) are chosen so
# every member clears the core cut: 220 <= sub_volume_radius - 30, i.e.
# sub_volume_radius >= 250; 280 leaves comfortable headroom.

_VC_SUB_VOLUME_RADIUS = 280.0
_VC_MAX_MEMBER_OBSERVER_DISTANCE = R_CENTER + R_SHELL  # 220.0, triangle-inequality bound
_VC_CORE_MARGIN_DEFAULT = 1.5 * R_SHELL  # shell_edges[-1]; matches core_margin=None


def _joint_gate_toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The velocity-centered half of the joint gate's inputs.

    Centers = the N_SHELL shell members (carrying their 3-D infall
    velocities); the single tracer = the former central density object.
    Every center therefore sees exactly ONE tracer, so the realized
    `pair_count` is N_SHELL (one per center).
    """
    s_center, s_neighbors, u, v_vec = _infall_shell_with_velocities()
    tracers = s_center[None, :]
    return s_neighbors, v_vec, tracers


def test_joint_sign_gate_group_and_velocity_centered_are_opposite():
    """Both estimators on the IDENTICAL toy infall configuration.

    Group-centered xi_Tu,1 must be negative (conventions.INFALL_DIPOLE_SIGN, the
    project's load-bearing sign gate); velocity-centered zeta_1 must be
    POSITIVE, since zeta_ell = (-1)**ell * xi_Tu,ell
    (conventions.nusser_multipole_sign) flips every odd multipole. If either sign
    is wrong, or the two magnitudes disagree, the orientation bookkeeping in
    `velocity_centered_shell_dipole` has a bug -- this is the test the whole
    velocity-centered estimator hangs on.
    """
    s_center, s_neighbors, u, v_vec = _infall_shell_with_velocities()
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    # --- group-centered xi_Tu: unchanged estimator, frozen orientation ------
    gc_result = shell_dipole(s_center, s_neighbors, shell_edges, weights=u)
    assert gc_result.dipole[0] < 0.0
    assert np.sign(gc_result.dipole[0]) == conventions.INFALL_DIPOLE_SIGN
    recovered_gc = 3.0 * gc_result.dipole[0] / gc_result.pair_count[0]
    assert recovered_gc == pytest.approx(V_INFALL, rel=0.1)

    # --- velocity-centered zeta: reversed orientation ------------------------
    # Pin the core-cut safety margin the toy geometry was built for: every
    # shell member must clear core_center_mask's cut
    # (sub_volume_radius - core_margin), or the "every member survives"
    # assumption below is untested luck rather than a guaranteed property of
    # the chosen constants.
    assert _VC_MAX_MEMBER_OBSERVER_DISTANCE <= _VC_SUB_VOLUME_RADIUS - _VC_CORE_MARGIN_DEFAULT

    vc_centers, vc_velocities, vc_tracers = _joint_gate_toy()
    vc_result = velocity_centered_shell_dipole(
        s_centers=vc_centers,
        v_centers=vc_velocities,
        s_tracers=vc_tracers,
        shell_edges=shell_edges,
        sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
    )

    assert vc_result.n_candidates == N_SHELL
    assert vc_result.n_centers == N_SHELL           # every member clears the core cut
    assert vc_result.pair_count.sum() == N_SHELL     # one tracer per center

    assert vc_result.dipole[0] > 0.0    # must FAIL LOUDLY if the orientation flips

    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
    recovered_vc = 3.0 * vc_result.dipole[0] / (y10_norm * vc_result.pair_count[0])
    assert recovered_vc == pytest.approx(-V_INFALL, rel=0.1)

    # Opposite signs -- zeta_ell = (-1)**ell * xi_Tu,ell made concrete.
    assert np.sign(vc_result.dipole[0]) == -np.sign(gc_result.dipole[0])
    assert abs(recovered_vc) == pytest.approx(abs(recovered_gc), rel=0.1)


# ---------------------------------------------------------------------------
# real_y10: pinned against scipy
# ---------------------------------------------------------------------------


class TestRealY10:
    """Pin `real_y10` against scipy's own normalized spherical harmonic."""

    @pytest.mark.parametrize(
        "theta", [0.0, np.pi / 6, np.pi / 4, np.pi / 2, 2 * np.pi / 3, np.pi]
    )
    def test_matches_scipy_sph_harm_y(self, theta):
        """real_y10(cos(theta)) == Re[Y_10(theta, phi)] for arbitrary phi.

        `scipy.special.sph_harm_y(n, m, theta, phi)` takes theta as the
        POLAR angle (from the z-axis) -- verified numerically against
        sqrt(3/4pi)*cos(theta) before writing this assertion (theta=0 gives
        sqrt(3/4pi), theta=pi gives -sqrt(3/4pi), matching the z-axis
        convention, not an azimuthal one). m=0 is phi-independent, so phi is
        arbitrary here; the imaginary part must vanish (Y_10 is real for
        m=0), checked explicitly rather than assumed.
        """
        phi = 1.234  # arbitrary; m=0 harmonics do not depend on it
        y = sph_harm_y(1, 0, theta, phi)

        np.testing.assert_allclose(np.imag(y), 0.0, atol=1e-12)
        np.testing.assert_allclose(real_y10(np.cos(theta)), np.real(y), atol=1e-12)

    def test_vectorized_over_many_angles(self):
        """The (N,) -> (N,) shape contract, checked in one call."""
        cos_theta = np.linspace(-1.0, 1.0, 17)
        out = real_y10(cos_theta)

        assert out.shape == (17,)
        np.testing.assert_allclose(out, np.sqrt(3.0 / (4.0 * np.pi)) * cos_theta)


# ---------------------------------------------------------------------------
# Correspondence: the estimator's amplitude vs. an independent construction
# ---------------------------------------------------------------------------


def test_amplitude_matches_independent_geometry_and_pins_the_reversal():
    """A_alpha,b pinned two ways on the joint-gate toy.

    1) An independently hand-assembled cos(theta), built through the same
       `pair_separation` / `unit_vector` / `mu_cosine` primitives the
       estimator uses internally, must match the estimator's own per-center
       amplitude exactly.
    2) Building mu with the FROZEN separation orientation
       (r = s_V - s_T, tracer -> center) instead of the REVERSED one the
       estimator actually uses (center -> tracer), while holding the SAME
       reference direction n_hat_V fixed, must give the exact negative --
       isolating the effect of the separation-vector reversal alone. (The
       companion substitution, n_hat_V vs n_hat_T, is exercised by the joint
       sign gate above, which uses the true n_hat_T-based group-centered
       estimator for its other half.)
    """
    vc_centers, vc_velocities, vc_tracers = _joint_gate_toy()
    s_center = vc_tracers[0]
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    result = velocity_centered_shell_dipole(
        s_centers=vc_centers,
        v_centers=vc_velocities,
        s_tracers=vc_tracers,
        shell_edges=shell_edges,
        sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
    )

    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    n_hat_V = unit_vector(vc_centers - observer)  # (N_SHELL, 3)

    # The reference direction is the FLOW-SIGNED axis, z_hat = sign(u) n_hat_V
    # (conventions.VELOCITY_AXIS_CONVENTION), assembled here from the raw dot
    # product rather than by calling geometry.radial_flow_axis, so this stays
    # an INDEPENDENT path rather than a re-run of the estimator's own helper.
    u_manual = np.einsum("ij,ij->i", vc_velocities, n_hat_V)  # (N_SHELL,)
    z_hat = np.sign(u_manual)[:, None] * n_hat_V              # (N_SHELL, 3)

    # The toy must actually exercise the flip, or this test would pass just as
    # well against the old unsigned axis: an infall shell has members falling
    # toward the observer AND away from it.
    assert np.any(u_manual > 0.0) and np.any(u_manual < 0.0)
    np.testing.assert_allclose(result.per_center_u, u_manual, atol=1e-10)
    np.testing.assert_allclose(result.per_center_speed, np.abs(u_manual), atol=1e-10)

    # (1) Independent hand-assembled path, through the SAME primitives the
    # estimator uses internally. pair_separation(s_center, vc_centers) with
    # the shared tracer in the "center" slot returns vc_centers - s_center,
    # i.e. tracer -> center (the FROZEN r = s_V - s_T orientation for this
    # pair); its negation is the REVERSED center -> tracer direction the
    # estimator actually uses.
    r_frozen_vec, _ = pair_separation(s_center, vc_centers)  # tracer -> center, (N_SHELL, 3)
    cos_theta = mu_cosine(unit_vector(-r_frozen_vec), z_hat)
    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
    a_manual = y10_norm * cos_theta

    np.testing.assert_allclose(result.per_center_amplitude[:, 0], a_manual, atol=1e-10)

    # (2) The frozen orientation itself (r = s_V - s_T, tracer -> center),
    # same reference direction z_hat, must be the exact negative.
    mu_frozen = mu_cosine(unit_vector(r_frozen_vec), z_hat)

    np.testing.assert_allclose(cos_theta, -mu_frozen, atol=1e-12)
    a_from_frozen = -y10_norm * mu_frozen
    np.testing.assert_allclose(result.per_center_amplitude[:, 0], a_from_frozen, atol=1e-10)

    # (3) The axis sign is what separates this from the unsigned-axis reading:
    # against the bare n_hat_V the amplitude differs by exactly sign(u), and
    # the per-center DIPOLE is invariant under that flip because the weight
    # |u| flips with it (velocity_centered_shell_dipole's "Axis sign" section).
    a_unsigned = y10_norm * mu_cosine(unit_vector(-r_frozen_vec), n_hat_V)
    np.testing.assert_allclose(
        result.per_center_amplitude[:, 0], np.sign(u_manual) * a_unsigned, atol=1e-10
    )
    np.testing.assert_allclose(
        result.per_center_dipole[:, 0], u_manual * a_unsigned, atol=1e-10
    )


# ---------------------------------------------------------------------------
# Flow-signed axis: z_hat = sign(u) n_hat_V, weight |u|
# ---------------------------------------------------------------------------


class TestFlowSignedAxis:
    """`conventions.VELOCITY_AXIS_CONVENTION`, pinned on real numbers.

    The axis is the direction the center is MOVING along its line of sight,
    not the line of sight itself, and the companion weight is the radial
    SPEED |u|. These pin the three consequences that are not visible in the
    dipole curve (which is invariant under the flip -- see
    `test_dipole_is_invariant_under_the_axis_flip`).
    """

    def test_inbound_center_is_axed_back_toward_the_observer(self):
        """A center approaching the observer must get z_hat = -n_hat_V.

        One center on the +x line of sight with a velocity pointing back at
        the observer. A tracer placed BETWEEN the observer and the center --
        i.e. ahead of the flow -- must give a POSITIVE amplitude, because it
        lies along +z_hat. Under the old unsigned axis the same tracer sat at
        cos_theta = -1 and gave a negative amplitude.
        """
        observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
        s_center = (observer + np.array([200.0, 0.0, 0.0]))[None, :]
        v_center = np.array([[-300.0, 0.0, 0.0]])  # inbound: u < 0

        tracer_ahead = s_center[0] - np.array([15.0, 0.0, 0.0])  # toward the observer
        result = velocity_centered_shell_dipole(
            s_centers=s_center,
            v_centers=v_center,
            s_tracers=tracer_ahead[None, :],
            shell_edges=np.array([10.0, 20.0]),
            sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
        )

        assert result.per_center_u[0] == pytest.approx(-300.0)
        assert result.per_center_speed[0] == pytest.approx(300.0)
        # cos_theta = +1: the tracer is directly ahead of the motion.
        assert result.per_center_amplitude[0, 0] == pytest.approx(np.sqrt(3.0 / (4.0 * np.pi)))
        assert result.dipole[0] > 0.0

    def test_monopole_is_the_speed_weighted_occupancy(self):
        """monopole == Sigma |u| N, hence non-negative -- not Sigma u N."""
        vc_centers, vc_velocities, vc_tracers = _joint_gate_toy()
        result = velocity_centered_shell_dipole(
            s_centers=vc_centers,
            v_centers=vc_velocities,
            s_tracers=vc_tracers,
            shell_edges=np.array([0.5 * R_SHELL, 1.5 * R_SHELL]),
            sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
        )

        np.testing.assert_allclose(
            result.monopole,
            (result.per_center_speed[:, None] * result.per_center_count).sum(axis=0),
        )
        np.testing.assert_allclose(result.per_center_speed, np.abs(result.per_center_u))
        assert np.all(result.monopole >= 0.0)
        # The signed sum it replaces is a genuinely different number here:
        # the toy's u values straddle zero, so it very nearly cancels.
        signed_monopole = (result.per_center_u[:, None] * result.per_center_count).sum(axis=0)
        assert abs(signed_monopole[0]) < result.monopole[0]

    def test_dipole_is_invariant_under_the_axis_flip(self):
        """|u| * A(z_hat) == u * A(n_hat_V), center by center.

        The identity behind `velocity_centered_shell_dipole`'s "Axis sign"
        section: the science curve does not move, which is what keeps the
        joint sign gate valid across the change. Checked against a hand-built
        unsigned-axis amplitude rather than asserted in prose.
        """
        vc_centers, vc_velocities, vc_tracers = _joint_gate_toy()
        result = velocity_centered_shell_dipole(
            s_centers=vc_centers,
            v_centers=vc_velocities,
            s_tracers=vc_tracers,
            shell_edges=np.array([0.5 * R_SHELL, 1.5 * R_SHELL]),
            sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
        )

        observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
        n_hat_V = unit_vector(vc_centers - observer)
        r_vec, _ = pair_separation(vc_tracers[0], vc_centers)  # tracer -> center
        y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
        a_unsigned = y10_norm * mu_cosine(unit_vector(-r_vec), n_hat_V)
        u_signed = np.einsum("ij,ij->i", vc_velocities, n_hat_V)

        np.testing.assert_allclose(
            result.per_center_dipole[:, 0], u_signed * a_unsigned, atol=1e-10
        )

    def test_transverse_velocity_leaves_the_axis_undefined_and_contributes_nothing(self):
        """u == 0 gives a zero axis and a zero weight -- no NaN, no crash.

        Documented degenerate case (`velocity_centered_shell_dipole`'s
        Notes): a center moving exactly across the line of sight has no
        radial motion to define an axis with, and no radial speed to weight
        by, so it contributes nothing rather than injecting a NaN.
        """
        observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
        s_center = (observer + np.array([200.0, 0.0, 0.0]))[None, :]
        v_center = np.array([[0.0, 400.0, 0.0]])  # exactly transverse: u == 0

        tracer = s_center[0] + np.array([0.0, 0.0, 15.0])
        result = velocity_centered_shell_dipole(
            s_centers=s_center,
            v_centers=v_center,
            s_tracers=tracer[None, :],
            shell_edges=np.array([10.0, 20.0]),
            sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
        )

        assert result.per_center_u[0] == pytest.approx(0.0)
        assert result.per_center_speed[0] == pytest.approx(0.0)
        assert result.pair_count[0] == 1.0          # occupancy is pure geometry
        assert result.per_center_amplitude[0, 0] == 0.0
        assert result.dipole[0] == 0.0
        assert result.monopole[0] == 0.0
        assert np.all(np.isfinite(result.dipole))


# ---------------------------------------------------------------------------
# Isotropic null: dipole vanishes, count does not
# ---------------------------------------------------------------------------

_ISO_N_TRACERS = 2048     # tracers per isotropic shell, mirrors N_SHELL
_ISO_R_CENTER = 300.0     # observer -> each isotropic center, h^-1 Mpc
_ISO_R_SHELL = 20.0       # isotropic shell radius around each center
_ISO_CONST_SPEED = 300.0  # coherent radial speed imposed at every center, km/s


def test_isotropic_shell_around_each_center_gives_vanishing_normalized_dipole():
    """Coherent u with a full isotropic shell of tracers cancels the dipole.

    Mirrors test_shell_dipole.py's group-centered isotropic null: several
    well-separated centers, each with its own near-uniform Fibonacci shell
    of tracers, and coherent u_alpha = CONST (every center's velocity is
    along its OWN line of sight, v_alpha = CONST * n_hat_V,alpha, so the
    projection is exactly CONST regardless of direction). Any residual
    dipole is then purely the small Fibonacci-sampling anisotropy the
    group-centered null already tolerates; pair_count is exact geometry and
    must match on the nose.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    directions = unit_vector(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    )
    n_centers = directions.shape[0]
    s_centers = observer + _ISO_R_CENTER * directions
    v_centers = _ISO_CONST_SPEED * directions  # v = CONST * n_hat_V by construction

    r_hat = _sphere_directions(_ISO_N_TRACERS)
    s_tracers = np.vstack([s_centers[k] + _ISO_R_SHELL * r_hat for k in range(n_centers)])

    shell_edges = np.array([0.5 * _ISO_R_SHELL, 1.5 * _ISO_R_SHELL])
    result = velocity_centered_shell_dipole(
        s_centers=s_centers,
        v_centers=v_centers,
        s_tracers=s_tracers,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,  # generous: no core cut in play here
    )

    assert result.n_centers == n_centers
    assert result.pair_count[0] == n_centers * _ISO_N_TRACERS  # exact geometry

    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
    normalized_dipole = 3.0 * result.dipole[0] / (y10_norm * result.pair_count[0])
    assert normalized_dipole == pytest.approx(0.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Reversed-orientation flip
# ---------------------------------------------------------------------------


def test_mirroring_tracers_flips_the_amplitude_sign():
    """Mirroring tracers through the center reverses A_alpha,b, not the count.

    s' = 2*s_alpha - s sends the center -> tracer direction n_hat_i to
    -n_hat_i for every tracer, hence Y_10(n_hat_i) -> -Y_10(n_hat_i); the
    RADIUS is unchanged (|s' - s_alpha| = |s - s_alpha|), so occupancy is
    untouched.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_center = observer + np.array([[250.0, 0.0, 0.0]])  # (1, 3), one center
    v_center = np.array([[123.0, -45.0, 67.0]])           # arbitrary nonzero velocity

    rng = np.random.default_rng(20260723)
    r_hat = unit_vector(rng.normal(size=(64, 3)))          # anisotropic tracer directions
    tracers = s_center[0] + 15.0 * r_hat                   # radius 15, inside [10, 20)

    shell_edges = np.array([10.0, 20.0])
    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )
    assert abs(result.per_center_amplitude[0, 0]) > 1e-6  # anisotropic: nonzero to begin with

    mirrored = 2.0 * s_center[0] - tracers
    result_mirrored = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=mirrored,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    np.testing.assert_allclose(result_mirrored.per_center_count, result.per_center_count)
    np.testing.assert_allclose(
        result_mirrored.per_center_amplitude, -result.per_center_amplitude, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Binning correctness
# ---------------------------------------------------------------------------


def _single_vc_center() -> tuple[np.ndarray, np.ndarray]:
    """One velocity-centered center with an arbitrary nonzero velocity."""
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_center = (observer + np.array([200.0, 0.0, 0.0]))[None, :]
    v_center = np.array([[100.0, 0.0, 0.0]])
    return s_center, v_center


def test_tracers_land_in_the_shell_their_radius_dictates():
    """Hand-placed tracers at known radii fall in the expected shells."""
    s_center, v_center = _single_vc_center()
    radii = np.array([5.0, 15.0, 25.0])
    tracers = s_center[0] + np.column_stack(
        (np.zeros_like(radii), radii, np.zeros_like(radii))
    )
    shell_edges = np.array([0.0, 10.0, 20.0, 30.0])

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    np.testing.assert_array_equal(result.pair_count, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(result.shell_centers, [5.0, 15.0, 25.0])


def test_tracers_outside_the_edge_range_are_excluded():
    """Below the innermost edge and beyond the outermost edge are both dropped."""
    s_center, v_center = _single_vc_center()
    radii = np.array([5.0, 15.0, 25.0, 35.0])  # 5 below edges[0]=10, 35 above 30
    tracers = s_center[0] + np.column_stack(
        (np.zeros_like(radii), radii, np.zeros_like(radii))
    )
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    np.testing.assert_array_equal(result.pair_count, [1.0, 1.0])
    assert result.pair_count.sum() == 2.0


def test_outermost_edge_is_included_in_the_last_shell():
    """A tracer sitting exactly on the outer edge is not silently dropped."""
    s_center, v_center = _single_vc_center()
    tracers = s_center[0] + np.array([[0.0, 30.0, 0.0]])  # r = 30 == edges[-1]
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    np.testing.assert_array_equal(result.pair_count, [0.0, 1.0])


# ---------------------------------------------------------------------------
# Empty shells / degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_shell_returns_zero_not_nan():
    s_center, v_center = _single_vc_center()
    tracers = s_center[0] + np.array([[0.0, 5.0, 0.0]])  # only in the first shell
    shell_edges = np.array([0.0, 10.0, 20.0])

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    np.testing.assert_array_equal(result.pair_count, [1.0, 0.0])
    assert result.dipole[1] == 0.0
    assert result.monopole[1] == 0.0
    assert np.all(np.isfinite(result.pair_count))
    assert np.all(np.isfinite(result.dipole))
    assert np.all(np.isfinite(result.monopole))


def test_zero_tracers_gives_all_zero_shells():
    s_center, v_center = _single_vc_center()
    tracers = np.empty((0, 3))
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    np.testing.assert_array_equal(result.pair_count, [0.0, 0.0])
    np.testing.assert_array_equal(result.dipole, [0.0, 0.0])
    np.testing.assert_array_equal(result.monopole, [0.0, 0.0])
    assert result.n_centers == 1


def test_zero_surviving_centers_is_a_valid_empty_result():
    """core_margin >= sub_volume_radius leaves nobody in the core -- a valid,
    NaN-free empty result, not an error."""
    s_center, v_center = _single_vc_center()
    tracers = s_center[0] + np.array([[0.0, 15.0, 0.0]])
    shell_edges = np.array([10.0, 20.0])

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=100.0,
        core_margin=100.0,  # >= sub_volume_radius: nobody survives
    )

    assert result.n_candidates == 1
    assert result.n_centers == 0
    np.testing.assert_array_equal(result.pair_count, [0.0])
    np.testing.assert_array_equal(result.dipole, [0.0])
    np.testing.assert_array_equal(result.monopole, [0.0])
    assert result.per_center_dipole.shape == (0, 1)
    assert result.per_center_amplitude.shape == (0, 1)
    assert result.per_center_count.shape == (0, 1)
    assert result.per_center_u.shape == (0,)
    assert np.all(np.isfinite(result.pair_count))
    assert np.all(np.isfinite(result.dipole))
    assert np.all(np.isfinite(result.monopole))


# ---------------------------------------------------------------------------
# Boundary test: the core cut suppresses the truncation bias
# ---------------------------------------------------------------------------

_BOUNDARY_R_SUB = 300.0        # h^-1 Mpc, mirrors notebook 04's R_SUB
_BOUNDARY_N_TRACERS = 60000    # generous sample: a stable outer-shell residual
_BOUNDARY_N_CANDIDATES = 4000  # candidate centers, subsampled from the tracers
_BOUNDARY_CONST_SPEED = 300.0  # coherent radial speed imposed at every center, km/s
_BOUNDARY_SHELL_STEP = 5.0     # h^-1 Mpc, mirrors the production script
_BOUNDARY_R_MAX = 60.0         # h^-1 Mpc, mirrors the production script
_BOUNDARY_SHELL_EDGES = np.arange(0.0, _BOUNDARY_R_MAX + _BOUNDARY_SHELL_STEP, _BOUNDARY_SHELL_STEP)
_BOUNDARY_SEED = 20260723


def _sample_ball(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """n points drawn UNIFORMLY inside a ball of `radius` around the observer.

    Inverse-CDF sampling on the radius (r ~ radius * U**(1/3), since the
    volume element grows as r^2 dr) combined with uniform-on-sphere
    directions gives an exactly uniform spatial density, not merely a
    plausible-looking cloud.
    """
    directions = unit_vector(rng.normal(size=(n, 3)))
    r = radius * rng.uniform(0.0, 1.0, size=n) ** (1.0 / 3.0)  # 1/3: inverse-CDF for r^2 dr
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    return observer + r[:, None] * directions


def test_core_margin_reduces_outer_shell_truncation_bias():
    """The core cut suppresses the boundary-truncation bias (`core_center_mask`'s
    docstring: the suspected mechanism behind the ~+13 km/s null offset in
    notebooks/04_first_mdpl2_run.ipynb).

    Tracers fill a ball of radius R_sub uniformly; candidate centers are a
    subsample of that SAME tracer population, each carrying a coherent
    radial velocity v_alpha = CONST * n_hat_V,alpha (so u_alpha = CONST for
    every center, regardless of position). The tracer FIELD itself has no
    imposed structure, so a center with a COMPLETE shell has an expected
    normalized dipole of zero; the only source of a nonzero stacked value is
    the geometric truncation bias from centers near the sub-volume boundary.
    Comparing the default core_margin (r_max, which removes boundary-affected
    centers) against core_margin=0 (which keeps all of them) isolates that
    bias in the OUTER shells, where the truncation lives.
    """
    rng = np.random.default_rng(_BOUNDARY_SEED)
    tracers = _sample_ball(rng, _BOUNDARY_N_TRACERS, _BOUNDARY_R_SUB)

    candidate_idx = rng.choice(_BOUNDARY_N_TRACERS, size=_BOUNDARY_N_CANDIDATES, replace=False)
    s_candidates = tracers[candidate_idx]
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    n_hat_V = unit_vector(s_candidates - observer)
    v_candidates = _BOUNDARY_CONST_SPEED * n_hat_V  # coherent radial signal

    n_bar = _BOUNDARY_N_TRACERS / ((4.0 / 3.0) * np.pi * _BOUNDARY_R_SUB**3)
    nbar_v_b = expected_shell_occupancy(n_bar, _BOUNDARY_SHELL_EDGES)
    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
    n_bins = len(_BOUNDARY_SHELL_EDGES) - 1

    def normalized_stack(core_margin):
        result = velocity_centered_shell_dipole(
            s_centers=s_candidates,
            v_centers=v_candidates,
            s_tracers=tracers,
            shell_edges=_BOUNDARY_SHELL_EDGES,
            sub_volume_radius=_BOUNDARY_R_SUB,
            core_margin=core_margin,
        )
        assert result.n_centers > 0
        return 3.0 * (result.dipole / result.n_centers) / (y10_norm * nbar_v_b)

    default_margin_stack = normalized_stack(None)  # core_margin -> r_max
    zero_margin_stack = normalized_stack(0.0)

    n_outer = n_bins // 2  # outer half of the shells, where the truncation bias lives
    residual_default = np.sum(np.abs(default_margin_stack[-n_outer:]))
    residual_zero = np.sum(np.abs(zero_margin_stack[-n_outer:]))

    assert residual_default < residual_zero


# ---------------------------------------------------------------------------
# Shuffle null
# ---------------------------------------------------------------------------


def test_shuffle_null_collapses_the_stacked_dipole():
    """Undoing the axis flip and permuting the SIGNED u kills the stacked
    dipole -- no second estimator pass needed, mirroring exactly what
    `dvcorr.pipeline.velocity_centered.normalize_result` builds.

    The undo step is load-bearing, not bookkeeping: the axis is
    z_hat = sign(u) n_hat_V, so `per_center_amplitude` already carries that
    sign. `test_permuting_the_speed_alone_is_not_a_null` below is the
    companion that shows what happens without it.
    """
    vc_centers, vc_velocities, vc_tracers = _joint_gate_toy()
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    result = velocity_centered_shell_dipole(
        s_centers=vc_centers,
        v_centers=vc_velocities,
        s_tracers=vc_tracers,
        shell_edges=shell_edges,
        sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
    )

    rng = np.random.default_rng(20260723)
    perm = rng.permutation(result.per_center_u.size)
    fixed_axis_amplitude = np.sign(result.per_center_u)[:, None] * result.per_center_amplitude
    shuffled_dipole = (result.per_center_u[perm][:, None] * fixed_axis_amplitude).sum(axis=0)

    shuffle_null_fraction = 0.2  # shuffled dipole must collapse below this fraction
    assert abs(shuffled_dipole[0]) < shuffle_null_fraction * abs(result.dipole[0])


def test_permuting_the_speed_alone_is_not_a_null():
    """The obvious one-line "null" -- permute the positive weight against the
    untouched amplitudes -- must NOT collapse, and this pins that.

    With the flow-signed axis the weight is |u_alpha| >= 0, so permuting it
    among centers leaves every A_alpha,b (and hence the alignment that
    produced the signal) exactly where it was; the recombined stack retains
    ~<|u|> * Sigma A, essentially the signal. This is the same trap
    `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null`
    documents for the velocity frame, and the reason `normalize_result`
    undoes the axis flip first. If this test ever starts collapsing, the
    axis sign has been dropped from `per_center_amplitude`.
    """
    vc_centers, vc_velocities, vc_tracers = _joint_gate_toy()
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    result = velocity_centered_shell_dipole(
        s_centers=vc_centers,
        v_centers=vc_velocities,
        s_tracers=vc_tracers,
        shell_edges=shell_edges,
        sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
    )

    rng = np.random.default_rng(20260723)
    perm = rng.permutation(result.per_center_speed.size)
    fake_null = (result.per_center_speed[perm][:, None] * result.per_center_amplitude).sum(axis=0)

    # Not a null at all: it keeps most of the signal rather than collapsing.
    retained_fraction = 0.5
    assert abs(fake_null[0]) > retained_fraction * abs(result.dipole[0])
    assert np.sign(fake_null[0]) == np.sign(result.dipole[0])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_finite_v_centers_raises(self):
        s_center, _ = _single_vc_center()
        bad_v = np.array([[np.nan, 0.0, 0.0]])
        with pytest.raises(ValueError):
            velocity_centered_shell_dipole(
                s_centers=s_center,
                v_centers=bad_v,
                s_tracers=s_center,
                shell_edges=np.array([0.0, 10.0]),
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
            )

    def test_mismatched_shapes_raises(self):
        s_center, _ = _single_vc_center()
        bad_v = np.zeros((2, 3))
        with pytest.raises(ValueError):
            velocity_centered_shell_dipole(
                s_centers=s_center,
                v_centers=bad_v,
                s_tracers=s_center,
                shell_edges=np.array([0.0, 10.0]),
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
            )

    def test_non_increasing_edges_raises(self):
        s_center, v_center = _single_vc_center()
        with pytest.raises(ValueError):
            velocity_centered_shell_dipole(
                s_centers=s_center,
                v_centers=v_center,
                s_tracers=s_center,
                shell_edges=np.array([20.0, 10.0]),
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
            )

    def test_edge_beyond_max_analysis_radius_raises(self):
        s_center, v_center = _single_vc_center()
        too_far = np.array([0.0, conventions.MAX_ANALYSIS_RADIUS + 1.0])
        with pytest.raises(ValueError):
            velocity_centered_shell_dipole(
                s_centers=s_center,
                v_centers=v_center,
                s_tracers=s_center,
                shell_edges=too_far,
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
            )

    def test_non_positive_sub_volume_radius_raises(self):
        s_center, v_center = _single_vc_center()
        with pytest.raises(ValueError):
            velocity_centered_shell_dipole(
                s_centers=s_center,
                v_centers=v_center,
                s_tracers=s_center,
                shell_edges=np.array([0.0, 10.0]),
                sub_volume_radius=0.0,
            )

    def test_negative_core_margin_raises(self):
        s_center, v_center = _single_vc_center()
        with pytest.raises(ValueError):
            velocity_centered_shell_dipole(
                s_centers=s_center,
                v_centers=v_center,
                s_tracers=s_center,
                shell_edges=np.array([0.0, 10.0]),
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
                core_margin=-1.0,
            )


class TestCenterStandardError:
    def test_single_center_returns_nan(self):
        values = np.array([[1.0, 2.0, 3.0]])
        sem = center_standard_error(values)

        assert sem.shape == (3,)
        assert np.all(np.isnan(sem))

    def test_matches_hand_checked_std_over_sqrt_n(self):
        values = np.array(
            [
                [1.0, 10.0],
                [3.0, 20.0],
                [5.0, 0.0],
            ]
        )
        sem = center_standard_error(values)
        expected = values.std(axis=0, ddof=1) / np.sqrt(3)

        np.testing.assert_allclose(sem, expected)

    def test_one_dimensional_input_raises_rather_than_guessing(self):
        """A 1-D (N_c,) array is genuinely ambiguous -- (N_c, 1) or (1, B)?

        `np.atleast_2d` would silently pick the (1, B) reading (one center,
        B shells), which for an actual "N_c centers, one shell" input quietly
        returns an all-NaN result (N_c=1 by that reading) instead of raising.
        This must be a ValueError, not a guess.
        """
        values_1d = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError):
            center_standard_error(values_1d)


class TestCoreCenterMask:
    def test_hand_checked_boundary(self):
        """Distance exactly at the cut is included (<=, not <)."""
        observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
        s_centers = observer + np.array(
            [[100.0, 0.0, 0.0], [150.0, 0.0, 0.0], [151.0, 0.0, 0.0]]
        )
        mask = core_center_mask(s_centers, sub_volume_radius=200.0, core_margin=50.0)

        np.testing.assert_array_equal(mask, [True, True, False])

    def test_shape_is_n_c(self):
        observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
        s_centers = observer + np.zeros((5, 3))
        mask = core_center_mask(s_centers, sub_volume_radius=100.0, core_margin=10.0)
        assert mask.shape == (5,)
