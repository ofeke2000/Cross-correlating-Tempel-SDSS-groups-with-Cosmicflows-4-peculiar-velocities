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

Immediately after the joint gate: the REDSHIFT-SPACE extension of the same
toy (`dvcorr.redshift_space.to_redshift_space`) -- the sign gate carried
through a redshift-space run of the identical estimator, and an
epsilon-continuity check that the redshift-space and real-space dipoles
converge as the imposed velocity shrinks to zero. See
`tests/test_redshift_space.py` for the transform-level tests (zero-velocity
limit, displacement direction, n_hat invariance, the through-observer flip
guard) and `tests/test_redshift_space_comparison.py` for the pipeline-level
ones (the shared center set, membership diagnostics).

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
from dvcorr.config import log_shell_edges
from dvcorr.geometry import mu_cosine, pair_separation, unit_vector
from dvcorr.pipeline.velocity_centered import (
    box_number_density,
    matched_gaussian_sample,
    normalize_stacked_dipole,
)
from dvcorr.redshift_space import to_redshift_space
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
# Redshift-space sign gate (dvcorr.redshift_space)
# ---------------------------------------------------------------------------
#
# Extends the SAME toy infall configuration used above: the velocity-object
# centers (`vc_centers`) and the single density tracer (`vc_tracers`) are run
# through `dvcorr.redshift_space.to_redshift_space` -- centers displaced by
# their own imposed infall velocity, the tracer left at rest (v = 0, so it is
# its own fixed point under the transform -- see
# `tests/test_redshift_space.py::test_zero_velocity_gives_identical_positions`;
# this toy never assigned the density tracer a velocity in the first place,
# so "at rest" is not a simplification of the toy, it is the toy) --
# then `velocity_centered_shell_dipole` (UNCHANGED) is run a second time on
# the displaced positions, with the SAME `v_centers`.
#
# Sign: this does NOT contradict hard rule 2 ("infall gives a negative
# dipole"). That rule is stated for the GROUP-CENTERED xi_Tu,1; this toy's
# velocity-centered zeta_1 is already POSITIVE for infall by construction
# (the joint sign gate above, and shell_dipole.py:166-173's "positive zeta_1
# is correct, negative is the bug"). Displacing into redshift space changes
# WHICH tracers fall in which shell (here, trivially, it changes which
# CENTERS' shells the one static tracer falls into), not the sign
# convention -- so the redshift-space run must ALSO be positive, and a
# negative sign here would be the same orientation bug the joint gate
# guards against, not a new redshift-space-specific one.


def test_redshift_space_infall_dipole_is_positive_and_suppressed_not_enhanced():
    """Redshift-space zeta_1 on the infall toy: sign preserved, amplitude SUPPRESSED.

    Measured on this exact toy (R_CENTER=200, R_SHELL=20, V_INFALL=-200,
    shell_edges=[10, 30]): pair counts are identical between the two runs
    (2048 both -- the single tracer is within range of every center in both
    spaces), and the redshift-space dipole is 0.957x the real-space one.
    This is a SUPPRESSION, not an enhancement: on a thin single shell of
    pure radial infall, redshift-space displacement squashes the shell
    members toward the (fixed) tracer along each center's own line of sight,
    reducing the realized Sigma Y_10 (the `2r/3R`-type geometric dilution
    that also drives the group-centered monopole leakage
    -- `dvcorr.geometry.mu_cosine`'s docstring). The Kaiser-boost intuition
    (RSD enhances clustering along the line of sight in the linear,
    broad-field regime) does not apply to this geometry, and this test
    makes no such claim -- it pins the SIGN and records the measured ratio,
    nothing about direction of amplitude change in general.
    """
    s_center, s_neighbors, u, v_vec = _infall_shell_with_velocities()
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])
    vc_centers, vc_velocities, vc_tracers = _joint_gate_toy()

    real_result = velocity_centered_shell_dipole(
        s_centers=vc_centers, v_centers=vc_velocities, s_tracers=vc_tracers,
        shell_edges=shell_edges, sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
    )

    center_transform = to_redshift_space(vc_centers, vc_velocities)
    tracer_transform = to_redshift_space(vc_tracers, np.zeros_like(vc_tracers))
    assert center_transform.n_dropped == 0
    assert tracer_transform.n_dropped == 0

    redshift_result = velocity_centered_shell_dipole(
        s_centers=center_transform.s_redshift, v_centers=vc_velocities,
        s_tracers=tracer_transform.s_redshift,
        shell_edges=shell_edges, sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
    )

    assert redshift_result.dipole[0] > 0.0   # sign gate: must FAIL LOUDLY if it flips
    assert redshift_result.pair_count.sum() == real_result.pair_count.sum() == N_SHELL

    ratio = redshift_result.dipole[0] / real_result.dipole[0]
    assert ratio == pytest.approx(0.957, abs=0.01)   # measured suppression, NOT an enhancement claim


def test_redshift_space_epsilon_continuity():
    """As the imposed velocity -> 0, the redshift-space run converges to the real-space one.

    Both dipoles are themselves O(eps) and vanish together as the velocity
    is scaled toward zero, so "the two runs agree to O(eps)" in an ABSOLUTE
    sense is trivially true and not the interesting claim. The real content
    is the RELATIVE deviation |1 - D_redshift/D_real|, which this project's
    author measured to be clean-linear on this exact toy:

        eps      ratio      (1-ratio)/eps
          1     0.957371     0.04263
        0.5     0.979389     0.04122
        0.1     0.995984     0.04016
       0.01     0.999601     0.03992
       1e-3     0.999960     0.03990

    i.e. C ~= 0.040 in |1 - ratio| <= C * eps; asserted below with headroom
    (0.06) rather than pinned to the measured constant.

    Why "no displacement but a non-trivial signal" cannot be constructed as
    a sharper null: displacement (`to_redshift_space`'s s_mag = r_mag +
    v_r/100) and the estimator's per-center weight (|u_alpha| = |v_r|,
    conventions.VELOCITY_AXIS_CONVENTION) are BOTH driven by the same v_r --
    there is no way to scale the imposed velocity toward zero (killing the
    displacement) while holding the dipole's own weight, and hence its
    signal, fixed.
    """
    s_center, s_neighbors, u, v_vec = _infall_shell_with_velocities()
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])
    vc_centers, _, vc_tracers = _joint_gate_toy()

    _EPSILON_SLOPE_HEADROOM = 0.06   # measured slope ~= 0.040; asserted with margin

    for eps in (1.0, 0.5, 0.1, 0.01, 1e-3):
        v_eps = eps * v_vec

        real_result = velocity_centered_shell_dipole(
            s_centers=vc_centers, v_centers=v_eps, s_tracers=vc_tracers,
            shell_edges=shell_edges, sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
        )
        center_transform = to_redshift_space(vc_centers, v_eps)
        tracer_transform = to_redshift_space(vc_tracers, np.zeros_like(vc_tracers))
        redshift_result = velocity_centered_shell_dipole(
            s_centers=center_transform.s_redshift, v_centers=v_eps,
            s_tracers=tracer_transform.s_redshift,
            shell_edges=shell_edges, sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
        )

        ratio = redshift_result.dipole[0] / real_result.dipole[0]
        relative_deviation = abs(1.0 - ratio)
        assert relative_deviation <= _EPSILON_SLOPE_HEADROOM * eps


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
# Non-uniform (geometric) shell_edges, end-to-end
# ---------------------------------------------------------------------------
#
# Nothing above exercises a non-uniform `shell_edges` through the estimator
# AND the normalization chain (dvcorr.pipeline.velocity_centered
# .normalize_stacked_dipole) together -- every test above uses either a
# single wide bin or evenly-spaced linear edges. Commit 2 flips the
# production default to spacing="log", min_radius=1.0, max_radius=64.0,
# n_bins=12 (edges 1, sqrt2, 2, 2 sqrt2, 4, ... 64), so these tests gate
# that change.

_GEOM_R_MIN = 1.0
_GEOM_R_MAX = 64.0
_GEOM_N_BINS = 12
_GEOM_SHELL_EDGES = log_shell_edges(_GEOM_R_MIN, _GEOM_R_MAX, _GEOM_N_BINS)
_GEOM_N_BAR = 3.7e-3  # arbitrary positive tracer number density, tracers per (h^-1 Mpc)^3


def test_geometric_edges_bin_tracers_correctly():
    """Hand-placed tracers at known radii fall into the correct, non-uniformly
    -spaced shells of `_GEOM_SHELL_EDGES`, and `result.shell_edges` round-trips
    the exact array passed in.
    """
    s_center, v_center = _single_vc_center()
    radii = np.array([1.2, 3.0, 50.0])  # in bins 0 ([1, sqrt2)), 3 ([2sqrt2, 4)), 11 ([32sqrt2, 64])
    tracers = s_center[0] + np.column_stack(
        (np.zeros_like(radii), radii, np.zeros_like(radii))
    )

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=_GEOM_SHELL_EDGES,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    expected_pair_count = np.zeros(_GEOM_N_BINS)
    expected_pair_count[[0, 3, 11]] = 1.0
    np.testing.assert_array_equal(result.pair_count, expected_pair_count)
    np.testing.assert_array_equal(result.shell_edges, _GEOM_SHELL_EDGES)


def test_box_number_density_is_the_box_mean_not_a_carved_density():
    """n_bar = n_halos / BOX_SIZE**3, and nothing else.

    Pins the normalization's denominator to the COSMIC MEAN of the box. The
    definition it replaced divided a carved count by the carve's own volume,
    which made every normalized zeta_hat_1 carry that one sphere's
    (1 + delta_sub) as an unknown flat factor. Two properties are asserted
    because both were false before:

      - the value is exactly the analytic box mean, and
      - it does not depend on `sub_volume_radius` in any way -- there is no
        radius argument left to mismatch a count against (the ~32% buffered
        -carve suppression `dvcorr.pipeline.redshift_space_comparison`
        documents was exactly that mismatch).
    """
    n_halos = 127_388_160  # the `full` MDPL2 catalog's row count

    n_bar = box_number_density(n_halos)

    assert n_bar == pytest.approx(n_halos / conventions.BOX_SIZE**3)
    # Linear in the count, and free of every other input.
    assert box_number_density(2 * n_halos) == pytest.approx(2.0 * n_bar)
    assert box_number_density(0) == 0.0


def test_expected_shell_occupancy_matches_analytic_volume_for_geometric_edges():
    """expected_shell_occupancy(n_bar, geom_edges) == n_bar * (4pi/3) * (r2**3 - r1**3)
    bin by bin -- pins that the denominator carries no hidden uniform-dr
    assumption that a non-uniform edge array would silently violate.
    """
    occupancy = expected_shell_occupancy(_GEOM_N_BAR, _GEOM_SHELL_EDGES)

    r1 = _GEOM_SHELL_EDGES[:-1]
    r2 = _GEOM_SHELL_EDGES[1:]
    expected = _GEOM_N_BAR * (4.0 / 3.0) * np.pi * (r2**3 - r1**3)

    np.testing.assert_allclose(occupancy, expected)


def test_empty_shell_under_geometric_edges_normalizes_to_zero_not_nan():
    """An inner geometric bin with zero tracers must come through
    `normalize_stacked_dipole` as `zeta_hat == 0.0` EXACTLY, with finite
    (non-NaN) SEM and finite `monopole_norm`.

    This is the specific failure mode log binning makes likely: the
    innermost bins are genuinely near-empty in a real MDPL2 run, and a 0/0
    against the (non-uniform) volume denominator would surface as a NaN
    silently propagating into the plotted curve, not a crash.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_centers = observer + np.array([[200.0, 0.0, 0.0], [0.0, 200.0, 0.0]])
    v_centers = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])

    # Tracers only at radius 10 (bin 6: [8, 8*sqrt2)) around each center --
    # nothing lands in bin 0 ([1, sqrt2)) for either center.
    tracers = np.vstack(
        [
            s_centers[0] + np.array([0.0, 10.0, 0.0]),
            s_centers[1] + np.array([10.0, 0.0, 0.0]),
        ]
    )

    result = velocity_centered_shell_dipole(
        s_centers=s_centers,
        v_centers=v_centers,
        s_tracers=tracers,
        shell_edges=_GEOM_SHELL_EDGES,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )
    assert result.n_centers == 2
    assert result.pair_count[0] == 0.0  # bin 0 genuinely empty

    normalized = normalize_stacked_dipole(
        shell_edges=result.shell_edges,
        dipole=result.dipole,
        monopole=result.monopole,
        per_center_dipole=result.per_center_dipole,
        null_dipole=result.dipole,               # any finite null suffices here
        null_per_center_dipole=result.per_center_dipole,
        n_centers=result.n_centers,
        n_bar=_GEOM_N_BAR,
    )

    assert normalized.zeta_hat[0] == 0.0
    assert np.isfinite(normalized.sem[0])
    assert np.isfinite(normalized.monopole_norm[0])


# ---------------------------------------------------------------------------
# The r = 0 self-pair, under a zero innermost edge
# ---------------------------------------------------------------------------
# `ShellConfig.include_zero_bin` prepends [0, min_radius) to the log ladder,
# so `shell_edges[0] == 0` and the innermost edge no longer excludes the
# coincident pair. The pipeline subsamples its centers FROM the tracer array,
# so that pair is not hypothetical: it is exactly one per center, and in the
# production binning it would outnumber the real pairs in [0, 1) by ~60x.
# These two tests pin the explicit `r_mag > 0` term that replaces the
# incidental protection the positive innermost edge used to provide.

_ZERO_BIN_EDGES = log_shell_edges(
    _GEOM_R_MIN, _GEOM_R_MAX, _GEOM_N_BINS, include_zero_bin=True
)


def test_self_pair_is_excluded_under_a_zero_innermost_edge():
    """A center that is ALSO one of its own tracers contributes nothing to the
    [0, min_radius) bin, while a genuine tracer at a small non-zero radius in
    the same bin is still counted.

    Both halves matter. Without the first, `pair_count[0]` and `monopole[0]`
    carry a pure self-correlation. Without the second, an over-broad guard
    (say `r_mag >= min_radius`) would pass the first while quietly emptying
    the bin the zero-bin option exists to fill.
    """
    s_center, v_center = _single_vc_center()
    genuine_radius = 0.5 * _GEOM_R_MIN  # inside [0, 1), comfortably clear of 0
    tracers = np.vstack(
        [
            s_center[0],  # the center itself: r = 0 exactly
            s_center[0] + np.array([0.0, genuine_radius, 0.0]),
        ]
    )

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=_ZERO_BIN_EDGES,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    assert result.shell_edges[0] == 0.0
    assert result.pair_count[0] == 1.0        # the genuine tracer only
    assert result.pair_count.sum() == 1.0     # and nothing leaked elsewhere


def test_self_pair_exclusion_is_a_no_op_when_the_innermost_edge_is_positive():
    """The `r_mag > 0` term changes NOTHING for a binning that starts above
    zero -- there, the self-pair was already below the innermost edge. This is
    what makes the term safe to apply unconditionally at every binning site
    rather than conditionally on `shell_edges[0] == 0`.
    """
    s_center, v_center = _single_vc_center()
    radii = np.array([1.2, 3.0, 50.0])
    offsets = np.column_stack((np.zeros_like(radii), radii, np.zeros_like(radii)))
    without_self = s_center[0] + offsets
    with_self = np.vstack([s_center[0], without_self])

    def _pair_count(tracers: np.ndarray) -> np.ndarray:
        return velocity_centered_shell_dipole(
            s_centers=s_center,
            v_centers=v_center,
            s_tracers=tracers,
            shell_edges=_GEOM_SHELL_EDGES,  # starts at 1.0, not 0.0
            sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
        ).pair_count

    np.testing.assert_array_equal(_pair_count(with_self), _pair_count(without_self))


# Edge sets for the binning-invariance gate below: _GATE_GEOM_EDGES tiles the
# EXACT SAME [10, 30] interval as the joint sign gate's own single wide bin
# (_GATE_LINEAR_EDGES == [0.5*R_SHELL, 1.5*R_SHELL]) into 4 geometric
# sub-shells, sharing both boundaries with it.
_GATE_LINEAR_EDGES = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])
_GATE_GEOM_N_BINS = 4
_GATE_GEOM_EDGES = log_shell_edges(
    _GATE_LINEAR_EDGES[0], _GATE_LINEAR_EDGES[-1], _GATE_GEOM_N_BINS
)
_GATE_N_BAR = 1.0e-4  # arbitrary; cancels out of the volume-weighted-average identity below

# Radii for the STRENGTHENED binning-invariance gate below: chosen to land
# one in each of _GATE_GEOM_EDGES's four sub-bins
# ([10, 13.16), [13.16, 17.32), [17.32, 22.79), [22.79, 30]), with 25 and 29
# deliberately sharing the outermost one -- occupying every sub-bin with a
# DIFFERENT dipole contribution is the whole point (see
# test_binning_invariance_on_a_coherent_infall_field's docstring for why the
# single-radius delta-function toy this generalizes was too weak a gate).
_GATE_MULTI_RADII = (11.0, 14.0, 18.0, 25.0, 29.0)
_GATE_N_PER_RADIUS = 512  # centers per radius shell -- enough for near-isotropy, small enough to stay fast
_GATE_N_TOTAL = len(_GATE_MULTI_RADII) * _GATE_N_PER_RADIUS


def _multi_radius_gate_toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generalizes `_joint_gate_toy` to several radii, so several
    `_GATE_GEOM_EDGES` sub-bins are occupied, not just one.

    `_joint_gate_toy` places every "center" on a SINGLE shell at r =
    R_SHELL -- a delta function in radius, so exactly one of
    `_GATE_GEOM_EDGES`'s four sub-bins is ever occupied and the
    volume-weighted-tiling identity below collapses to a single term. Here,
    one Fibonacci-sphere shell of centers
    (`tests.test_geometry._sphere_directions`, `_GATE_N_PER_RADIUS` centers
    per shell) is built around the SAME `s_center` for EACH radius in
    `_GATE_MULTI_RADII`, using the identical recipe
    `tests.test_geometry._infall_shell_with_velocities` uses for its one
    shell -- same imposed infall speed `V_INFALL`, same direction
    convention -- just repeated once per radius.

    Every center, at every radius, sees the SAME single tracer, `s_center`
    itself, at EXACTLY its own shell's radius (the vector from a shell-k
    member at `s_center + radius_k * r_hat` back to `s_center` has magnitude
    `radius_k` by construction): there is exactly one (center, tracer) pair
    per center and no cross-shell contamination is possible, so this is a
    genuine multi-radius generalization of the delta-function toy's
    cross-contamination-free construction, not a departure from it.

    Returns
    -------
    vc_centers : ndarray, shape (_GATE_N_TOTAL, 3)
    vc_velocities : ndarray, shape (_GATE_N_TOTAL, 3)
    vc_tracers : ndarray, shape (1, 3)
        A single tracer, `s_center`.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_center = observer + np.array([R_CENTER, 0.0, 0.0])

    centers, velocities = [], []
    for radius in _GATE_MULTI_RADII:
        r_hat = _sphere_directions(_GATE_N_PER_RADIUS)
        centers.append(s_center + radius * r_hat)
        velocities.append(V_INFALL * r_hat)

    vc_centers = np.vstack(centers)
    vc_velocities = np.vstack(velocities)
    vc_tracers = s_center[None, :]
    return vc_centers, vc_velocities, vc_tracers


def _assert_binning_invariance(
    vc_centers: np.ndarray,
    vc_velocities: np.ndarray,
    vc_tracers: np.ndarray,
    expected_n_total: int,
    expected_n_occupied_geom_bins: int,
) -> None:
    """Shared body of the binning-invariance gate.

    Parametrized by the toy handed in and how many of `_GATE_GEOM_EDGES`'s
    four sub-bins it occupies -- the only two things
    `test_binning_invariance_on_a_coherent_infall_field` (multi-radius) and
    `test_binning_invariance_delta_function_toy` (single-radius) differ on.

    Runs `velocity_centered_shell_dipole` under both `_GATE_LINEAR_EDGES`
    (one wide bin) and `_GATE_GEOM_EDGES` (the same [10, 30] interval tiled
    into 4 geometric sub-bins), then checks that `normalize_stacked_dipole`
    divides the raw stacked dipole by `n_bar * V_b` (a shell VOLUME)
    consistently: the raw dipole is additive across any tiling of bins (same
    set of pairs, differently partitioned), so algebraically

        zeta_hat_linear == Sum_k zeta_hat_geom_k * (V_geom_k / V_linear)

    i.e. the normalized dipole in the wide linear bin must equal the
    VOLUME-WEIGHTED AVERAGE of the normalized dipole across the geometric
    sub-bins that tile it. This holds regardless of the underlying tracer
    field -- it only requires `expected_shell_occupancy`'s volume to scale
    correctly with non-uniform bin width, which is exactly what commit 2's
    log-spacing default depends on. Verified numerically to agree with the
    estimator's actual output to ~1e-13 relative before the `rel=1e-9`
    tolerance below was chosen. If this fails, DO NOT loosen the tolerance
    -- it is a real finding about the normalization arithmetic under
    non-uniform edges, not test noise.

    Sign: read `dvcorr.estimators.shell_dipole`'s module docstring before
    asserting -- CLAUDE.md hard rule 2 ("infall gives a negative dipole") is
    stated for the GROUP-centered xi_Tu,1. This is the VELOCITY-centered
    zeta_1, related by zeta_ell = (-1)**ell * xi_Tu,ell
    (conventions.nusser_multipole_sign), so it is POSITIVE for infall; do
    not "fix" it if it comes out positive.
    """
    result_linear = velocity_centered_shell_dipole(
        s_centers=vc_centers,
        v_centers=vc_velocities,
        s_tracers=vc_tracers,
        shell_edges=_GATE_LINEAR_EDGES,
        sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
    )
    result_geom = velocity_centered_shell_dipole(
        s_centers=vc_centers,
        v_centers=vc_velocities,
        s_tracers=vc_tracers,
        shell_edges=_GATE_GEOM_EDGES,
        sub_volume_radius=_VC_SUB_VOLUME_RADIUS,
    )
    assert result_linear.n_centers == result_geom.n_centers == expected_n_total

    normalized_linear = normalize_stacked_dipole(
        shell_edges=result_linear.shell_edges,
        dipole=result_linear.dipole,
        monopole=result_linear.monopole,
        per_center_dipole=result_linear.per_center_dipole,
        null_dipole=result_linear.dipole,
        null_per_center_dipole=result_linear.per_center_dipole,
        n_centers=result_linear.n_centers,
        n_bar=_GATE_N_BAR,
    )
    normalized_geom = normalize_stacked_dipole(
        shell_edges=result_geom.shell_edges,
        dipole=result_geom.dipole,
        monopole=result_geom.monopole,
        per_center_dipole=result_geom.per_center_dipole,
        null_dipole=result_geom.dipole,
        null_per_center_dipole=result_geom.per_center_dipole,
        n_centers=result_geom.n_centers,
        n_bar=_GATE_N_BAR,
    )

    assert np.sum(result_geom.pair_count > 0.0) == expected_n_occupied_geom_bins
    assert result_geom.pair_count.sum() == result_linear.pair_count.sum() == expected_n_total

    r1 = _GATE_GEOM_EDGES[:-1]
    r2 = _GATE_GEOM_EDGES[1:]
    v_geom = (4.0 / 3.0) * np.pi * (r2**3 - r1**3)  # 4pi/3: sphere volume, pure math
    v_linear = (
        (4.0 / 3.0) * np.pi * (_GATE_LINEAR_EDGES[-1] ** 3 - _GATE_LINEAR_EDGES[0] ** 3)
    )
    np.testing.assert_allclose(v_geom.sum(), v_linear)  # the two edge sets truly tile

    volume_weighted_average = np.sum(normalized_geom.zeta_hat * v_geom) / v_linear
    assert normalized_linear.zeta_hat[0] == pytest.approx(volume_weighted_average, rel=1e-9)

    # Sign gate: velocity-centered zeta_1 is POSITIVE for infall.
    assert normalized_linear.zeta_hat[0] > 0.0
    assert np.sign(normalized_linear.zeta_hat[0]) == -conventions.INFALL_DIPOLE_SIGN


def test_binning_invariance_on_a_coherent_infall_field():
    """The gate: non-uniform (geometric) shell_edges must preserve the whole
    normalization chain, not just the raw estimator.

    Uses `_multi_radius_gate_toy`: `_GATE_MULTI_RADII` = (11, 14, 18, 25, 29)
    h^-1 Mpc, chosen to land in ALL FOUR of `_GATE_GEOM_EDGES`'s sub-bins
    (25 and 29 share the outermost one). This is a STRENGTHENED version of
    an earlier form of this test that placed every center on a single shell
    at r = R_SHELL = 20 -- a delta function in radius occupying only ONE of
    the four sub-bins, which left the volume-weighted-tiling identity
    resting on a single term: corrupting the volume of any of the three
    EMPTY sub-bins by even a factor of a million left the identity
    unaffected (relerr ~1e-16), so a per-bin volume bug in an empty bin
    would have passed silently -- only corrupting the one occupied bin
    failed. With every sub-bin genuinely occupied and contributing a
    DIFFERENT dipole amount, a volume error in ANY sub-bin now breaks the
    identity (self-checked by temporarily corrupting one non-innermost
    occupied bin's volume by 5% and confirming this test fails, then
    reverting). See `test_binning_invariance_delta_function_toy` for the
    single-bin composition case this generalizes, kept as a separate,
    simpler check, and `_assert_binning_invariance`'s docstring for the
    volume-weighted-average identity both tests share.
    """
    vc_centers, vc_velocities, vc_tracers = _multi_radius_gate_toy()
    _assert_binning_invariance(
        vc_centers, vc_velocities, vc_tracers,
        expected_n_total=_GATE_N_TOTAL,
        expected_n_occupied_geom_bins=_GATE_GEOM_N_BINS,
    )


def test_binning_invariance_delta_function_toy():
    """The single-occupied-bin case `test_binning_invariance_on_a_coherent_infall_field`
    generalizes: every center sees the SAME single tracer at EXACTLY
    r = R_SHELL = 20 (`_joint_gate_toy`, a delta function in radius), so
    only one of `_GATE_GEOM_EDGES`'s four sub-bins is ever occupied and the
    other three tile empty volume. Kept as a separate, simpler check that
    the estimator -> `normalize_stacked_dipole` chain composes correctly
    even in this degenerate case; deliberately NOT the only version of this
    gate any more (see the other test's docstring for why an
    all-empty-but-one gate under-tests the volume arithmetic).
    """
    vc_centers, vc_velocities, vc_tracers = _joint_gate_toy()
    _assert_binning_invariance(
        vc_centers, vc_velocities, vc_tracers,
        expected_n_total=N_SHELL,
        expected_n_occupied_geom_bins=1,
    )


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
_BOUNDARY_SHELL_STEP = 5.0     # h^-1 Mpc, deliberately NOT the production default (see below)
_BOUNDARY_R_MAX = 60.0         # h^-1 Mpc, deliberately NOT the production default (see below)
# Built by hand via np.arange, not RunConfig/ShellConfig -- so this test is
# unaffected by commit 2's production default flipping to spacing="log", and
# is deliberately kept LINEAR here rather than switched to match it. `n_outer
# = n_bins // 2` below only means "outer half of the shells IN RADIUS" for a
# LINEAR binning; under log edges the same slice would instead mean "outer
# sqrt(r_max/r_min) factor", quietly changing what this test measures without
# any assertion or edge-count changing to reveal it.
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

    # outer half of the shells, where the truncation bias lives -- meaningful
    # as "outer half IN RADIUS" only because _BOUNDARY_SHELL_EDGES above is
    # deliberately LINEAR; under log edges this slice would instead mean
    # "outer sqrt(r_max/r_min) factor", silently changing what this test
    # measures (see the comment on _BOUNDARY_SHELL_EDGES).
    n_outer = n_bins // 2
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
# Matched-Gaussian null
# ---------------------------------------------------------------------------


def test_matched_gaussian_sample_matches_mean_and_spread():
    """The distributional contract `matched_gaussian_sample` exists to hold:
    the draw reproduces the sample's mean and its ddof=1 spread, per column.

    This is what makes the Gaussian null comparable to the shuffle null at
    all -- a permutation preserves those two moments EXACTLY, so if the draw
    did not match them the gap between the two null curves would be a
    first/second-moment difference rather than the distribution-SHAPE
    difference it is advertised to isolate. Tolerances are loose because a
    finite draw scatters around its parameters; they are tight enough to
    catch a wrong convention (a zeroed mean, or ddof=0 vs ddof=1 at small N).
    """
    rng = np.random.default_rng(20260804)
    n_centers = 4000
    # Deliberately NOT Gaussian and NOT zero-mean: an exponential tail plus an
    # offset, i.e. the shape the real u distribution has and the draw must
    # nonetheless match in its first two moments.
    sample = rng.exponential(scale=180.0, size=n_centers) - 60.0

    drawn = matched_gaussian_sample(sample, seed=11)

    assert drawn.shape == sample.shape
    mean_tolerance = 0.05 * sample.std(ddof=1)  # 5% of a sigma
    spread_tolerance = 0.05  # fractional
    assert abs(drawn.mean() - sample.mean()) < mean_tolerance
    assert abs(drawn.std(ddof=1) / sample.std(ddof=1) - 1.0) < spread_tolerance


def test_matched_gaussian_sample_matches_each_column_of_a_vector_sample():
    """The (N, 3) path -- the velocity frame's
    (`dvcorr.pipeline.velocity_frame_comparison.run_gaussian_velocity_null`)
    -- matches per COMPONENT, so the sample's bulk-flow VECTOR survives.

    A pooled single sigma, or a zeroed mean, would make the null isotropic
    and delete that bulk motion; this pins the documented choice, since an
    isotropic null already exists (`run_random_axis_null`) and the whole
    point of this one is to be the non-isotropic counterpart.
    """
    rng = np.random.default_rng(20260804)
    n_centers = 4000
    bulk_flow = np.array([250.0, -80.0, 0.0])
    per_component_sigma = np.array([300.0, 450.0, 200.0])
    sample = bulk_flow + per_component_sigma * rng.normal(size=(n_centers, 3))

    drawn = matched_gaussian_sample(sample, seed=12)

    assert drawn.shape == sample.shape
    mean_tolerance = 0.05 * sample.std(axis=0, ddof=1)
    spread_tolerance = 0.05
    assert np.all(np.abs(drawn.mean(axis=0) - sample.mean(axis=0)) < mean_tolerance)
    assert np.all(
        np.abs(drawn.std(axis=0, ddof=1) / sample.std(axis=0, ddof=1) - 1.0) < spread_tolerance
    )


def test_matched_gaussian_sample_is_seed_deterministic_and_seed_sensitive():
    """Same seed -> same draw (a run is reproducible); different seed ->
    different draw (the seed ladder in `RunConfig.gaussian_null_seed` buys
    genuinely independent nulls, not one stream viewed twice).
    """
    sample = np.random.default_rng(20260804).normal(size=500)

    assert np.array_equal(
        matched_gaussian_sample(sample, seed=46), matched_gaussian_sample(sample, seed=46)
    )
    assert not np.array_equal(
        matched_gaussian_sample(sample, seed=46), matched_gaussian_sample(sample, seed=47)
    )


def test_matched_gaussian_sample_degenerates_to_nan_below_two_rows():
    """N < 2 -> all-NaN, mirroring `center_standard_error`'s own convention
    rather than raising: a single-center run is supported everywhere else in
    this pipeline and must degrade to a missing null curve, not to an
    exception thrown from inside `normalize_result`.
    """
    assert np.all(np.isnan(matched_gaussian_sample(np.array([3.0]), seed=1)))
    assert np.all(np.isnan(matched_gaussian_sample(np.zeros((1, 3)), seed=1)))


def test_gaussian_null_collapses_the_stacked_dipole():
    """The Gaussian null is a NULL: recombining a matched draw against the
    fixed-axis amplitude collapses the stack, exactly as the shuffle does.

    Same toy and same threshold as
    `test_shuffle_null_collapses_the_stacked_dipole`, deliberately -- the two
    nulls are meant to be interchangeable as floors and only to differ in
    the distribution they assume, so a failure here means the draw has
    stopped decoupling the velocity from the geometry.
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

    fixed_axis_amplitude = np.sign(result.per_center_u)[:, None] * result.per_center_amplitude
    u_gaussian = matched_gaussian_sample(result.per_center_u, seed=20260804)
    gaussian_dipole = (u_gaussian[:, None] * fixed_axis_amplitude).sum(axis=0)

    gaussian_null_fraction = 0.2  # same bar as the shuffle null's
    assert abs(gaussian_dipole[0]) < gaussian_null_fraction * abs(result.dipole[0])


def test_every_null_in_the_three_pipelines_gets_its_own_seed():
    """The seed ladder documented on `RunConfig.gaussian_null_seed` is
    actually distinct, end to end.

    Six random streams now exist across the three pipelines -- the center
    draw plus five nulls -- and every docstring that mentions one of them
    claims it never silently reuses another. That claim is one careless
    default away from being false, and a collision would not raise: it would
    quietly make two "independent" null curves correlated, which is exactly
    the kind of thing a reader would take at face value on a figure.
    """
    # Imported here rather than at module scope: this is the only test in
    # this file that needs the two downstream pipelines, and importing them
    # at the top would make this module's import graph misrepresent what it
    # is a test of.
    from dvcorr.pipeline.redshift_space_comparison import RedshiftSpaceRunConfig
    from dvcorr.pipeline.velocity_frame_comparison import ComparisonRunConfig

    comparison_cfg = ComparisonRunConfig()
    redshift_cfg = RedshiftSpaceRunConfig()

    seeds = [
        comparison_cfg.seed,                              # candidate centers
        comparison_cfg.shuffle_seed,                      # obs frame, u-shuffle
        comparison_cfg.axis_null_seed,                    # vel frame, random-axis
        comparison_cfg.gaussian_null_seed,                # obs frame, gaussian
        comparison_cfg.velocity_gaussian_null_seed,       # vel frame, gaussian
        redshift_cfg.redshift_shuffle_seed,               # redshift run, u-shuffle
        redshift_cfg.redshift_gaussian_null_seed,         # redshift run, gaussian
    ]

    assert len(set(seeds)) == len(seeds), f"seed collision in the null ladder: {seeds}"


def test_normalize_stacked_dipole_rejects_half_a_gaussian_null():
    """Passing one of the two Gaussian-null arrays without the other raises,
    rather than producing a curve with no band (or a band with no curve).
    """
    shell_edges = np.array([0.0, 10.0])
    zeros_b = np.zeros(1)
    zeros_cb = np.zeros((2, 1))
    common = dict(
        shell_edges=shell_edges,
        dipole=zeros_b,
        monopole=zeros_b,
        per_center_dipole=zeros_cb,
        null_dipole=zeros_b,
        null_per_center_dipole=zeros_cb,
        n_centers=2,
        n_bar=1.0,
    )

    with pytest.raises(ValueError):
        normalize_stacked_dipole(**common, gaussian_null_dipole=zeros_b)
    with pytest.raises(ValueError):
        normalize_stacked_dipole(**common, gaussian_null_per_center_dipole=zeros_cb)

    # Neither given: NaN curve, not an exception -- the documented
    # arithmetic-only call pattern.
    normalized = normalize_stacked_dipole(**common)
    assert np.all(np.isnan(normalized.zeta_hat_gaussian))
    assert np.all(np.isnan(normalized.sem_gaussian))


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


# ---------------------------------------------------------------------------
# Candidate draw: order independence
# ---------------------------------------------------------------------------


class TestDrawIsOrderIndependent:
    """`draw_candidates_from_arrays` selects HALOS, not file rows.

    Synthetic and catalog-free, so this holds regardless of which files are on
    disk; `tests/test_catalog_equivalence.py` asserts the same property against
    the two real catalogs. Both matter: this one pins the mechanism, that one
    pins that the mechanism actually reconciles the files.
    """

    @staticmethod
    def _population(n: int = 4000, seed: int = 11):
        rng = np.random.default_rng(seed)
        observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
        pos = observer + rng.uniform(-100.0, 100.0, size=(n, 3))
        vel = rng.normal(scale=300.0, size=(n, 3))
        mvir = conventions.PARTICLE_MASS * rng.integers(2, 10_000, size=n).astype(float)
        is_distinct = rng.random(n) > 0.12
        return pos, vel, mvir, is_distinct

    def test_shuffled_rows_give_the_same_candidates(self) -> None:
        from dvcorr.pipeline.velocity_centered import RunConfig, draw_candidates_from_arrays

        pos, vel, mvir, is_distinct = self._population()
        cfg = RunConfig(n_candidate_centers=200)

        first = draw_candidates_from_arrays(cfg, pos, vel, mvir, is_distinct)

        shuffle = np.random.default_rng(99).permutation(pos.shape[0])
        second = draw_candidates_from_arrays(
            cfg, pos[shuffle], vel[shuffle], mvir[shuffle], is_distinct[shuffle]
        )

        # Identical halos, in identical order -- so every per-row array agrees.
        np.testing.assert_array_equal(first.s, second.s)
        np.testing.assert_array_equal(first.v, second.v)
        np.testing.assert_array_equal(first.mvir, second.mvir)
        np.testing.assert_array_equal(first.is_distinct, second.is_distinct)

    def test_draw_index_still_indexes_the_callers_arrays(self) -> None:
        """`draw_index` refers to INPUT rows, not to the canonical ordering --
        otherwise a caller using it against its own arrays would silently get
        the wrong halos."""
        from dvcorr.pipeline.velocity_centered import RunConfig, draw_candidates_from_arrays

        pos, vel, mvir, is_distinct = self._population()
        cfg = RunConfig(n_candidate_centers=200)
        drawn = draw_candidates_from_arrays(cfg, pos, vel, mvir, is_distinct)

        np.testing.assert_array_equal(pos[drawn.draw_index], drawn.s)
        np.testing.assert_array_equal(mvir[drawn.draw_index], drawn.mvir)

    def test_a_box_face_halo_matches_its_periodic_image(self) -> None:
        """Coordinates at BOX_SIZE and at 0.0 are the same point, so the two
        representations must sort together -- the case the two real catalogs
        actually disagree on."""
        from dvcorr.pipeline.velocity_centered import RunConfig, draw_candidates_from_arrays

        pos, vel, mvir, is_distinct = self._population()
        pos = pos.copy()
        pos[0] = [conventions.BOX_SIZE, 10.0, 20.0]
        wrapped = pos.copy()
        wrapped[0] = [0.0, 10.0, 20.0]

        cfg = RunConfig(n_candidate_centers=200)
        a = draw_candidates_from_arrays(cfg, pos, vel, mvir, is_distinct)
        b = draw_candidates_from_arrays(cfg, wrapped, vel, mvir, is_distinct)
        np.testing.assert_array_equal(a.mvir, b.mvir)
