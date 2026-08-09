"""
test_velocity_frame_dipole.py
--------------------------------
Unit tests for the observer-free velocity-frame dipole
(`dvcorr.estimators.velocity_frame_dipole.velocity_frame_shell_dipole`) and
its pipeline (`dvcorr.pipeline.velocity_frame_comparison`).

Mirrors tests/test_velocity_centered_dipole.py in tone and structure. The
centerpiece is the sign gate: coherent infall must give a POSITIVE
velocity-frame dipole, agreeing in sign with the observer-frame zeta_1 on the
identical toy configuration -- the two frames must never disagree in sign,
since they coincide exactly in the pure-radial-flow limit (see
`dvcorr.estimators.velocity_frame_dipole`'s module docstring). The rest pin:
the frame-agreement limit for purely radial flow (both outbound and
inbound); the pipeline's degenerate-center handling
(`select_shared_centers`'s speed floor, and the estimator's own zero-speed
`ValueError`); that `run_both_frames` hands the two estimators the identical
center set, row-aligned; the null tests for both frames (an uncorrelated
density field, and the pipeline's random-axis null on a genuine signal);
binning/empty-input/input-validation mechanics mirroring the observer-frame
test file; and the `per_center_axis_angle` diagnostic's shape, range, and a
hand-checked case.
"""

from __future__ import annotations

import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.config import ShellConfig
from dvcorr.geometry import unit_vector
from dvcorr.estimators.shell_dipole import (
    expected_shell_occupancy,
    velocity_centered_shell_dipole,
)
from dvcorr.estimators.velocity_frame_dipole import velocity_frame_shell_dipole
from dvcorr.pipeline.velocity_centered import (
    matched_gaussian_sample,
    null_realization_seeds,
)
from dvcorr.pipeline.velocity_frame_comparison import (
    ComparisonRunConfig,
    gaussian_velocity_null_dipoles,
    random_axis_null_dipoles,
    run_both_frames,
    select_shared_centers,
)

# Reuse the SAME infall toy the geometry/group-centered/observer-frame
# velocity-centered sign gates are built on, rather than rebuilding a
# parallel one that could drift out of agreement with it. (_sphere_directions
# is one of the shared toys but this file does not need it -- its own
# anisotropic patterns are built with rng.normal instead, see e.g.
# _frame_agreement_centers_and_tracers -- so it is deliberately not imported,
# to avoid an unused import.)
from tests.test_geometry import (
    _infall_shell_with_velocities,
    R_CENTER,
    R_SHELL,
    V_INFALL,
    N_SHELL,
)


def _sample_ball(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """n points drawn UNIFORMLY inside a ball of `radius` around the observer.

    Reimplemented locally rather than imported: this is
    tests/test_velocity_centered_dipole.py's private helper of the same
    name, not one of the shared toys in tests/test_geometry.py that this
    file is told to import cross-file, so it is reproduced here verbatim
    (never weakened) per the task's instructions. Inverse-CDF sampling on
    the radius (r ~ radius * U**(1/3), since the volume element grows as
    r^2 dr) combined with uniform-on-sphere directions gives an exactly
    uniform spatial density, not merely a plausible-looking cloud.
    """
    directions = unit_vector(rng.normal(size=(n, 3)))
    r = radius * rng.uniform(0.0, 1.0, size=n) ** (1.0 / 3.0)  # 1/3: inverse-CDF for r^2 dr
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    return observer + r[:, None] * directions


# ---------------------------------------------------------------------------
# 1. Sign gate
# ---------------------------------------------------------------------------
#
# Geometry: every shell member sits at most R_CENTER + R_SHELL = 220 h^-1 Mpc
# from the observer (triangle inequality). `sub_volume_radius` and the
# DEFAULT `core_margin` (None -> shell_edges[-1] = 1.5 * R_SHELL = 30.0) are
# chosen, identically to test_velocity_centered_dipole.py's joint gate, so
# every member clears the core cut.

_VF_SUB_VOLUME_RADIUS = 280.0
_VF_MAX_MEMBER_OBSERVER_DISTANCE = R_CENTER + R_SHELL  # 220.0, triangle-inequality bound
_VF_CORE_MARGIN_DEFAULT = 1.5 * R_SHELL  # shell_edges[-1]; matches core_margin=None
_VF_MAGNITUDE_ATOL = 1e-8  # see the exactness note in the test docstring below


def _joint_gate_toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The velocity-frame half of the sign-gate toy.

    Centers = the N_SHELL shell members (carrying their 3-D infall
    velocities); the single tracer = the former central density object.
    Reimplemented locally (mirrors, but does not import,
    tests/test_velocity_centered_dipole.py's private `_joint_gate_toy` of
    the same construction) since that helper is private to its own file.
    """
    s_center, s_neighbors, u, v_vec = _infall_shell_with_velocities()
    tracers = s_center[None, :]
    return s_neighbors, v_vec, tracers


def test_sign_gate_velocity_frame_dipole_is_positive_and_agrees_with_observer_frame():
    """Velocity-frame dipole[0] positive under infall; agrees in SIGN with
    the observer-frame zeta_1 on the identical toy.

    Derivation (see dvcorr.estimators.velocity_frame_dipole's module
    docstring, Sign section, for the general argument -- this test makes it
    concrete): v_alpha = V_INFALL * r_hat_alpha with V_INFALL < 0, so
    v_hat_alpha = -r_hat_alpha, pointing from the shell member straight AT
    the central tracer. The center -> tracer direction n_hat_i (from
    `pair_separation` with the member in the "center" slot) is ALSO
    -r_hat_alpha (the tracer sits at the far end of that same radius).
    Hence cos_theta = z_hat_alpha . n_hat_i = (-r_hat_alpha).(-r_hat_alpha)
    = +1 EXACTLY for every member -- not approximately, and not merely on
    average over an isotropic shell, unlike the observer-frame gate (below).
    A = sqrt(3/4pi) * 1 > 0, and the scalar |v_alpha| > 0, so every center
    contributes positively and the stacked dipole is positive.

    Magnitude note -- this is where this test deliberately departs from a
    NAIVE transplant of the observer-frame gate's recovery formula, and why:
    the observer-frame joint gate (test_velocity_centered_dipole.py) reads
    off 3 * dipole[0] / (y10_norm * pair_count[0]) and recovers -V_INFALL to
    a LOOSE (rel=0.1) tolerance. That "3" undoes the <mu^2> = 1/3 dilution
    that occurs there because the observer-frame axis n_hat_V,alpha is
    fixed by each member's POSITION and is only approximately aligned with
    -r_hat_alpha (exactly so only in the distant-observer limit) -- the
    "3" is a shell-averaging correction, and the recovery is only
    approximate because of the finite-R leakage that correction is
    compensating for.
    Neither applies here. The velocity frame's axis z_hat_alpha comes from
    v_alpha, i.e. from the SAME r_hat_alpha that also places the tracer --
    the construction is exactly SELF-ALIGNED (see
    `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null`'s
    docstring for the general statement of this property), so cos_theta is
    identically 1 for every member with NO shell-averaging and NO
    finite-distance approximation anywhere in this toy. Multiplying by 3
    here would not undo a dilution that never happened -- it would introduce
    a spurious factor of 3. The correct, and now EXACT (not merely loose),
    recovery formula is therefore the L_0-style ratio without the L_1
    prefactor: dipole[0] / (y10_norm * pair_count[0]) == -V_INFALL, checked
    to a tight tolerance since there is no approximation left to be loose
    about.
    """
    shell_edges = np.array([0.5 * R_SHELL, 1.5 * R_SHELL])

    # Pin the core-cut safety margin the toy geometry was built for, exactly
    # as test_velocity_centered_dipole.py's joint gate does.
    assert _VF_MAX_MEMBER_OBSERVER_DISTANCE <= _VF_SUB_VOLUME_RADIUS - _VF_CORE_MARGIN_DEFAULT

    vf_centers, vf_velocities, vf_tracers = _joint_gate_toy()
    vf_result = velocity_frame_shell_dipole(
        s_centers=vf_centers,
        v_centers=vf_velocities,
        s_tracers=vf_tracers,
        shell_edges=shell_edges,
        sub_volume_radius=_VF_SUB_VOLUME_RADIUS,
    )

    assert vf_result.n_candidates == N_SHELL
    assert vf_result.n_centers == N_SHELL           # every member clears the core cut
    assert vf_result.pair_count.sum() == N_SHELL      # one tracer per center

    assert vf_result.dipole[0] > 0.0    # must FAIL LOUDLY if the orientation flips

    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
    recovered_vf = vf_result.dipole[0] / (y10_norm * vf_result.pair_count[0])
    assert recovered_vf == pytest.approx(-V_INFALL, abs=_VF_MAGNITUDE_ATOL)  # == |V_INFALL|, exact

    # Agreement in SIGN with the observer-frame zeta_1 on the identical toy
    # -- the two frames must not disagree, and this is the guard against a
    # silent flip introduced by the rotated axis.
    vc_result = velocity_centered_shell_dipole(
        s_centers=vf_centers,
        v_centers=vf_velocities,
        s_tracers=vf_tracers,
        shell_edges=shell_edges,
        sub_volume_radius=_VF_SUB_VOLUME_RADIUS,
    )
    assert vc_result.dipole[0] > 0.0
    assert np.sign(vf_result.dipole[0]) == np.sign(vc_result.dipole[0])


# ---------------------------------------------------------------------------
# 2. Frame-agreement limit: purely radial flow
# ---------------------------------------------------------------------------

_FA_N_CENTERS = 6
_FA_R_CENTER = 300.0     # h^-1 Mpc, observer -> each center
_FA_N_TRACERS = 64       # anisotropic tracers per center
_FA_SHELL_EDGES = np.array([5.0, 15.0, 25.0])
_FA_SPEED = 250.0        # km/s, |c|
_FA_ATOL = 1e-6          # tight: no approximation involved, see the test docstring


def _frame_agreement_centers_and_tracers(
    rng: np.random.Generator, speed_sign: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Well-separated centers on the coordinate axes, each with a purely
    radial velocity v_alpha = speed_sign * _FA_SPEED * n_hat_V,alpha, plus an
    anisotropic (not perfectly isotropic) set of tracers around each.

    Centers are placed along +-x, +-y, +-z (not random directions) so that
    even the outermost shell (r <= 25) around one center cannot overlap
    another center's KDTree query (adjacent centers are >= 300*sqrt(2) ~ 424
    h^-1 Mpc apart) -- deterministic isolation, not a probabilistic one.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    directions = unit_vector(
        np.array(
            [
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            ]
        )
    )
    s_centers = observer + _FA_R_CENTER * directions
    n_hat_V = unit_vector(s_centers - observer)  # == directions, exactly
    v_centers = speed_sign * _FA_SPEED * n_hat_V  # purely radial

    tracer_dirs = unit_vector(rng.normal(size=(_FA_N_TRACERS, 3)))  # anisotropic pattern
    tracer_radii = rng.uniform(
        _FA_SHELL_EDGES[0] + 1.0, _FA_SHELL_EDGES[-1] - 1.0, size=_FA_N_TRACERS
    )
    offsets = tracer_radii[:, None] * tracer_dirs
    s_tracers = np.vstack([s_centers[k] + offsets for k in range(_FA_N_CENTERS)])

    return s_centers, v_centers, s_tracers


def test_frame_agreement_for_purely_radial_flow_outbound_and_inbound():
    """v_alpha = c * n_hat_V,alpha (pure radial flow) makes the two frames
    agree EXACTLY, for either sign of c.

    Outbound (c > 0): both axes are +n_hat_V,alpha -- the velocity frame's
    v_hat_alpha trivially, the observer frame's because sign(u_alpha) = +1
    (conventions.VELOCITY_AXIS_CONVENTION) -- so they coincide
    (per_center_axis_angle ~= 0), and both scalars are |c|.

    Inbound (c < 0, the COMPANION case, same test): BOTH axes flip together,
    to -n_hat_V,alpha -- the velocity frame's because v_hat_alpha points back
    at the observer, the observer frame's because sign(u_alpha) = -1. They
    coincide again, so per_center_axis_angle is ~0 HERE TOO, not ~pi. Both
    scalars are again |c| (|v_alpha| and |u_alpha| respectively). This is the
    whole point of signing the axis: the two frames agree about which way an
    inbound center is moving, and the old ~pi reading was an artifact of
    axing the observer frame on the unsigned line of sight.

    Either way, pure-radial flow gives exact frame agreement regardless of
    direction; the frames can only diverge through the TRANSVERSE velocity
    component, which this toy deliberately has none of. The dipole equality
    below is asserted for both signs and pins that.
    """
    rng = np.random.default_rng(20260724)

    for speed_sign in (1.0, -1.0):
        s_centers, v_centers, s_tracers = _frame_agreement_centers_and_tracers(rng, speed_sign)

        obs_result = velocity_centered_shell_dipole(
            s_centers=s_centers,
            v_centers=v_centers,
            s_tracers=s_tracers,
            shell_edges=_FA_SHELL_EDGES,
            sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
        )
        vf_result = velocity_frame_shell_dipole(
            s_centers=s_centers,
            v_centers=v_centers,
            s_tracers=s_tracers,
            shell_edges=_FA_SHELL_EDGES,
            sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
        )

        assert obs_result.n_centers == _FA_N_CENTERS
        assert vf_result.n_centers == _FA_N_CENTERS

        np.testing.assert_allclose(
            vf_result.per_center_speed, np.abs(obs_result.per_center_u), atol=_FA_ATOL
        )
        np.testing.assert_allclose(vf_result.per_center_count, obs_result.per_center_count)
        np.testing.assert_allclose(vf_result.per_center_dipole, obs_result.per_center_dipole, atol=_FA_ATOL)
        np.testing.assert_allclose(vf_result.dipole, obs_result.dipole, atol=_FA_ATOL)

        # Zero for BOTH signs: the observer-frame axis carries sign(u_alpha),
        # so an inbound center's two axes flip together rather than apart.
        np.testing.assert_allclose(vf_result.per_center_axis_angle, 0.0, atol=_FA_ATOL)
        # The observer frame's own weight is the radial SPEED, and for a purely
        # radial flow it is the full speed.
        np.testing.assert_allclose(
            obs_result.per_center_speed, vf_result.per_center_speed, atol=_FA_ATOL
        )


# ---------------------------------------------------------------------------
# 3. Degenerate centers -- pipeline level
# ---------------------------------------------------------------------------


def test_select_shared_centers_drops_zero_and_slow_speed_centers():
    """A zero-speed candidate and one just under the floor are both dropped;
    the funnel is reported via n_dropped_slow, and nothing crashes.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    min_center_speed = 5.0
    cfg = ComparisonRunConfig(min_center_speed=min_center_speed)

    s_candidates = observer + np.array(
        [[50.0, 0.0, 0.0], [60.0, 0.0, 0.0], [70.0, 0.0, 0.0]]
    )
    v_candidates = np.array(
        [
            [0.0, 0.0, 0.0],    # exactly zero speed
            [1.0, 0.0, 0.0],    # speed 1.0, below the floor of 5.0
            [100.0, 0.0, 0.0],  # well above the floor
        ]
    )

    centers = select_shared_centers(cfg, s_candidates, v_candidates, observer)

    assert centers.n_candidates == 3
    assert centers.n_core == 3       # all well inside any reasonable core cut
    assert centers.n_centers == 1
    assert centers.n_dropped_slow == 2
    np.testing.assert_allclose(centers.s_centers, [s_candidates[2]])
    np.testing.assert_allclose(centers.v_centers, [v_candidates[2]])


def test_velocity_frame_shell_dipole_raises_on_zero_speed_surviving_center():
    """The contract the min_center_speed floor exists to protect: a
    zero-speed center that reaches the estimator directly (bypassing
    `select_shared_centers`) must raise, not silently contribute A=0.
    """
    s_center = (np.asarray(conventions.OBSERVER_POSITION, dtype=float) + np.array([50.0, 0.0, 0.0]))[None, :]
    v_center = np.array([[0.0, 0.0, 0.0]])
    tracers = s_center + np.array([[0.0, 5.0, 0.0]])

    with pytest.raises(ValueError):
        velocity_frame_shell_dipole(
            s_centers=s_center,
            v_centers=v_center,
            s_tracers=tracers,
            shell_edges=np.array([0.0, 10.0]),
            sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
        )


def test_zero_speed_candidate_outside_the_core_does_not_raise():
    """The zero-speed guard is scoped to SURVIVING centers only (after the
    core cut), per `velocity_frame_shell_dipole`'s Raises docstring: a
    zero-speed candidate that the core cut drops for an unrelated reason
    (too close to the sub-volume boundary) must be silently excluded, like
    any other core-cut casualty, not raise. This is the companion to
    `test_velocity_frame_shell_dipole_raises_on_zero_speed_surviving_center`,
    which only pins the positive (raises) case.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    sub_volume_radius = 100.0
    shell_edges = np.array([0.0, 10.0])
    core_margin = shell_edges[-1]  # matches the default core_margin=None -> r_max

    s_centers = observer + np.array(
        [
            [50.0, 0.0, 0.0],  # distance 50 <= 100-10=90: survives the core cut
            [95.0, 0.0, 0.0],  # distance 95 > 90: dropped BY THE CORE CUT, not the speed guard
        ]
    )
    v_centers = np.array(
        [
            [20.0, 0.0, 0.0],  # nonzero: this surviving center must not trip the guard
            [0.0, 0.0, 0.0],   # zero speed, but never reaches the guard: dropped first
        ]
    )
    tracers = s_centers[0][None, :] + np.array([[0.0, 5.0, 0.0]])

    result = velocity_frame_shell_dipole(
        s_centers=s_centers,
        v_centers=v_centers,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=sub_volume_radius,
        core_margin=core_margin,
    )

    assert result.n_candidates == 2
    assert result.n_centers == 1  # only the nonzero-speed, in-core candidate survives


def test_select_shared_centers_speed_floor_boundary():
    """The `>=` floor is inclusive for a positive `min_center_speed` (a
    center exactly AT the floor survives), while `min_center_speed == 0.0`
    still drops an exact-zero-speed center (the `speeds > 0.0` conjunct
    this task's review added -- see `select_shared_centers`'s docstring).
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_candidates = observer + np.array([[50.0, 0.0, 0.0], [60.0, 0.0, 0.0]])

    # Positive floor: a speed exactly AT the floor is kept, not dropped.
    floor = 5.0
    v_candidates = np.array([[floor, 0.0, 0.0], [floor - 0.001, 0.0, 0.0]])
    cfg = ComparisonRunConfig(min_center_speed=floor)
    centers = select_shared_centers(cfg, s_candidates, v_candidates, observer)
    assert centers.n_centers == 1
    np.testing.assert_allclose(centers.s_centers, [s_candidates[0]])

    # Zero floor: an exact-zero-speed candidate is still dropped, per the
    # documented "0.0 means drop only exactly-zero-speed centers" contract.
    v_candidates_zero_floor = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cfg_zero_floor = ComparisonRunConfig(min_center_speed=0.0)
    centers_zero_floor = select_shared_centers(
        cfg_zero_floor, s_candidates, v_candidates_zero_floor, observer
    )
    assert centers_zero_floor.n_centers == 1
    assert centers_zero_floor.n_dropped_slow == 1
    np.testing.assert_allclose(centers_zero_floor.s_centers, [s_candidates[1]])


def test_select_shared_centers_raises_runtime_error_not_zero_division_on_empty_input():
    """An empty candidate array must raise `RuntimeError` (checked BEFORE the
    core-cut survival percentage is computed), never a bare
    `ZeroDivisionError` with a misleading traceback -- the same hazard
    `dvcorr.pipeline.velocity_centered.load_and_carve`'s docstring calls out.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    cfg = ComparisonRunConfig()
    empty_s = np.empty((0, 3))
    empty_v = np.empty((0, 3))

    with pytest.raises(RuntimeError):
        select_shared_centers(cfg, empty_s, empty_v, observer)


# ---------------------------------------------------------------------------
# 4. Same center set for both frames
# ---------------------------------------------------------------------------

_RB_N_CANDIDATES = 40
_RB_N_TRACERS = 2000
_RB_SUB_VOLUME_RADIUS = 300.0
_RB_ATOL = 1e-8


def test_run_both_frames_shares_the_same_center_set():
    """`run_both_frames` hands both estimators the identical center set:
    equal n_centers, row-aligned per-center scalars (independently
    recomputed), and IDENTICAL occupancy (per_center_count) -- occupancy is
    pure geometry and cannot depend on which axis is used.
    """
    rng = np.random.default_rng(20260725)
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    cfg = ComparisonRunConfig(sub_volume_radius=_RB_SUB_VOLUME_RADIUS)

    s_candidates = _sample_ball(rng, _RB_N_CANDIDATES, _RB_SUB_VOLUME_RADIUS)
    v_candidates = rng.normal(scale=200.0, size=(_RB_N_CANDIDATES, 3))
    # Guard against a pathological near-zero draw so this test exercises the
    # ordinary path, not the (separately tested) zero-speed ValueError.
    too_slow = np.linalg.norm(v_candidates, axis=1) < cfg.min_center_speed
    v_candidates[too_slow] += 50.0

    tracers = _sample_ball(rng, _RB_N_TRACERS, _RB_SUB_VOLUME_RADIUS)

    centers = select_shared_centers(cfg, s_candidates, v_candidates, observer)
    results = run_both_frames(cfg, centers, tracers, observer)

    assert results.obs_result.n_centers == centers.n_centers
    assert results.vel_result.n_centers == centers.n_centers

    n_hat_V = unit_vector(centers.s_centers - observer)
    u_expected = np.einsum("ij,ij->i", centers.v_centers, n_hat_V)
    speed_expected = np.linalg.norm(centers.v_centers, axis=1)

    np.testing.assert_allclose(results.obs_result.per_center_u, u_expected, atol=_RB_ATOL)
    np.testing.assert_allclose(results.vel_result.per_center_speed, speed_expected, atol=_RB_ATOL)

    np.testing.assert_array_equal(
        results.obs_result.per_center_count, results.vel_result.per_center_count
    )


# ---------------------------------------------------------------------------
# 5. Null tests
# ---------------------------------------------------------------------------

_NULL_N_CENTERS = 10
_NULL_R_CENTER = 200.0       # h^-1 Mpc; well inside _NULL_R_SUB - shell_max (buffer 50)
_NULL_RADIAL_SPEED = 300.0   # km/s, the radial component (this is u_alpha for the obs frame)
_NULL_TRANSVERSE_SPEED = 300.0  # km/s, transverse component -- see docstring note below
_NULL_N_TRACERS = 120000     # generous: keeps per-shell shot noise manageable
_NULL_R_SUB = 300.0
_NULL_SHELL_EDGES = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
_NULL_ZERO_TOL = 0.2         # normalized dipole must stay well below this fraction of the speed scale


def test_uncorrelated_density_field_gives_null_dipole_in_both_frames():
    """Tracers drawn uniformly at random -- a density field UNCORRELATED
    with the centers' (fixed) velocities/axes -- must leave both frames'
    normalized dipoles consistent with zero.

    Two things this test deliberately does NOT do, and why:

    1. It does not permute tracer POSITIONS among themselves as a "null".
       That permutation is a no-op: the tracer catalog is the same set of
       points either way, so nothing about the density field actually
       changes. The null that matters is an uncorrelated density field,
       built directly here by sampling tracers independently of the
       centers, not by shuffling a correlated one.
    2. For the velocity frame specifically, a permutation of the per-center
       SCALAR |v_alpha| would not be a null at all -- see
       `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null`'s
       docstring for the full argument (the statistic is SELF-ALIGNED: the
       axis and the scalar are built from the SAME velocity vector, so
       permuting only the scalar leaves every center's axis-density
       alignment untouched and the recombined sum stays close to the real
       signal). That is exactly why the pipeline null randomizes the AXIS
       instead of the scalar -- exercised separately below on a
       configuration with a genuine nonzero signal to collapse.

    A THIRD deliberate choice: `v_centers` carries a TRANSVERSE component
    (`_NULL_TRANSVERSE_SPEED`, comparable in magnitude to the radial one),
    not a purely radial velocity. A purely radial `v_centers` would make
    z_hat_alpha == n_hat_V,alpha exactly (the frame-agreement limit,
    `test_frame_agreement_for_purely_radial_flow_outbound_and_inbound`), so
    the two frames' results would be bit-identical here and this test would
    only independently exercise ONE frame's null behavior, not both. The
    transverse component breaks that degeneracy so this test genuinely
    checks the velocity frame's own axis (not a copy of the observer
    frame's) against an uncorrelated density field.
    """
    rng = np.random.default_rng(20260726)
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)

    directions = unit_vector(rng.normal(size=(_NULL_N_CENTERS, 3)))
    s_centers = observer + _NULL_R_CENTER * directions
    n_hat_V = unit_vector(s_centers - observer)

    # A transverse component (Gram-Schmidt: strip the radial projection out
    # of an independent random draw, then renormalize) so z_hat_alpha and
    # n_hat_V,alpha genuinely differ -- see the docstring's third point.
    raw = rng.normal(size=(_NULL_N_CENTERS, 3))
    radial_part = np.einsum("ij,ij->i", raw, n_hat_V)[:, None] * n_hat_V
    transverse_dir = unit_vector(raw - radial_part)
    v_centers = _NULL_RADIAL_SPEED * n_hat_V + _NULL_TRANSVERSE_SPEED * transverse_dir

    tracers = _sample_ball(rng, _NULL_N_TRACERS, _NULL_R_SUB)

    obs_result = velocity_centered_shell_dipole(
        s_centers=s_centers,
        v_centers=v_centers,
        s_tracers=tracers,
        shell_edges=_NULL_SHELL_EDGES,
        sub_volume_radius=_NULL_R_SUB,
    )
    vf_result = velocity_frame_shell_dipole(
        s_centers=s_centers,
        v_centers=v_centers,
        s_tracers=tracers,
        shell_edges=_NULL_SHELL_EDGES,
        sub_volume_radius=_NULL_R_SUB,
    )

    assert obs_result.n_centers == _NULL_N_CENTERS
    assert vf_result.n_centers == _NULL_N_CENTERS

    n_bar = _NULL_N_TRACERS / ((4.0 / 3.0) * np.pi * _NULL_R_SUB**3)
    nbar_v_b = expected_shell_occupancy(n_bar, _NULL_SHELL_EDGES)
    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))

    obs_normalized = (obs_result.dipole / obs_result.n_centers) / (y10_norm * nbar_v_b)
    vf_normalized = (vf_result.dipole / vf_result.n_centers) / (y10_norm * nbar_v_b)

    # Tolerance scaled to the total speed |v_alpha| = sqrt(radial^2 + transverse^2),
    # the natural scale of both u_alpha (bounded by it) and |v_alpha| (equal to it).
    speed_scale = np.hypot(_NULL_RADIAL_SPEED, _NULL_TRANSVERSE_SPEED)
    tol = _NULL_ZERO_TOL * speed_scale
    np.testing.assert_allclose(obs_normalized, 0.0, atol=tol)
    np.testing.assert_allclose(vf_normalized, 0.0, atol=tol)


_CLUSTER_N_CENTERS = 12
_CLUSTER_R_CENTER = 250.0
_CLUSTER_SPEED = 300.0
_CLUSTER_N_TRACERS_PER_CENTER = 400
_CLUSTER_SHELL_MIN = 10.0
_CLUSTER_SHELL_MAX = 30.0
_CLUSTER_SUB_VOLUME_RADIUS = 300.0
_CLUSTER_AXIS_MIX = 0.6       # concentration of tracers toward the flow axis; 0=isotropic, 1=on-axis
_CLUSTER_COLLAPSE_FRACTION = 0.2  # the null must land well below this fraction of the signal
_CLUSTER_N_REALIZATIONS = 8       # enough for a mean and a spread on a 12-center toy
#: How many standard errors the null's across-realization MEAN may sit from
#: zero before the construction is judged not to be a null. Three: the mean of
#: R draws is asymptotically normal about zero, so this is a ~99.7% band.
_NULL_MEAN_SIGMA_TOLERANCE = 3.0


def _clustered_signal_centers_and_tracers(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centers with a purely radial velocity and tracers deliberately biased
    toward each center's own flow axis -- a genuine, nonzero velocity-frame
    signal (density excess ahead of the flow, per the sign derivation), for
    `run_random_axis_null` to be checked against.

    The bias is built by mixing each isotropic tracer direction with the
    center's own flow axis before renormalizing (a simple, Gram-Schmidt-free
    way to concentrate directions around an arbitrary axis without needing
    to construct its orthonormal complement) -- not a physically motivated
    profile, just a deterministic way to guarantee a clear, nonzero signal.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    directions = unit_vector(rng.normal(size=(_CLUSTER_N_CENTERS, 3)))
    s_centers = observer + _CLUSTER_R_CENTER * directions
    n_hat_V = unit_vector(s_centers - observer)
    v_centers = _CLUSTER_SPEED * n_hat_V  # purely radial: z_hat_alpha == n_hat_V,alpha

    tracer_rows = []
    for k in range(_CLUSTER_N_CENTERS):
        isotropic_dirs = unit_vector(rng.normal(size=(_CLUSTER_N_TRACERS_PER_CENTER, 3)))
        biased_dirs = unit_vector(
            _CLUSTER_AXIS_MIX * n_hat_V[k] + (1.0 - _CLUSTER_AXIS_MIX) * isotropic_dirs
        )
        radii = rng.uniform(
            _CLUSTER_SHELL_MIN, _CLUSTER_SHELL_MAX, size=_CLUSTER_N_TRACERS_PER_CENTER
        )
        tracer_rows.append(s_centers[k] + radii[:, None] * biased_dirs)
    s_tracers = np.vstack(tracer_rows)

    return s_centers, v_centers, s_tracers


def _clustered_cfg() -> ComparisonRunConfig:
    """The single-shell config all four null tests below share."""
    return ComparisonRunConfig(
        sub_volume_radius=_CLUSTER_SUB_VOLUME_RADIUS,
        shells=ShellConfig(
            min_radius=_CLUSTER_SHELL_MIN,
            max_radius=_CLUSTER_SHELL_MAX,
            radii_step=_CLUSTER_SHELL_MAX - _CLUSTER_SHELL_MIN,  # a single shell
        ),
        n_null_realizations=_CLUSTER_N_REALIZATIONS,
    )


def _clustered_signal_run(rng: np.random.Generator):
    """Centers, tracers, and the signal estimator pass they produce."""
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    cfg = _clustered_cfg()
    s_centers, v_centers, s_tracers = _clustered_signal_centers_and_tracers(rng)
    centers = select_shared_centers(cfg, s_centers, v_centers, observer)

    signal_result = velocity_frame_shell_dipole(
        s_centers=centers.s_centers,
        v_centers=centers.v_centers,
        s_tracers=s_tracers,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )
    return cfg, centers, s_tracers, observer, signal_result


def test_random_axis_null_collapses_the_velocity_frame_dipole_on_a_clustered_signal():
    """`random_axis_null_dipoles` collapses the velocity-frame dipole relative
    to a genuine, nonzero signal built by biasing tracers toward each center's
    own flow direction (see `_clustered_signal_centers_and_tracers`).

    Checked on the across-realization MEAN, which is what reaches the figure,
    and then on every individual realization -- the second is the stronger
    statement, and it is the one that would catch a null that collapses only
    on average because its draws cancel.
    """
    rng = np.random.default_rng(20260727)
    cfg, centers, _s_tracers, _observer, signal_result = _clustered_signal_run(rng)
    assert centers.n_centers == _CLUSTER_N_CENTERS  # nothing dropped: fast, well-placed centers

    null_dipoles = random_axis_null_dipoles(cfg, centers, signal_result)
    assert null_dipoles.shape == (cfg.n_null_realizations, signal_result.dipole.size)

    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
    signal_normalized = signal_result.dipole / (y10_norm * signal_result.pair_count)
    null_normalized = null_dipoles / (y10_norm * signal_result.pair_count)

    assert abs(signal_normalized[0]) > 0.0  # sanity: the signal is genuinely nonzero
    bar = _CLUSTER_COLLAPSE_FRACTION * abs(signal_normalized[0])
    null_mean = null_normalized.mean(axis=0)[0]
    assert abs(null_mean) < bar

    # ... and the mean is consistent with ZERO given the realizations' own
    # scatter, which is the sharper statement. Note what is deliberately NOT
    # asserted: that every INDIVIDUAL realization clears `bar`. Several do not
    # -- on a 12-center toy a single draw lands a few times the bar away from
    # zero, in either direction -- and demanding otherwise would be demanding
    # that a random variable equal its own expectation. That mistake, made on
    # the production figure rather than here, is what made a single-realization
    # null curve read as a systematic trend at small r.
    null_sem = null_normalized[:, 0].std(ddof=1) / np.sqrt(null_normalized.shape[0])
    assert abs(null_mean) < _NULL_MEAN_SIGMA_TOLERANCE * null_sem


def test_random_axis_null_realization_equals_a_full_estimator_pass():
    """The recombination is EXACT, not an approximation: realization k of
    `random_axis_null_dipoles` equals what `velocity_frame_shell_dipole`
    returns when re-run on that realization's replacement velocities.

    This is the invariant that lets the null be built from cached direction
    sums instead of a second estimator pass
    (`VelocityFrameShellDipoleResult.per_center_direction_sum`), and it
    subsumes what two older tests checked separately -- that the null keeps
    each center's own speed attached, and that it leaves positions (and hence
    occupancy) untouched. Neither could survive a mismatch here.
    """
    rng = np.random.default_rng(20260729)
    cfg, centers, s_tracers, observer, signal_result = _clustered_signal_run(rng)

    null_dipoles = random_axis_null_dipoles(cfg, centers, signal_result)

    # Rebuild realization 0's velocities exactly as the function does, then
    # pay for the estimator pass this construction exists to avoid.
    child = null_realization_seeds(cfg.axis_null_seed, cfg.n_null_realizations)[0]
    speeds = np.linalg.norm(centers.v_centers, axis=1)
    v_null = speeds[:, None] * unit_vector(
        np.random.default_rng(child).normal(size=centers.v_centers.shape)
    )
    reference = velocity_frame_shell_dipole(
        s_centers=centers.s_centers,
        v_centers=v_null,
        s_tracers=s_tracers,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )

    np.testing.assert_allclose(null_dipoles[0], reference.dipole, rtol=1e-10, atol=1e-10)
    # Speed preserved by the draw, and occupancy (pure geometry) untouched.
    np.testing.assert_allclose(reference.per_center_speed, speeds, atol=1e-8)
    np.testing.assert_array_equal(
        reference.per_center_count, signal_result.per_center_count
    )


def test_gaussian_velocity_null_collapses_the_velocity_frame_dipole():
    """`gaussian_velocity_null_dipoles` is a NULL on the same clustered signal
    the random-axis null is checked against, and to the same bar.

    Both of this frame's nulls have to clear this: whatever else they differ
    on (`gaussian_velocity_null_dipoles`' docstring lays out exactly what each
    preserves), a construction that did not decouple the axis from the
    density field would not be a null at all and would have no business on
    the figure.
    """
    rng = np.random.default_rng(20260804)
    cfg, centers, _s_tracers, _observer, signal_result = _clustered_signal_run(rng)

    gaussian_null_dipoles = gaussian_velocity_null_dipoles(cfg, centers, signal_result)

    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
    signal_normalized = signal_result.dipole / (y10_norm * signal_result.pair_count)
    null_normalized = gaussian_null_dipoles / (y10_norm * signal_result.pair_count)

    assert abs(signal_normalized[0]) > 0.0
    bar = _CLUSTER_COLLAPSE_FRACTION * abs(signal_normalized[0])
    null_mean = null_normalized.mean(axis=0)[0]
    assert abs(null_mean) < bar

    # Same two-part bar as the random-axis null, and the same reason for not
    # testing individual realizations -- see that test.
    null_sem = null_normalized[:, 0].std(ddof=1) / np.sqrt(null_normalized.shape[0])
    assert abs(null_mean) < _NULL_MEAN_SIGMA_TOLERANCE * null_sem


def test_gaussian_velocity_null_detaches_the_speed_from_its_center():
    """The two velocity-frame nulls differ in EXACTLY the documented way.

    Speed: `random_axis_null_dipoles` keeps each center's own |v_alpha|
    attached to that center (asserted in
    `test_random_axis_null_realization_equals_a_full_estimator_pass`); this
    one does NOT -- it draws a fresh vector, so the per-center speeds differ
    center-by-center while the population's per-component mean and spread are
    matched. If this ever starts preserving the speeds, the two nulls have
    silently become the same test and the figure's dotted curve stops
    carrying information.

    The recombination itself is checked the same way as the random-axis
    null's: realization 0 must equal a full estimator pass on the same drawn
    velocities.
    """
    rng = np.random.default_rng(20260804)
    cfg, centers, s_tracers, observer, signal_result = _clustered_signal_run(rng)

    gaussian_null_dipoles = gaussian_velocity_null_dipoles(cfg, centers, signal_result)

    child = null_realization_seeds(
        cfg.velocity_gaussian_null_seed, cfg.n_null_realizations
    )[0]
    drawn_v = matched_gaussian_sample(centers.v_centers, int(child))

    # Speed detached from its center (contrast the random-axis null, which
    # preserves it exactly). The moment-MATCHING contract of the draw is not
    # re-checked here: this toy has ~12 centers, where a matched draw's mean
    # scatters by sigma/sqrt(12), so it is pinned at large N in
    # `tests/test_velocity_centered_dipole.py::test_matched_gaussian_sample_matches_each_column_of_a_vector_sample`.
    original_speed = np.linalg.norm(centers.v_centers, axis=1)
    assert not np.allclose(np.linalg.norm(drawn_v, axis=1), original_speed, atol=1e-8)

    reference = velocity_frame_shell_dipole(
        s_centers=centers.s_centers,
        v_centers=drawn_v,
        s_tracers=s_tracers,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )
    np.testing.assert_allclose(
        gaussian_null_dipoles[0], reference.dipole, rtol=1e-10, atol=1e-10
    )
    # Position untouched: identical occupancy against the ORIGINAL velocities.
    np.testing.assert_array_equal(
        reference.per_center_count, signal_result.per_center_count
    )


def test_per_center_direction_sum_projects_onto_the_amplitude():
    """A_alpha,b == sqrt(3/4pi) * (z_hat_alpha . S_alpha,b), exactly.

    The linearity that every null in this frame rests on: Y_10 is linear in
    the direction cosine, so summing over a shell's tracers commutes with
    projecting onto the axis. If this ever fails, the cached direction sums
    are not the amplitude's factorization and every null built from them is
    measuring something else.
    """
    rng = np.random.default_rng(20260806)
    _cfg, centers, _s_tracers, _observer, signal_result = _clustered_signal_run(rng)

    z_hat = unit_vector(centers.v_centers)
    projected = np.sqrt(3.0 / (4.0 * np.pi)) * np.einsum(
        "ac,abc->ab", z_hat, signal_result.per_center_direction_sum
    )

    np.testing.assert_allclose(
        projected, signal_result.per_center_amplitude, rtol=1e-10, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 6. Mechanical contracts: binning, empty/degenerate inputs, validation
# ---------------------------------------------------------------------------


def _single_vf_center() -> tuple[np.ndarray, np.ndarray]:
    """One velocity-frame center with an arbitrary nonzero velocity."""
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_center = (observer + np.array([200.0, 0.0, 0.0]))[None, :]
    v_center = np.array([[100.0, 0.0, 0.0]])
    return s_center, v_center


def test_tracers_land_in_the_shell_their_radius_dictates():
    """Hand-placed tracers at known radii fall in the expected shells."""
    s_center, v_center = _single_vf_center()
    radii = np.array([5.0, 15.0, 25.0])
    tracers = s_center[0] + np.column_stack((np.zeros_like(radii), radii, np.zeros_like(radii)))
    shell_edges = np.array([0.0, 10.0, 20.0, 30.0])

    result = velocity_frame_shell_dipole(
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
    s_center, v_center = _single_vf_center()
    radii = np.array([5.0, 15.0, 25.0, 35.0])  # 5 below edges[0]=10, 35 above 30
    tracers = s_center[0] + np.column_stack((np.zeros_like(radii), radii, np.zeros_like(radii)))
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = velocity_frame_shell_dipole(
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
    s_center, v_center = _single_vf_center()
    tracers = s_center[0] + np.array([[0.0, 30.0, 0.0]])  # r = 30 == edges[-1]
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = velocity_frame_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    np.testing.assert_array_equal(result.pair_count, [0.0, 1.0])


def test_empty_shell_returns_zero_not_nan():
    s_center, v_center = _single_vf_center()
    tracers = s_center[0] + np.array([[0.0, 5.0, 0.0]])  # only in the first shell
    shell_edges = np.array([0.0, 10.0, 20.0])

    result = velocity_frame_shell_dipole(
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
    s_center, v_center = _single_vf_center()
    tracers = np.empty((0, 3))
    shell_edges = np.array([10.0, 20.0, 30.0])

    result = velocity_frame_shell_dipole(
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
    s_center, v_center = _single_vf_center()
    tracers = s_center[0] + np.array([[0.0, 15.0, 0.0]])
    shell_edges = np.array([10.0, 20.0])

    result = velocity_frame_shell_dipole(
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
    assert result.per_center_speed.shape == (0,)
    assert result.per_center_axis_angle.shape == (0,)
    assert np.all(np.isfinite(result.pair_count))
    assert np.all(np.isfinite(result.dipole))
    assert np.all(np.isfinite(result.monopole))


class TestValidation:
    def test_non_finite_v_centers_raises(self):
        s_center, _ = _single_vf_center()
        bad_v = np.array([[np.nan, 0.0, 0.0]])
        with pytest.raises(ValueError):
            velocity_frame_shell_dipole(
                s_centers=s_center,
                v_centers=bad_v,
                s_tracers=s_center,
                shell_edges=np.array([0.0, 10.0]),
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
            )

    def test_mismatched_shapes_raises(self):
        s_center, _ = _single_vf_center()
        bad_v = np.zeros((2, 3))
        with pytest.raises(ValueError):
            velocity_frame_shell_dipole(
                s_centers=s_center,
                v_centers=bad_v,
                s_tracers=s_center,
                shell_edges=np.array([0.0, 10.0]),
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
            )

    def test_non_increasing_edges_raises(self):
        s_center, v_center = _single_vf_center()
        with pytest.raises(ValueError):
            velocity_frame_shell_dipole(
                s_centers=s_center,
                v_centers=v_center,
                s_tracers=s_center,
                shell_edges=np.array([20.0, 10.0]),
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
            )

    def test_edge_beyond_max_analysis_radius_raises(self):
        s_center, v_center = _single_vf_center()
        too_far = np.array([0.0, conventions.MAX_ANALYSIS_RADIUS + 1.0])
        with pytest.raises(ValueError):
            velocity_frame_shell_dipole(
                s_centers=s_center,
                v_centers=v_center,
                s_tracers=s_center,
                shell_edges=too_far,
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
            )

    def test_non_positive_sub_volume_radius_raises(self):
        s_center, v_center = _single_vf_center()
        with pytest.raises(ValueError):
            velocity_frame_shell_dipole(
                s_centers=s_center,
                v_centers=v_center,
                s_tracers=s_center,
                shell_edges=np.array([0.0, 10.0]),
                sub_volume_radius=0.0,
            )

    def test_negative_core_margin_raises(self):
        s_center, v_center = _single_vf_center()
        with pytest.raises(ValueError):
            velocity_frame_shell_dipole(
                s_centers=s_center,
                v_centers=v_center,
                s_tracers=s_center,
                shell_edges=np.array([0.0, 10.0]),
                sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
                core_margin=-1.0,
            )


# ---------------------------------------------------------------------------
# 7. per_center_axis_angle contract
# ---------------------------------------------------------------------------


def test_per_center_axis_angle_hand_checked_perpendicular_case():
    """v perpendicular to n_hat_V,alpha must give delta_alpha == pi/2 exactly.

    This is delta's CEILING, not a point in its interior: cos(delta) =
    |u_alpha| / |v_alpha|, which is zero exactly when the flow is purely
    transverse -- the case the observer frame is blind to.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_center = (observer + np.array([200.0, 0.0, 0.0]))[None, :]  # n_hat_V = +x
    v_center = np.array([[0.0, 50.0, 0.0]])                        # perpendicular to +x
    tracers = s_center[0] + np.array([[0.0, 0.0, 5.0]])

    result = velocity_frame_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=tracers,
        shell_edges=np.array([0.0, 10.0]),
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    assert result.per_center_axis_angle.shape == (1,)
    np.testing.assert_allclose(result.per_center_axis_angle, [np.pi / 2.0], atol=1e-12)


def test_per_center_axis_angle_shape_and_range_for_a_batch():
    """(N_c,) shape and the [0, pi/2] range hold over a generic batch.

    pi/2, NOT pi: the observer-frame axis the angle is measured against is
    sign(u_alpha) * n_hat_V,alpha (conventions.VELOCITY_AXIS_CONVENTION), so
    cos(delta) = |u_alpha| / |v_alpha| >= 0 and delta cannot exceed a right
    angle. A random batch like this one -- isotropic velocities, isotropic
    positions -- would routinely produce delta > pi/2 against the UNSIGNED
    line of sight, so this is a real discriminator between the two readings,
    not a vacuous bound. `velocity_frame_shell_dipole` enforces the same
    ceiling internally; the assertion here is the independent check.
    """
    rng = np.random.default_rng(20260728)
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    n = 25
    s_centers = observer + 200.0 * unit_vector(rng.normal(size=(n, 3)))
    v_centers = rng.normal(scale=150.0, size=(n, 3))
    too_slow = np.linalg.norm(v_centers, axis=1) < 1.0
    v_centers[too_slow] += 50.0  # avoid the (separately tested) zero-speed edge case
    tracers = _sample_ball(rng, 500, 250.0)

    result = velocity_frame_shell_dipole(
        s_centers=s_centers,
        v_centers=v_centers,
        s_tracers=tracers,
        shell_edges=np.array([5.0, 15.0]),
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )

    assert result.per_center_axis_angle.shape == (result.n_centers,)
    assert np.all(result.per_center_axis_angle >= 0.0)
    assert np.all(result.per_center_axis_angle <= np.pi / 2.0 + 1e-9)

    # Independent reconstruction: cos(delta) == |u| / |v|, exactly.
    n_hat_V = unit_vector(s_centers - observer)
    u = np.einsum("ij,ij->i", v_centers, n_hat_V)
    speed = np.linalg.norm(v_centers, axis=1)
    expected = np.arccos(np.clip(np.abs(u) / speed, -1.0, 1.0))
    np.testing.assert_allclose(result.per_center_axis_angle, expected, atol=1e-9)

    # Not a vacuous bound: against the UNSIGNED line of sight this same batch
    # would put centers past a right angle, which is the reading being fixed.
    unsigned = np.arccos(np.clip(u / speed, -1.0, 1.0))
    assert np.any(unsigned > np.pi / 2.0)
