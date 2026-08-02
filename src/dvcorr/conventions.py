"""
conventions.py
---------------
Single source of truth for the FROZEN conventions of this project.

This is the simulation-validation arm of the Tempel x CF4 analysis: we
cross-correlate halo peculiar velocities with a halo density field inside the
MDPL2 box, as a controlled reproduction of the velocity-density dipole
estimator of Nusser (2017). The point of running it in a simulation is that
the answer is known in advance, so every sign and every geometric definition
can be checked against a case where we cannot be fooled.

Everything in this module is load-bearing. A silent change to any constant
here does not raise an error anywhere downstream -- it changes the sign or the
normalization of the final science number. Change these only deliberately,
and re-run `pytest` (the sign gate in tests/test_geometry.py) afterwards.


The conventions, in prose
=========================

Pair separation vector.
    For a density/Tempel-like object T and a velocity object V,

        r = s_V - s_T

    The vector points FROM the density object TO the velocity object. T is the
    center; V is the neighbor. Every pair in the project is oriented this way,
    including in the box, where s_T and s_V are periodic box coordinates and the
    difference must be taken with the minimum-image convention (see below).

Angular cosine.
    mu = n_hat_T . r_hat

    n_hat_T is the observer -> central-object radial direction, i.e. the line of
    sight of the CENTRAL (density) object, not of the neighbor. mu is therefore
    a group-centered cosine: mu = +1 puts the neighbor directly behind the
    center as seen by the observer, mu = -1 puts it directly in front.

Velocity-object polar axis (z_hat).
    For any construction that puts a VELOCITY object at the center of a shell
    and expands the surrounding density field in multipoles, the polar axis is
    the direction that object is MOVING, not the direction it lies in:

        u      = v . n_hat_V                  (signed radial peculiar velocity)
        z_hat  = sign(u) * n_hat_V

    A halo receding from the observer has z_hat = +n_hat_V; a halo APPROACHING
    the observer has z_hat = -n_hat_V. The line of sight n_hat_V only fixes the
    axis up to a sign; the peculiar velocity is what picks the sign, and the
    dipole is a statement about where density sits relative to the FLOW ("the
    overdensity lies ahead of the motion"), which is meaningless against an
    unsigned axis.

    The companion scalar weight is therefore the radial SPEED |u|, never the
    signed u. The sign has been moved out of the scalar and into the axis;
    applying it in both places multiplies it in twice, and sum |u| * A(z_hat)
    would then collapse to noise -- the u <-> A sign correlation is exactly
    what makes the dipole survive the stack. One definition site:
    `dvcorr.geometry.radial_flow_axis`.

    The observer-free variant (`dvcorr.estimators.velocity_frame_dipole`) is
    the same statement with the full 3-D velocity in place of its radial
    projection: z_hat = v / |v|, weight |v|. In the distant-observer limit of a
    purely radial flow the two coincide, sign included.

Reference frame.
    CMB frame throughout. In the box this means halo velocities are used as
    stored, with no observer-motion subtraction. The Local-Group-frame transform
    (subtracting the observer halo's own velocity) is a deliberately separate,
    later step -- it is a different statistic, not a cosmetic reframing, because
    it removes the observer's own bulk motion from every u.

Legendre polynomials.
    Written L_ell. L_0 = 1, L_1 = mu, L_2 = (3 mu^2 - 1)/2. The symbol P is
    RESERVED for the matter power spectrum P_m(k) and is never used for Legendre
    polynomials anywhere in this project. This is not stylistic: the dipole
    prediction contains both objects in the same expression.

Sign convention (the one that bites).
    Coherent infall onto the central object produces a NEGATIVE dipole.

    The reason is the sign of the product u * L_1(mu), not the sign of mu alone.
    mu is a geometric cosine and is positive over half the shell by construction.
    What is negative over the WHOLE shell is the product:

      - Far side (mu > 0): the neighbor lies beyond the center and falls back
        toward it, i.e. toward the observer, so its observer-centered radial
        velocity is u < 0. Product u*mu < 0.
      - Near side (mu < 0): the neighbor lies in front of the center and falls
        away from the observer, so u > 0. Product u*mu < 0.

    Both hemispheres contribute with the same sign; that is exactly why the
    dipole survives the shell average while the monopole cancels. Under this
    project's orientation r = s_V - s_T with r_hat pointing outward from the
    center, the infall profile v_inf(r) is itself negative, and

        xi_Tu,1(r) = xi_Tvr(r) = v_inf(r) < 0.

    Reversing the pair vector (writing r = s_T - s_V) sends mu -> -mu and
    therefore flips the sign of ALL odd multipoles while leaving even ones
    untouched. Nothing crashes; the dipole simply comes back positive and the
    growth-rate constraint comes out with the wrong sign. This is the single
    most likely silent error in the project.

    Note on the literature: Nusser (2017) centers on the VELOCITY object and
    takes the density dipole in shells around it, i.e. the opposite role
    assignment. That is the same two-point information with the separation
    reversed, so his dipole and ours differ by a factor (-1)^ell -- monopole
    unchanged, dipole flipped. When comparing to his figures, apply that factor
    explicitly rather than adjusting a sign somewhere in the estimator.

Observer position.
    A single module-level constant, OBSERVER_POSITION, default = box center.
    There is exactly one observer per analysis run. Everything observer-centered
    (n_hat_T, the radial velocity u) is defined relative to it, so it must not
    be passed around as an ad-hoc argument with its own default in three
    different modules.

Periodic boundary conditions.
    Inherited hard rule from the bulk-flow project: the box is periodic, and
    every spatial calculation -- separations, neighbor queries, lines of sight
    -- must use the minimum-image convention. A plain Euclidean difference on
    box coordinates is a bug even when it does not look like one. The corollary
    is MAX_ANALYSIS_RADIUS: minimum-image geometry is only well defined out to
    half the box, so no shell may extend beyond BOX_SIZE / 2.

Units.
    Positions: comoving h^-1 Mpc. Velocities: km/s. Masses: h^-1 M_sun.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Simulation box (MDPL2)
# --------------------------------------------------------------------------

#: Comoving side length of the periodic simulation box, h^-1 Mpc.
BOX_SIZE: float = 1000.0

#: Reduced Hubble parameter of the MDPL2 cosmology (H0 = 67.77 km/s/Mpc).
HUBBLE_PARAM: float = 0.6777

#: Largest pair separation for which minimum-image geometry is unambiguous.
#: Shell edges must not exceed this. Derived, not tunable.
MAX_ANALYSIS_RADIUS: float = BOX_SIZE / 2.0

#: MDPL2 dark-matter particle mass, h^-1 M_sun (3840^3 particles in a
#: 1000 h^-1 Mpc box). Rockstar's `mvir` is EXACTLY quantized in multiples of
#: this -- every mass in the catalog satisfies mvir = N_particles * PARTICLE_MASS
#: with integer N -- so `mvir / PARTICLE_MASS` recovers the particle count
#: exactly rather than approximately. That is what makes the particle-count
#: axis on the mass-funnel histogram (dvcorr.pipeline.mass_diagnostics) a
#: relabelling of the mass axis rather than a second, lossy estimate.
PARTICLE_MASS: float = 1.5054e9

#: Particle count below which a Rockstar halo's mass is not considered
#: resolved. The conventional completeness floor for this kind of catalog.
RESOLVED_PARTICLE_COUNT: int = 20

#: Particle count below which a halo's *velocity* is not considered converged.
#: Higher than RESOLVED_PARTICLE_COUNT because velocity converges more slowly
#: with particle number than mass does -- a halo can have a usable mass and an
#: unusable peculiar velocity. Both are drawn on the mass-funnel histogram so
#: the population entering the estimator can be read against them.
CONVERGED_PARTICLE_COUNT: int = 100


# --------------------------------------------------------------------------
# Observer
# --------------------------------------------------------------------------

#: The single observer for an analysis run, in box coordinates (h^-1 Mpc).
#: Default is the box center. Read-only: n_hat_T and u are defined relative to
#: this, so an in-place edit would silently redefine the whole statistic.
OBSERVER_POSITION: np.ndarray = np.array(
    [BOX_SIZE / 2.0, BOX_SIZE / 2.0, BOX_SIZE / 2.0], dtype=float
)
OBSERVER_POSITION.flags.writeable = False


# --------------------------------------------------------------------------
# Reference frame
# --------------------------------------------------------------------------

#: Velocity reference frame. "cmb" uses halo velocities as stored; "lg" would
#: subtract the observer halo's own velocity. Only "cmb" is implemented.
REFERENCE_FRAME: str = "cmb"


# --------------------------------------------------------------------------
# Pair-geometry orientation
# --------------------------------------------------------------------------

#: Documented orientation of the pair separation vector. Constant so that the
#: convention is greppable and assertable, not only readable in a docstring.
PAIR_SEPARATION_CONVENTION: str = "r = s_V - s_T  (density/Tempel object -> velocity object)"

#: Documented definition of the angular cosine.
MU_CONVENTION: str = "mu = n_hat_T . r_hat  (n_hat_T = observer -> central object)"

#: Documented definition of the velocity-object polar axis, and of the scalar
#: that travels with it. See the prose section above; implemented once, in
#: `dvcorr.geometry.radial_flow_axis`.
VELOCITY_AXIS_CONVENTION: str = (
    "z_hat = sign(u) n_hat_V, weight |u|  (axis along the radial peculiar "
    "velocity, not along the line of sight)"
)

#: Expected sign of the measured dipole under coherent infall. The sign gate in
#: tests/test_geometry.py asserts against this rather than a bare literal.
INFALL_DIPOLE_SIGN: int = -1

#: Factor relating this project's group-centered multipoles to the
#: velocity-centered multipoles of Nusser (2017): xi_Tu,ell = (-1)^ell xi_uT,ell.
def nusser_multipole_sign(ell: int) -> int:
    """Return (-1)^ell, the factor converting Nusser (2017) multipoles to ours.

    Parameters
    ----------
    ell : int
        Multipole order.

    Returns
    -------
    int
        +1 for even ell (monopole unchanged), -1 for odd ell (dipole flipped).
    """
    return -1 if ell % 2 else 1


# --------------------------------------------------------------------------
# Catalog columns
# --------------------------------------------------------------------------

#: Semantic name -> column name in the MDPL2/Rockstar halo dataframe.
#: Code addresses halo properties through these keys only; a catalog with
#: different headers is handled by remapping here, never by renaming in place.
HALO_COLUMNS: dict[str, str] = {
    "id": "rockstarid",
    "parent_id": "pid",
    "mass": "mvir",
    "radius": "rvir",
    "scale_radius": "rs",
    "x": "x",
    "y": "y",
    "z": "z",
    "vx": "vx",
    "vy": "vy",
    "vz": "vz",
}

#: Convenience groupings, so downstream code never hardcodes ("x", "y", "z").
POSITION_COLUMNS: tuple[str, str, str] = (
    HALO_COLUMNS["x"],
    HALO_COLUMNS["y"],
    HALO_COLUMNS["z"],
)
VELOCITY_COLUMNS: tuple[str, str, str] = (
    HALO_COLUMNS["vx"],
    HALO_COLUMNS["vy"],
    HALO_COLUMNS["vz"],
)
