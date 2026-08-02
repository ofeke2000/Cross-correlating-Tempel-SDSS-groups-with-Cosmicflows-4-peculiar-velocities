"""
shell_dipole.py
---------------
Shell-binned monopole and dipole around a single central object.

WHICH ESTIMATOR THIS IS (read before comparing to Nusser)
---------------------------------------------------------
This computes the GROUP-CENTRED density-velocity correlation xi_Tu: the density
object T sits at the center and the VELOCITY objects V are the neighbors,
entering through per-neighbor weights w_i = u_i (their radial peculiar
velocities). It is NOT Nusser (2017)'s velocity-centered estimator (his eq. 23-24,
sometimes written zeta), where a single velocity object sits at the center with a
scalar velocity u_alpha, the DENSITY tracers are counted in the shell as
Q_alpha = sum_beta w_beta mu_alpha,beta, and the statistic is Q_alpha * u_alpha.
The two are the same two-point information with the separation reversed and are
related multipole by multipole by

    xi_Tu,ell = (-1)^ell * xi_uT,ell          (conventions.nusser_multipole_sign)

so their monopoles agree and their dipoles differ in sign. The distinction is
not cosmetic here: in the group-centered form the velocity is a PER-NEIGHBOUR
weight (there are N of them), whereas a SCALAR central velocity would be correct
only for the velocity-centered form. A scalar central velocity applied to this
construction vanishes on an isotropic shell (sum mu ~ 0) and is a bug, not a null
result. The project's methodology note names xi_Tu as the primary target; this
module is that target, and per-neighbor weights generalize cleanly to
inverse-variance weighting later.

What it accumulates
-------------------
For the central density object T we bin its neighbors into concentric radial
shells and accumulate, per shell b, the two Legendre moments of the neighbor
weights, alongside the raw pair count:

    pair_count(r_b) = sum over neighbors in shell b of  1        (the pair count)
    monopole(r_b)   = sum over neighbors in shell b of  w_i      (the L_0 moment)
    dipole(r_b)     = sum over neighbors in shell b of  w_i mu_i (the L_1 moment)

with L_0(mu) = 1 and L_1(mu) = mu. All three are RAW sums: nothing here is
divided by a pair count, a shell volume, or a number density. Normalisation is
a deliberate downstream physics choice (see CLAUDE.md hard rule), so the raw
numerators and the raw pair count are returned side by side for the caller to
combine explicitly. The usual combination is

    xi_Tu,0(r_b) = monopole(r_b) / pair_count(r_b)     (mean radial velocity)
    xi_Tu,1(r_b) = 3 * dipole(r_b) / pair_count(r_b)    (the 3 = 2*ell + 1)

pair_count is Sigma 1 (pure geometry / occupancy) while monopole is Sigma w_i.
They coincide only when the weights are uniform (w_i = 1); once w_i = u_i the
monopole is the velocity monopole Sigma u, a different quantity from the count,
which is exactly why both are kept.

Why the monopole travels with the dipole
-----------------------------------------
The velocity monopole (Sigma u, normalized by pair_count) is not a by-product to
be discarded -- it is the geometry diagnostic (CLAUDE.md hard rule 6). In an
ideal spherical, angularly complete shell around a static center it vanishes by
the near/far cancellation, and any departure measures the same effects that also
bias the dipole: finite-distance leakage of order (2r / 3R) v_inf, incomplete
angular coverage of the shell, and residual bulk motion of the central object.
Reporting a dipole without its companion monopole hides the one number that says
whether the dipole is trustworthy, so this function makes returning one without
the other impossible.

Meaning of the weights
----------------------
The same accumulator serves two statistics, distinguished only by w_i:

  - w_i = 1 (uniform, the default): the dipole is the geometric dipole moment
    of the neighbor distribution in the shell -- Nusser's Q_alpha, the
    density-side quantity. Around a random center it averages to zero; a
    non-zero value means the neighbors are distributed anisotropically about
    the line of sight. With uniform weights monopole == count.

  - w_i = u_i (the neighbor's observer-centered radial peculiar velocity): the
    dipole is the unnormalized numerator of the density-velocity dipole
    xi_Tu,1(r_b). This is the science signal, and its monopole Σu_i is the
    finite-distance leakage / bulk-motion diagnostic above.

Sign
----
Under the frozen convention (dvcorr.conventions) r points outward from the central
object and mu = n_hat_T . r_hat. For coherent infall, u and mu have opposite
signs on both the near and the far side of the shell, so every pair contributes
u * mu < 0 and the dipole is NEGATIVE. If a validated infall mock returns a
positive dipole, the bug is an orientation flip in `pair_separation`, not a
physics result.

Periodic boundary conditions
----------------------------
This function does NOT wrap coordinates. Consistent with `geometry.py`, the
periodic minimum image is resolved one level up, where the neighbor sub-volume
around each center is carved and unwrapped into that center's local continuous
frame. The neighbors handed in must already be the images nearest s_center.
The only periodicity obligation enforced here is that the outermost shell edge
must not exceed conventions.MAX_ANALYSIS_RADIUS, beyond which the nearest image --
and hence the carving itself -- is not unique.

Nusser (2017) centers on the velocity object rather than the density object, so
his dipole is the negative of this one; see conventions.nusser_multipole_sign before
comparing amplitudes.


VELOCITY-CENTRED ESTIMATOR (zeta) -- the production target
------------------------------------------------------------
This module also implements Nusser (2017)'s eq. 23-24 velocity-centered
statistic zeta directly, alongside xi_Tu above, rather than only describing
it. A single VELOCITY object alpha sits at the center, carrying its own
scalar observer-frame radial velocity u_alpha = v_alpha . n_hat_V,alpha
(n_hat_V,alpha = observer -> alpha), and DENSITY tracers i are counted in
shells around it. The polar axis of that center's expansion is its
DIRECTION OF MOTION, not its direction from the observer
(conventions.VELOCITY_AXIS_CONVENTION):

    z_hat_alpha = sign(u_alpha) * n_hat_V,alpha

so a center approaching the observer is axed on -n_hat_V,alpha. Per shell b,
the accumulator is the real l=1, m=0 spherical harmonic summed over that
shell's tracers,

    A_alpha,b = sum_{i in b} Y_10(n_hat_i)
              = sqrt(3/4pi) * sum_{i in b} (z_hat_alpha . n_hat_i)

with n_hat_i the unit vector FROM the center alpha TO tracer i (center ->
tracer -- see Orientation below), and the statistic stacked over centers is
Sigma_alpha |u_alpha| * A_alpha,b -- the radial SPEED, because the sign has
been moved into z_hat_alpha and entering it twice would cancel the signal
(`velocity_centered_shell_dipole`, `real_y10`). Since flipping z_hat flips
A, |u_alpha| * A(z_hat_alpha) == u_alpha * A(n_hat_V,alpha) identically, so
the stacked dipole is numerically the same as the unsigned-axis reading; the
monopole, the per-center amplitudes, and the shuffle null are not (see
`velocity_centered_shell_dipole`'s "Axis sign" section). CF4 will eventually
supply the velocity centers alpha (its groups' peculiar velocities); that is
why this estimator, not xi_Tu above, is the production target -- xi_Tu is the
simulation-validation cross-check.

Orientation: the REVERSED separation, not the frozen r
----------------------------------------------------------
dvcorr.conventions freezes r = s_V - s_T (density -> velocity) and mu = n_hat_T . r_hat
(n_hat_T = observer -> density object). NEITHER symbol is redefined here, or
anywhere. This velocity-centered path instead reuses the same primitives on a
deliberately different construction:

  - the REVERSED separation, center -> tracer. For the physical pair (density
    tracer, velocity center alpha), the frozen orientation would put the
    density object in the center slot and point r = s_V - s_T outward from
    it; here the VELOCITY object is the center, so the arrow direction flips
    -- it is the exact negative of the frozen r for the same pair;
  - the reference direction z_hat_alpha = sign(u_alpha) * n_hat_V,alpha (the
    center's own direction of motion along its line of sight) in place of
    n_hat_T (the density object's line of sight).

Both substitutions are deliberate, and together are what "velocity-centered"
means. The resulting local cosine is therefore NOT conventions.MU_CONVENTION's
frozen mu -- same primitives (`pair_separation`, `unit_vector`, `mu_cosine`),
different orientation and different reference direction, by design. See
`velocity_centered_shell_dipole`'s docstring for the exact construction.

Multipole relation and its sign consequence
----------------------------------------------
The two constructions are the same two-point information with the pair
reversed, so multipole by multipole

    zeta_ell = (-1)**ell * xi_Tu,ell          (conventions.nusser_multipole_sign)

monopoles agree (ell even); dipoles are opposite in sign (ell odd). Hard rule
2 says coherent infall gives a NEGATIVE group-centered xi_Tu,1; it follows that
coherent infall gives a POSITIVE velocity-centered zeta_1. Do not "fix" a
positive zeta_1 from an infall mock -- that positive sign is the CORRECT
result of this estimator, and a negative one from infall is the orientation
bug. See the joint sign gate in tests/test_velocity_centered_dipole.py, which
runs both estimators on the identical toy infall configuration and asserts
the opposite signs explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from dvcorr import conventions
from dvcorr.geometry import mu_cosine, pair_separation, radial_flow_axis, unit_vector

#: Centers per batched `query_ball_point` call. Batching is what lets SciPy
#: use its `workers` thread pool at all: a single-point query is serial no
#: matter what, so the per-center loop this replaces could only ever use one
#: core. Memory scales as batch size x neighbors-per-ball, and on the full
#: MDPL2 catalog a ball at r_max = 64 holds ~1.4e5 tracers, so a batch of 32
#: costs a few million indices -- large enough to saturate the pool, small
#: enough not to rival the tree itself.
_NEIGHBOR_BATCH_CENTERS: int = 32

#: `workers` argument for batched neighbor queries; -1 means every core.
_NEIGHBOR_QUERY_WORKERS: int = -1


def _neighbors_by_center(
    tree: cKDTree | None, centers: np.ndarray, radius: float
):
    """Yield `(center_row, neighbor_indices)` for every center, batched.

    A generator so callers keep their per-center loop body unchanged while the
    underlying queries run in parallel batches. Yields exactly one pair per row
    of `centers`, in row order, including empty results -- so a caller can rely
    on the row index rather than tracking its own counter.

    Parameters
    ----------
    tree : cKDTree | None
        Tracer tree. None (no tracers) yields an empty index array per center,
        so the caller needs no separate None branch.
    centers : ndarray, shape (N_c, 3)
        Query points.
    radius : float
        Ball radius, h^-1 Mpc -- the outermost shell edge.

    Yields
    ------
    center_row : int
        Row of `centers` the indices belong to.
    neighbor_indices : ndarray, shape (k,), int
        Rows of the tree's data within `radius`. Unsorted: callers here bin by
        separation and sum, and neither depends on neighbor order.
    """
    n_centers = centers.shape[0]
    if tree is None:
        empty = np.empty(0, dtype=int)
        for a in range(n_centers):
            yield a, empty
        return

    for start in range(0, n_centers, _NEIGHBOR_BATCH_CENTERS):
        stop = min(start + _NEIGHBOR_BATCH_CENTERS, n_centers)
        batch = tree.query_ball_point(
            centers[start:stop],
            radius,
            workers=_NEIGHBOR_QUERY_WORKERS,
            return_sorted=False,
        )
        for offset, idx in enumerate(batch):
            yield start + offset, np.asarray(idx, dtype=int)


@dataclass(frozen=True)
class ShellDipoleResult:
    """Per-shell Legendre moments of the neighbor weights, all raw sums.

    Every array is aligned bin-for-bin and has length B = len(shell_edges) - 1,
    so the caller may combine them elementwise. Nothing is normalized.

    Attributes
    ----------
    shell_edges : ndarray, shape (B + 1,)
        The radial bin boundaries passed in, h^-1 Mpc.
    shell_centers : ndarray, shape (B,)
        Midpoints 0.5 * (edge_low + edge_high) of each shell, h^-1 Mpc. A
        plain midpoint, not a pair-count-weighted effective radius.
    pair_count : ndarray, shape (B,)
        Raw pair count per shell, sum of 1 over neighbors in the shell. Pure
        geometry / occupancy, and the natural normalization denominator for both
        multipoles. Empty shells are 0, never NaN.
    monopole : ndarray, shape (B,)
        L_0 moment, sum of w_i over the shell. Equals `pair_count` when weights
        are uniform; equals the velocity monopole Σu_i -- the finite-distance
        (2r/3R) leakage / bulk-motion diagnostic, the numerator of xi_Tu,0 --
        when weights are the radial velocities. Empty shells are 0.0.
    dipole : ndarray, shape (B,)
        L_1 moment, sum of w_i * mu_i over the shell -- the unnormalized
        numerator of xi_Tu,1. Empty shells are 0.0, never NaN.

    Notes
    -----
    Downstream normalization divides `dipole` (and `monopole`) by `pair_count`,
    so it MUST guard against empty shells: pair_count == 0 gives dipole == 0.0
    here, and 0.0 / 0 is a NaN the caller has to mask, not a value this function
    can invent.
    """

    shell_edges: np.ndarray
    shell_centers: np.ndarray
    pair_count: np.ndarray
    monopole: np.ndarray
    dipole: np.ndarray


def shell_dipole(
    s_center: np.ndarray,
    s_neighbors: np.ndarray,
    shell_edges: np.ndarray,
    weights: np.ndarray | None = None,
    observer: np.ndarray | None = None,
) -> ShellDipoleResult:
    """Accumulate the monopole and dipole of neighbors in radial shells.

    Parameters
    ----------
    s_center : ndarray, shape (3,)
        Position of the central density object s_T, comoving coordinates,
        h^-1 Mpc, in the same unwrapped frame as `s_neighbors`.
    s_neighbors : ndarray, shape (N, 3)
        Positions of the N neighbor (velocity) objects s_V, same units and
        frame. Neighbors outside [shell_edges[0], shell_edges[-1]] are
        excluded, not an error.
    shell_edges : ndarray, shape (B + 1,)
        Strictly increasing shell boundaries in h^-1 Mpc, defining B shells.
        Bins are left-closed / right-open, [edge_low, edge_high), with the
        outermost edge included in the last shell. The outermost edge must not
        exceed conventions.MAX_ANALYSIS_RADIUS.
    weights : ndarray, shape (N,), optional
        Per-neighbor weight w_i. None (the default) means uniform weights of
        1, giving the purely geometric shell dipole. Pass the radial peculiar
        velocities u_i to obtain the numerator of xi_Tu,1. Must be finite: a
        missing peculiar velocity is missing data (CLAUDE.md hard rule 5) and
        the object must be dropped by the caller, never entered as u = 0 -- so
        a non-finite weight is a ValueError here, not a silent zero.
    observer : ndarray, shape (3,), optional
        Observer position defining n_hat_T = observer -> s_center. Defaults to
        conventions.OBSERVER_POSITION. There is one observer per run; pass this
        explicitly only in tests.

    Returns
    -------
    ShellDipoleResult
        Struct of raw per-shell sums (pair_count, monopole, dipole) plus the
        shell geometry (edges, centers). See the class docstring; monopole and
        dipole are always returned together (hard rule 6).

    Raises
    ------
    ValueError
        If shell_edges is not one-dimensional with at least two entries, is not
        strictly increasing, or has an outermost edge exceeding
        conventions.MAX_ANALYSIS_RADIUS; or if `weights` is given with a shape that
        does not match s_neighbors, or contains a non-finite value.

    Notes
    -----
    A neighbor coincident with the center (r_mag ~ 0) has no direction:
    `unit_vector` returns a zero row, so its mu is 0 and it contributes its
    weight to `count` and `monopole` but nothing to `dipole`. It is only
    reached at all when shell_edges[0] == 0; otherwise it falls below the
    innermost edge and is excluded. Either way it never injects a NaN.

    Fully vectorized: the binning is a single `np.digitize` plus weighted
    `np.bincount` calls, with no Python-level loop over neighbors.
    """
    observer = (
        conventions.OBSERVER_POSITION if observer is None
        else np.asarray(observer, dtype=float)
    )

    edges = np.asarray(shell_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("shell_edges must be 1-D with at least two entries.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("shell_edges must be strictly increasing.")
    if edges[-1] > conventions.MAX_ANALYSIS_RADIUS:
        raise ValueError(
            f"outermost shell edge {edges[-1]} exceeds "
            f"conventions.MAX_ANALYSIS_RADIUS = {conventions.MAX_ANALYSIS_RADIUS}."
        )
    n_bins = edges.size - 1

    s_center = np.asarray(s_center, dtype=float)
    s_neighbors = np.atleast_2d(np.asarray(s_neighbors, dtype=float))
    n_neighbors = s_neighbors.shape[0]

    if weights is None:
        w = np.ones(n_neighbors, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_neighbors,):
            raise ValueError(
                f"weights shape {w.shape} does not match "
                f"({n_neighbors},) neighbors."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "weights contains a non-finite value; a missing peculiar "
                "velocity is missing data and the object must be dropped by "
                "the caller, never entered as 0 (CLAUDE.md hard rule 5)."
            )

    # Geometry: line of sight of the CENTRAL object, then the group-centered
    # cosine of each neighbor. r = s_V - s_T points outward from the center.
    n_T_hat = unit_vector(s_center - observer)
    r_vec, r_mag = pair_separation(s_center, s_neighbors)
    mu = mu_cosine(unit_vector(r_vec), n_T_hat)

    # Bin by separation. Left-closed/right-open interior boundaries; the
    # outermost edge is folded into the last shell so a neighbor sitting
    # exactly on it is not silently dropped.
    in_range = (r_mag >= edges[0]) & (r_mag <= edges[-1])
    bin_index = np.digitize(r_mag, edges) - 1
    bin_index = np.where(bin_index == n_bins, n_bins - 1, bin_index)

    b = bin_index[in_range]
    w_in = w[in_range]
    mu_in = mu[in_range]

    # One pass, three weighted histograms sharing the same bin assignment.
    pair_count = np.bincount(b, minlength=n_bins)[:n_bins].astype(float)
    monopole = np.bincount(b, weights=w_in, minlength=n_bins)[:n_bins]
    dipole = np.bincount(b, weights=w_in * mu_in, minlength=n_bins)[:n_bins]

    shell_centers = 0.5 * (edges[:-1] + edges[1:])

    return ShellDipoleResult(
        shell_edges=edges,
        shell_centers=shell_centers,
        pair_count=pair_count,
        monopole=monopole,
        dipole=dipole,
    )


# ---------------------------------------------------------------------------
# Velocity-centered estimator (Nusser 2017 eq. 23-24, "zeta")
# ---------------------------------------------------------------------------
#
# See the module docstring above for the physics, the orientation reversal,
# and the zeta_ell = (-1)**ell * xi_Tu,ell relation. The functions below are
# the production-target implementation; CF4 will eventually supply the
# velocity centers.


def real_y10(direction_cosine: np.ndarray) -> np.ndarray:
    """Real l=1, m=0 spherical harmonic, as a function of the direction cosine.

    Y_10(theta, phi) = sqrt(3 / (4*pi)) * cos(theta) is independent of phi
    (m=0), so it is written directly as a function of cos(theta) rather than
    of the two spherical angles.

    Parameters
    ----------
    direction_cosine : ndarray, shape (N,)
        cos(theta) for N directions, e.g. the output of `mu_cosine`. Scalars
        (0-d arrays / Python floats) pass through fine elementwise, but the
        stable contract downstream is (N,).

    Returns
    -------
    ndarray, shape (N,)
        Y_10 evaluated at each direction cosine.

    Notes
    -----
    sqrt(3 / (4*pi)) is the Y_10 normalization constant -- pure mathematics
    (CLAUDE.md hard rule 4's exemption for constants that are part of a
    derivation), kept inline rather than promoted to dvcorr.conventions or a settings
    dataclass.

    Pinned in tests/test_velocity_centered_dipole.py against
    `scipy.special.sph_harm_y(1, 0, theta, phi)` (m=0 is phi-independent) at
    a handful of angles. Implemented directly with numpy here; scipy is not
    used in this implementation, only in the test that checks it.

    Deliberately l=1, m=0 ONLY: m=+-1 are out of scope and NOT implemented.
    They would require fixing the transverse (x_hat, y_hat) axes of each
    center's rotated frame -- a real complication that the m=0 term never
    needs, since it depends only on z_hat = n_hat_V,alpha and the transverse
    axes are unconstrained and irrelevant to it (see
    `velocity_centered_shell_dipole`'s docstring, "Rotated per-center
    frame").
    """
    direction_cosine = np.asarray(direction_cosine, dtype=float)
    return np.sqrt(3.0 / (4.0 * np.pi)) * direction_cosine  # sqrt(3/4pi): Y_10 normalization


def core_center_mask(
    s_centers: np.ndarray,
    sub_volume_radius: float,
    core_margin: float,
    observer: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask selecting centers whose FULL shell fits inside the sub-volume.

    A candidate center survives when its distance from the observer leaves at
    least `core_margin` of clearance to the sub-volume boundary:

        |s_alpha - observer| <= sub_volume_radius - core_margin

    Parameters
    ----------
    s_centers : ndarray, shape (3,) or (N_c, 3)
        Candidate velocity-object positions, comoving h^-1 Mpc.
    sub_volume_radius : float
        Radius of the spherical sub-volume carved around the observer,
        h^-1 Mpc (e.g. R_sub = 300 in the first MDPL2 run,
        notebooks/04_first_mdpl2_run.ipynb cell 8).
    core_margin : float
        Minimum required clearance to the sub-volume boundary, h^-1 Mpc.
        Typically the outermost shell edge r_max, so a surviving center's
        entire shell -- out to r_max in every direction -- is guaranteed to
        lie inside the carved sub-volume.
    observer : ndarray, shape (3,), optional
        Observer position. Defaults to conventions.OBSERVER_POSITION.

    Returns
    -------
    ndarray, shape (N_c,), bool
        True where the center survives the core cut. Always (N_c,), even for
        a single input center.

    Notes
    -----
    Why this matters: a center near the sub-volume boundary has a shell that
    is truncated by the carve -- tracers beyond the boundary are simply
    absent from the catalog, not merely unobserved. The missing cap is not
    random: it sits systematically at the outward pole of the center's own
    line of sight (large n_hat_V,alpha . n_hat_i), so Sigma Y_10 is pulled
    away from zero by geometry ALONE, correlated with u_alpha through the
    shared n_hat_V,alpha, and this correlation does not average out across
    many boundary centers -- it is the suspected mechanism behind the
    ~+13 km/s null offset seen in the exploratory first MDPL2 run
    (notebooks/04_first_mdpl2_run.ipynb). With R_sub = 300 and
    core_margin = r_max = 64 (dvcorr.pipeline.velocity_centered's default log
    binning), the surviving core ball has radius 236 -- a (236/300)**3 ~=
    49% volume cut, deliberately visible via
    `VelocityCenteredShellDipoleResult.n_candidates` /
    `.n_centers` rather than hidden inside the estimator. Measured on the
    real MDPL2 run: n_candidates = 4000, n_centers = 1933 (48.3% survive).
    """
    observer = (
        conventions.OBSERVER_POSITION if observer is None
        else np.asarray(observer, dtype=float)
    )
    s_centers = np.atleast_2d(np.asarray(s_centers, dtype=float))
    distance = np.linalg.norm(s_centers - observer, axis=-1)
    return distance <= (sub_volume_radius - core_margin)


def expected_shell_occupancy(
    number_density: float,
    shell_edges: np.ndarray,
) -> np.ndarray:
    """Expected tracer occupancy per shell from a uniform global number density.

    n_bar * V_b -- the Nusser (2017) eq. 24 normalization denominator: the
    EXPECTED occupancy from a uniform tracer field over the sub-volume, not
    the realized pair count (`VelocityCenteredShellDipoleResult.pair_count`).
    Dividing the raw stacked dipole by this, rather than by the realized
    count, is what the eq. 24 normalization calls for.

    Parameters
    ----------
    number_density : float
        Global tracer number density n_bar over the sub-volume, tracers per
        (h^-1 Mpc)^3. Computed by the CALLER (e.g.
        N_carved / ((4*pi/3) * R_sub**3)); never estimated inside this
        function.
    shell_edges : ndarray, shape (B + 1,)
        Radial shell boundaries, h^-1 Mpc, matching the binning used by
        `velocity_centered_shell_dipole`.

    Returns
    -------
    ndarray, shape (B,)
        n_bar * V_b per shell, with V_b = (4*pi/3) * (r_out**3 - r_in**3) the
        volume of a spherical shell -- pure mathematics, kept inline with a
        comment (CLAUDE.md hard rule 4's exemption).

    Notes
    -----
    The matching plotting abscissa -- the first moment of this same
    `r**2 dr` weight -- is `dvcorr.config.shells.volume_weighted_shell_radii`.
    """
    edges = np.asarray(shell_edges, dtype=float)
    shell_volume = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)  # 4pi/3: sphere volume
    return number_density * shell_volume


def center_standard_error(per_center_values: np.ndarray) -> np.ndarray:
    """Across-center standard error of the mean, per shell.

    std(per_center_values, axis=0, ddof=1) / sqrt(N_c).

    Parameters
    ----------
    per_center_values : ndarray, shape (N_c, B)
        Any per-center, per-shell quantity -- typically
        `VelocityCenteredShellDipoleResult.per_center_dipole`, optionally
        already rescaled by a normalization factor at the call site (the
        rescaling is linear, so it commutes with the standard-error
        calculation).

    Returns
    -------
    ndarray, shape (B,)
        Standard error per shell. NaN in every shell (never a crash, never a
        silent zero) when N_c < 2 -- a standard error needs at least two
        samples.

    Raises
    ------
    ValueError
        If `per_center_values` is not 2-D. A 1-D `(N_c,)` array is genuinely
        ambiguous here -- it could mean N_c centers in a single shell (the
        `(N_c, 1)` reading) or one center across B shells (the `(1, B)`
        reading) -- so it is rejected rather than silently disambiguated.
        `np.atleast_2d` would pick the `(1, B)` reading (prepending a length-1
        axis), which for a genuine `(N_c,)` input of many centers in one
        shell silently returns an all-NaN result (N_c = 1 by that reading, so
        every shell reports "not enough centers") instead of the intended
        standard error. Reshape explicitly at the call site --
        `values[:, None]` for "N_c centers, one shell" or `values[None, :]`
        for "one center, B shells" -- rather than relying on this function to
        guess.

    Notes
    -----
    **This treats centers as independent samples, which they are not.** Two
    centers closer together than a shell's radius share tracers, and on
    scales approaching the typical inter-center separation the per-center
    measurements are correlated through shared large-scale velocity
    coherence. The independence assumption therefore UNDERESTIMATES the true
    uncertainty -- more so on large scales, but everywhere at some level.
    This is a diagnostic error bar ("how noisy is the stack"), not a
    calibrated confidence interval; a proper covariance, including
    shell-to-shell and center-to-center off-diagonal terms, needs mock
    catalogs (the not-yet-built `mocks/` arm), which is the eventual
    replacement.
    """
    values = np.asarray(per_center_values, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"per_center_values must be 2-D with shape (N_c, B), got ndim="
            f"{values.ndim} (shape {values.shape}). See this function's "
            "Raises docstring: a 1-D array is ambiguous, not auto-promoted."
        )
    n_centers = values.shape[0]
    if n_centers < 2:
        return np.full(values.shape[1], np.nan)
    return values.std(axis=0, ddof=1) / np.sqrt(n_centers)


@dataclass(frozen=True)
class VelocityCenteredShellDipoleResult:
    """Per-shell raw sums of the velocity-centered (Nusser eq. 23-24) statistic.

    Mirrors `ShellDipoleResult` in spirit -- everything raw, nothing
    normalized -- but stacked over MULTIPLE centers (one per surviving
    velocity object) rather than describing a single central object's
    neighbors, and carries the per-center breakdown needed for an error
    band and a shuffle null without a second estimator pass.

    Attributes
    ----------
    shell_edges : ndarray, shape (B + 1,)
        The radial bin boundaries passed in, h^-1 Mpc.
    shell_centers : ndarray, shape (B,)
        Midpoints 0.5 * (edge_low + edge_high) of each shell, h^-1 Mpc.
    pair_count : ndarray, shape (B,)
        Realised Sigma_alpha N_alpha,b, occupancy summed over centers. Pure
        geometry, the realized (not expected) denominator; contrast
        `expected_shell_occupancy`'s n_bar * V_b. Empty shells are 0.0.
    monopole : ndarray, shape (B,)
        Sigma_alpha |u_alpha| * N_alpha,b, the l=0 companion (CLAUDE.md hard
        rule 6): the SPEED-weighted occupancy. Because the weight is |u| and
        not the signed u (conventions.VELOCITY_AXIS_CONVENTION -- the sign
        lives in z_hat now), this monopole sits at roughly the mean radial
        SPEED <|u|>, not near zero: a positive-definite weight has no
        near/far cancellation available to it. It is still the
        incomplete-shell / occupancy diagnostic -- on a clustered tracer
        field it inherits the 1 + xi_hh(r) occupancy shape, so read it as
        "what survives after the occupancy ratio is divided out", exactly as
        `dvcorr.estimators.velocity_frame_dipole`'s Monopole section
        describes for the observer-free frame. Empty shells are 0.0.
    dipole : ndarray, shape (B,)
        Sigma_alpha |u_alpha| * A_alpha,b, the raw stacked l=1 numerator --
        zeta_1's unnormalized numerator. Empty shells are 0.0, never NaN.
        Numerically UNCHANGED by the signed axis: |u_alpha| * A(z_hat) ==
        u_alpha * A(n_hat_V), since flipping z_hat flips A. The signed axis
        is a correction to what the statistic MEANS (and to the monopole and
        the null), not to this curve.
    per_center_dipole : ndarray, shape (N_c, B)
        |u_alpha| * A_alpha,b, retained per surviving center per shell.
        `dipole == per_center_dipole.sum(axis=0)` by construction; kept so
        `center_standard_error` and a shuffle null can be built without
        re-running the estimator.
    per_center_amplitude : ndarray, shape (N_c, B)
        A_alpha,b = Sigma_i Y_10(n_hat_i) per surviving center per shell --
        the pure-geometry accumulator, before the |u_alpha| weight is
        applied. Measured against z_hat_alpha = sign(u_alpha) * n_hat_V,alpha,
        so its sign is relative to the center's own direction of MOTION.
        Recombining `per_center_speed[:, None] * per_center_amplitude` and
        summing over centers reproduces `dipole`.

        A shuffle null must NOT simply permute a scalar against this array:
        the axis is now derived from the same u_alpha the weight comes from,
        so permuting |u| alone would leave every A_alpha,b (and therefore the
        alignment that produced the signal) untouched -- the same trap
        `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null`
        documents for the velocity frame. Undo the flip first,
        `np.sign(per_center_u)[:, None] * per_center_amplitude`, to recover
        the fixed-axis amplitude, then permute the SIGNED per_center_u
        against it -- which re-randomizes the axis and the weight together.
        See `dvcorr.pipeline.velocity_centered.normalize_result`.
    per_center_count : ndarray, shape (N_c, B)
        N_alpha,b, the realized tracer occupancy per surviving center per
        shell. `pair_count == per_center_count.sum(axis=0)` by construction.
        Pure geometry: unaffected by the axis sign.
    per_center_u : ndarray, shape (N_c,)
        SIGNED u_alpha = v_alpha . n_hat_V,alpha for each surviving center,
        km/s. Retained signed (not as the |u| the estimator weights by)
        because its sign is the only record of which way z_hat_alpha was
        flipped, which a null test needs in order to undo the flip.
    per_center_speed : ndarray, shape (N_c,)
        |u_alpha|, the radial SPEED actually used as the per-center weight,
        km/s. Positive-definite. Mirrors
        `dvcorr.estimators.velocity_frame_dipole.VelocityFrameShellDipoleResult
        .per_center_speed`, which holds the full |v_alpha| for the
        observer-free frame.
    n_candidates : int
        Number of candidate centers passed in, before the core cut.
    n_centers : int
        Number of surviving centers, N_c, after `core_center_mask`. The
        `n_centers / n_candidates` ratio makes the core cut's volume loss
        (~49% for R_sub = 300, core_margin = r_max = 64, measured 48.3% --
        n_candidates = 4000, n_centers = 1933 -- on the real MDPL2 run)
        visible rather than silent.

    Notes
    -----
    Cross-check invariants that hold by construction:
    `dipole == per_center_dipole.sum(axis=0)`,
    `pair_count == per_center_count.sum(axis=0)`,
    `monopole == (per_center_speed[:, None] * per_center_count).sum(axis=0)`,
    `per_center_speed == np.abs(per_center_u)`.
    """

    shell_edges: np.ndarray
    shell_centers: np.ndarray
    pair_count: np.ndarray
    monopole: np.ndarray
    dipole: np.ndarray
    per_center_dipole: np.ndarray
    per_center_amplitude: np.ndarray
    per_center_count: np.ndarray
    per_center_u: np.ndarray
    per_center_speed: np.ndarray
    n_candidates: int
    n_centers: int


def velocity_centered_shell_dipole(
    s_centers: np.ndarray,
    v_centers: np.ndarray,
    s_tracers: np.ndarray,
    shell_edges: np.ndarray,
    sub_volume_radius: float,
    core_margin: float | None = None,
    observer: np.ndarray | None = None,
) -> VelocityCenteredShellDipoleResult:
    """Stack the velocity-centered zeta_1 statistic (Nusser 2017 eq. 23-24)
    over many velocity-object centers.

    For each surviving center alpha: n_hat_V,alpha = observer -> s_alpha,
    u_alpha = v_alpha . n_hat_V,alpha, and the polar axis

        z_hat_alpha = sign(u_alpha) * n_hat_V,alpha

    -- the direction the center is MOVING along its line of sight, not the
    line of sight itself (conventions.VELOCITY_AXIS_CONVENTION; one
    definition site, `dvcorr.geometry.radial_flow_axis`). For each shell b,
    sum the real l=1, m=0 spherical harmonic over the tracers in that shell,

        A_alpha,b = sum_{i in b} Y_10(n_hat_i)
                  = sqrt(3/4pi) * sum_{i in b} (z_hat_alpha . n_hat_i)

    with n_hat_i the unit vector FROM alpha TO tracer i (see Orientation
    below), then accumulate |u_alpha| * A_alpha,b -- the SPEED, since the
    sign is already carried by z_hat_alpha -- stacked over centers.

    Axis sign -- why the plotted dipole does not move, and what does
    ------------------------------------------------------------------
    Flipping z_hat_alpha flips A_alpha,b, and the weight |u_alpha| carries
    the matching flip, so

        |u_alpha| * A(z_hat_alpha) == u_alpha * A(n_hat_V,alpha)

    identically, center by center. `dipole` (and hence zeta_hat_1) is
    therefore NUMERICALLY UNCHANGED by the signed axis. That invariance is
    the point, not a reason to skip the change: it says the two readings
    agree on the science curve, and it is what makes the sign gate below
    still valid. What DOES change is everything the axis sign was previously
    smuggled into:

      - `monopole` becomes Sigma |u_alpha| N_alpha,b, sitting near <|u|>
        instead of near zero (see the dataclass docstring);
      - `per_center_amplitude` is now measured against the flow direction,
        so its sign is physical rather than an artifact of which side of the
        observer the center happens to sit on;
      - the shuffle null must permute the SIGNED u_alpha after undoing the
        flip, since permuting a positive-definite weight against an
        unchanged amplitude is not a null at all (dataclass docstring;
        `dvcorr.pipeline.velocity_centered.normalize_result`).

    Orientation -- READ THIS BEFORE CHANGING ANYTHING HERE
    ----------------------------------------------------------
    dvcorr.conventions's frozen r = s_V - s_T and mu = n_hat_T . r_hat are NOT
    redefined here, or anywhere. This function instead reuses the SAME
    primitives, unmodified, on a deliberately REVERSED construction:

        r_vec, r_mag = pair_separation(s_alpha, s_tracers_near)

    passes the velocity center in the "center" slot, so
    r_vec = s_tracer - s_alpha points center -> tracer -- the exact negative
    of the frozen r = s_V - s_T for the same physical pair (there, the
    density object is the center and r points center -> velocity object;
    here the velocity object is the center, so the analogous arrow flips).
    This IS n_hat_i's direction, straight out of `pair_separation`.

        cos_theta = mu_cosine(unit_vector(r_vec), n_hat_V_alpha)

    `mu_cosine` is a plain row-wise dot product; passing z_hat_alpha (this
    center's own flow-signed axis, `radial_flow_axis`) instead of n_hat_T
    (the frozen convention's density-object line of sight) is the second
    deliberate substitution. The resulting cos_theta is therefore NOT
    conventions.MU_CONVENTION's frozen mu -- same primitives, different
    orientation and different reference direction, by design.

    Multipole relation: zeta_ell = (-1)**ell * xi_Tu,ell
    (conventions.nusser_multipole_sign). Monopoles agree; dipoles are opposite in
    sign. Consequence: coherent infall gives xi_Tu,1 < 0 (hard rule 2) and
    therefore zeta_1 > 0. A NEGATIVE zeta_1 from an infall mock is the
    orientation bug, not a result -- see the joint sign gate in
    tests/test_velocity_centered_dipole.py.

    Rotated per-center frame
    -----------------------------
    Each center's frame has z_hat = sign(u_alpha) * n_hat_V,alpha; only the
    l=1, m=0 term is computed, and it depends on z_hat alone -- the transverse (x_hat, y_hat)
    axes are unconstrained and never needed, so no rotation matrix is ever
    formed (see `real_y10`'s docstring on why m=+-1 is out of scope).

    Parameters
    ----------
    s_centers : ndarray, shape (3,) or (N_cand, 3)
        Candidate velocity-object positions, comoving h^-1 Mpc.
    v_centers : ndarray, shape (3,) or (N_cand, 3)
        Candidate 3-D peculiar velocities, km/s, same row order as
        `s_centers`. Must be finite everywhere: a missing peculiar velocity
        is missing data (CLAUDE.md hard rule 5) and the candidate must be
        dropped by the CALLER before this function is reached, never entered
        as v = 0 -- a non-finite entry is a ValueError here, not a silent
        drop.
    s_tracers : ndarray, shape (N_t, 3)
        Density tracer positions, comoving h^-1 Mpc, in the same continuous
        (already carved, no wrapping needed) frame as `s_centers` -- same PBC
        contract as `shell_dipole`: this function does NOT wrap, so the
        caller's sub-volume must sit clear of the box faces.
    shell_edges : ndarray, shape (B + 1,)
        Strictly increasing shell boundaries, h^-1 Mpc, defining B shells.
        Left-closed / right-open, [edge_low, edge_high), outermost edge
        folded into the last shell. Outermost edge must not exceed
        conventions.MAX_ANALYSIS_RADIUS.
    sub_volume_radius : float
        Radius of the spherical sub-volume carved around the observer,
        h^-1 Mpc. Must be > 0 and <= conventions.MAX_ANALYSIS_RADIUS.
    core_margin : float, optional
        Minimum clearance to the sub-volume boundary required of a surviving
        center (see `core_center_mask`). None (the default) uses
        `shell_edges[-1]`, i.e. r_max -- the natural choice, guaranteeing a
        surviving center's full shell fits inside the sub-volume. Must be
        >= 0.
    observer : ndarray, shape (3,), optional
        Observer position. Defaults to conventions.OBSERVER_POSITION.

    Returns
    -------
    VelocityCenteredShellDipoleResult
        Struct of raw per-shell sums plus the per-center breakdown and shell
        geometry. See the class docstring; monopole and dipole are always
        returned together (hard rule 6). Zero surviving centers is a VALID
        result, not an error: every per-shell sum is 0.0 (never NaN) and the
        per-center arrays have shape (0, B) / (0,).

    Raises
    ------
    ValueError
        If shell_edges is not 1-D with at least two entries, is not strictly
        increasing, or has an outermost edge exceeding
        conventions.MAX_ANALYSIS_RADIUS; if sub_volume_radius is not in
        (0, conventions.MAX_ANALYSIS_RADIUS]; if core_margin < 0; if
        v_centers.shape does not match s_centers.shape; or if v_centers
        contains a non-finite value.

    Notes
    -----
    A tracer coincident with its center (r_mag ~ 0 -- e.g. the center itself,
    when centers are drawn from the same catalog as the tracers) has no
    direction: `unit_vector` returns a zero row, so cos_theta = 0 and it
    contributes to `per_center_count` (occupancy) but nothing to
    `per_center_amplitude` (A). Identical documented contract to
    `shell_dipole`'s coincident-neighbor case. It is only reached when
    shell_edges[0] == 0; otherwise r_mag = 0 falls below the innermost edge
    and is excluded.

    A center whose velocity is exactly TRANSVERSE (u_alpha == 0) has an
    undefined flow axis: `radial_flow_axis` returns a zero row, so every
    cos_theta is 0 and A_alpha,b = 0. Its weight |u_alpha| is 0 as well, so
    it contributes nothing to `dipole` or `monopole` -- exactly as it did
    before the axis was signed (its signed weight was 0 then too). Not a new
    degenerate case, and not an error here; it still occupies a row in the
    `per_center_*` arrays and so dilutes `center_standard_error`, the same
    caveat as any zero-weight center.

    One `scipy.spatial.cKDTree` is built on `s_tracers` and reused across all
    surviving centers; the per-center loop is over CENTRES (each needs its
    own `query_ball_point` call), not over tracers -- the inner work for each
    center (`pair_separation`, `unit_vector`, `mu_cosine`, `np.digitize` +
    `np.bincount`) is fully vectorized over that center's tracers, mirroring
    `shell_dipole`'s binning semantics exactly (left-closed/right-open,
    outermost edge folded into the last shell, below-innermost-edge
    excluded).
    """
    observer = (
        conventions.OBSERVER_POSITION if observer is None
        else np.asarray(observer, dtype=float)
    )

    edges = np.asarray(shell_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("shell_edges must be 1-D with at least two entries.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("shell_edges must be strictly increasing.")
    if edges[-1] > conventions.MAX_ANALYSIS_RADIUS:
        raise ValueError(
            f"outermost shell edge {edges[-1]} exceeds "
            f"conventions.MAX_ANALYSIS_RADIUS = {conventions.MAX_ANALYSIS_RADIUS}."
        )
    n_bins = edges.size - 1

    if not (0.0 < sub_volume_radius <= conventions.MAX_ANALYSIS_RADIUS):
        raise ValueError(
            "sub_volume_radius must be > 0 and <= conventions.MAX_ANALYSIS_RADIUS "
            f"({conventions.MAX_ANALYSIS_RADIUS}), got {sub_volume_radius}."
        )

    if core_margin is None:
        core_margin = float(edges[-1])
    if core_margin < 0.0:
        raise ValueError(f"core_margin must be >= 0, got {core_margin}.")

    s_centers = np.atleast_2d(np.asarray(s_centers, dtype=float))
    v_centers = np.atleast_2d(np.asarray(v_centers, dtype=float))
    s_tracers = np.atleast_2d(np.asarray(s_tracers, dtype=float))
    n_candidates = s_centers.shape[0]

    if v_centers.shape != s_centers.shape:
        raise ValueError(
            f"v_centers shape {v_centers.shape} does not match "
            f"s_centers shape {s_centers.shape}."
        )
    if not np.all(np.isfinite(v_centers)):
        raise ValueError(
            "v_centers contains a non-finite value; a missing peculiar "
            "velocity is missing data and the candidate must be dropped by "
            "the caller, never entered as 0 (CLAUDE.md hard rule 5)."
        )

    survives = core_center_mask(s_centers, sub_volume_radius, core_margin, observer=observer)
    s_survivors = s_centers[survives]
    v_survivors = v_centers[survives]
    n_centers = s_survivors.shape[0]

    shell_centers = 0.5 * (edges[:-1] + edges[1:])

    per_center_count = np.zeros((n_centers, n_bins), dtype=float)
    per_center_amplitude = np.zeros((n_centers, n_bins), dtype=float)
    per_center_u = np.zeros(n_centers, dtype=float)

    if n_centers > 0:
        n_hat_V = unit_vector(s_survivors - observer)                # (N_c, 3)
        # z_hat = sign(u) * n_hat_V: the axis points along the center's own
        # RADIAL MOTION, not along its line of sight
        # (conventions.VELOCITY_AXIS_CONVENTION, one definition site in
        # geometry.radial_flow_axis). per_center_u stays SIGNED; the weight
        # applied below is its absolute value.
        z_hat, per_center_u = radial_flow_axis(v_survivors, n_hat_V)

        tree = cKDTree(s_tracers) if s_tracers.shape[0] > 0 else None

        for a, idx in _neighbors_by_center(tree, s_survivors, edges[-1]):
            if idx.size == 0:
                continue
            s_alpha = s_survivors[a]
            s_near = s_tracers[idx]

            # Reversed orientation: the velocity center in the "center" slot
            # of pair_separation gives r_vec = s_tracer - s_alpha, center ->
            # tracer -- see the Orientation section of this docstring.
            r_vec, r_mag = pair_separation(s_alpha, s_near)
            cos_theta = mu_cosine(unit_vector(r_vec), z_hat[a])

            in_range = (r_mag >= edges[0]) & (r_mag <= edges[-1])
            bin_index = np.digitize(r_mag, edges) - 1
            bin_index = np.where(bin_index == n_bins, n_bins - 1, bin_index)

            b = bin_index[in_range]
            y10_in = real_y10(cos_theta[in_range])

            per_center_count[a] = np.bincount(b, minlength=n_bins)[:n_bins].astype(float)
            per_center_amplitude[a] = np.bincount(b, weights=y10_in, minlength=n_bins)[:n_bins]

    # The SPEED |u_alpha| is the weight; the sign already lives in z_hat, and
    # entering it twice would cancel the statistic.
    per_center_speed = np.abs(per_center_u)

    per_center_dipole = per_center_speed[:, None] * per_center_amplitude

    pair_count = per_center_count.sum(axis=0)
    monopole = (per_center_speed[:, None] * per_center_count).sum(axis=0)
    dipole = per_center_dipole.sum(axis=0)

    return VelocityCenteredShellDipoleResult(
        shell_edges=edges,
        shell_centers=shell_centers,
        pair_count=pair_count,
        monopole=monopole,
        dipole=dipole,
        per_center_dipole=per_center_dipole,
        per_center_amplitude=per_center_amplitude,
        per_center_count=per_center_count,
        per_center_u=per_center_u,
        per_center_speed=per_center_speed,
        n_candidates=n_candidates,
        n_centers=n_centers,
    )
