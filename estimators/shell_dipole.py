"""
shell_dipole.py
---------------
Shell-binned monopole and dipole around a single central object.

WHICH ESTIMATOR THIS IS (read before comparing to Nusser)
---------------------------------------------------------
This computes the GROUP-CENTRED density-velocity correlation xi_Tu: the density
object T sits at the centre and the VELOCITY objects V are the neighbours,
entering through per-neighbour weights w_i = u_i (their radial peculiar
velocities). It is NOT Nusser (2017)'s velocity-centred estimator (his eq. 23-24,
sometimes written zeta), where a single velocity object sits at the centre with a
scalar velocity u_alpha, the DENSITY tracers are counted in the shell as
Q_alpha = sum_beta w_beta mu_alpha,beta, and the statistic is Q_alpha * u_alpha.
The two are the same two-point information with the separation reversed and are
related multipole by multipole by

    xi_Tu,ell = (-1)^ell * xi_uT,ell          (config.nusser_multipole_sign)

so their monopoles agree and their dipoles differ in sign. The distinction is
not cosmetic here: in the group-centred form the velocity is a PER-NEIGHBOUR
weight (there are N of them), whereas a SCALAR central velocity would be correct
only for the velocity-centred form. A scalar central velocity applied to this
construction vanishes on an isotropic shell (sum mu ~ 0) and is a bug, not a null
result. The project's methodology note names xi_Tu as the primary target; this
module is that target, and per-neighbour weights generalise cleanly to
inverse-variance weighting later.

What it accumulates
-------------------
For the central density object T we bin its neighbours into concentric radial
shells and accumulate, per shell b, the two Legendre moments of the neighbour
weights, alongside the raw pair count:

    pair_count(r_b) = sum over neighbours in shell b of  1        (the pair count)
    monopole(r_b)   = sum over neighbours in shell b of  w_i      (the L_0 moment)
    dipole(r_b)     = sum over neighbours in shell b of  w_i mu_i (the L_1 moment)

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
The velocity monopole (Sigma u, normalised by pair_count) is not a by-product to
be discarded -- it is the geometry diagnostic (CLAUDE.md hard rule 6). In an
ideal spherical, angularly complete shell around a static centre it vanishes by
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
    of the neighbour distribution in the shell -- Nusser's Q_alpha, the
    density-side quantity. Around a random centre it averages to zero; a
    non-zero value means the neighbours are distributed anisotropically about
    the line of sight. With uniform weights monopole == count.

  - w_i = u_i (the neighbour's observer-centred radial peculiar velocity): the
    dipole is the unnormalised numerator of the density-velocity dipole
    xi_Tu,1(r_b). This is the science signal, and its monopole Σu_i is the
    finite-distance leakage / bulk-motion diagnostic above.

Sign
----
Under the frozen convention (config.py) r points outward from the central
object and mu = n_hat_T . r_hat. For coherent infall, u and mu have opposite
signs on both the near and the far side of the shell, so every pair contributes
u * mu < 0 and the dipole is NEGATIVE. If a validated infall mock returns a
positive dipole, the bug is an orientation flip in `pair_separation`, not a
physics result.

Periodic boundary conditions
----------------------------
This function does NOT wrap coordinates. Consistent with `geometry.py`, the
periodic minimum image is resolved one level up, where the neighbour sub-volume
around each centre is carved and unwrapped into that centre's local continuous
frame. The neighbours handed in must already be the images nearest s_center.
The only periodicity obligation enforced here is that the outermost shell edge
must not exceed config.MAX_ANALYSIS_RADIUS, beyond which the nearest image --
and hence the carving itself -- is not unique.

Nusser (2017) centres on the velocity object rather than the density object, so
his dipole is the negative of this one; see config.nusser_multipole_sign before
comparing amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from geometry import mu_cosine, pair_separation, unit_vector


@dataclass(frozen=True)
class ShellDipoleResult:
    """Per-shell Legendre moments of the neighbour weights, all raw sums.

    Every array is aligned bin-for-bin and has length B = len(shell_edges) - 1,
    so the caller may combine them elementwise. Nothing is normalised.

    Attributes
    ----------
    shell_edges : ndarray, shape (B + 1,)
        The radial bin boundaries passed in, h^-1 Mpc.
    shell_centers : ndarray, shape (B,)
        Midpoints 0.5 * (edge_low + edge_high) of each shell, h^-1 Mpc. A
        plain midpoint, not a pair-count-weighted effective radius.
    pair_count : ndarray, shape (B,)
        Raw pair count per shell, sum of 1 over neighbours in the shell. Pure
        geometry / occupancy, and the natural normalisation denominator for both
        multipoles. Empty shells are 0, never NaN.
    monopole : ndarray, shape (B,)
        L_0 moment, sum of w_i over the shell. Equals `pair_count` when weights
        are uniform; equals the velocity monopole Σu_i -- the finite-distance
        (2r/3R) leakage / bulk-motion diagnostic, the numerator of xi_Tu,0 --
        when weights are the radial velocities. Empty shells are 0.0.
    dipole : ndarray, shape (B,)
        L_1 moment, sum of w_i * mu_i over the shell -- the unnormalised
        numerator of xi_Tu,1. Empty shells are 0.0, never NaN.

    Notes
    -----
    Downstream normalisation divides `dipole` (and `monopole`) by `pair_count`,
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
    """Accumulate the monopole and dipole of neighbours in radial shells.

    Parameters
    ----------
    s_center : ndarray, shape (3,)
        Position of the central density object s_T, comoving coordinates,
        h^-1 Mpc, in the same unwrapped frame as `s_neighbors`.
    s_neighbors : ndarray, shape (N, 3)
        Positions of the N neighbour (velocity) objects s_V, same units and
        frame. Neighbours outside [shell_edges[0], shell_edges[-1]] are
        excluded, not an error.
    shell_edges : ndarray, shape (B + 1,)
        Strictly increasing shell boundaries in h^-1 Mpc, defining B shells.
        Bins are left-closed / right-open, [edge_low, edge_high), with the
        outermost edge included in the last shell. The outermost edge must not
        exceed config.MAX_ANALYSIS_RADIUS.
    weights : ndarray, shape (N,), optional
        Per-neighbour weight w_i. None (the default) means uniform weights of
        1, giving the purely geometric shell dipole. Pass the radial peculiar
        velocities u_i to obtain the numerator of xi_Tu,1. Must be finite: a
        missing peculiar velocity is missing data (CLAUDE.md hard rule 5) and
        the object must be dropped by the caller, never entered as u = 0 -- so
        a non-finite weight is a ValueError here, not a silent zero.
    observer : ndarray, shape (3,), optional
        Observer position defining n_hat_T = observer -> s_center. Defaults to
        config.OBSERVER_POSITION. There is one observer per run; pass this
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
        config.MAX_ANALYSIS_RADIUS; or if `weights` is given with a shape that
        does not match s_neighbors, or contains a non-finite value.

    Notes
    -----
    A neighbour coincident with the centre (r_mag ~ 0) has no direction:
    `unit_vector` returns a zero row, so its mu is 0 and it contributes its
    weight to `count` and `monopole` but nothing to `dipole`. It is only
    reached at all when shell_edges[0] == 0; otherwise it falls below the
    innermost edge and is excluded. Either way it never injects a NaN.

    Fully vectorised: the binning is a single `np.digitize` plus weighted
    `np.bincount` calls, with no Python-level loop over neighbours.
    """
    observer = (
        config.OBSERVER_POSITION if observer is None
        else np.asarray(observer, dtype=float)
    )

    edges = np.asarray(shell_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("shell_edges must be 1-D with at least two entries.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("shell_edges must be strictly increasing.")
    if edges[-1] > config.MAX_ANALYSIS_RADIUS:
        raise ValueError(
            f"outermost shell edge {edges[-1]} exceeds "
            f"config.MAX_ANALYSIS_RADIUS = {config.MAX_ANALYSIS_RADIUS}."
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
                f"({n_neighbors},) neighbours."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "weights contains a non-finite value; a missing peculiar "
                "velocity is missing data and the object must be dropped by "
                "the caller, never entered as 0 (CLAUDE.md hard rule 5)."
            )

    # Geometry: line of sight of the CENTRAL object, then the group-centred
    # cosine of each neighbour. r = s_V - s_T points outward from the centre.
    n_T_hat = unit_vector(s_center - observer)
    r_vec, r_mag = pair_separation(s_center, s_neighbors)
    mu = mu_cosine(unit_vector(r_vec), n_T_hat)

    # Bin by separation. Left-closed/right-open interior boundaries; the
    # outermost edge is folded into the last shell so a neighbour sitting
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
