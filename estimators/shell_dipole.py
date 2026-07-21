"""
shell_dipole.py
---------------
Shell-binned monopole and dipole around a single central object.

This is the estimator core of Nusser (2017), eq. (23), re-expressed in this
project's group-centred convention. For a central object T we bin its
neighbours into concentric radial shells and accumulate, per shell b,

    Q(r_b)     = sum over neighbours in shell b of  w_i * L_1(mu_i)
               = sum over neighbours in shell b of  w_i * mu_i
    count(r_b) = sum over neighbours in shell b of  w_i            (the L_0 term)

Q is the dipole accumulator and count is the monopole accumulator: the two
differ only by the Legendre weight L_ell(mu), so they are computed in one pass
over the same pair list and returned side by side. That pairing is the point of
the function. The monopole is not a by-product to be discarded -- it is the
geometry diagnostic. In an ideal spherical, angularly complete shell it should
vanish once normalised, and any departure measures exactly the effects that
also bias the dipole: finite-distance leakage of order (2r / 3R) v_inf,
incomplete angular coverage of the shell, and residual bulk motion of the
central object. Reporting a dipole without its companion monopole hides the
one number that says whether the dipole is trustworthy.

Meaning of the weights
----------------------
The same accumulator serves two statistics, distinguished only by w_i:

  - w_i = 1 (uniform): Q is the geometric dipole moment of the neighbour
    distribution in the shell -- Nusser's Q_alpha, the density-side quantity.
    Around a random centre it averages to zero; a non-zero value means the
    neighbours are distributed anisotropically about the line of sight.

  - w_i = u_i (the neighbour's observer-centred radial peculiar velocity):
    Q is the unnormalised numerator of the density-velocity dipole
    xi_Tu,1(r_b). This is the science signal.

The full estimator normalises Q by the shell's pair count and subtracts the
same measurement made around random centre positions, which removes the mean
velocity and the leading window effect. Neither the normalisation nor the
random subtraction happens here; this function only accumulates.

Sign
----
Under the frozen convention (config.py) r points outward from the central
object and mu = n_hat_T . r_hat. For coherent infall, u and mu have opposite
signs on both the near and the far side of the shell, so every pair contributes
u * mu < 0 and the dipole is NEGATIVE. If a validated infall mock returns a
positive dipole, the bug is an orientation flip in `pair_separation`, not a
physics result.
"""

from __future__ import annotations

import numpy as np


def shell_dipole(
    s_center: np.ndarray,
    s_neighbors: np.ndarray,
    shell_edges: np.ndarray,
    weights: np.ndarray | None = None,
    observer: np.ndarray | None = None,
    box_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate the monopole and dipole of neighbours in radial shells.

    Parameters
    ----------
    s_center : ndarray, shape (3,)
        Position of the central density object s_T, comoving box coordinates,
        h^-1 Mpc.
    s_neighbors : ndarray, shape (N, 3)
        Positions of the N neighbour objects s_V, same units. Neighbours
        outside the outermost shell edge are ignored, not an error.
    shell_edges : ndarray, shape (B + 1,)
        Monotonically increasing shell boundaries in h^-1 Mpc, defining B
        shells. The outermost edge must not exceed config.MAX_ANALYSIS_RADIUS,
        beyond which the periodic minimum image is ambiguous.
    weights : ndarray, shape (N,), optional
        Per-neighbour weight w_i. None means uniform weights of 1, giving the
        purely geometric shell dipole. Pass the radial peculiar velocities u_i
        to obtain the numerator of xi_Tu,1. Must not contain NaN: a missing
        peculiar velocity is missing data and the object must be dropped by the
        caller, never entered as u = 0.
    observer : ndarray, shape (3,), optional
        Observer position defining n_hat_T. Defaults to
        config.OBSERVER_POSITION. There is one observer per run; pass this
        explicitly only in tests.
    box_size : float, optional
        Periodic box side, h^-1 Mpc. Defaults to config.BOX_SIZE.

    Returns
    -------
    monopole : ndarray, shape (B,)
        Per-shell sum of w_i * L_0(mu_i) = sum of w_i. With uniform weights
        this is the raw pair count in the shell.
    dipole : ndarray, shape (B,)
        Per-shell sum of w_i * L_1(mu_i) = sum of w_i * mu_i, i.e. Q(r_b).

    Both arrays have length B = len(shell_edges) - 1 and are aligned bin for
    bin, so a caller may divide them elementwise. Empty shells return 0.0 in
    both arrays rather than NaN; whether an empty shell is usable is the
    caller's decision, and the pair count makes it visible.

    Raises
    ------
    ValueError
        If shell_edges is not increasing, if the outermost edge exceeds
        config.MAX_ANALYSIS_RADIUS, or if `weights` is given with a length
        that does not match s_neighbors.

    Notes
    -----
    Separations use the periodic minimum-image convention throughout
    (`geometry.pair_separation`), as does the observer -> centre direction that
    defines n_hat_T.

    Nusser (2017) centres on the velocity object rather than the density
    object, so his dipole is the negative of this one; see
    config.nusser_multipole_sign before comparing amplitudes.
    """
    raise NotImplementedError
