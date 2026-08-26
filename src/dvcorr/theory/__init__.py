"""
dvcorr.theory
-------------
Linear-theory predictions for the density-velocity cross-correlation, in the
VELOCITY-CENTERED convention of Nusser (2017).

What lives here
===============
The analytic half of the project: everything that turns a linear matter power
spectrum `P(k)` into a predicted multipole, with no catalog, no observer, and
no estimator anywhere in sight.

- `transforms.py` -- the spherical-Bessel k-integrals every prediction is
  built from (`bessel_moment`), plus the two master integrals `A` and `B` and
  a brute-quadrature reference to validate the FFTLog against.
- `gorski.py` -- the Gorski (1988) velocity correlation functions
  `Psi_perp`, `Psi_par`, `Psi_minus` and the two derivatives the dipole needs.
- `dipole.py` -- the public first-moment API `xi_udelta_dipole`, its scalar
  amplitude, and the one place `a H` converts h^-1 Mpc to km/s.
- `wide_angle.py` -- exact (non-distant-observer) geometry and the numerical
  Legendre moments that need it.

Separation convention -- read this before using any of it
=========================================================
This subpackage speaks NUSSER'S geometry, not the project's, and the two are
mirror images:

    Nusser:   y = r_tilde - r     velocity object -> density point
    dvcorr:   r = s_V - s_T       density object  -> velocity object

Same pair, same magnitude (`|y| == |r|`), opposite orientation, and the
centering swaps with it. Odd multipoles therefore differ in sign:

    xi_Tu,ell = conventions.nusser_multipole_sign(ell) * xi_uT,ell

so this project's group-centered dipole is `-1` times what
`dipole.xi_udelta_dipole` returns. Nothing here applies that factor silently:
the separation variable is spelled `y` throughout precisely so that a caller
mixing it with the pipeline's `r` has to notice. Convert explicitly, at the
call site, via `dvcorr.conventions.nusser_multipole_sign` (CLAUDE.md hard
rule 1). `xi_udelta_dipole` is POSITIVE for coherent infall; the pipeline's
`xi_Tu,1` is NEGATIVE for it, and both statements are the same physics.

Multipole normalization -- the factor of 3
==========================================
Nusser normalizes multipoles WITHOUT the `(2 ell + 1)` factor,

    <xi>_ell = (1/2) Int_-1^1 xi(mu) L_ell(mu) dmu

where the community standard carries `(2 ell + 1) / 2`. For the dipole that is
a factor of 3, entirely capable of being absorbed into a "measured bias" and
never noticed. Both are available, by name, through
`dipole.CONVENTION_NUSSER` / `dipole.CONVENTION_STANDARD`; the default is
Nusser's, since his paper is what these curves are checked against.

Relation to `dvcorr.pipeline.galaxy_bias`
=========================================
`galaxy_bias.zeta_one_linear_prediction` is the production real-space dipole
and predates this subpackage; it is the SAME number as
`xi_udelta_dipole(..., space=SPACE_REAL)` up to the velocity-centered sign it
applies internally, and the notebook
`intuition/nusser_dipole_real_and_redshift_space.ipynb` asserts that agreement
to machine precision. What this subpackage adds is everything the production
path does not need: the Gorski functions in their own right, redshift space,
and exact wide-angle geometry.

Layering
========
Imports `dvcorr.conventions` and `dvcorr.config` only, never
`dvcorr.pipeline` or `dvcorr.estimators` -- `pipeline.galaxy_bias` imports
THIS way round, for the shared `bessel_moment`. `mcfit` and `scipy` are
imported lazily inside the functions that need them, matching
`galaxy_bias`'s treatment of its optional dependencies.

Units
=====
`y` and `r_obs` in h^-1 Mpc, `k` in h Mpc^-1, `P(k)` in (h^-1 Mpc)^3. Every
`Psi` comes out in (h^-1 Mpc)^2 and every `Psi'` and multipole in h^-1 Mpc;
exactly one factor of `a H` (in km/s per h^-1 Mpc, i.e. `H(z) / h`, which is
exactly 100 at z = 0) converts any of them to km/s, and it is applied in one
place -- `dipole.hubble_h_units`, via the `aH` argument.
"""

from __future__ import annotations

__all__ = ["dipole", "gorski", "transforms", "wide_angle"]
