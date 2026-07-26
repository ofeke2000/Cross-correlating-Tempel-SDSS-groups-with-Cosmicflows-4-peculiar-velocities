# Summary — science context and pipeline

> **Maintenance status:** snapshot overview, **not actively maintained yet**. The live
> documents right now are [architecture.md](architecture.md) (kept in sync with the code)
> and CLAUDE.md. Update this file only when asked. For the authoritative conventions and the
> full narrative, see [../README.md](../README.md) and the methodological note in
> `literature/`.

## The question

Measure the **density–velocity cross-correlation dipole** ξ_Tu,1(r) between Tempel et al.
(2017) SDSS groups (the density tracers, T) and Cosmicflows-4 peculiar velocities (the
velocity objects, V). In linear theory this dipole is proportional to the growth rate of
structure, so it is a probe of gravity on large scales. We follow Nusser (2017) but in a
**group-centered** construction (center on the density object T) rather than his
velocity-centered one; the two differ by (−1)^ℓ per multipole.

## Current phase — simulation validation

Before trusting the estimator on data, reproduce it inside the **MDPL2** N-body box, where
halo peculiar velocities and the halo density field are both known exactly and the sign and
amplitude of the answer are known in advance. The whole point is a controlled check: a
spherical-infall mock must return a **negative** dipole that recovers the imposed infall
speed. This is the sign gate the project is built around.

## Conventions (frozen — see `config.py`)

- Pair separation `r = s_V − s_T`, pointing from the density object to the velocity object.
- Cosine `µ = n̂_T · r̂`, using the *central* (density) object's line of sight.
- CMB frame; observer at a single fixed position (default box center).
- Infall → negative dipole. Reversing the pair vector silently flips all odd multipoles.
- Legendre `L_ℓ`; `P` is reserved for the matter power spectrum `P_m(k)`.

## Pipeline (target)

1. **Selection** — CF4-like masks and the selection function φ(r); random catalogs.
2. **Geometry** — group-centered pair separations and cosines (`geometry.py`, done).
3. **Estimator** — per-shell monopole and dipole accumulators (`estimators/shell_dipole.py`,
   done); normalize `ξ_Tu,1 = 3·dipole/count`, report the monopole alongside.
4. **Mocks** — MDPL2 observers, halo selection matched to the survey, mock covariance.
5. **Inference** — compare to the linear-theory prediction with a conservative scale cut
   (r ≳ 20 h⁻¹ Mpc), mock-based covariance.

## Null tests

- Velocity shuffle: permuting u across positions must kill the cross-correlation.
- Isotropic shell: the dipole vanishes while the count does not.
- Periodic-wrap: a center near a box face must agree once the sub-volume is carved.
- Finite-distance monopole leakage `U₀ = (2r/3R) v_inf` as a diagnostic, not an assumed zero.
