# Architecture — per-file responsibilities

The live map of what each module is responsible for. **This document is kept in sync with
the code (CLAUDE.md hard rule 8): any change to structure — a new module, a moved
responsibility, a changed public signature — updates this file in the same task.** If the
code and this file disagree, the code is right and this file has a bug.

Status tags: **[done]** implemented and tested · **[stub]** signature/docstring only ·
**[empty]** directory reserved, no code yet.

---

## Top-level modules

### `config.py` — frozen conventions **[done]**
Single source of truth for every convention that carries a sign or a normalisation. Nothing
here is computed; it is all constants and one helper. Downstream code imports from here and
never restates a convention locally.

- Box: `BOX_SIZE`, `HUBBLE_PARAM`, `MAX_ANALYSIS_RADIUS = BOX_SIZE / 2`.
- Observer: `OBSERVER_POSITION` (read-only `(3,)`, default box centre).
- Frame: `REFERENCE_FRAME = "cmb"`.
- Orientation strings: `PAIR_SEPARATION_CONVENTION` (`r = s_V − s_T`), `MU_CONVENTION`
  (`µ = n̂_T · r̂`) — greppable, assertable copies of the docstring conventions.
- Sign: `INFALL_DIPOLE_SIGN = -1`; `nusser_multipole_sign(ell) -> (-1)**ell`.
- Catalogue columns: `HALO_COLUMNS`, `POSITION_COLUMNS`, `VELOCITY_COLUMNS`.

### `geometry.py` — pure geometry primitives **[done]**
Stateless, array-vectorised free functions; the layer the sign convention physically lives
in, kept small enough to be fully unit-tested. Shape contracts: `(3,)` one vector,
`(N, 3)` many, `(N,)` scalars-per-object; a function documented `(N,)` returns `(N,)` even
for `N = 1`.

- `unit_vector(vec)` — normalise `(3,)` or `(N, 3)` along the last axis; **zero-length rows
  return zeros** (masked divide, no NaN, no raise).
- `pair_separation(s_center, s_others) -> (r_vec, r_mag)` — `r = s_others − s_center`, the
  frozen `r = s_V − s_T`. Plain Euclidean; **no minimum-image** (see PBC note below).
- `mu_cosine(r_hat, n_T_hat)` — `µ = n̂_T · r̂`, row-wise dot, clipped to `[-1, 1]`. Does
  not normalise its inputs.

**PBC contract (important):** these primitives do *not* apply the minimum-image convention.
Periodicity is discharged upstream by carving a sub-volume around each centre and unwrapping
it into that centre's continuous frame; coordinates reaching `geometry.py` are already the
nearest image of their centre. Re-adding a minimum-image reduction here would corrupt
already-unwrapped separations. (This refines CLAUDE.md hard rule 3: PBC is honoured at the
carving step, not in the primitives. The carving step is not yet implemented — see
research notes.)

---

## `estimators/`

### `estimators/shell_dipole.py` — shell monopole + dipole **[done]**
The estimator core. Computes the **group-centred** density–velocity correlation **ξ_Tu**:
density object T at the centre, velocity objects V as neighbours entering through
per-neighbour weights `w_i = u_i`. This is *not* Nusser (2017)'s velocity-centred estimator
(eq. 23–24, a single central velocity with density tracers counted in the shell); the two
are related by `ξ_Tu,ℓ = (−1)^ℓ ξ_uT,ℓ`. A scalar central velocity would be correct only for
the velocity-centred form and vanishes on an isotropic shell here — hence per-neighbour
weights. Bins one central object's neighbours into radial shells and accumulates the L₀ and
L₁ Legendre moments of the weights.

- `ShellDipoleResult` (frozen dataclass): `shell_edges (B+1,)`, `shell_centers (B,)`,
  `pair_count (B,)`, `monopole (B,)`, `dipole (B,)`. All **raw sums, un-normalised**.
- `shell_dipole(s_center, s_neighbors, shell_edges, weights=None, observer=None)`:
  - `n̂_T = unit_vector(s_center − observer)`; `r, r_mag = pair_separation(...)`;
    `µ = mu_cosine(unit_vector(r), n̂_T)`.
  - Bins by `r_mag` (`np.digitize` + weighted `np.bincount`, no per-neighbour loop);
    left-closed/right-open shells, outer edge folded into the last shell; out-of-range
    neighbours excluded.
  - `pair_count = Σ1` (occupancy), `monopole = Σw_i`, `dipole = Σ w_i µ_i`.
  - Weights are **per-neighbour** (the velocity objects V carry the velocities); `None`
    means uniform weights → geometric dipole. Non-finite weight → `ValueError` (hard rule 5).
  - Empty shell → `0.0`, never NaN. Coincident neighbour (`r_mag≈0`) → `µ=0`, counts but
    adds nothing to the dipole.
  - Normalisation is the caller's: `ξ_Tu,1 = 3·dipole/pair_count`,
    `ξ_Tu,0 = monopole/pair_count`. `pair_count` (Σ1) and `monopole` differ once `w_i = u_i`
    — then `monopole = Σu` is the velocity monopole (the finite-R `2r/3R` leakage / bulk-motion
    diagnostic, ≈0 by near/far cancellation), which is why both are returned.

---

## `selection/` — masks, selection functions, random catalogues **[empty]**
Reserved for CF4-like angular/radial masks, the selection function φ(r), and random
catalogue generation. Port target: the old repo's `masks.py` (CF4-like selection).

## `mocks/` — MDPL2 observers, halo selection, mock covariance **[empty]**
Reserved for placing observers in the MDPL2 box, matching halo selection to the survey, and
building the mock-based covariance. Port target: the old repo's `overdensity.py` (periodic
KDTree overdensity), `data_loader.py`.

---

## `tests/`

- `tests/test_geometry.py` **[done]** — the sign gate plus primitive-level contracts.
  - `test_spherical_infall_gives_negative_dipole` — the load-bearing gate: infall → negative
    dipole through `shell_dipole`, recovering `V_INFALL` via `3·dipole/count`.
  - `test_infall_dipole_is_negative_from_geometry_alone` — the same sign, primitives only.
  - `test_reversing_the_pair_vector_flips_the_dipole`, `test_mu_is_a_cosine_and_spans_the_shell`.
  - `TestUnitVector`, `TestPairSeparation`, `TestMuCosine`, `TestPrimitivesCompose` — shapes,
    norms, orientation, clipping, zero handling.
  - Shared toy helpers `_sphere_directions`, `_infall_shell` (reused by the estimator tests).
- `tests/test_shell_dipole.py` **[done]** — estimator-level sign gate (reuses `_infall_shell`),
  isotropic/velocity-shuffle nulls, binning correctness, empty-shell/NaN edge cases, and
  input validation.

## `notebooks/` — exploration only **[empty]**
Nothing load-bearing.

## `Imports from old repo/` — reference dump (read only)
The bulk-flow project's working code, not a package and not importable. Copy and adapt
deliberately; never wire live imports into it.
