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
Single source of truth for every convention that carries a sign or a normalization. Nothing
here is computed; it is all constants and one helper. Downstream code imports from here and
never restates a convention locally.

- Box: `BOX_SIZE`, `HUBBLE_PARAM`, `MAX_ANALYSIS_RADIUS = BOX_SIZE / 2`.
- Observer: `OBSERVER_POSITION` (read-only `(3,)`, default box center).
- Frame: `REFERENCE_FRAME = "cmb"`.
- Orientation strings: `PAIR_SEPARATION_CONVENTION` (`r = s_V − s_T`), `MU_CONVENTION`
  (`µ = n̂_T · r̂`) — greppable, assertable copies of the docstring conventions.
- Sign: `INFALL_DIPOLE_SIGN = -1`; `nusser_multipole_sign(ell) -> (-1)**ell`.
- Catalog columns: `HALO_COLUMNS`, `POSITION_COLUMNS`, `VELOCITY_COLUMNS`.

### `settings.py` — tunable settings **[done]**
Counterpart to `config.py`: everything here is a first-pass, tunable knob (file locations,
MDPL2 cosmology metadata, shell binning, mocks/selection placeholders), never a sign or
normalization convention. Anything with a sign lives in `config.py` and is imported here,
never restated. Four small dataclasses plus an aggregator:

- `PathsConfig` (mutable) — catalog and output locations, all derived from `project_root`
  (`Path(__file__).resolve().parent`, i.e. the repo root) rather than hardcoded absolutes, so
  the settings are portable across checkouts. `data_dir`, `output_dir`, and the four catalog
  paths (`mdpl2_catalog`, `cf4_groups_catalog`, `cf4_velocities_catalog`,
  `sdss_tempel_catalog`) default to `None` and are resolved in `__post_init__`. Method
  `ensure_output_dir()` creates `output_dir`. Does not load the ~4M-row MDPL2 catalog — a
  path only.
- `CosmologyConfig` (frozen dataclass) — MDPL2 simulation cosmology (`H0`, `Om0`, `Ode0`,
  `Ob0`, `sigma8`, `ns`, `growth_index`, `flat`); simulation metadata, not a `config.py`-style
  convention, but `__post_init__` raises `ValueError` if `h` (`H0/100`, the definition of the
  reduced Hubble parameter) disagrees with `config.HUBBLE_PARAM` beyond `_H_CONSISTENCY_TOL`,
  keeping the two single sources of truth from drifting apart. `to_colossus_dict()` for
  colossus's constructor shape. CF4's distance-ladder `H0_CF4` is deliberately out of scope
  until the CF4 data arm starts.
- `ShellConfig` — radial shell binning for `estimators.shell_dipole.shell_dipole`:
  `min_radius`, `max_radius`, `radii_step`, `sigma_star` (small-scale velocity noise, km/s).
  `__post_init__` validates ordering and that `max_radius <= config.MAX_ANALYSIS_RADIUS`.
  Properties `shell_edges (B+1,)` and `shell_centers (B,)`; edges are built min→max in steps
  of `radii_step` with `max_radius` appended exactly (avoiding `np.arange` overshoot) and
  clipped to `config.MAX_ANALYSIS_RADIUS`.
- `SelectionConfig` — lean placeholder knobs for the not-yet-built `mocks/`/`selection/` arms:
  `mass_min` (default `1e12`, matching the catalog's mvir floor), `mass_max`,
  `number_of_observers`, `observer_selection` (`"random"` or `"virgo"`), all validated in
  `__post_init__`.
- `Settings` — aggregates one instance of each via `default_factory` (so sub-configs are
  independent across instances); `default_settings()` returns a fresh instance rather than a
  shared module-level singleton.

### `geometry.py` — pure geometry primitives **[done]**
Stateless, array-vectorized free functions; the layer the sign convention physically lives
in, kept small enough to be fully unit-tested. Shape contracts: `(3,)` one vector,
`(N, 3)` many, `(N,)` scalars-per-object; a function documented `(N,)` returns `(N,)` even
for `N = 1`.

- `unit_vector(vec)` — normalize `(3,)` or `(N, 3)` along the last axis; **zero-length rows
  return zeros** (masked divide, no NaN, no raise).
- `pair_separation(s_center, s_others) -> (r_vec, r_mag)` — `r = s_others − s_center`, the
  frozen `r = s_V − s_T`. Plain Euclidean; **no minimum-image** (see PBC note below).
- `mu_cosine(r_hat, n_T_hat)` — `µ = n̂_T · r̂`, row-wise dot, clipped to `[-1, 1]`. Does
  not normalize its inputs.

**PBC contract (important):** these primitives do *not* apply the minimum-image convention.
Periodicity is discharged upstream by carving a sub-volume around each center and unwrapping
it into that center's continuous frame; coordinates reaching `geometry.py` are already the
nearest image of their center. Re-adding a minimum-image reduction here would corrupt
already-unwrapped separations. (This refines CLAUDE.md hard rule 3: PBC is honoured at the
carving step, not in the primitives. The carving step is not yet implemented — see
research notes.)

---

## `estimators/`

### `estimators/shell_dipole.py` — shell monopole + dipole **[done]**
The estimator core. Computes the **group-centered** density–velocity correlation **ξ_Tu**:
density object T at the center, velocity objects V as neighbors entering through
per-neighbor weights `w_i = u_i`. This is *not* Nusser (2017)'s velocity-centered estimator
(eq. 23–24, a single central velocity with density tracers counted in the shell); the two
are related by `ξ_Tu,ℓ = (−1)^ℓ ξ_uT,ℓ`. A scalar central velocity would be correct only for
the velocity-centered form and vanishes on an isotropic shell here — hence per-neighbor
weights. Bins one central object's neighbors into radial shells and accumulates the L₀ and
L₁ Legendre moments of the weights.

- `ShellDipoleResult` (frozen dataclass): `shell_edges (B+1,)`, `shell_centers (B,)`,
  `pair_count (B,)`, `monopole (B,)`, `dipole (B,)`. All **raw sums, un-normalized**.
- `shell_dipole(s_center, s_neighbors, shell_edges, weights=None, observer=None)`:
  - `n̂_T = unit_vector(s_center − observer)`; `r, r_mag = pair_separation(...)`;
    `µ = mu_cosine(unit_vector(r), n̂_T)`.
  - Bins by `r_mag` (`np.digitize` + weighted `np.bincount`, no per-neighbor loop);
    left-closed/right-open shells, outer edge folded into the last shell; out-of-range
    neighbors excluded.
  - `pair_count = Σ1` (occupancy), `monopole = Σw_i`, `dipole = Σ w_i µ_i`.
  - Weights are **per-neighbor** (the velocity objects V carry the velocities); `None`
    means uniform weights → geometric dipole. Non-finite weight → `ValueError` (hard rule 5).
  - Empty shell → `0.0`, never NaN. Coincident neighbor (`r_mag≈0`) → `µ=0`, counts but
    adds nothing to the dipole.
  - Normalisation is the caller's: `ξ_Tu,1 = 3·dipole/pair_count`,
    `ξ_Tu,0 = monopole/pair_count`. `pair_count` (Σ1) and `monopole` differ once `w_i = u_i`
    — then `monopole = Σu` is the velocity monopole (the finite-R `2r/3R` leakage / bulk-motion
    diagnostic, ≈0 by near/far cancellation), which is why both are returned.

**Velocity-centered estimator (ζ, Nusser 2017 eq. 23–24) — the production target [done]**
Alongside `xi_Tu`, this module also implements the velocity-centered statistic directly, in
the same file (the module docstring is extended additively; `shell_dipole` /
`ShellDipoleResult` are untouched — frozen as the simulation-validation cross-check). A
velocity object α is the center; density tracers are counted in shells around it. Uses the
REVERSED separation (center → tracer, the negative of the frozen `r = s_V − s_T`) and the
reference direction `n̂_V` (not the frozen `n̂_T`) — same primitives (`pair_separation`,
`unit_vector`, `mu_cosine`), different orientation, by design. Related to `xi_Tu` by
`ζ_ℓ = (−1)^ℓ ξ_Tu,ℓ` (`config.nusser_multipole_sign`): monopoles agree, dipoles flip —
coherent infall gives `ξ_Tu,1 < 0` and therefore `ζ_1 > 0`.

- `real_y10(direction_cosine) -> (N,)` — real ℓ=1, m=0 spherical harmonic,
  `√(3/4π)·cosθ`; pure numpy, pinned against `scipy.special.sph_harm_y`. m=±1 out of scope
  (would need the per-center frame's transverse axes, never needed by m=0).
- `core_center_mask(s_centers, sub_volume_radius, core_margin, observer=None) -> (N_c,) bool`
  — keeps candidates whose full shell (out to `core_margin`, typically `r_max`) fits inside
  the spherical sub-volume: `|s_α − observer| ≤ sub_volume_radius − core_margin`. Suppresses
  the boundary-truncation bias suspected behind the first MDPL2 run's ~+13 km/s null offset.
- `expected_shell_occupancy(number_density, shell_edges) -> (B,)` — `n̄·V_b`, the Nusser
  eq. 24 normalization denominator (expected, not realized, occupancy); `n̄` supplied by the
  caller.
- `center_standard_error(per_center_values) -> (B,)` — `std(axis=0, ddof=1)/√N_c` across
  centers; NaN for `N_c < 2`. Treats centers as independent (documented understatement of
  the true uncertainty; mock covariance is the eventual replacement).
- `VelocityCenteredShellDipoleResult` (frozen dataclass): `shell_edges (B+1,)`,
  `shell_centers (B,)`, `pair_count (B,)`, `monopole (B,)`, `dipole (B,)` — all raw sums,
  stacked over centers — plus `per_center_dipole/amplitude/count (N_c, B)` and
  `per_center_u (N_c,)` for a downstream error band / shuffle null without a second pass,
  and `n_candidates`, `n_centers` (int) so the core cut's volume loss is visible.
- `velocity_centered_shell_dipole(s_centers, v_centers, s_tracers, shell_edges, sub_volume_radius, core_margin=None, observer=None)`:
  applies `core_center_mask`; builds one `cKDTree` on `s_tracers`, loops over surviving
  centers (`query_ball_point`), vectorized inner work per center
  (`pair_separation`→`unit_vector`→`mu_cosine`→`np.digitize`+`np.bincount`, same binning
  semantics as `shell_dipole`). `v_centers` must be finite everywhere (hard rule 5). Zero
  surviving centers is a valid all-zero/empty-array result, never NaN.

---

## `selection/` — masks, selection functions, random catalogs **[empty]**
Reserved for CF4-like angular/radial masks, the selection function φ(r), and random
catalog generation. Port target: the old repo's `masks.py` (CF4-like selection).

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
- `tests/test_velocity_centered_dipole.py` **[done]** — tests for the velocity-centered ζ
  estimator. Joint sign gate: `shell_dipole` and `velocity_centered_shell_dipole` run on the
  identical toy infall configuration (`tests.test_geometry._infall_shell_with_velocities`,
  the sibling helper added next to `_infall_shell`) and must come out with opposite signs.
  Also: `real_y10` pinned against `scipy.special.sph_harm_y`; the reversed construction pinned
  against an independent hand-assembled path and against the frozen orientation's exact
  negative; isotropic-shell null; mirrored-tracer sign flip; binning correctness; empty-shell
  / zero-tracer / zero-surviving-center edge cases; a boundary test showing the default core
  cut reduces the outer-shell truncation-bias residual versus `core_margin=0`; a shuffle
  null; input validation; `center_standard_error` and `core_center_mask` unit checks.
- `tests/test_settings.py` **[done]** — `settings.py` dataclass tests: `PathsConfig` derived
  paths and `ensure_output_dir` (via `tmp_path`, never the real `output/`); `CosmologyConfig`
  frozen-ness, `h`/`config.HUBBLE_PARAM` agreement, `to_colossus_dict`, and the inconsistent-H0
  `ValueError`; `ShellConfig` edge/center shape and monotonicity, the `MAX_ANALYSIS_RADIUS`
  ceiling, invalid-input `ValueError`s, and a toy `shell_dipole` call on the produced edges;
  `SelectionConfig` valid/invalid construction; `Settings`/`default_settings` independence
  across instances.

## `notebooks/` — exploration only
Nothing load-bearing; diagnostics that import the real modules, never reimplement them.

- `04_first_mdpl2_run.ipynb` **[done]** — first `shell_dipole` run on real MDPL2 halos:
  load → carve a boundary-safe sub-volume (periodicity guard) → single-central diagnostics
  (count ~ r², Q = Σμ ≈ 0, normalized dipole) → stack over centrals (raw-sum then normalize)
  → velocity-shuffle null. Exploratory, not a science result.
- `05_velocity_centered_dipole.ipynb` **[done]** — exploratory twin of
  `scripts/plot_velocity_centered_dipole.py`; every code cell calls one of that script's
  stage functions (`load_and_carve`, `draw_candidates`, `run_estimator`,
  `global_number_density`, `normalize_result`, `make_figure`) in sequence — nothing is
  reimplemented. One stage per cell, short markdown between them (what a positive ζ̂₁ means
  under the velocity-centered sign convention, the monopole panel as the trust diagnostic,
  the shuffle null). Final cell saves the PNG via the same `paths.output_dir` / `cfg` logic as
  `main()`, then displays the returned `Figure` inline (never `plt.show`). Not executed as
  part of authoring — outputs are cleared.

## `scripts/` — runnable, load-bearing scripts
Unlike `notebooks/`, code here is meant to be run directly (`python -m scripts.<name>`), not
explored interactively; results are load-bearing. Public functions are also the ones
`notebooks/05_velocity_centered_dipole.ipynb` calls, so notebook and script never drift apart.

- `plot_velocity_centered_dipole.py` **[done]** — measures the velocity-centered ζ dipole on
  real MDPL2 halos and plots it, as composable stage functions plus a `main()` that chains
  them. Importing the module selects NO matplotlib backend (no `matplotlib.use` at import
  time) — a notebook that imports the stages below keeps whatever backend it already has;
  `main()` is the only place that calls `matplotlib.use("Agg")`.
  - `RunConfig` (dataclass, hard rule 4): `sub_volume_radius`; `shells` — a COMPOSED
    `settings.ShellConfig` (`field(default_factory=_default_shells)`), reusing its validation
    (ordering, `radii_step > 0`, `max_radius <= config.MAX_ANALYSIS_RADIUS`) and its
    `shell_edges` property rather than mirroring those fields and their checks a second time;
    `_default_shells()`'s `min_radius` defaults to `radii_step`, not 0 — candidates are
    subsampled from the tracer array, so every candidate is its own tracer at r = 0, and
    excluding that self-pair keeps it out of the monopole diagnostic panel; also
    `n_candidate_centers`, `seed`/`shuffle_seed`, output filename, `dpi`.
  - `load_and_carve(cfg, paths) -> (pos, vel)` — loads only `config.POSITION_COLUMNS +
    config.VELOCITY_COLUMNS`, carves a sphere of `sub_volume_radius` around
    `config.OBSERVER_POSITION` (plain Euclidean, no wrapping needed). Raises `RuntimeError`
    for a zero-row catalog or a zero-halo carve, hoisted before any percentage is printed
    from either count (no bare `ZeroDivisionError`).
  - `draw_candidates(cfg, pos, vel) -> (s_candidates, v_candidates)` — seeded subsample of the
    carved halos; `RuntimeError` if that would be zero candidates.
  - `run_estimator(cfg, s_candidates, v_candidates, s_tracers, observer) ->
    VelocityCenteredShellDipoleResult` — calls `velocity_centered_shell_dipole`, prints
    `n_candidates` vs `n_centers` (the core cut's visible loss), `RuntimeError` if zero
    centers survive.
  - `global_number_density(n_tracers, sub_volume_radius) -> float` — n̄ = N /
    ((4π/3)·R_sub³).
  - `NormalizedDipole` (frozen dataclass) / `normalize_result(result, n_bar, shuffle_seed) ->
    NormalizedDipole` — turns a raw result + n̄ into `zeta_hat`, `sem`, `zeta_hat_shuffle`,
    `sem_shuffle`, `monopole_norm`, all `(B,)`. Normalises via `expected_shell_occupancy`
    visibly at the call site (`ζ̂₁ = 3·(dipole/n_centers) / (√(3/4π)·n̄V_b)`), SEM via
    `center_standard_error`, and the shuffle null via a seeded permutation of `per_center_u`
    recombined with `per_center_amplitude` (no second estimator pass).
  - `make_figure(cfg, result, normalized) -> Figure` — the two-panel figure (ζ̂₁ with SEM +
    shuffle null; the normalized monopole companion below, hard rule 6); builds and returns
    only, never saves or calls `plt.show`.
  - `main()` — `matplotlib.use("Agg")`, then chains the stages above and saves the PNG under
    `paths.output_dir`. Never runs at import time (guarded by `if __name__ == "__main__"`)
    since the MDPL2 catalog load is long.

## `Imports from old repo/` — reference dump (read only)
The bulk-flow project's working code, not a package and not importable. Copy and adapt
deliberately; never wire live imports into it.
