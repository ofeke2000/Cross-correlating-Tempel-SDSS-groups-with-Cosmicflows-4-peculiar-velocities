# Architecture — per-file responsibilities

The live map of what each module is responsible for. **This document is kept in sync with
the code (CLAUDE.md hard rule 8): any change to structure — a new module, a moved
responsibility, a changed public signature — updates this file in the same task.** If the
code and this file disagree, the code is right and this file has a bug.

Status tags: **[done]** implemented and tested · **[stub]** signature/docstring only ·
**[empty]** directory reserved, no code yet.

The project is an editable-installed package, `dvcorr`, laid out under `src/` (setuptools
src layout, declared in `pyproject.toml`). `import dvcorr…` works identically in scripts,
notebooks, and tests — no `sys.path` bootstrap, no run-from-root requirement. See "Working
model" below for how `src/dvcorr/`, `scripts/`, and `notebooks/` relate.

---

## `pyproject.toml` — packaging

Declares `dvcorr` (setuptools build backend, `[tool.setuptools.packages.find] where =
["src"]`), runtime dependencies (numpy, pandas, scipy, healpy, astropy, matplotlib, h5py),
a `dev` extra (`pytest`), and `[tool.pytest.ini_options] testpaths = ["tests"]`.
`requirements.txt` is a one-liner, `-e .[dev]`, so the dependency list has exactly one home.
Install once per checkout: `pip install -e .` into the project `.venv`.

---

## `src/dvcorr/` — the package

### `src/dvcorr/__init__.py` — package root **[done]**
Deliberately minimal: docstring + `__version__ = "0.1.0"` only. No submodule imports here,
so `import dvcorr` never pulls in scipy/matplotlib/pandas as a side effect. Import
submodules explicitly (`from dvcorr import conventions`, `from dvcorr.geometry import …`).

### `src/dvcorr/conventions.py` — frozen conventions **[done]**
(Was `config.py` at the repo root; renamed on the move into the package — CLAUDE.md hard
rule 1's authority, same content, new name.) Single source of truth for every convention
that carries a sign or a normalization. Nothing here is computed; it is all constants and
one helper. Downstream code imports `from dvcorr import conventions` and never restates a
convention locally. A leaf module: no intra-project imports.

- Box: `BOX_SIZE`, `HUBBLE_PARAM`, `MAX_ANALYSIS_RADIUS = BOX_SIZE / 2`.
- Observer: `OBSERVER_POSITION` (read-only `(3,)`, default box center).
- Frame: `REFERENCE_FRAME = "cmb"`.
- Orientation strings: `PAIR_SEPARATION_CONVENTION` (`r = s_V − s_T`), `MU_CONVENTION`
  (`µ = n̂_T · r̂`) — greppable, assertable copies of the docstring conventions.
- Sign: `INFALL_DIPOLE_SIGN = -1`; `nusser_multipole_sign(ell) -> (-1)**ell`.
- Catalog columns: `HALO_COLUMNS`, `POSITION_COLUMNS`, `VELOCITY_COLUMNS`.

### `src/dvcorr/geometry.py` — pure geometry primitives **[done]**
Stateless, array-vectorized free functions; the layer the sign convention physically lives
in, kept small enough to be fully unit-tested. A leaf module (no imports from elsewhere in
`dvcorr`; its docstrings refer to `dvcorr.conventions` by name but do not import it). Shape
contracts: `(3,)` one vector, `(N, 3)` many, `(N,)` scalars-per-object; a function documented
`(N,)` returns `(N,)` even for `N = 1`.

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
carving step, not in the primitives. The carving step lives in `pipeline/velocity_centered.py`
for the velocity-centered path — see below.)

---

## `src/dvcorr/config/` — tunable settings

Counterpart to `dvcorr.conventions`: everything here is a first-pass, tunable knob (file
locations, MDPL2 cosmology metadata, shell binning, mocks/selection placeholders), never a
sign or normalization convention. Anything with a sign lives in `dvcorr.conventions` and is
imported here, never restated. One dataclass per file, mirroring the old bulk-flow repo's
`src/config/` pattern (`<thing>_config.py` + an aggregator):

### `src/dvcorr/config/paths.py` — `PathsConfig` **[done]**
Catalog and output locations, all derived from `project_root` rather than hardcoded
absolutes, so the settings are portable across checkouts. `project_root` defaults to
`Path(__file__).resolve().parents[3]` — this file lives at
`<repo-root>/src/dvcorr/config/paths.py`, so climbing three parents (config → dvcorr → src
→ repo-root) reaches the repo root even though the module itself sits three levels deep;
holds under the editable install since `__file__` still points at the in-place source tree.
`data_dir`, `output_dir`, and the four catalog paths (`mdpl2_catalog`, `cf4_groups_catalog`,
`cf4_velocities_catalog`, `sdss_tempel_catalog`) default to `None` and are resolved in
`__post_init__`. Method `ensure_output_dir()` creates `output_dir`. Does not load the ~4M-row
MDPL2 catalog — a path only.

### `src/dvcorr/config/cosmology.py` — `CosmologyConfig` **[done]**
Frozen dataclass — MDPL2 simulation cosmology (`H0`, `Om0`, `Ode0`, `Ob0`, `sigma8`, `ns`,
`growth_index`, `flat`); simulation metadata, not a `dvcorr.conventions`-style convention,
but `__post_init__` raises `ValueError` if `h` (`H0/100`, the definition of the reduced
Hubble parameter) disagrees with `dvcorr.conventions.HUBBLE_PARAM` beyond
`_H_CONSISTENCY_TOL` (module-level constant, `1e-9`), keeping the two single sources of truth
from drifting apart. `to_colossus_dict()` for colossus's constructor shape. CF4's
distance-ladder `H0_CF4` is deliberately out of scope until the CF4 data arm starts.

### `src/dvcorr/config/shells.py` — `ShellConfig` **[done]**
Radial shell binning for `dvcorr.estimators.shell_dipole.shell_dipole`: `min_radius`,
`max_radius`, `radii_step`, `sigma_star` (small-scale velocity noise, km/s). `__post_init__`
validates ordering and that `max_radius <= dvcorr.conventions.MAX_ANALYSIS_RADIUS`.
Properties `shell_edges (B+1,)` and `shell_centers (B,)`; edges are built min→max in steps
of `radii_step` with `max_radius` appended exactly (avoiding `np.arange` overshoot) and
clipped to `dvcorr.conventions.MAX_ANALYSIS_RADIUS`.

### `src/dvcorr/config/selection.py` — `SelectionConfig` **[done]**
Lean placeholder knobs for the not-yet-built `mocks/`/`selection/` arms: `mass_min` (default
`1e12`, matching the catalog's mvir floor), `mass_max`, `number_of_observers`,
`observer_selection` (`"random"` or `"virgo"`), all validated in `__post_init__`. No
dependency on `dvcorr.conventions`.

### `src/dvcorr/config/settings.py` — `Settings` aggregator **[done]**
`Settings` aggregates one instance of each of the four dataclasses above via
`default_factory` (so sub-configs are independent across instances); `default_settings()`
returns a fresh instance rather than a shared module-level singleton.

### `src/dvcorr/config/__init__.py` — re-export surface **[done]**
`from dvcorr.config import Settings, default_settings, PathsConfig, CosmologyConfig,
ShellConfig, SelectionConfig` — a clean import surface over the five files above.

---

## `src/dvcorr/estimators/`

### `src/dvcorr/estimators/shell_dipole.py` — shell monopole + dipole **[done]**
The estimator core. Imports `from dvcorr import conventions` and
`from dvcorr.geometry import mu_cosine, pair_separation, unit_vector`. Computes the
**group-centered** density–velocity correlation **ξ_Tu**: density object T at the center,
velocity objects V as neighbors entering through per-neighbor weights `w_i = u_i`. This is
*not* Nusser (2017)'s velocity-centered estimator (eq. 23–24, a single central velocity with
density tracers counted in the shell); the two are related by
`ξ_Tu,ℓ = (−1)^ℓ ξ_uT,ℓ`. A scalar central velocity would be correct only for the
velocity-centered form and vanishes on an isotropic shell here — hence per-neighbor weights.
Bins one central object's neighbors into radial shells and accumulates the L₀ and L₁
Legendre moments of the weights.

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
`ζ_ℓ = (−1)^ℓ ξ_Tu,ℓ` (`dvcorr.conventions.nusser_multipole_sign`): monopoles agree, dipoles
flip — coherent infall gives `ξ_Tu,1 < 0` and therefore `ζ_1 > 0`.

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

**ℓ=0 exposure, consumed unchanged by the two-frame comparison:**
`VelocityCenteredShellDipoleResult.monopole` → `NormalizedDipole.monopole_norm`
(`dvcorr.pipeline.velocity_centered.normalize_stacked_dipole`) → the monopole panel of
`dvcorr.pipeline.velocity_frame_comparison.make_comparison_figure` is the exact plumbing
already built for the single-frame figure (`make_figure`'s bottom panel) — the comparison
pipeline needed no new plumbing for the observer-frame monopole, only a second (twin-axis)
curve alongside it for the velocity frame.

### `src/dvcorr/estimators/velocity_frame_dipole.py` — velocity-frame dipole **[done]**
An OBSERVER-FREE variant of `velocity_centered_shell_dipole`, in its own file (imports
`from dvcorr import conventions`, `from dvcorr.geometry import mu_cosine, pair_separation,
unit_vector`, and reuses `core_center_mask` / `real_y10` from `shell_dipole.py` UNCHANGED —
`shell_dipole.py` itself was not touched). Same reversed-orientation construction
(`pair_separation(s_alpha, s_near)` with the center in the "center" slot, i.e. center→tracer)
and the same `real_y10`/binning machinery as the observer frame; the ONLY differences are the
axis and the scalar:

- axis: `z_hat_alpha = unit_vector(v_alpha)` (the center's OWN flow direction) instead of
  `n_hat_V,alpha = unit_vector(s_alpha − observer)`.
- scalar: `speed_alpha = |v_alpha|` (positive-definite) instead of the observer-frame radial
  projection `u_alpha = v_alpha · n̂_V,alpha`.

**Observer role:** the observer enters this estimator in exactly two places, both
deliberate — (a) `core_center_mask`, reused identically to the observer frame so both frames
see the identical candidate set before either estimator runs, and (b) the diagnostic
`per_center_axis_angle = arccos(clip(z_hat_alpha · n̂_V,alpha, −1, 1))`, which never feeds
back into `dipole`/`monopole`/`per_center_dipole`/`per_center_amplitude`. Outside those two
places there is no observer in the module at all — that is the entire point of the variant.

- `VelocityFrameShellDipoleResult` (frozen dataclass): mirrors
  `VelocityCenteredShellDipoleResult` field-for-field (`shell_edges`, `shell_centers`,
  `pair_count`, `monopole`, `dipole`, the `per_center_*` breakdown, `n_candidates`,
  `n_centers`), with `per_center_speed (N_c,)` in place of `per_center_u`, plus
  `per_center_axis_angle (N_c,)` (radians, `[0, π]`) — the pure diagnostic above. Invariants
  by construction: `dipole == per_center_dipole.sum(axis=0)`,
  `pair_count == per_center_count.sum(axis=0)`,
  `monopole == (per_center_speed[:, None] * per_center_count).sum(axis=0)`.
- `velocity_frame_shell_dipole(s_centers, v_centers, s_tracers, shell_edges, sub_volume_radius, core_margin=None, observer=None)`:
  copies `velocity_centered_shell_dipole`'s validation block, `cKDTree` loop, and binning
  semantics exactly. NEW validation: after the core cut, any SURVIVING center with
  `|v_alpha| == 0.0` raises `ValueError` naming
  `dvcorr.pipeline.velocity_centered.RunConfig.min_center_speed` as the place such centers
  are dropped upstream — `unit_vector` would otherwise silently return a zero row (A=0,
  speed=0), diluting the across-center mean/SEM rather than crashing (the same class of bug
  as hard rule 5). Zero surviving centers is a valid, NaN-free, all-zero `(0, B)`/`(0,)`
  result, identical contract to the observer frame.
- **Sign:** coherent infall gives a POSITIVE velocity-frame dipole — the SAME sign as the
  observer-frame ζ₁ (unlike the `(−1)^ℓ` flip between ζ and group-centered ξ_Tu), because the
  two frames coincide exactly in the pure-radial-flow limit (see
  `tests/test_velocity_frame_dipole.py`'s frame-agreement test). A negative dipole from an
  infall mock is an orientation bug, not a result.
- **Monopole:** `Σ_alpha |v_alpha| · N_alpha,b` is FLAT in `r` for a complete shell but offset
  to roughly the mean halo speed `⟨|v|⟩` (hundreds of km/s), not to zero, because `|v|` is
  positive-definite and cannot cancel across centers the way the observer frame's signed
  `u_alpha` does. The diagnostic content is the ABSENCE of the observer frame's r-dependent
  `2r/3R` finite-distance leakage trend, not proximity to zero — see the module docstring's
  Monopole section for the full argument, consumed by
  `dvcorr.pipeline.velocity_frame_comparison.make_comparison_figure`'s twin-axis monopole
  panel.

---

## `src/dvcorr/pipeline/` — reusable stage functions

### `src/dvcorr/pipeline/velocity_centered.py` **[done]**
The single source of truth for the velocity-centered ζ measurement pipeline, extracted out
of the script that used to contain it (`scripts/plot_velocity_centered_dipole.py`) so that
both the script and `notebooks/05_velocity_centered_dipole.ipynb` consume the same functions
instead of one reimplementing the other's logic (the "one library, two thin consumers" model
— see "Working model" below). Imports `from dvcorr import conventions`,
`from dvcorr.config import PathsConfig, ShellConfig`, and
`from dvcorr.estimators.shell_dipole import …`.

**Matplotlib backend discipline:** this module imports `matplotlib.pyplot` at module level
(for `make_figure`) but **never calls `matplotlib.use(...)`**. Only
`scripts/plot_velocity_centered_dipole.py` selects a backend (`Agg`), and it does so *before*
importing this module — so a notebook that imports `make_figure` directly keeps whatever
backend (e.g. an inline one) it already has.

- `RunConfig` (dataclass, hard rule 4): `sub_volume_radius`; `shells` — a COMPOSED
  `dvcorr.config.ShellConfig` (`field(default_factory=_default_shells)`), reusing its
  validation (ordering, `radii_step > 0`, `max_radius <= dvcorr.conventions.MAX_ANALYSIS_RADIUS`)
  and its `shell_edges` property rather than mirroring those fields and their checks a second
  time; `_default_shells()`'s `min_radius` defaults to `radii_step`, not 0 — candidates are
  subsampled from the tracer array, so every candidate is its own tracer at r = 0, and
  excluding that self-pair keeps it out of the monopole diagnostic panel; also
  `n_candidate_centers`, `seed`/`shuffle_seed`, output filename, `dpi`, and
  `min_center_speed` (default `1.0` km/s, `__post_init__`-validated `>= 0`) — the minimum
  `|v_alpha|` for a center to have a well-defined flow direction, consumed by
  `dvcorr.pipeline.velocity_frame_comparison.select_shared_centers` to drop centers from
  BOTH frames' shared center set at the orchestration level (never inside an estimator). It
  lives here rather than on `ComparisonRunConfig` because it filters the center set both
  frames consume, not a comparison-only knob.
- `load_and_carve(cfg, paths) -> (pos, vel)` — loads only
  `dvcorr.conventions.POSITION_COLUMNS + dvcorr.conventions.VELOCITY_COLUMNS`, carves a
  sphere of `sub_volume_radius` around `dvcorr.conventions.OBSERVER_POSITION` (plain
  Euclidean, no wrapping needed). Raises `RuntimeError` for a zero-row catalog or a
  zero-halo carve, hoisted before any percentage is printed from either count (no bare
  `ZeroDivisionError`).
- `draw_candidates(cfg, pos, vel) -> (s_candidates, v_candidates)` — seeded subsample of the
  carved halos; `RuntimeError` if that would be zero candidates.
- `run_estimator(cfg, s_candidates, v_candidates, s_tracers, observer) ->
  VelocityCenteredShellDipoleResult` — calls `velocity_centered_shell_dipole`, prints
  `n_candidates` vs `n_centers` (the core cut's visible loss), `RuntimeError` if zero
  centers survive.
- `global_number_density(n_tracers, sub_volume_radius) -> float` — n̄ = N /
  ((4π/3)·R_sub³).
- `NormalizedDipole` (frozen dataclass) — `zeta_hat`, `sem`, `zeta_hat_shuffle`, `sem_shuffle`,
  `monopole_norm`, all `(B,)`.
- `shell_dipole_norm_scale(shell_edges, n_bar) -> ndarray (B,)` — the ONE home for the
  per-shell scale `3 / (√(3/4π)·n̄V_b)` (via `expected_shell_occupancy`). Extracted (in a
  post-review pass on this task) specifically because `normalize_stacked_dipole` (below) AND
  `dvcorr.pipeline.velocity_frame_comparison.normalize_comparison`'s per-center summary both
  need the identical factor — before the extraction each computed its own copy of the same
  three lines, a drift risk between the plotted stack and the per-center breakdown that is
  supposed to sum back to it.
- `normalize_stacked_dipole(shell_edges, dipole, monopole, per_center_dipole, null_dipole,
  null_per_center_dipole, n_centers, n_bar) -> NormalizedDipole` — the ONE home for the
  normalization arithmetic (`ζ̂₁ = 3·(dipole/n_centers) / (√(3/4π)·n̄V_b)` via
  `shell_dipole_norm_scale`, SEM via `center_standard_error`), factored out so
  `dvcorr.pipeline.velocity_frame_comparison` can reuse it for the velocity frame, whose
  NULL is a different construction (a random-axis re-run, not a scalar permutation) — the
  null is therefore a parameter (`null_dipole`/`null_per_center_dipole`), supplied
  already-built by the caller, rather than constructed inside this function.
  `zeta_hat_shuffle`/`sem_shuffle` on the returned `NormalizedDipole` name "whatever null the
  caller built", not specifically a scalar permutation.
- `normalize_result(result, n_bar, shuffle_seed) -> NormalizedDipole` — UNCHANGED public
  signature, return type, and numerical output. Now builds the velocity-shuffle null (a
  seeded permutation of `per_center_u` recombined with `per_center_amplitude`, no second
  estimator pass) and delegates the arithmetic to `normalize_stacked_dipole`.
- `make_figure(cfg, result, normalized) -> Figure` — the two-panel figure (ζ̂₁ with SEM +
  shuffle null; the normalized monopole companion below, hard rule 6); builds and returns
  only, never saves or calls `plt.show`.

`src/dvcorr/pipeline/__init__.py` is a one-line docstring; no public re-exports needed — there
are now two pipeline modules (`velocity_centered.py` and `velocity_frame_comparison.py`, the
latter additive on top of the former, not a fork of it), each imported directly by name.

### `src/dvcorr/pipeline/velocity_frame_comparison.py` **[done]**
The single source of truth for the two-frame comparison (observer-frame ζ₁ vs. the
observer-free velocity-frame dipole), consumed by
`scripts/plot_velocity_frame_comparison.py`. Deliberately does NOT reimplement
`velocity_centered.py`'s earlier stages: `RunConfig` (subclassed), `NormalizedDipole`,
`normalize_result`, and `normalize_stacked_dipole` are imported from there; `load_and_carve`,
`draw_candidates`, and `global_number_density` are consumed directly by the script instead
(this module's own stage functions start one step later, from already-loaded candidates).
Same matplotlib-backend discipline as `velocity_centered.py` (never calls `matplotlib.use`).

- `ComparisonRunConfig(RunConfig)` — adds `comparison_output_name`,
  `angle_diagnostic_output_name`, and `axis_null_seed` (distinct from `seed`/`shuffle_seed` so
  the random-axis null never reuses another stream); `min_center_speed` stays on the parent
  `RunConfig` since it filters the shared center set, not a comparison-only knob.
- `SharedCenterSet` (frozen dataclass) / `select_shared_centers(cfg, s_candidates,
  v_candidates, observer) -> SharedCenterSet` — applies `core_center_mask` (identical function
  and `core_margin = cfg.shells.max_radius`, matching each estimator's own default), THEN the
  speed floor `keep = (speeds > 0.0) & (speeds >= cfg.min_center_speed)` (the explicit
  `speeds > 0.0` conjunct is a post-review fix: `speeds >= cfg.min_center_speed` alone is
  trivially true when `min_center_speed == 0.0`, since a norm can never be negative, which
  would silently keep exact-zero-speed centers despite that field's documented "0.0 means
  drop only exactly-zero-speed centers" contract); reports the funnel (`n_candidates` →
  `n_core` → `n_centers`, with `n_dropped_slow = n_core - n_centers` a returned field, not
  merely printed); `RuntimeError` if `s_candidates` is empty (checked BEFORE the survival
  percentage is printed, to avoid a bare `ZeroDivisionError`) or if zero centers survive both
  cuts. Selecting once here and handing the identical arrays to both estimators is what
  guarantees the only difference between the two frames is the axis and the scalar.
- `run_both_frames(cfg, centers, s_tracers, observer) -> FrameRunResults` — runs
  `velocity_centered_shell_dipole` and `velocity_frame_shell_dipole` on the IDENTICAL
  `centers.s_centers`/`v_centers`; asserts both estimators' `n_centers == centers.n_centers`
  (`RuntimeError` if not — the row-alignment invariant the comparison depends on); also runs
  `run_random_axis_null`.
- `run_random_axis_null(cfg, centers, s_tracers, observer) -> VelocityFrameShellDipoleResult`
  — replaces each center's velocity DIRECTION with an isotropic random unit vector
  (`np.random.default_rng(cfg.axis_null_seed)`) while keeping its speed and position, then
  reruns `velocity_frame_shell_dipole`. Why this null and not a scalar shuffle: the
  velocity-frame statistic is SELF-ALIGNED (the axis is built from the same vector that
  supplies the scalar), so permuting `|v_alpha|` among centers leaves every center's
  axis-density alignment intact and retains essentially the signal itself (`|v|` is
  positive-definite, unlike the observer frame's `u_alpha` which averages to ~0). Randomizing
  the AXIS is the actual guard against align-then-measure bias; it costs one extra estimator
  pass, deliberately.
- `FrameRunResults` (frozen dataclass): `obs_result`, `vel_result`, `vel_null_result`,
  `centers`.
- `normalize_velocity_frame_result(result, null_result, n_bar) -> NormalizedDipole` — thin
  delegate to `normalize_stacked_dipole` for the velocity frame; its `zeta_hat_shuffle` holds
  the random-axis null.
- `FrameComparison` (frozen dataclass): `obs`, `vel` (both `NormalizedDipole`),
  `shell_centers`, `per_center_delta` (radians), `per_center_dipole_difference`.
- `normalize_comparison(cfg, results, n_bar) -> FrameComparison` — normalizes both frames and
  builds the per-center frame-gap breakdown: each center's own normalized curve
  (`per_center_dipole * norm_scale_b`, no `/n_centers`) averaged over shells, differenced
  vel − obs, so the per-center decomposition sums back to the plotted stack
  (`mean_alpha(summary_alpha) == mean_b(zeta_hat_b)`).
- `make_comparison_figure(cfg, results, comparison) -> Figure` — the main deliverable: top
  panel overlays both frames' ζ̂₁ with SEM bands and their own (differently-constructed) nulls;
  bottom panel plots both frames' monopoles on a TWIN y-axis (obs left, vel right, each
  y-label colored to match), because the two monopoles sit at incomparable offsets (near zero
  vs. near `⟨|v|⟩`) by construction — a shared axis would hide the presence/absence of the
  r-dependent leakage trend that is the comparison's core finite-distance diagnostic.
- `make_angle_diagnostic_figure(cfg, comparison) -> Figure` — top panel bins
  `per_center_dipole_difference` by `per_center_delta` (`_N_ANGLE_BINS` bins over `[0, π]`,
  degrees on the axis) with an SEM errorbar over a scatter of individual centers; bottom panel
  histograms `delta` against the isotropic `P(delta) ∝ sin(delta)` reference. A few high-delta
  outliers driving the top-panel gap ⇒ bulk-flow contamination; a smooth spread ⇒ genuine
  projection geometry; an excess at low delta relative to the isotropic reference ⇒ flow
  directions aligned with the lines of sight (residual bulk motion).

---

## `src/dvcorr/selection/` — masks, selection functions, random catalogs **[empty]**
Reserved for CF4-like angular/radial masks, the selection function φ(r), and random
catalog generation. Port target: the old repo's `masks.py` (CF4-like selection).

## `src/dvcorr/mocks/` — MDPL2 observers, halo selection, mock covariance **[empty]**
Reserved for placing observers in the MDPL2 box, matching halo selection to the survey, and
building the mock-based covariance. Port target: the old repo's `overdensity.py` (periodic
KDTree overdensity), `data_loader.py`.

---

## `scripts/` — runnable, load-bearing scripts

Unlike `notebooks/`, code here is meant to be run directly (`python -m scripts.<name>`), not
explored interactively; results are load-bearing. `scripts/` is NOT part of the installed
`dvcorr` package (it sits outside `src/`); it is executed with `python -m` from the repo
root, importing the installed `dvcorr` package like any other consumer.

### `scripts/plot_velocity_centered_dipole.py` — thin driver **[done]**
All algorithmic content lives in `dvcorr.pipeline.velocity_centered` (see above); this script
only wires it together. Module body: `import matplotlib; matplotlib.use("Agg")` — called
here, at import time, BEFORE importing `dvcorr.pipeline.velocity_centered` (which imports
`matplotlib.pyplot` at module level) — then the `dvcorr.*` imports, then a single `main()`
that chains `load_and_carve` → `draw_candidates` → `run_estimator` →
`global_number_density` → `normalize_result` → `make_figure` and saves the PNG under
`PathsConfig().output_dir`. Never runs at import time (guarded by
`if __name__ == "__main__"`) since the MDPL2 catalog load is long; importing the module is
fast and safe (no data load), used as a smoke test.

Usage: `.venv/bin/python -m scripts.plot_velocity_centered_dipole`.

### `scripts/plot_velocity_frame_comparison.py` — thin driver **[done]**
Mirrors `plot_velocity_centered_dipole.py`'s structure exactly. All algorithmic content lives
in `dvcorr.pipeline.velocity_frame_comparison` (plus `load_and_carve`/`draw_candidates`/
`global_number_density` reused from `dvcorr.pipeline.velocity_centered`); this script only
wires it together. Module body: `import matplotlib; matplotlib.use("Agg")` before importing
either pipeline module, then a single `main()` that chains `load_and_carve` →
`draw_candidates` → `select_shared_centers` → `run_both_frames` → `global_number_density` →
`normalize_comparison` → `make_comparison_figure` and `make_angle_diagnostic_figure`, saving
BOTH PNGs (`cfg.comparison_output_name`, `cfg.angle_diagnostic_output_name`) under
`PathsConfig().output_dir`. Never runs at import time (`if __name__ == "__main__"`); importing
it is fast and safe, used as a smoke test.

Usage: `.venv/bin/python -m scripts.plot_velocity_frame_comparison`.

---

## `tests/`

- `tests/test_geometry.py` **[done]** — the sign gate plus primitive-level contracts.
  Imports `from dvcorr import conventions`, `from dvcorr.geometry import …`,
  `from dvcorr.estimators.shell_dipole import shell_dipole`.
  - `test_spherical_infall_gives_negative_dipole` — the load-bearing gate: infall → negative
    dipole through `shell_dipole`, recovering `V_INFALL` via `3·dipole/count`.
  - `test_infall_dipole_is_negative_from_geometry_alone` — the same sign, primitives only.
  - `test_reversing_the_pair_vector_flips_the_dipole`, `test_mu_is_a_cosine_and_spans_the_shell`.
  - `TestUnitVector`, `TestPairSeparation`, `TestMuCosine`, `TestPrimitivesCompose` — shapes,
    norms, orientation, clipping, zero handling.
  - Shared toy helpers `_sphere_directions`, `_infall_shell` (reused by the estimator tests).
- `tests/test_shell_dipole.py` **[done]** — estimator-level sign gate (reuses `_infall_shell`
  via `from tests.test_geometry import …`), isotropic/velocity-shuffle nulls, binning
  correctness, empty-shell/NaN edge cases, and input validation. Imports
  `from dvcorr import conventions`, `from dvcorr.geometry import unit_vector`,
  `from dvcorr.estimators.shell_dipole import shell_dipole`.
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
- `tests/test_velocity_frame_dipole.py` **[done]** — tests for the observer-free
  velocity-frame dipole (`dvcorr.estimators.velocity_frame_dipole`) and its pipeline
  (`dvcorr.pipeline.velocity_frame_comparison`). Imports the shared toys from
  `tests.test_geometry` (`_infall_shell_with_velocities`, `_sphere_directions`, `R_CENTER`,
  `R_SHELL`, `V_INFALL`, `N_SHELL`); reimplements (does not import) the private
  `_joint_gate_toy`/`_sample_ball` helpers from `test_velocity_centered_dipole.py`. Sign gate:
  the velocity-frame dipole is positive under infall and agrees in sign with the
  observer-frame ζ₁ on the identical toy — magnitude pinned via `dipole[0] /
  (√(3/4π)·pair_count[0]) == |V_INFALL|` (note: WITHOUT the observer-frame gate's `3×`
  prefactor, and to a TIGHT rather than loose tolerance — that toy's tracer sits exactly on
  the flow axis, `cos_theta ≡ 1` for every center by construction with no `⟨mu²⟩ = 1/3`
  shell-averaging to undo, so the `3×` used by the observer-frame recovery formula would be a
  spurious factor here, not a correction). Frame-agreement limit: purely radial flow makes the
  two frames agree exactly for both outbound and inbound velocities (the two sign flips in the
  inbound case cancel). Pipeline-level degenerate-center handling
  (`select_shared_centers`'s speed floor, reported via `n_dropped_slow`; the estimator's own
  zero-speed `ValueError`). `run_both_frames` row-alignment: identical `n_centers`,
  independently-recomputed per-center scalars, and identical `per_center_count` between
  frames. Null tests: an uncorrelated (uniformly random) density field gives both frames a
  null-consistent dipole; `run_random_axis_null` collapses the velocity-frame dipole on a
  clustered configuration with a genuine nonzero signal (documenting why a scalar permutation
  is not a valid null for this self-aligned statistic). Binning/empty-input/input-validation
  mechanics mirroring `test_velocity_centered_dipole.py`. `per_center_axis_angle` shape/range
  contract plus a hand-checked perpendicular case (`delta == pi/2`).
- `tests/test_settings.py` **[done]** — `dvcorr.config` dataclass tests (filename predates
  the `settings.py` → `dvcorr/config/*.py` split; left as-is, not renamed in this pass):
  `PathsConfig` derived paths and `ensure_output_dir` (via `tmp_path`, never the real
  `output/`); `CosmologyConfig` frozen-ness, `h`/`dvcorr.conventions.HUBBLE_PARAM` agreement,
  `to_colossus_dict`, and the inconsistent-H0 `ValueError`; `ShellConfig` edge/center shape
  and monotonicity, the `MAX_ANALYSIS_RADIUS` ceiling, invalid-input `ValueError`s, and a toy
  `shell_dipole` call on the produced edges; `SelectionConfig` valid/invalid construction;
  `Settings`/`default_settings` independence across instances. Imports
  `from dvcorr import conventions`, `from dvcorr.estimators.shell_dipole import shell_dipole`,
  `from dvcorr.config import (CosmologyConfig, PathsConfig, SelectionConfig, Settings,
  ShellConfig, default_settings)`.
- `tests/__init__.py` — present so `from tests.test_geometry import …` resolves when pytest
  is run from the repo root (`testpaths = ["tests"]` in `pyproject.toml`).

## `notebooks/` — exploration only
Nothing load-bearing; diagnostics that import the real modules, never reimplement them. No
`sys.path` bootstrap: `dvcorr` is editable-installed, so `import dvcorr…` works directly.

- `04_first_mdpl2_run.ipynb` **[done]** — first `shell_dipole` run on real MDPL2 halos:
  load → carve a boundary-safe sub-volume (periodicity guard) → single-central diagnostics
  (count ~ r², Q = Σμ ≈ 0, normalized dipole) → stack over centrals (raw-sum then normalize)
  → velocity-shuffle null. Imports `dvcorr.conventions`, `dvcorr.config.PathsConfig`,
  `dvcorr.geometry.unit_vector`, `dvcorr.estimators.shell_dipole.shell_dipole`. Its inline
  carve→measure→normalize→null logic is a pre-refactor duplicate of
  `dvcorr.pipeline.velocity_centered`'s pattern (for the group-centered estimator, not yet
  extracted into `dvcorr.pipeline`) — deliberately left as-is; de-duplicating it is a
  follow-up, not part of this reorganization. Exploratory, not a science result.
- `05_velocity_centered_dipole.ipynb` **[done]** — exploratory twin of
  `scripts/plot_velocity_centered_dipole.py`; every code cell calls one of
  `dvcorr.pipeline.velocity_centered`'s stage functions (`load_and_carve`, `draw_candidates`,
  `run_estimator`, `global_number_density`, `normalize_result`, `make_figure`) in sequence —
  nothing is reimplemented, and both the notebook and the script import from the same
  library module. One stage per cell, short markdown between them (what a positive ζ̂₁ means
  under the velocity-centered sign convention, the monopole panel as the trust diagnostic,
  the shuffle null). Final cell saves the PNG via the same `paths.output_dir` / `cfg` logic as
  the script's `main()`, then displays the returned `Figure` inline (never `plt.show`). Not
  executed as part of authoring — outputs are cleared.
- `06_velocity_frame_comparison.ipynb` **[done]** — exploratory twin of
  `scripts/plot_velocity_frame_comparison.py`, and the notebook that actually *runs* the
  two-frame comparison. Every code cell calls one stage of
  `dvcorr.pipeline.velocity_frame_comparison` (`select_shared_centers`, `run_both_frames`,
  `normalize_comparison`, `make_comparison_figure`, `make_angle_diagnostic_figure`) plus the
  three stages it imports unchanged from `dvcorr.pipeline.velocity_centered`
  (`load_and_carve`, `draw_candidates`, `global_number_density`) — nothing is reimplemented,
  and the notebook and the script drive the identical library functions. One stage per cell,
  markdown between them covering: the observer-free axis construction; the **three**
  contributors to an amplitude gap (self-alignment/no ⟨cos²θ⟩=1/3 dilution ≈ 3×, `|v|` vs
  `v·n̂_V`, then axis rotation proper — not attributable to rotation alone); why the velocity
  frame needs a **random-axis** null rather than a scalar shuffle; the twin-axis monopole
  panel (both monopoles inherit the `1+ξ_hh(r)` occupancy ratio and so both decline — the
  frames separate only after dividing it out, leaving the observer-frame `2r/3R` trend
  against a velocity-frame residual flat at ⟨|v|⟩); and the
  δ_α diagnostic's reading (few high-δ outliers ⇒ bulk-flow contamination, smooth spread ⇒
  projection geometry; low-δ excess over the isotropic `sin δ` reference ⇒ residual bulk
  motion). Saves both PNGs via the same `paths.output_dir` / `cfg` logic as the script's
  `main()` and displays each returned `Figure` inline (never `plt.show`). Unlike 04/05 this
  one **is** executed and its outputs are kept — it is the record of the comparison run.

## `Imports from old repo/` — reference dump (read only)
The bulk-flow project's working code, not a package and not importable. Copy and adapt
deliberately; never wire live imports into it. `src/dvcorr/config/`'s one-dataclass-per-file
plus aggregator pattern mirrors this repo's `src/config/` (`app_config.py`, `paths_config.py`,
`cosmology_config.py`, `bulkflow_config.py`, …).

---

## Working model — how `src/dvcorr/`, `scripts/`, and `notebooks/` relate

One library, two thin consumers:

- **`src/dvcorr/` is the single source of truth.** All reusable logic — pure geometry,
  estimators, pipeline stage functions, plotting helpers — lives here, is importable, and is
  tested. Nothing load-bearing lives outside it.
- **`scripts/` are thin, headless, reproducible drivers.** A script wires library stage
  functions and writes outputs (PNG/HDF5). Orchestration only, no algorithm.
  `python -m scripts.<name>`.
- **`notebooks/` are interactive exploration and presentation.** They import the *same*
  library stage functions and call them, adding plots and narrative. They never reimplement
  pipeline logic (notebook 05 models this fully; notebook 04 predates the pattern and is the
  documented exception above).
- **Graduation rule:** if you write a function in a notebook or script you'd want to reuse,
  it moves into `src/dvcorr/` and gets imported back.
- Because `dvcorr` is editable-installed, `import dvcorr…` behaves identically everywhere.
