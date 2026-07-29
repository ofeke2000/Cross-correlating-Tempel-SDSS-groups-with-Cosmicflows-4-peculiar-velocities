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
- `radial_flow_axis(v, n_hat_los) -> (z_hat, u)` — **the one definition site of `ẑ`** for
  every velocity-centered construction (`dvcorr.conventions.VELOCITY_AXIS_CONVENTION`):
  `u = v · n̂_los` (signed), `ẑ = sign(u) · n̂_los`. The axis follows the object's **motion**
  along the line of sight, not the line of sight itself — a halo approaching the observer has
  `ẑ = −r̂`. The companion weight is the **speed** `|u|`, never the signed `u`: the sign lives
  in the axis now, and entering it twice cancels the statistic. `u == 0` (purely transverse
  motion) returns a zero axis row — undefined, not invented; that object also carries weight
  `|u| = 0` and so contributes nothing.

**PBC contract (important):** these primitives do *not* apply the minimum-image convention.
Periodicity is discharged upstream by carving a sub-volume around each center and unwrapping
it into that center's continuous frame; coordinates reaching `geometry.py` are already the
nearest image of their center. Re-adding a minimum-image reduction here would corrupt
already-unwrapped separations. (This refines CLAUDE.md hard rule 3: PBC is honoured at the
carving step, not in the primitives. The carving step lives in `pipeline/velocity_centered.py`
for the velocity-centered path — see below.)

### `src/dvcorr/redshift_space.py` — the redshift-space coordinate transform **[done]**
The redshift-space run's ONLY new geometry: displace a halo along its own observer line of
sight by its own radial peculiar velocity, `sᵢ = observer + (|rᵢ − observer| + v_r,ᵢ/100)·n̂ᵢ`
with `n̂ᵢ = unit_vector(rᵢ − observer)` and `v_r,ᵢ = vᵢ · n̂ᵢ`. A coordinate transform only —
no estimator, no pipeline knowledge (centers/tracers/shells/n̄ all live one level up, in
`dvcorr.pipeline.redshift_space_comparison`) — and NOT wired into
`velocity_centered_shell_dipole`, which is unchanged and stays agnostic to which positions it
receives; that agnosticism is a design property this module is built to preserve. Imports
`from dvcorr import conventions`, `from dvcorr.geometry import radial_flow_axis, unit_vector`.

- The `/100` is the SAME definitional literal as `dvcorr.config.cosmology
  .CosmologyConfig.h` (H0 = 100 h km/s/Mpc, so km/s → h⁻¹ Mpc regardless of h's numeric
  value) — kept inline with a comment, not promoted to `conventions`.
- **Exactness, not approximation:** MDPL2 snapshot 125 is z ≈ 0, so this displacement is
  EXACT in comoving box coordinates — there is no redshift-to-distance conversion (no
  cz/H0, no ∫dz/E(z)) anywhere in this transform for it to be a shortcut for; that question
  belongs to the future real-data (CF4/Tempel) pipeline. No sub-percent accuracy figure is
  claimed anywhere in this module.
- **n̂ is invariant.** The displacement is purely radial, so it cannot rotate the line of
  sight (except via the sign flip the flip guard below catches); consequently
  `u_α = v_α · n̂_V,α` is IDENTICAL whether `n̂_V,α` is computed from a center's real or
  redshift-space position — no carry-over parameter, no new argument anywhere downstream.
  Verified numerically on the project's shared infall toy: n̂ agrees to 3.3e-16, u_α to
  8.5e-14 (floating-point round-off).
- `FLIP_GUARD_FLOOR` (module constant, 1.0 h⁻¹ Mpc) — a **strictly positive** floor on
  `s_mag = |r − observer| + v_r/100`, not `s_mag < 0`: at `s_mag ≈ 0`, `geometry.unit_vector`
  returns a zero row silently (no NaN), so a naive `< 0` guard would let a near-zero,
  effectively-undefined direction through. `s_mag` is an OBSERVER-CENTERED RADIAL DISTANCE, not
  a pair separation, so it has no relationship to the shell binning — re-justified against the
  quantity it actually bounds: a negligible volume fraction of the carved sub-volume
  (`R_sub = 300 h⁻¹Mpc`, of which a 1 h⁻¹Mpc ball is `(1/300)³ ≈ 4e-8`, expected occupancy
  `n̄·(4π/3)·1³ ≈ 0.017` halos — dropping essentially nothing that is not a genuine
  through-observer flip, independently of whatever radial shell binning is in use) and well
  above floating-point round-off on `BOX_SIZE ~ 1e3` coordinates. No cross-reference to any
  `dvcorr.pipeline.velocity_centered` shell-binning constant — that cross-module coupling was
  removed; the floor stands on its own justification.
- `radial_velocity(positions, velocities, observer=None) -> (N,)` — thin wrapper over
  `geometry.radial_flow_axis`, keeping only its `u` return value (the flow-signed axis this
  module never needs). Used to derive `v_margin` BEFORE the flip guard applies.
- `RedshiftSpaceTransform` (frozen dataclass): `s_redshift (N_kept, 3)`,
  `kept_mask (N,) bool` (aligned with the ORIGINAL input, so `positions[kept_mask]` is the
  real-space counterpart of `s_redshift` row for row), `n_input`, `n_dropped`,
  `flip_guard_floor`.
- `to_redshift_space(positions, velocities, observer=None, flip_guard_floor=FLIP_GUARD_FLOOR)
  -> RedshiftSpaceTransform` — the transform. `velocities` must be finite everywhere (hard
  rule 5); `flip_guard_floor` must be `> 0`.
- `v_margin_from_statistic(v_r, statistic, percentile) -> float` — `"max"` (a TRUE bound,
  `max(|v_r|)`) or `"percentile"` (cheaper, not a true bound — the caller must additionally
  enforce it as a global cut, see `redshift_space_comparison.py` below). Defined on `|v_r|`,
  never the 3-D speed `|v|` (a percentile of `|v|` would under-cover the actual radial
  displacement, since `|v| ≥ |v_r|` always).

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
`max_radius`, `radii_step`, `sigma_star` (small-scale velocity noise, km/s), plus `spacing`
and `n_bins` (below). `__post_init__` validates ordering, `radii_step > 0`,
`max_radius <= dvcorr.conventions.MAX_ANALYSIS_RADIUS`, `n_bins` a positive integer,
`spacing` one of `_VALID_SPACINGS`, and — the log-mode guard — `min_radius > 0` whenever
`spacing == SPACING_LOG` (`np.geomspace` is undefined at zero; enforced here, not inside
`log_shell_edges` itself).

- `SPACING_LINEAR = "linear"`, `SPACING_LOG = "log"` — named constants, not bare strings, so
  a typo'd spacing raises in `__post_init__` instead of silently falling through (hard rules 4
  and 9).
- `linear_shell_edges(min_radius, max_radius, radii_step) -> (B+1,)` — `min_radius` to
  `max_radius` in steps of `radii_step`; `max_radius` appended exactly (any interior edge
  `>= max_radius` from `np.arange`'s float-accumulation overshoot is dropped first, so the
  append cannot duplicate the outer edge), clipped to `MAX_ANALYSIS_RADIUS`.
- `log_shell_edges(min_radius, max_radius, n_bins) -> (B+1,)` — `n_bins` shells geometrically
  spaced via `np.geomspace`, which pins both endpoints exactly (no drop-then-append dance
  needed, unlike the linear builder), clipped to `MAX_ANALYSIS_RADIUS` belt-and-braces. Caller
  obligation (`min_radius > 0`) enforced at `ShellConfig.__post_init__`, not here.
- `volume_weighted_shell_radii(shell_edges) -> (B,)` — `r_eff = (3/4)*(r2**4-r1**4)/(r2**3-r1**3)`,
  the plotting abscissa: the first moment of the same `r**2 dr` weight
  `dvcorr.estimators.shell_dipole.expected_shell_occupancy`'s `n_bar*V_b` denominator already
  divides by (see that entry below for the cross-reference), so ordinate and abscissa agree.
  NOT the geometric mean (that is only correct for a weight uniform in log r, which this is
  not) and NOT a pair-count-weighted mean (would pick up the realization-dependent clustering
  factor `1+xi(r)` the denominator doesn't carry, and give two curves on a comparison figure
  different x-coordinates). `edges[0] == 0` is fine, giving the closed form `0.75*r2`.
- `ShellConfig.shell_edges (B+1,)` — dispatches to `log_shell_edges` or `linear_shell_edges` on
  `spacing`.
- `ShellConfig.shell_centers (B,)` — the plain midpoint `0.5*(edge_low+edge_high)`, deliberately
  NOT volume-weighted, and deliberately left alone everywhere except the four plot sites listed
  under `pipeline/velocity_centered.py` below (three existing tests pin `shell_centers ==
  [5, 15, 25]` and must keep passing untouched). `shell_effective_radii` (below) is the
  volume-weighted quantity actually used for plotting; `r_eff` names that quantity throughout
  the pipeline modules.
- `ShellConfig.shell_effective_radii (B,)` — thin property wrapper over
  `volume_weighted_shell_radii(self.shell_edges)`.

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
ShellConfig, SelectionConfig, SPACING_LINEAR, SPACING_LOG, linear_shell_edges,
log_shell_edges, volume_weighted_shell_radii` — a clean import surface over the five
dataclass files above, plus the spacing constants and free functions `shells.py` also
carries (see that entry above).

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
**flow-signed axis** `ẑ_α = sign(u_α)·n̂_V,α` from `geometry.radial_flow_axis` (not the frozen
`n̂_T`, and not the bare `n̂_V`) — same primitives (`pair_separation`, `unit_vector`,
`mu_cosine`), different orientation, by design. Related to `xi_Tu` by
`ζ_ℓ = (−1)^ℓ ξ_Tu,ℓ` (`dvcorr.conventions.nusser_multipole_sign`): monopoles agree, dipoles
flip — coherent infall gives `ξ_Tu,1 < 0` and therefore `ζ_1 > 0`.

**Axis sign — what it moves and what it does not.** The per-center weight is the radial
**speed** `|u_α|`, since the sign now rides on `ẑ_α`. Flipping `ẑ` flips `A_α,b`, and the
weight flips with it, so `|u_α|·A(ẑ_α) ≡ u_α·A(n̂_V,α)`: **`dipole` and the plotted `ζ̂₁` are
numerically unchanged** by the signed axis, which is what keeps the joint sign gate valid
across the change. What *does* change: `monopole` becomes `Σ|u_α|N_α,b` and so sits near
`⟨|u|⟩` rather than near zero; `per_center_amplitude` is measured against the flow direction,
so its sign is physical rather than an artifact of which side of the observer the center sits
on; and the shuffle null must undo the flip before permuting (see `normalize_result` below).

- `real_y10(direction_cosine) -> (N,)` — real ℓ=1, m=0 spherical harmonic,
  `√(3/4π)·cosθ`; pure numpy, pinned against `scipy.special.sph_harm_y`. m=±1 out of scope
  (would need the per-center frame's transverse axes, never needed by m=0).
- `core_center_mask(s_centers, sub_volume_radius, core_margin, observer=None) -> (N_c,) bool`
  — keeps candidates whose full shell (out to `core_margin`, typically `r_max`) fits inside
  the spherical sub-volume: `|s_α − observer| ≤ sub_volume_radius − core_margin`. Suppresses
  the boundary-truncation bias suspected behind the first MDPL2 run's ~+13 km/s null offset.
- `expected_shell_occupancy(number_density, shell_edges) -> (B,)` — `n̄·V_b`, the Nusser
  eq. 24 normalization denominator (expected, not realized, occupancy); `n̄` supplied by the
  caller. Matching first moment of the same `r²dr` weight:
  `dvcorr.config.shells.volume_weighted_shell_radii` is `∫r·r²dr / ∫r²dr` over the same shell,
  the abscissa for whatever this function normalizes as an ordinate.
- `center_standard_error(per_center_values) -> (B,)` — `std(axis=0, ddof=1)/√N_c` across
  centers; NaN for `N_c < 2`. Treats centers as independent (documented understatement of
  the true uncertainty; mock covariance is the eventual replacement).
- `VelocityCenteredShellDipoleResult` (frozen dataclass): `shell_edges (B+1,)`,
  `shell_centers (B,)`, `pair_count (B,)`, `monopole (B,)`, `dipole (B,)` — all raw sums,
  stacked over centers — plus `per_center_dipole/amplitude/count (N_c, B)`,
  `per_center_u (N_c,)` (the **signed** `u_α`, retained because its sign is the only record
  of which way `ẑ_α` was flipped, which the null needs to undo it) and
  `per_center_speed (N_c,)` (`|u_α|`, the weight actually applied) for a downstream error
  band / shuffle null without a second pass, and `n_candidates`, `n_centers` (int) so the
  core cut's volume loss is visible. Invariants by construction:
  `monopole == (per_center_speed[:, None] * per_center_count).sum(axis=0)` and
  `per_center_speed == np.abs(per_center_u)`.
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

- axis: `z_hat_alpha = unit_vector(v_alpha)` (the center's **full** flow direction) instead of
  the observer frame's radial projection of it, `ẑ_obs,α = sign(u_α)·n̂_V,α`.
- scalar: `speed_alpha = |v_alpha|` instead of the observer frame's radial speed `|u_alpha|`.

Both frames are therefore the same construction — axis along the motion, weight the speed
along that axis — differing only in full 3-vector vs. radial projection.

**Observer role:** the observer enters this estimator in exactly two places, both
deliberate — (a) `core_center_mask`, reused identically to the observer frame so both frames
see the identical candidate set before either estimator runs, and (b) the diagnostic
`per_center_axis_angle = arccos(clip(z_hat_alpha · ẑ_obs,α, −1, 1))` — measured against the
observer frame's **signed** axis, so `cos δ = |u_α|/|v_α| ≥ 0` and `δ ∈ [0, π/2]`, a ceiling
the function **checks** (`RuntimeError` past float slack) rather than assumes. It never feeds
back into `dipole`/`monopole`/`per_center_dipole`/`per_center_amplitude`. Outside those two
places there is no observer in the module at all — that is the entire point of the variant.

- `VelocityFrameShellDipoleResult` (frozen dataclass): mirrors
  `VelocityCenteredShellDipoleResult` field-for-field (`shell_edges`, `shell_centers`,
  `pair_count`, `monopole`, `dipole`, the `per_center_*` breakdown including
  `per_center_speed (N_c,)`, `n_candidates`, `n_centers`) — the frames differ in what
  `per_center_speed` HOLDS (`|v_α|` here, `|u_α|` there), and there is no `per_center_u` here
  (no axis sign to record). Adds `per_center_axis_angle (N_c,)` (radians, `[0, π/2]`) — the
  pure diagnostic above, with no observer-frame analogue. Invariants
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
- **Monopole:** `Σ_alpha |v_alpha| · N_alpha,b` is offset to roughly the mean halo speed
  `⟨|v|⟩` (hundreds of km/s), not to zero, because `|v|` is positive-definite and has no
  near/far cancellation available to it. Since the axis was signed, the observer frame's
  monopole shares that property (its weight is likewise a speed, `|u_α|`, so it sits near
  `⟨|u|⟩`) — the two are now the same *kind* of quantity, a projection factor apart. Neither
  is flat on a clustered field: both inherit the `1+ξ_hh(r)` occupancy ratio, and dividing it
  out leaves the **same** speed–density correlation in both (−7.9% / −7.8% on the first MDPL2
  run with the signed axis).
- ⚠️ **The `2r/3R` finite-distance leakage is no longer visible in either monopole.** It is a
  *signed* effect that lived in `Σu_α N_α,b` — a quantity near zero, so an ~11 km/s trend
  stood out against it — and `|u|` discards exactly that sign. This is a real loss of ℓ=0
  diagnostic content, and it is why `per_center_u` is retained **signed**: the leakage
  diagnostic is recoverable as `(per_center_u[:, None] * per_center_count).sum(axis=0)`,
  normalized like `monopole`. Notebook 06's summary cell prints it alongside the
  speed-weighted pair, and the trend still survives the occupancy division there. Promote that
  line to a library function if it becomes a routine pipeline step rather than a notebook
  read-out. See the module docstring's Monopole section for the full argument.

---

## `src/dvcorr/pipeline/` — reusable stage functions

### `src/dvcorr/pipeline/velocity_centered.py` **[done]**
The single source of truth for the velocity-centered ζ measurement pipeline, extracted out
of the script that used to contain it (`scripts/plot_velocity_centered_dipole.py`) so that
both the script and `notebooks/05_velocity_centered_dipole.ipynb` consume the same functions
instead of one reimplementing the other's logic (the "one library, two thin consumers" model
— see "Working model" below). Imports `from dvcorr import conventions`,
`from dvcorr.config import PathsConfig, ShellConfig`, and
`from dvcorr.estimators.shell_dipole import …` (now including `core_center_mask`, needed by
`select_shared_centers` below).

Also the shared base BOTH comparison pipelines (`velocity_frame_comparison.py`,
`redshift_space_comparison.py`) import from: `SharedCenterSet` and `select_shared_centers`
moved here FROM `velocity_frame_comparison.py` in the redshift-space task (they were
originally written there, for the observer/velocity-frame comparison only) once the
redshift-space comparison needed them too — leaving them in `velocity_frame_comparison.py`
would have made the redshift-space pipeline depend on it, when both are equally built on top
of this module. `velocity_frame_comparison.py` now imports both names from here and
re-exports them, so existing code importing them FROM `velocity_frame_comparison` is
unaffected.

**Matplotlib backend discipline:** this module imports `matplotlib.pyplot` at module level
(for `make_figure`) but **never calls `matplotlib.use(...)`**. Only
`scripts/plot_velocity_centered_dipole.py` selects a backend (`Agg`), and it does so *before*
importing this module — so a notebook that imports `make_figure` directly keeps whatever
backend (e.g. an inline one) it already has.

- `RunConfig` (dataclass, hard rule 4): `sub_volume_radius`; `shells` — a COMPOSED
  `dvcorr.config.ShellConfig` (`field(default_factory=_default_shells)`), reusing its
  validation (ordering, `radii_step > 0`, `max_radius <= dvcorr.conventions.MAX_ANALYSIS_RADIUS`,
  the log-mode `min_radius > 0` guard) and its `shell_edges` property rather than mirroring
  those fields and their checks a second time; also `n_candidate_centers`, `seed`/
  `shuffle_seed`, output filename, `dpi`, and `min_center_speed` (default `1.0` km/s,
  `__post_init__`-validated `>= 0`) — the minimum `|v_alpha|` for a center to have a
  well-defined flow direction, consumed by `select_shared_centers` (below) to drop centers
  from a shared center set at the orchestration level (never inside an estimator). It lives
  here rather than on a comparison-only subclass because it filters the center set every
  comparison pipeline consumes, not a comparison-only knob.
- `_default_shells()`'s default binning is now LOG spacing: `_DEFAULT_SPACING = SPACING_LOG`,
  `_DEFAULT_MIN_RADIUS = 1.0`, `_DEFAULT_MAX_RADIUS = 64.0`, `_DEFAULT_N_BINS = 12` (module
  constants, replacing the old `_DEFAULT_RADII_STEP = 5.0` / `_DEFAULT_MAX_RADIUS = 60.0`
  linear pair — `_DEFAULT_RADII_STEP` no longer exists anywhere in the codebase).
  `radii_step` is left at `ShellConfig`'s own class default (inert under `SPACING_LOG`).
  Resulting edges: `1, 1.41, 2, 2.83, 4, 5.66, 8, 11.31, 16, 22.63, 32, 45.25, 64` — a ratio of
  exactly √2 between consecutive edges (exact powers of two and their halves), FIVE of the
  twelve bins below 5.66 h⁻¹Mpc.
  - The r = 0 self-pair — candidates are subsampled from the tracer array, so every candidate
    is its own tracer at r = 0, and an unguarded `min_radius = 0` bin would collect that pure
    self-correlation into `pair_count[0]`/`monopole[0]`, polluting exactly the hard-rule-6
    monopole diagnostic panel — is excluded STRUCTURALLY now, not by convention:
    `ShellConfig.__post_init__` hard-raises on `min_radius <= 0` under `SPACING_LOG`.
  - `min_radius = 1.0`: a statistical floor (≥ 50 expected uniform-field pairs across the stack,
    `n_bar·V_b·N_c` with n̄ ≈ 4e-3 (h⁻¹Mpc)⁻³ and N_c = 1933, measured on the real run, gives
    r₁ ≥ 0.94) is not the binding one — the catalog is `pid = -1` (distinct halos only), so
    every center carries a hard exclusion hole of `R_vir(host)` (0.20 h⁻¹Mpc at 1e12 M☉/h,
    0.94 at 1e14, 2.0 at 1e15), and the massive centers that dominate the `|u|`-weighted stack
    have the LARGEST holes. MDPL2's force softening (5 h⁻¹kpc) sits ~100× below this and is NOT
    the limit. The innermost one or two bins are the exclusion / one-halo regime, deliberately
    SHOWN (alongside the monopole companion, hard rule 6) rather than hidden behind one wide bin.
  - `max_radius = 64.0` (not 60): every `query_ball_point(s_alpha, edges[-1])` ball grows by
    `(64/60)³ ≈ 1.21`, and `core_margin` (`select_shared_centers`'s default, below) tightens
    from 240 to 236 h⁻¹Mpc, so surviving centers go from ~2048 to ~1933 of 4000 candidates
    (−5.6%), measured on the real run (48.3% of 4000 candidates survive; the carve itself keeps
    464764 of 4093751 halos within R_sub = 300) — worth stating since it moves the `N_c` printed
    in every figure title. Runtime is independent of bin COUNT (`np.bincount` is O(pairs)), so
    12 log bins cost the same as the old 11 linear ones; the cost above is entirely from
    `r_max`, not `n_bins`.
- `_load_all_halos(paths) -> (pos_all, vel_all)` — reads
  `dvcorr.conventions.POSITION_COLUMNS + dvcorr.conventions.VELOCITY_COLUMNS` for every halo
  in the catalog, uncarved. Factored out of `load_and_carve` (which calls this, then carves)
  so `dvcorr.pipeline.redshift_space_comparison.load_and_carve_buffered` can build its OWN
  two-pass carve (plain `sub_volume_radius`, then buffered) from a SINGLE CSV read rather than
  re-reading the ~4M-row catalog once per carve. `RuntimeError` on a zero-row catalog.
- `load_and_carve(cfg, paths) -> (pos, vel)` — calls `_load_all_halos`, then keeps halos
  within `sub_volume_radius` of `dvcorr.conventions.OBSERVER_POSITION` (plain Euclidean, no
  wrapping needed). Raises `RuntimeError` for a zero-row catalog or a zero-halo carve, hoisted
  before any percentage is printed from either count (no bare `ZeroDivisionError`).
- `draw_candidates(cfg, pos, vel) -> (s_candidates, v_candidates)` — seeded subsample of the
  carved halos; `RuntimeError` if that would be zero candidates.
- `SharedCenterSet` (frozen dataclass) / `select_shared_centers(cfg, s_candidates,
  v_candidates, observer, core_margin=None) -> SharedCenterSet` — MOVED here from
  `velocity_frame_comparison.py` (see above); the only substantive change from the move is
  that `core_margin` is now an EXPLICIT argument (previously hardcoded to
  `cfg.shells.max_radius` inside the function body) defaulting to `None` ->
  `cfg.shells.max_radius`, and `cfg`'s type hint loosened from `ComparisonRunConfig` to base
  `RunConfig` (the function only ever touched `cfg.sub_volume_radius` and
  `cfg.min_center_speed`, both base fields). Applies `core_center_mask` (the IDENTICAL
  function every estimator applies internally) THEN the speed floor
  `keep = (speeds > 0.0) & (speeds >= cfg.min_center_speed)` (the explicit `speeds > 0.0`
  conjunct matters specifically for `min_center_speed == 0.0`, otherwise trivially true and
  silently keeping exact-zero-speed centers despite the documented "0.0 means drop only
  exactly-zero" contract). Reports the funnel (`n_candidates` → `n_core` → `n_centers`,
  `n_dropped_slow = n_core - n_centers` a returned field). `RuntimeError` if `s_candidates` is
  empty (checked before the survival percentage is printed) or if zero centers survive both
  cuts. Selecting once and handing the identical arrays to every estimator run on the result
  is what guarantees the only difference between runs built on the same `SharedCenterSet` is
  whatever each run deliberately varies.
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
- `normalize_result(result, n_bar, shuffle_seed) -> NormalizedDipole` — builds the
  velocity-shuffle null and delegates the arithmetic to `normalize_stacked_dipole`. The null
  **undoes the axis flip first** — `np.sign(per_center_u) * per_center_amplitude` recovers the
  amplitude against the fixed `+n̂_V,α` — and then permutes the **signed** `per_center_u`
  against that; still no second estimator pass. The undo is load-bearing, not bookkeeping:
  permuting the positive-definite `per_center_speed` against untouched amplitudes is **not a
  null at all** (it retains the signal — the same trap `run_random_axis_null` documents for
  the velocity frame, pinned by `test_permuting_the_speed_alone_is_not_a_null`). Numerically
  the null curve is unchanged from the pre-signed-axis version, since `sign(u)·A = A(n̂_V)`.
- `apply_radial_axis_scale(ax, shells) -> None` — put a radial x-axis in log or linear scale to
  match `shells.spacing`, taking a `ShellConfig` (not a `RunConfig`) so it works unchanged for
  all three config types (`RunConfig`, `ComparisonRunConfig`, `RedshiftSpaceRunConfig`).
  `SPACING_LINEAR` returns immediately — existing linear figures are unchanged, the stated
  backward-compatibility guarantee. `SPACING_LOG`: `ax.set_xscale("log")`;
  `matplotlib.ticker.LogLocator(base=_LOG_TICK_BASE, subs=_LOG_TICK_SUBS)` (module constants
  `_LOG_TICK_BASE = 10.0`, `_LOG_TICK_SUBS = (1.0, 2.0, 5.0)`) with a `ScalarFormatter` on the major axis (labels read
  "1, 2, 5, 10, 20, 50" rather than bare powers of ten — matplotlib's default `LogLocator`
  would label only `10^0`/`10^1` over a sub-decade range like `[1, 64]`) and `NullLocator()` on
  the minor axis; `ax.set_xlim(edges[0], edges[-1])` from `shells.shell_edges` — necessary,
  not cosmetic, since log autoscale pads MULTIPLICATIVELY and would otherwise push the left
  edge toward ~0.1. Called on the BOTTOM axis of each shared-x figure (the one carrying
  `set_xlabel`); `sharex=True` propagates to the panel(s) above. Defined here, the shared base
  both comparison pipelines import from; `velocity_frame_comparison.py` and
  `redshift_space_comparison.py` (below) import and call it, not redefine it.
- `apply_dipole_axis_scale(ax, shells) -> None` — companion to `apply_radial_axis_scale`, but
  for a dipole/monopole y-axis instead of the radial x-axis; same backward-compatibility
  guarantee (`SPACING_LINEAR` returns immediately, doing nothing). Under `SPACING_LOG`:
  `ax.set_yscale("symlog", linthresh=_SYMLOG_LINTHRESH)` with module constant
  `_SYMLOG_LINTHRESH = 10.0` km/s. SYMLOG, not plain log, because the shuffle null (and the
  frame/axis-rotation differences the comparison figures plot) crosses zero and goes negative
  — a plain log y-axis would silently drop those points. `linthresh = 10.0` km/s: below it the
  curves sit at or below their own SEM, so a linear region there loses no visible structure;
  above it the real run's ~2 decades of dynamic range (670 km/s at r ≈ 1.2 down to 20 km/s at
  r ≈ 56) get real vertical space instead of being compressed into a sliver at the top of a
  linear axis — before this, the per-shell normalization scale spanned a factor of 92,700
  across the 12 log bins (57 under the old linear default), squashing everything beyond
  r ≈ 8 h⁻¹Mpc, where the ~13 km/s signal this project measures actually lives, into ~7% of
  panel height. `axhline(0.0)` renders correctly under symlog (defined at zero, unlike plain
  log). Called on every dipole/monopole axis under log spacing — both panels of `make_figure`
  (below); `make_comparison_figure`'s dipole panel and BOTH sides of its twin-axis monopole
  panel (`velocity_frame_comparison.py`); both panels of `make_redshift_comparison_figure` and
  `make_single_center_figure` (`redshift_space_comparison.py`) — **not**
  `make_angle_diagnostic_figure`, whose axes are angular (degrees), not radial-dipole y-axes.
  Same shared-base pattern as `apply_radial_axis_scale`: defined once here, imported by the
  other two pipeline modules, not redefined.
- `_binning_description(shells) -> str` — one-line binning summary for a figure title (e.g.
  `"r in [1, 64] h⁻¹Mpc, 12 log bins"` or `"r in [20, 150] h⁻¹Mpc, step=10 h⁻¹Mpc"`), so a saved
  PNG states its own binning instead of only `r_max`. Formats both endpoints with `:.4g`, not
  `:.2g` — `:.2g` rendered 150.0 as `"1.5e+02"` (scientific notation), a mismatch with this very
  worked example that a review caught; `:.4g` stays in plain decimal form across the range this
  project can reach (1 to `conventions.MAX_ANALYSIS_RADIUS` = 500 h⁻¹Mpc) while still trimming
  trailing zeros. Same shared-base pattern as `apply_radial_axis_scale`: defined once here,
  imported by the other three figure builders.
- `make_figure(cfg, result, normalized) -> Figure` — the two-panel figure (ζ̂₁ with SEM +
  shuffle null; the normalized monopole companion below, hard rule 6); builds and returns
  only, never saves or calls `plt.show`. The monopole panel carries **no zero reference
  line**: with the `|u_α|` weight the curve sits near `⟨|u|⟩`, and a `y=0` line would only
  compress the axis and imply a reference that no longer applies. Plotting abscissa is
  `dvcorr.config.volume_weighted_shell_radii(result.shell_edges)` (`r_eff`), NOT
  `result.shell_centers` (the plain midpoint, left alone — see the `ShellConfig` entry above);
  `apply_radial_axis_scale(ax_mono, cfg.shells)` and `apply_dipole_axis_scale(ax, cfg.shells)`
  (both panels) are called on the respective axes, and the title now includes
  `_binning_description(cfg.shells)` in place of the old bare `r_max=…`.

`src/dvcorr/pipeline/__init__.py` is a one-line docstring; no public re-exports needed — there
are now three pipeline modules (`velocity_centered.py`, `velocity_frame_comparison.py`, and
`redshift_space_comparison.py`, the latter two additive on top of the first, not forks of
it), each imported directly by name.

### `src/dvcorr/pipeline/velocity_frame_comparison.py` **[done]**
The single source of truth for the two-frame comparison (observer-frame ζ₁ vs. the
observer-free velocity-frame dipole), consumed by
`scripts/plot_velocity_frame_comparison.py`. Deliberately does NOT reimplement
`velocity_centered.py`'s earlier stages: `RunConfig` (subclassed), `NormalizedDipole`,
`normalize_result`, `normalize_stacked_dipole`, `SharedCenterSet`, and `select_shared_centers`
are all imported from there; `load_and_carve`, `draw_candidates`, and `global_number_density`
are consumed directly by the script instead (this module's own stage functions start one step
later, from already-loaded candidates). Same matplotlib-backend discipline as
`velocity_centered.py` (never calls `matplotlib.use`).

`SharedCenterSet` / `select_shared_centers` USED TO be defined in this module; they moved to
`velocity_centered.py` in the redshift-space task once `redshift_space_comparison.py` needed
them too (see that module's entry above) — this module now imports both names and re-exports
them unchanged, so `from dvcorr.pipeline.velocity_frame_comparison import select_shared_centers`
(as `tests/test_velocity_frame_dipole.py` and `notebooks/06_velocity_frame_comparison.ipynb`
already do) still resolves.

- `ComparisonRunConfig(RunConfig)` — adds `comparison_output_name`,
  `angle_diagnostic_output_name`, and `axis_null_seed` (distinct from `seed`/`shuffle_seed` so
  the random-axis null never reuses another stream); `min_center_speed` stays on the parent
  `RunConfig` since it filters the shared center set, not a comparison-only knob.
- `run_both_frames(cfg, centers, s_tracers, observer) -> FrameRunResults` — runs
  `velocity_centered_shell_dipole` and `velocity_frame_shell_dipole` on the IDENTICAL
  `centers.s_centers`/`v_centers`; asserts both estimators' `n_centers == centers.n_centers`
  (`RuntimeError` if not — the row-alignment invariant the comparison depends on); also runs
  `run_random_axis_null`.
- `run_random_axis_null(cfg, centers, s_tracers, observer) -> VelocityFrameShellDipoleResult`
  — replaces each center's velocity DIRECTION with an isotropic random unit vector
  (`np.random.default_rng(cfg.axis_null_seed)`) while keeping its speed and position, then
  reruns `velocity_frame_shell_dipole`. Why this null and not a scalar shuffle: the
  velocity-frame statistic is FULLY SELF-ALIGNED (the axis is built entirely from the same
  vector that supplies the scalar), so permuting `|v_alpha|` among centers leaves every
  center's axis-density alignment intact and retains essentially the signal itself. The
  observer frame is only PARTIALLY self-aligned — it takes a *sign* from `u_α` and its *line*
  from the center's position — so once that sign is divided back out, a signed-scalar
  permutation still works there (`normalize_result`); here there is no position-derived
  remainder to hold fixed. Randomizing the AXIS is the actual guard against
  align-then-measure bias; it costs one extra estimator pass, deliberately.
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
  (`mean_alpha(summary_alpha) == mean_b(zeta_hat_b)`). This identity is pure index exchange and
  holds for ANY `shell_edges` binning — it does not break under log spacing. What changes is
  interpretation: the shell average (`.mean(axis=1)`) is a mean over the shell INDEX b, UNWEIGHTED
  — "uniform in r" under the old linear default, "uniform in log r" once
  `cfg.shells.spacing == SPACING_LOG`, where five of twelve bins sit below 5.66 h⁻¹Mpc, making
  this per-center summary noticeably more sensitive to the one-halo/exclusion regime. An
  `n_bar·V_b`-weighted mean is deliberately NOT used instead — it would break the very identity
  this function's docstring promises, since the plotted `mean_b(zeta_hat_b)` is itself
  unweighted; a `summary_radial_window` knob is the right follow-up if this degrades in
  practice.
- `make_comparison_figure(cfg, results, comparison) -> Figure` — the main deliverable: top
  panel overlays both frames' ζ̂₁ with SEM bands and their own (differently-constructed) nulls;
  bottom panel plots both frames' monopoles on a TWIN y-axis (obs left, vel right, each
  y-label colored to match). Neither curve sits at zero any more — both frames weight by a
  speed (`⟨|u|⟩` and `⟨|v|⟩` respectively) — but they remain a projection factor apart, so a
  shared axis would compress the smaller one and hide the presence/absence of the r-dependent
  leakage trend that is the comparison's core finite-distance diagnostic. Neither panel
  carries a zero reference line. Plotting abscissa is
  `dvcorr.config.volume_weighted_shell_radii(results.obs_result.shell_edges)` (`r_eff`), not
  `comparison.shell_centers` (left alone); `apply_radial_axis_scale(ax_mono_obs, cfg.shells)`
  (imported from `velocity_centered.py`, not redefined) is called on the bottom axis — the
  `twinx()` velocity-frame monopole axis shares it automatically — and the title includes
  `_binning_description(cfg.shells)` (also imported) in place of the old bare `r_max=…`.
  `apply_dipole_axis_scale` (also imported, symlog y under `SPACING_LOG`) is called on THREE
  axes, not one: `ax_dipole`, `ax_mono_obs`, AND `ax_mono_vel` — `twinx()` only shares the
  X-axis, so the velocity-frame monopole's own y-scale needs its own call.
- `make_angle_diagnostic_figure(cfg, comparison) -> Figure` — top panel bins
  `per_center_dipole_difference` by `per_center_delta` (`_N_ANGLE_BINS` bins over `[0, π/2]`,
  degrees on the axis, `_MAX_ANGLE_DEG = 90`) with an SEM errorbar over a scatter of individual
  centers; bottom panel histograms `delta` against the isotropic reference, which for the
  signed axis is `P(delta) ∝ sin(delta)` on `[0, π/2]` (`cos δ = |c|`, `c` uniform on
  `[-1, 1]`). A few high-delta
  outliers driving the top-panel gap ⇒ bulk-flow contamination; a smooth spread ⇒ genuine
  projection geometry; an excess at low delta relative to the isotropic reference ⇒ flow
  directions aligned with the lines of sight (residual bulk motion). Its axes are angular
  (degrees), not radial-dipole y-axes, so neither `apply_radial_axis_scale` nor
  `apply_dipole_axis_scale` is ever called here. Its docstring now also carries the log-spacing
  sensitivity note `normalize_comparison` already had: on an unclustered synthetic,
  `std(summary_alpha)` across centers goes 14.9 → 234.3 switching linear → log spacing, with
  the three innermost log bins alone supplying 75% of that variance (bin 0 alone 33%) — this
  figure's per-center y-quantity is that same unweighted `summary_alpha`, so under log spacing
  it risks becoming a plot of which centers happened to catch a tracer inside ~2 h⁻¹Mpc rather
  than the bulk-flow-vs-projection-geometry diagnostic it is meant to be.

### `src/dvcorr/pipeline/redshift_space_comparison.py` **[done]**
The single source of truth for the real-space vs. redshift-space comparison: the SAME
`velocity_centered_shell_dipole` (UNCHANGED — no new argument, no carry-over parameter) run
twice on a shared center set, once on real positions and once on redshift-space positions
(`dvcorr.redshift_space.to_redshift_space`). Consumed by
`scripts/plot_redshift_space_comparison.py` and
`notebooks/07_redshift_space_comparison.ipynb`. Reuses `RunConfig`, `draw_candidates`,
`global_number_density`, `NormalizedDipole`, `normalize_result`, `shell_dipole_norm_scale`,
`SharedCenterSet`, `select_shared_centers`, and `_load_all_halos` from `velocity_centered.py`
— does NOT reuse `load_and_carve` (needs a different, two-pass carve; see
`load_and_carve_buffered` below) but DOES reuse the CSV-reading helper it is built on, so the
two-pass carve costs one catalog read, not two. Same matplotlib-backend discipline as the
other two pipeline modules.

**Volume construction — two separate problems, two different fixes** (see the module
docstring for the full argument):

- **Tracers** leak across the `R_sub` boundary in BOTH directions once displaced. Fix: carve
  the tracer buffer at `R_sub + v_margin` (`load_and_carve_buffered`, ~32% more rows at
  `v_margin`'s "max" default — negligible), displace the FULL buffer, THEN restrict to
  `R_sub` (`build_tracer_spaces`). Restricting first would discard exactly the tracers the
  buffer exists to keep.
- **Centers** use a uniform core margin `r_max + v_margin`, applied to REAL positions —
  **never** to a center's displaced position (that would be velocity-conditioned selection:
  near the far boundary it drops outward-movers and keeps inward-movers, skewing ⟨u⟩ with
  position). `v_margin` is a configurable statistic on `|v_r|` (`"max"`, the true bound, or
  `"percentile"`, cheaper but requiring an additional GLOBAL `|v_r|` cut on centers when
  selected — legitimate because it is sign-symmetric and position-independent, unlike the
  displaced-position cut). A **triangle-inequality argument**
  (`select_redshift_shared_centers`'s docstring) shows the widened real-position margin
  ALREADY GUARANTEES every surviving center's displaced position clears the plain `r_max`
  margin `velocity_centered_shell_dipole` re-applies internally — so no displaced-position
  check is ever needed, and the through-observer flip guard applied on top is a safety net,
  not the mechanism keeping the two runs' center sets aligned.
- **Cost:** the single-frame run's own core margin is `r_max = 64`, giving a core radius of
  236 h⁻¹Mpc (candidate volume fraction ≈ 49%, measured 48.3%: n_candidates = 4000,
  n_centers = 1933). With `v_margin` at its "max" default, the core radius shrinks further to
  ~206 h⁻¹Mpc (candidate volume fraction ≈ 49% → ≈ 32%, roughly a third fewer centers than the
  currently published single-frame run) — the documented, deliberate price of a like-for-like
  comparison.

**n̄ normalization — both runs share the REAL-space n̄** (decision documented in the module
docstring): computed from `TracerSpaces.n_real_inside` (the POST-restriction count), never
the buffered count (that substitution silently inflates n̄ by ~32% and suppresses ζ̂₁ by the
same amount in BOTH runs, with no shape change to reveal it — the exact bug
`scripts/plot_velocity_centered_dipole.py:64`'s `n_bar` call site would hit if handed a
buffered carve). Both realized counts (`n_real_inside_r_sub`, `n_redshift_inside_r_sub`) are
logged and returned, expected to agree to well under a percent.

- `RedshiftSpaceRunConfig(RunConfig)` — adds `v_margin_statistic` (`"max"`/`"percentile"`),
  `v_margin_percentile` (default 99.9), `flip_guard_floor` (default
  `dvcorr.redshift_space.FLIP_GUARD_FLOOR`), `redshift_shuffle_seed` (distinct from
  `shuffle_seed`), `comparison_output_name`, `single_center_output_name`, and
  `example_center_index` (which shared center the single-center figure plots).
- `BufferedCarve` (frozen dataclass) / `load_and_carve_buffered(cfg, paths) -> BufferedCarve`
  — one `_load_all_halos` read, two in-memory carves: a plain `sub_volume_radius` pass (its
  population's `|v_r|` sets `v_margin`, via `dvcorr.redshift_space.radial_velocity` +
  `v_margin_from_statistic`), then the buffered `sub_volume_radius + v_margin` pass. Exposes
  `pos_core`/`vel_core` (the plain-radius population — the correct source for candidate
  centers) alongside `pos_buffer`/`vel_buffer`, so no script/notebook cell needs to
  re-derive that subset by re-filtering the buffer.
- `TracerSpaces` (frozen dataclass) / `build_tracer_spaces(cfg, buffer, observer) ->
  TracerSpaces` — displaces the FULL buffer
  (`dvcorr.redshift_space.to_redshift_space`), then restricts both the real and displaced
  arrays to `R_sub`. Carries STABLE buffer-row ids (`real_ids`/`redshift_ids`) alongside each
  restricted array — required because the two are different positional subsets of the same
  buffer, so a raw positional-index comparison between them would be nonsense (see
  `membership_diagnostics` below). Logs the real/redshift flux-check counts and the
  through-observer-guard drop count.
- `RedshiftCenterSet` (frozen dataclass) / `select_redshift_shared_centers(cfg, s_candidates,
  v_candidates, observer, v_margin_kms, v_margin_mpc) -> RedshiftCenterSet` — three cuts:
  `select_shared_centers` with `core_margin = r_max + v_margin` (on real positions) → the
  optional global `|v_r|` cut (percentile statistic only) → the through-observer flip guard
  on the survivors' own displacement, applied so BOTH runs see the identical survivor set.
  Returns row-aligned `s_centers_real`/`s_centers_redshift` plus a SINGLE shared `v_centers`
  (the concrete statement of n̂ invariance: `velocity_centered_shell_dipole` recomputes
  `n_hat_V,alpha` from whichever `s_centers` it gets, and that recomputation is provably
  identical in either space).
- `RedshiftSpaceFrameResults` (frozen dataclass): `real_result`, `redshift_result` (both
  `VelocityCenteredShellDipoleResult`), `centers`, `tracers`.
- `run_both_spaces(cfg, centers, tracers, observer) -> RedshiftSpaceFrameResults` — calls
  `velocity_centered_shell_dipole` twice, unchanged, differing ONLY in which position arrays
  are passed; asserts both calls' `n_centers == centers.n_centers` (`RuntimeError` if not —
  mirrors `run_both_frames`'s row-alignment check, guaranteed to hold here by
  `select_redshift_shared_centers`'s triangle-inequality argument).
- `RedshiftSpaceComparison` (frozen dataclass): `real`, `redshift` (both `NormalizedDipole`),
  `shell_centers`, `n_bar`, `n_real_inside_r_sub`, `n_redshift_inside_r_sub`.
- `normalize_redshift_comparison(cfg, results) -> RedshiftSpaceComparison` — `normalize_result`
  on each raw result, both with the SAME real-space `n_bar`
  (`global_number_density(results.tracers.n_real_inside, cfg.sub_volume_radius)`), distinct
  shuffle seeds (`cfg.shuffle_seed`, `cfg.redshift_shuffle_seed`).
- `MembershipDiagnostics` (frozen dataclass) / `membership_diagnostics(s_centers,
  s_tracers_real, real_ids, s_tracers_redshift, redshift_ids, shell_edges) ->
  MembershipDiagnostics` — STANDALONE, deliberately NOT part of the estimator (runs its own
  `cKDTree` query against each tracer array, so the estimator never needs an
  N_c × N_members index set). Per shell: `net_change` (Σ_α (N_redshift,α,b − N_real,α,b),
  aggregated and per-center) and churn (`churn_only_real`/`churn_only_redshift`/
  `churn_intersection`, aggregated and per-center) — a shell with unchanged count but fully
  swapped membership shows up in churn, not in net change. All set algebra is on the STABLE
  buffer-row ids, never on `query_ball_point`'s positional output (which is positional in
  whichever array was queried, and the two tracer arrays are different subsets).
- `make_redshift_comparison_figure(cfg, results, comparison) -> Figure` — center-averaged,
  (3, 1) two-panel layout (hard rule 6): top panel overlays both spaces' ζ̂₁ with SEM bands
  and their own (`u`-shuffle) nulls; bottom panel plots both monopoles on a SINGLE shared
  y-axis (unlike the observer/velocity-frame comparison's twin axis — both curves here are
  the same kind of quantity on the same `u_α` values, differing only through shell occupancy,
  so there is no projection-factor scale gap to hide behind separate axes). Plotting abscissa
  is `dvcorr.config.volume_weighted_shell_radii(results.real_result.shell_edges)` (`r_eff`),
  not `comparison.shell_centers` (left alone); `apply_radial_axis_scale(ax_mono, cfg.shells)`
  and `_binning_description(cfg.shells)` in the title, both imported from `velocity_centered.py`.
  `apply_dipole_axis_scale(ax, cfg.shells)` (also imported) is called on BOTH `ax_dipole` and
  `ax_mono` — a single shared y-axis here (unlike the twin-axis comparison figure), so one call
  per panel suffices.
- `make_single_center_figure(cfg, results, n_bar) -> Figure` — the same (3, 1) layout for ONE
  center (`cfg.example_center_index`), real and redshift overlaid, NO error band. Dipole:
  `per_center_dipole[a] * shell_dipole_norm_scale(...)` (no `/n_centers`, only one center);
  monopole: the RAW `per_center_speed[a] * per_center_count[a]`, deliberately un-normalized
  (a single-center shape read-out, not a comparison to the ensemble expectation). Same abscissa
  and axis-scale/title wiring as `make_redshift_comparison_figure` above: `r_eff` from
  `results.real_result.shell_edges` (already read there for `norm_scale`, `shell_centers` left
  alone), `apply_radial_axis_scale(ax_mono, cfg.shells)`, `apply_dipole_axis_scale` on both
  `ax_dipole` and `ax_mono`, `_binning_description(cfg.shells)`.

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

### `scripts/plot_redshift_space_comparison.py` — thin driver **[done]**
Mirrors the other two scripts' structure exactly. All algorithmic content lives in
`dvcorr.pipeline.redshift_space_comparison` (plus `draw_candidates` reused from
`dvcorr.pipeline.velocity_centered`); this script only wires it together. Module body:
`import matplotlib; matplotlib.use("Agg")` before importing the pipeline module, then a
single `main()` that chains `load_and_carve_buffered` → `build_tracer_spaces` →
`draw_candidates` (on `buffer.pos_core`/`vel_core`) → `select_redshift_shared_centers` →
`run_both_spaces` → `normalize_redshift_comparison` → `make_redshift_comparison_figure` and
`make_single_center_figure`, saving BOTH PNGs (`cfg.comparison_output_name`,
`cfg.single_center_output_name`) under `PathsConfig().output_dir`. Never runs at import time
(`if __name__ == "__main__"`); importing it is fast and safe, used as a smoke test.

Usage: `.venv/bin/python -m scripts.plot_redshift_space_comparison`.

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
  - `TestRadialFlowAxis` — the axis convention at its definition site: outbound gives
    `ẑ = +n̂_los`, inbound gives `ẑ = −n̂_los`, `v · ẑ = |u| ≥ 0` over a random batch, and
    `u == 0` returns a zero axis row rather than an invented direction.
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
  `TestFlowSignedAxis` covers the estimator-level consequences of the signed axis: an inbound
  center is axed back toward the observer (a tracer ahead of the flow gives a POSITIVE
  amplitude); `monopole == Σ|u|N ≥ 0`, distinct from the signed sum it replaces; the dipole is
  invariant under the flip (`|u|·A(ẑ) ≡ u·A(n̂_V)`, checked against a hand-built unsigned
  amplitude); and a purely transverse center (`u == 0`) contributes nothing without a NaN.
  `test_permuting_the_speed_alone_is_not_a_null` is the companion to the shuffle-null test:
  it asserts the naive positive-weight permutation does NOT collapse, which is *why*
  `normalize_result` undoes the axis flip first. Extended in the redshift-space task,
  immediately after the joint gate, with the REDSHIFT-SPACE sign gate and an
  epsilon-continuity check on the identical toy (`dvcorr.redshift_space.to_redshift_space`):
  the redshift-space run is positive (does not contradict hard rule 2 — that rule is for the
  group-centered ξ_Tu,1, this is ζ₁) with pair counts identical between spaces (2048 both) and
  a measured ratio of 0.957× — asserted as a SIGN + measured-ratio pin, with an explicit
  comment that this is a suppression, not an enhancement, and no directional amplitude claim
  is made. The continuity check asserts the RELATIVE deviation `|1 - D_redshift/D_real| <=
  0.06 * eps` (measured slope ≈ 0.040, asserted with headroom) across `eps` from `1` to
  `1e-3` — the absolute-agreement version would be trivially true, since both dipoles vanish
  together as `eps -> 0`.
  Geometric-edge (non-uniform bin width) tests: `test_geometric_edges_bin_tracers_correctly`
  (hand-placed tracers land in the correct bins of `log_shell_edges`-built edges);
  `test_expected_shell_occupancy_matches_analytic_volume_for_geometric_edges`;
  `test_empty_shell_under_geometric_edges_normalizes_to_zero_not_nan` (a genuinely empty
  geometric bin normalizes to `0.0`, not `NaN`). The binning-invariance gate —
  `normalize_stacked_dipole`'s volume-weighted-tiling identity, `zeta_hat_linear == Sum_k
  zeta_hat_geom_k * (V_geom_k / V_linear)` for any tiling of the same [10, 30] interval — is
  split into TWO tests sharing one body (`_assert_binning_invariance`):
  `test_binning_invariance_on_a_coherent_infall_field`, the STRENGTHENED version
  (`_multi_radius_gate_toy`, centers on 5 concentric shells at r = 11/14/18/25/29 h⁻¹Mpc around
  a single tracer, occupying all four of `_GATE_GEOM_EDGES`'s sub-bins with different dipole
  contributions), and `test_binning_invariance_delta_function_toy`, the original single-radius
  version (`_joint_gate_toy`, one occupied sub-bin) kept as a separate, simpler check. The
  strengthening matters: with only one sub-bin occupied, corrupting any of the three EMPTY
  sub-bins' volumes by even 10⁶× left the identity unaffected (relerr ~1e-16) — a real
  per-bin-volume bug in an empty bin would have passed silently. Self-checked by temporarily
  corrupting one non-innermost OCCUPIED bin's volume by 5% in `expected_shell_occupancy` and
  confirming `test_binning_invariance_on_a_coherent_infall_field` fails (it does) while
  `test_binning_invariance_delta_function_toy` still passes (that bin is empty in its toy) —
  then reverting.
- `tests/test_redshift_space.py` **[done]** — transform-level tests for
  `dvcorr.redshift_space`: the zero-velocity limit (`to_redshift_space` is the identity when
  every velocity is zero); displacement direction (outward/inward velocity moves the halo
  further from / closer to the observer); n̂ (and hence `u`) invariance under the transform,
  pinned on the shared infall toy with an explicit `atol` reproducing the measured 3.3e-16 /
  8.5e-14 precision; the through-observer flip guard (a constructed near-observer,
  fast-inward halo is dropped and counted, and the guard's floor is a strict `>`, not `>=`);
  `radial_velocity` against a hand-computed projection; `v_margin_from_statistic`'s two
  statistics and its `ValueError` on an unknown one; input validation (mismatched shapes,
  non-finite velocity, non-positive `flip_guard_floor`); `RedshiftSpaceTransform`'s
  frozen-dataclass contract.
- `tests/test_redshift_space_comparison.py` **[done]** — pipeline-level tests for
  `dvcorr.pipeline.redshift_space_comparison`, built on small synthetic populations
  (`_synthetic_buffer`, a hand-built stand-in for `load_and_carve_buffered`'s output — never
  the real MDPL2 catalog). Shared center set: `s_centers_real`/`s_centers_redshift` are
  ROW-ALIGNED (an independent re-run of `to_redshift_space` on `s_centers_real`/`v_centers`
  reproduces `s_centers_redshift` exactly, which can only hold if the two arrays describe the
  same centers in the same order); `run_both_spaces` preserves `n_centers` and does not raise
  on an ordinary synthetic population; the percentile `v_margin_statistic` drops centers with
  `|v_r|` above it globally. Membership diagnostics: an all-zero-velocity population gives
  zero net change AND zero churn in every shell (paired with, not a replacement for,
  `test_redshift_space.py`'s position-level zero-velocity check — this one additionally
  guards against a broken pipeline silently reusing the same tracer array for both spaces);
  a deliberately adversarial case with mismatched array lengths/order but overlapping stable
  ids shows churn is computed on those ids, not on `query_ball_point`'s positional output.
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
  two frames agree exactly for both outbound and inbound velocities — and with the signed
  axis their axes now *coincide* in both cases (`delta ≈ 0` for inbound too, where the
  unsigned reading reported `≈ π`). Pipeline-level degenerate-center handling
  (`select_shared_centers`'s speed floor, reported via `n_dropped_slow`; the estimator's own
  zero-speed `ValueError`). `run_both_frames` row-alignment: identical `n_centers`,
  independently-recomputed per-center scalars, and identical `per_center_count` between
  frames. Null tests: an uncorrelated (uniformly random) density field gives both frames a
  null-consistent dipole; `run_random_axis_null` collapses the velocity-frame dipole on a
  clustered configuration with a genuine nonzero signal (documenting why a scalar permutation
  is not a valid null for this self-aligned statistic). Binning/empty-input/input-validation
  mechanics mirroring `test_velocity_centered_dipole.py`. `per_center_axis_angle` shape and
  `[0, π/2]` range contract — reconstructed independently as `arccos(|u|/|v|)`, and shown to
  be a real discriminator (the same random batch exceeds `π/2` against the *unsigned* line of
  sight) — plus a hand-checked perpendicular case (`delta == pi/2`, delta's ceiling).
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
- `tests/test_plot_wiring.py` **[done]** — pins the log/linear radial x-axis wiring
  `apply_radial_axis_scale` adds to `dvcorr.pipeline.velocity_centered.make_figure`: a log
  `ShellConfig` gives `ax.get_xscale() == "log"` with `ax.get_xlim()` pinned to the outer shell
  edges; a linear one gives `"linear"`, unchanged. Also pins the companion y-axis wiring,
  `apply_dipole_axis_scale`: a log `ShellConfig` gives BOTH `make_figure` panels
  (`ax_dipole`, `ax_mono`) `get_yscale() == "symlog"`; a linear one gives `"linear"` on both,
  unchanged — same backward-compatibility guarantee as the x-axis case. Builds a tiny
  one-center/zero-tracer synthetic result (shape-correct, values irrelevant) rather than
  running the real pipeline; calls `matplotlib.use("Agg")` itself at module top, before
  importing `dvcorr.pipeline.velocity_centered` — the consumer selects the backend, the library
  never
  does (that module's own backend-discipline note).
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
  frames separate only after dividing it out, leaving the observer-frame `2r/3R` trend on top
  of its ⟨|u|⟩ offset against a velocity-frame residual flat at ⟨|v|⟩); and the
  δ_α diagnostic's reading (δ now spans `[0, π/2]`, since both frames' axes follow the
  motion — few high-δ outliers ⇒ bulk-flow contamination, smooth spread ⇒ projection
  geometry; low-δ excess over the isotropic `sin δ` reference ⇒ residual bulk motion). Saves both PNGs via the same `paths.output_dir` / `cfg` logic as the script's
  `main()` and displays each returned `Figure` inline (never `plt.show`). Unlike 04/05 this
  one **is** executed and its outputs are kept — it is the record of the comparison run.
- `07_redshift_space_comparison.ipynb` **[done]** — exploratory twin of
  `scripts/plot_redshift_space_comparison.py`. Every code cell calls one stage of
  `dvcorr.pipeline.redshift_space_comparison` (`load_and_carve_buffered`,
  `build_tracer_spaces`, `select_redshift_shared_centers`, `run_both_spaces`,
  `normalize_redshift_comparison`, `membership_diagnostics`, `make_redshift_comparison_figure`,
  `make_single_center_figure`) plus `draw_candidates` reused unchanged from
  `dvcorr.pipeline.velocity_centered` — nothing is reimplemented. One stage per cell, markdown
  between them covering: why tracers need a buffered carve while centers need a WIDENED
  real-position margin instead (two different fixes for two different boundary problems); why
  selecting centers on their displaced position would be a real bias, not just inelegant, and
  why the widened margin makes that check unnecessary by construction (the
  triangle-inequality argument); the shared, real-space n̄ decision and the ~32%-error trap it
  avoids; the center-averaged comparison figure's single (not twin) monopole y-axis, and why
  that is legitimate here specifically; the single-center example figure; and the membership
  diagnostics' net-change-vs-churn distinction. **Not executed as part of authoring** —
  outputs are cleared, mirroring notebook 05's own convention (the MDPL2 catalog load is long
  and this task explicitly does not require running it).

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
