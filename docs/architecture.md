# Architecture — index

This file is an **index**, not a description. Every module below carries its own
module-level docstring (and, for estimators/pipeline modules, exhaustive function/class
docstrings) that is the definition site for what it does, its shapes, arguments, and
behavior — read the file, not this page, for that detail.

**Governing rule (CLAUDE.md hard rule 8):** touched only when a file is added, removed,
or changes responsibility. Never updated to restate something a docstring already says.
If this index and the code disagree, the code is right and this file has a bug.

**Status tags:** **[done]** implemented and tested · **[stub]** signature/docstring only ·
**[empty]** directory reserved, no code yet.

The project is an editable-installed package, `dvcorr`, laid out under `src/` (setuptools
src layout, `pyproject.toml`). `import dvcorr…` works identically in scripts, notebooks,
and tests — no `sys.path` bootstrap, no run-from-root requirement.

---

## Module index

`pyproject.toml` — packaging: declares `dvcorr` (`src` layout), runtime deps, the `dev`
extra (`pytest`), and `testpaths = ["tests"]`. `requirements.txt` is `-e .[dev]`.

### `src/dvcorr/`

- `__init__.py` **[done]** — package root; `__version__` only, no submodule imports.
- `conventions.py` **[done]** — frozen conventions: pair orientation, µ, the observer,
  reference frame, sign, catalog columns (hard rule 1).
- `geometry.py` **[done]** — pure geometry primitives (`unit_vector`, `pair_separation`,
  `radial_flow_axis`, `mu_cosine`); the layer the sign convention lives in.
- `redshift_space.py` **[done]** — the redshift-space coordinate transform: displace a
  halo along its own line of sight by its own radial velocity.

### `src/dvcorr/config/` — tunable settings (counterpart to `conventions`'s frozen ones)

- `paths.py` **[done]** — `PathsConfig`: catalog/output locations, repo-root-relative;
  `halo_catalog(name)` resolves a catalog name to its CSV or Parquet path.
- `cosmology.py` **[done]** — `CosmologyConfig`: MDPL2 cosmology metadata; self-checks
  against `conventions.HUBBLE_PARAM`.
- `shells.py` **[done]** — `ShellConfig` + free builders (`linear_shell_edges`,
  `log_shell_edges`, `volume_weighted_shell_radii`): radial shell binning.
- `selection.py` **[done]** — `SelectionConfig`: placeholder knobs for `mocks/`/`selection/`.
- `catalog.py` **[done]** — `CatalogConfig`: which halo catalog a run reads (`CATALOG_FULL`,
  `CATALOG_MVIR12`) and which halos it keeps from it (mass bounds, subhalos).
- `settings.py` **[done]** — `Settings` aggregator + `default_settings()` factory.
- `__init__.py` **[done]** — re-export surface over the above.

### `src/dvcorr/estimators/`

- `__init__.py` **[done]** — package docstring only.
- `shell_dipole.py` **[done]** — the group-centered estimator `shell_dipole` (ξ_Tu, the
  simulation-validation cross-check) **and** the velocity-centered estimator
  `velocity_centered_shell_dipole` (ζ, Nusser 2017 eq. 23–24, the production target),
  plus shared helpers (`real_y10`, `core_center_mask`, `expected_shell_occupancy`,
  `center_standard_error`).
- `velocity_frame_dipole.py` **[done]** — `velocity_frame_shell_dipole`: an observer-free
  variant of the velocity-centered estimator (axis = full 3-D flow direction instead of
  its radial projection). Reuses `core_center_mask`/`real_y10` from `shell_dipole.py`.

### `src/dvcorr/pipeline/` — reusable stage functions, consumed by `scripts/` and `notebooks/`

- `__init__.py` **[done]** — package docstring only; no re-exports, each module is
  imported by name.
- `catalog_conversion.py` **[done]** — one-time CSV → Parquet conversion of the halo
  catalogs, streamed and cache-and-skip; both catalogs get one shared schema.
- `mass_diagnostics.py` **[done]** — the mass funnel: log₁₀(m_vir) distribution of the
  halos surviving each selection stage, as a printed table and a two-panel figure.
- `velocity_centered.py` **[done]** — the shared base: load/carve, candidate drawing,
  shared-center selection (`SharedCenterSet`/`select_shared_centers`), running
  `velocity_centered_shell_dipole`, normalization (`NormalizedDipole`,
  `normalize_stacked_dipole`, `normalize_result`, the velocity-shuffle null), and
  the two-panel figure. Both comparison pipelines
  below build on this module rather than duplicating it.
- `velocity_frame_comparison.py` **[done]** — observer-frame ζ₁ vs. the observer-free
  velocity-frame dipole, on an identical shared center set; the random-axis null
  (`run_random_axis_null`) and the two comparison figures.
- `redshift_space_comparison.py` **[done]** — the unchanged `velocity_centered_shell_dipole`
  run twice (real vs. redshift-space positions, `dvcorr.redshift_space`), on a shared
  center set built with a widened real-position margin; the buffered tracer carve,
  membership diagnostics, and the two comparison figures.

### `src/dvcorr/selection/` **[empty]**

Reserved for CF4-like angular/radial masks, the selection function φ(r), random catalogs.
Port target: the old repo's `masks.py`.

### `src/dvcorr/mocks/` **[empty]**

Reserved for MDPL2 mock observers, halo selection, mock covariance. Port target: the old
repo's `overdensity.py` (periodic KDTree overdensity), `data_loader.py`.

### `scripts/` — thin, headless, reproducible drivers (`python -m scripts.<name>`)

- `plot_velocity_centered_dipole.py` **[done]** — drives `pipeline.velocity_centered`, saves one PNG.
- `plot_velocity_frame_comparison.py` **[done]** — drives `pipeline.velocity_frame_comparison`, saves two PNGs.
- `plot_redshift_space_comparison.py` **[done]** — drives `pipeline.redshift_space_comparison`, saves two PNGs.
- `convert_mdpl2_catalog.py` **[done]** — drives `pipeline.catalog_conversion`; run once per
  catalog before any pipeline run.

Each sets `matplotlib.use("Agg")` at import time, before importing its pipeline module;
none contain algorithmic content (see Cross-cutting contracts below).

### `tests/`

- `test_geometry.py` **[done]** — the sign gate (infall → negative dipole); the
  load-bearing test the project hangs on.
- `test_shell_dipole.py` **[done]** — estimator-level sign gate, nulls, binning, edge cases
  for `shell_dipole`.
- `test_velocity_centered_dipole.py` **[done]** — joint sign gate (ξ_Tu vs. ζ), flow-signed
  axis, shuffle null, log-binning invariance gate, redshift-space sign gate.
- `test_redshift_space.py` **[done]** — transform-level tests for `dvcorr.redshift_space`.
- `test_redshift_space_comparison.py` **[done]** — pipeline-level tests for
  `redshift_space_comparison` (shared-center alignment, membership diagnostics).
- `test_velocity_frame_dipole.py` **[done]** — sign gate and frame-agreement limit for the
  observer-free dipole; random-axis null.
- `test_settings.py` **[done]** — `dvcorr.config` dataclass tests.
- `test_catalog_conversion.py` **[done]** — conversion schema, row order, cache-and-skip,
  on tiny synthetic CSVs.
- `test_catalog_equivalence.py` **[done]** — the superset relation between the two
  catalogs, against the real files; skipped when they are absent.
- `test_mass_diagnostics.py` **[done]** — mass-funnel construction guards and figure build.
- `test_plot_wiring.py` **[done]** — pins the log/linear axis-scale wiring.
- `__init__.py` — present so `from tests.test_geometry import …` resolves.

### `notebooks/` — exploration only; nothing load-bearing

- `04_first_mdpl2_run.ipynb` **[done]** — first `shell_dipole` run on real MDPL2 halos;
  pre-refactor, still carries an inline duplicate of the pipeline (documented exception to
  the graduation rule below).
- `05_velocity_centered_dipole.ipynb` **[done]** — exploratory twin of
  `scripts/plot_velocity_centered_dipole.py`; the model notebook for the working model.
- `06_velocity_frame_comparison.ipynb` **[done]** — exploratory twin of
  `scripts/plot_velocity_frame_comparison.py`; executed, outputs kept.
- `07_redshift_space_comparison.ipynb` **[done]** — exploratory twin of
  `scripts/plot_redshift_space_comparison.py`; not executed, outputs cleared.

### `Imports from old repo/` — reference dump, read only

Not a package, not importable. Copy and adapt deliberately (hard rule 0). `vector3d.py`,
`overdensity.py`, `masks.py`, `data_loader.py` are the pieces most likely worth porting.

---

## Cross-cutting contracts

Statements below span more than one file, or refine a CLAUDE.md hard rule. Nothing here
duplicates a single module's own docstring.

**PBC / carving contract (refines hard rule 3).** `geometry.py`'s primitives, and every
estimator built on them (`shell_dipole.py`, `velocity_frame_dipole.py`), do **not** apply
the minimum-image convention — they are plain Euclidean geometry on whatever coordinates
they are handed. Periodicity is discharged upstream, once, at the carving step:
`dvcorr.pipeline.velocity_centered.load_and_carve` (and `load_and_carve_buffered` in the
redshift-space pipeline) carve a sub-volume around the observer and hand down coordinates
already the nearest image of their center, continuous with it. Re-adding a minimum-image
reduction inside `geometry.py` or an estimator would silently corrupt already-unwrapped
separations — it would look like a fix and would in fact wrap a legitimate neighbor back
across the box. The only obligation the primitives themselves enforce is that no shell
may exceed `conventions.MAX_ANALYSIS_RADIUS = BOX_SIZE / 2`, beyond which the nearest
image is not unique and the carving itself is ill-defined.

**The r = 0 coincident pair is excluded, at every binning.** All four sites that bin a
separation into shells — `shell_dipole` and `velocity_centered_shell_dipole` in
`estimators/shell_dipole.py`, `velocity_frame_shell_dipole` in
`estimators/velocity_frame_dipole.py`, and `_ids_by_shell` in
`pipeline/redshift_space_comparison.py` — carry an `r_mag > 0` term in their in-range
mask. A zero-length separation has no direction, so `geometry.unit_vector` returns a zero
row and its µ comes out 0: an ordinary-looking cosine that never disturbs the dipole but
does enter `pair_count` and the monopole as a **self-correlation rather than a
correlation**, inflating exactly the occupancy hard rule 6 has the monopole diagnose. This
was incidental protection while every binning began above zero;
`ShellConfig.include_zero_bin` makes `shell_edges[0] == 0` a production binning, and since
centers are subsampled *from* the tracer array there is then exactly one such pair per
center. The term is unconditional, not gated on `shell_edges[0] == 0`, so the guarantee
holds for any binning and cannot be undone by a config change. A new binning site inherits
this obligation.

**n̄ is the box mean, not the carve's density.** The eq. 24 denominator
`expected_shell_occupancy(n_bar, edges)` = n̄·V_b is fed one n̄ throughout:
`pipeline/velocity_centered.box_number_density(n_total)` — the catalog's post-cut halo
count over `BOX_SIZE**3`. Every call site builds it that way
(`CarvedHalos.n_total`, `BufferedCarve.n_total`), and no stage function derives it
internally: the three normalize stages (`normalize_result`, `normalize_comparison`,
`normalize_redshift_comparison`) all take n̄ as an argument, so the choice stays visible at
the call site. It is deliberately *not* the carved count over the carve's volume: the
sub-volume is one realization of the density field, n̄·(1+δ_sub), and that unknown factor
would multiply every shell of ζ̂₁ flat in r. The box mean is exact here because the box is
the whole universe of the simulation, and being carve-blind it also removes the
count/radius mismatch the buffered carve in `redshift_space_comparison.py` used to expose.
The one remaining obligation: the count must be the *same population* that enters the
shells — post mass/subhalo cut, pre-carve.

**Catalog contract.** Two MDPL2 catalogs are supported and neither supersedes the other:
`CATALOG_FULL` (~127M halos, down to 2 particles, subhalos included) and `CATALOG_MVIR12`
(4,093,751 distinct halos at m_vir ≥ 1e12). A run names one in its `CatalogConfig`, and
`_load_all_halos` is the only place that name becomes arrays. The CSVs are the source of
truth; the pipeline reads the **Parquet** files `pipeline/catalog_conversion.py` writes
from them, and both catalogs are written with one identical schema so the loader has a
single code path. `CATALOG_FULL` filtered at `mvir ≥ 1e12` with subhalos excluded
reproduces `CATALOG_MVIR12` exactly — asserted by `tests/test_catalog_equivalence.py`,
which is what keeps the two interchangeable.

**Box coordinates are half-open: [0, `BOX_SIZE`).** Rockstar emits a small number of halos
with a coordinate at exactly `BOX_SIZE` (196 rows in the raw full catalog, 0 in the pre-cut
one) — the same point as `0.0` under PBC, but a different representative, which made the two
files disagree on shared halos. Conversion folds every position with `np.mod`, adopting the
pre-cut catalog's convention; the fold is exact and a no-op on interior coordinates.
Downstream code may therefore rely on the half-open interval, and
`test_box_coordinates_are_half_open` is what keeps that true. Any catalog reaching the
pipeline by some other route must be folded the same way.

**The seeded draw selects halos, not file rows.** `draw_candidates_from_arrays` sorts the
carved population into a canonical order — lexicographic on box-folded position — before
indexing it with `rng.choice`. Without that, the seed fixes a set of row *numbers*, and
the two catalogs' different tie orders make the same seed select different halos from the
identical population (measured: 291 of 4000 in common, multipoles differing ~1σ per shell).
With it, the same halos give the same centers whichever file they were read from, so a
comparison between the two catalogs isolates the catalogs rather than the sampling. The
folding is a canonicalization of *identity*, not a minimum-image reduction, and never
reaches the coordinates handed to an estimator — it is belt-and-braces now that conversion
folds at ingest, and the guarantee for any population that did not come through it. Asserted
synthetically in `test_velocity_centered_dipole.py` and
against the real files in `test_catalog_equivalence.py`. Two runs over the same halos in
different array order still differ at ~1e-16 in the dipole, from float addition order.

**Import layering.** `conventions.py` and `geometry.py` are leaf modules: no intra-project
imports (their docstrings name each other but do not import them). `config/*` imports only
`conventions`. `estimators/*` import `conventions` and `geometry`; `velocity_frame_dipole.py`
additionally imports `core_center_mask`/`real_y10` from `shell_dipole.py` rather than
redefining them. `pipeline/velocity_centered.py` is the shared base of `pipeline/`: it
imports `config` and `estimators`, and both comparison pipelines
(`velocity_frame_comparison.py`, `redshift_space_comparison.py`) import their shared
machinery (`RunConfig`, `SharedCenterSet`, `select_shared_centers`, `_load_all_halos`,
`NormalizedDipole`, `normalize_stacked_dipole`, `matched_gaussian_sample`,
`shell_dipole_norm_scale`, `_binning_description`) from it — **neither comparison
pipeline imports from the other**. `scripts/` and `notebooks/` import from `dvcorr.pipeline`/`dvcorr.config`
/`dvcorr.conventions`; nothing under `src/dvcorr/` ever imports from `scripts/` or
`notebooks/` — the dependency runs only one way.

**Every null gets its own random stream.** Seeds are allocated once, across all three
pipelines, and no two collide: `seed` = 42 (candidate centers), `shuffle_seed` = 43,
`ComparisonRunConfig.axis_null_seed` = 44, `RedshiftSpaceRunConfig.redshift_shuffle_seed`
= 45, `RunConfig.gaussian_null_seed` = 46,
`ComparisonRunConfig.velocity_gaussian_null_seed` = 47,
`RedshiftSpaceRunConfig.redshift_gaussian_null_seed` = 48. This is a contract rather than
a per-file detail because the configs inherit across modules and a collision would not
raise — it would quietly correlate two curves the figures present as independent nulls.
Pinned by `tests/test_velocity_centered_dipole.py::test_every_null_in_the_three_pipelines_gets_its_own_seed`;
a new null adds the next integer and extends that test.

Which construction each of those seeds drives is the owning module's business, not this
file's — but the one fact that spans them: every frame carries **two** nulls, its own
primary one plus a matched-Gaussian draw (`matched_gaussian_sample`) built at the
sample's own mean and ddof=1 spread, so the pair differs only in distribution shape.

**Matplotlib backend discipline.** Only the three `scripts/*.py` entry points call
`matplotlib.use("Agg")`, and each does so before importing its pipeline module. No module
under `dvcorr.pipeline` ever calls `matplotlib.use`, so a notebook importing a pipeline
module directly keeps whatever backend it already has (e.g. inline).

**Library, scripts, notebooks — one source of truth, two thin consumers.**
`src/dvcorr/` holds every piece of reusable logic — geometry, estimators, pipeline stage
functions, plotting helpers — and is the only place it is defined. `scripts/` are thin,
headless, reproducible drivers: orchestration only, no algorithm, run as
`python -m scripts.<name>`. `notebooks/` are interactive exploration and presentation:
they import the same library stage functions and add plots/narrative, never reimplementing
pipeline logic (`05_velocity_centered_dipole.ipynb` is the model; `04_first_mdpl2_run.ipynb`
predates the pattern and is the documented exception). **Graduation rule:** a function
written in a notebook or script that is worth reusing moves into `src/dvcorr/` and is
imported back — a script or notebook is never the definition site.
