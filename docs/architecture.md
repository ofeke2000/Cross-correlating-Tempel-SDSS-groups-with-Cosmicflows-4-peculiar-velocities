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

**Catalog contract.** Two MDPL2 catalogs are supported and neither supersedes the other:
`CATALOG_FULL` (~127M halos, down to 2 particles, subhalos included) and `CATALOG_MVIR12`
(4,093,751 distinct halos at m_vir ≥ 1e12). A run names one in its `CatalogConfig`, and
`_load_all_halos` is the only place that name becomes arrays. The CSVs are the source of
truth; the pipeline reads the **Parquet** files `pipeline/catalog_conversion.py` writes
from them, and both catalogs are written with one identical schema so the loader has a
single code path. `CATALOG_FULL` filtered at `mvir ≥ 1e12` with subhalos excluded
reproduces `CATALOG_MVIR12` exactly — asserted by `tests/test_catalog_equivalence.py`,
which is what keeps the two interchangeable. The two files disagree about the boundary
coordinate of exactly four halos (`BOX_SIZE` in one, `0.0` in the other, same object under
PBC), so box coordinates are **not** guaranteed to lie in the half-open interval
[0, `BOX_SIZE`); code that needs that convention must fold first.

**Row order is not a convention.** Candidate centers are drawn by row index
(`draw_candidates_from_arrays`), so the *sample* a run measures depends on the order rows
happen to sit in the catalog file — while the *population* does not. The two catalogs
store the same halos in different tie orders, so a run on one and a run on the equivalent
cut of the other select the same halos but different centers, and their multipoles differ
by ordinary center-sampling noise (~1σ per shell, not a bug). Compare catalogs on their
populations, or at fixed center count with the sampling scatter accounted for; do not
expect identical numbers.

**Import layering.** `conventions.py` and `geometry.py` are leaf modules: no intra-project
imports (their docstrings name each other but do not import them). `config/*` imports only
`conventions`. `estimators/*` import `conventions` and `geometry`; `velocity_frame_dipole.py`
additionally imports `core_center_mask`/`real_y10` from `shell_dipole.py` rather than
redefining them. `pipeline/velocity_centered.py` is the shared base of `pipeline/`: it
imports `config` and `estimators`, and both comparison pipelines
(`velocity_frame_comparison.py`, `redshift_space_comparison.py`) import their shared
machinery (`RunConfig`, `SharedCenterSet`, `select_shared_centers`, `_load_all_halos`,
`NormalizedDipole`, `normalize_stacked_dipole`, `shell_dipole_norm_scale`,
`_binning_description`) from it — **neither comparison pipeline imports from the
other**. `scripts/` and `notebooks/` import from `dvcorr.pipeline`/`dvcorr.config`
/`dvcorr.conventions`; nothing under `src/dvcorr/` ever imports from `scripts/` or
`notebooks/` — the dependency runs only one way.

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
