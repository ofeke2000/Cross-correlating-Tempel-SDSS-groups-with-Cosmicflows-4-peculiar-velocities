# CLAUDE.md

Research code measuring the **density–velocity cross-correlation dipole** ξ_Tu,1 between
Tempel et al. (2017) SDSS groups and Cosmicflows-4 peculiar velocities, following
Nusser (2017) in a group-centered rather than velocity-centered construction.

The current phase is the **simulation-validation arm**: reproduce the estimator inside the
MDPL2 box, where halo peculiar velocities and the halo density field are both known exactly
and the answer can be checked before it is trusted on data.

Read these for context before non-trivial work:

- [docs/architecture.md](docs/architecture.md) — **the live document**: module index and
  cross-module contracts (PBC carving, n̄, seeds, normalization, import layering). It points
  you at the right file; the file's docstring tells you what it does.
- [README.md](README.md) — science goals, frozen conventions table, planned data-arm
  pipeline, null tests.
- [docs/summary.md](docs/summary.md) and [docs/research_notes.md](docs/research_notes.md) —
  early scaffolds, **not maintained** (they still say conventions live in `config.py`).
  Read for orientation only; where they disagree with `architecture.md` or the code, they
  are wrong.
- `literature/` (gitignored — local copies only, not in the repository):
  - `tempel_cf4_velocity_correlations.pdf` — the project's methodological note; **the
    authority on every convention and sign**
  - `1703.05324v2.pdf` — Nusser (2017), the estimator being reproduced (eq. 23–24 is the
    velocity-centered shell dipole ζ; eq. 13–17 the linear-theory multipoles)
  - `1988ApJ...332L...7G.pdf` — Górski (1988), the Ψ∥/Ψ⊥ velocity-correlation formalism
    (implemented in `dvcorr.theory.gorski`)
  - `1704.04477v1.pdf` — Tempel et al. (2017), the group catalog

  Each PDF has a `.txt` twin with `=== page N ===` markers. **Search the `.txt` first**
  (grep for an equation number or phrase) and read only the matching page of the PDF
  (`Read` with `pages: "N"`) when you need an equation — the text extraction garbles
  math, and reading a whole paper costs ~100k tokens.

## Code layout

Editable-installed package (`pip install -e .`, setuptools src layout, `pyproject.toml`):
`import dvcorr…` works identically in scripts, notebooks, and tests, no `sys.path` hacks.

```text
pyproject.toml        package metadata, deps, pytest config — the one dependency source
src/dvcorr/
  conventions.py       frozen conventions — sign, pair orientation, observer, box, columns
  config/              tunable settings, one dataclass per file — paths, cosmology (+ f(z),
                       aH), shell binning, selection, catalog — plus a Settings aggregator
  geometry.py          pure geometry primitives (unit_vector, pair_separation,
                       radial_flow_axis, mu_cosine)
  redshift_space.py    the redshift-space coordinate transform (displace along own LOS)
  estimators/          shell_dipole.py — group-centered xi_Tu and velocity-centered zeta
                       (Nusser eq. 23-24), monopole + dipole per radial shell;
                       velocity_frame_dipole.py — observer-free variant (axis = v_hat)
  theory/              linear-theory predictions in Nusser's velocity-centered orientation:
                       transforms.py (one FFTLog, bessel_moment), gorski.py (Ψ∥, Ψ⊥),
                       dipole.py (real/redshift-space dipole), wide_angle.py
  pipeline/            velocity_centered.py — the shared base (load/carve, draw, run,
                       normalize, nulls, plot); velocity_frame_comparison.py;
                       redshift_space_comparison.py; dm_snapshot.py (Gadget-4 300 h⁻¹ Mpc
                       particle box loaders); robustness.py; frame_ratios.py;
                       galaxy_bias.py (b of the tracers, Corrfunc/CAMB, needs `[bias]`);
                       halo_class_comparison.py; catalog_conversion.py (CSV→Parquet);
                       mass_diagnostics.py (mass funnel)
  selection/           masks, selection functions, random catalogs (still empty)
  mocks/               MDPL2 observers, halo selection, mock covariance (still empty)
tests/                 unit tests, including the spherical-infall sign gates
notebooks/             exploration only — nothing load-bearing (04 … 10, DM_simulation)
intuition/             pedagogical parameter-scan notebooks; consume the library, no script
                       twin, nothing imports from them
scripts/               thin drivers: plot_velocity_centered_dipole, plot_velocity_frame_
                       comparison, plot_redshift_space_comparison, plot_halo_class_
                       comparison, convert_mdpl2_catalog
data/                  input catalogs + DM_simulation.hdf5 (gitignored, see README)
literature/            papers and the methodological note (gitignored, local only)
.claude/agents/        subagent definitions with pinned models (implementer, reviewer)
.claude/hooks/         venv activation, post-edit sign gate + rule-5 check, stale-notebook stop
.claude/rules/         path-scoped rules, loaded only when matching files are read (plotting)
Imports from old repo/   reference code from the bulk-flow project; not importable, read only
```

`Imports from old repo/` is a reference dump, not a package. Copy and adapt from it
deliberately — do not wire imports into it. `vector3d.py`, `overdensity.py` (periodic
KDTree overdensity), `masks.py` (CF4-like selection), and `data_loader.py` are the pieces
most likely to be worth porting.

### Library, scripts, notebooks — one source of truth, two thin consumers

`src/dvcorr/` is the single source of truth: all reusable logic — geometry, estimators,
pipeline stage functions, plotting helpers — lives here, importable and tested. The other
two directories are thin consumers and hold **no** load-bearing logic.

- **`scripts/`** are thin, headless, reproducible drivers: they wire library stage functions
  together and write outputs (PNG/HDF5). Orchestration only, no algorithm. Run as
  `python -m scripts.<name>`.
- **`notebooks/`** are for interactive exploration and presentation: they import the *same*
  library stage functions and call them, adding plots and narrative — they never reimplement
  pipeline logic. `notebooks/05_velocity_centered_dipole.ipynb` is the model to follow;
  `notebooks/04_first_mdpl2_run.ipynb` still carries a duplicated copy of the pipeline and is
  the exception to be fixed.

Graduation rule: any function written in a script or notebook that is worth reusing moves
into `src/dvcorr/` and is imported back — a script or notebook is never the definition site.
This is what keeps the same logic from being duplicated across the two.

## Running

Work inside the project virtual environment (`requirements.txt` is just `-e .[dev]`;
`pyproject.toml` is the dependency source). Create it once, then activate it for every
session; don't `pip install` into the system interpreter. Inside Claude Code a SessionStart
hook (`.claude/hooks/session_start.sh`) activates it for every Bash call — no `source`
prefix needed — and warns when the venv cannot import `dvcorr` (as after the Ubuntu
24.04 upgrade removed the Python 3.10 it was first built on; it is now on 3.12).

```bash
python -m venv .venv                    # once
source .venv/bin/activate               # every session
pip install -e .[dev]                  # once, inside the .venv — editable install, dvcorr + pytest
pip install -e .[bias]                 # optional: Corrfunc, colossus, camb, mcfit — needed by
                                       # dvcorr.theory (mcfit) and pipeline.galaxy_bias; Corrfunc
                                       # needs a shared GSL at build time (see pyproject.toml)

python -m scripts.convert_mdpl2_catalog # once — CSV -> Parquet, both catalogs (name one to limit)
pytest                                 # ~1 min; tests/test_catalog_equivalence.py reads the real
                                       # catalogs and skips when they are absent
```

The sign gates are live tests, not stubs: [tests/test_geometry.py](tests/test_geometry.py)
(group-centered infall → negative dipole), `tests/test_shell_dipole.py`,
`tests/test_velocity_centered_dipole.py` (joint gate: ξ_Tu,1 < 0 and ζ₁ > 0 on the same toy)
and `tests/test_velocity_frame_dipole.py`. They must pass before any result is trusted.
**Never weaken an assertion to make a test pass**; if an assertion looks wrong, that is a
conversation about conventions, not an edit.

Two halo catalogs are supported and **both stay available** — a run picks one with
`RunConfig(catalog=CatalogConfig(name=...))`:

- `full` — `data/mdpl2_rockstar_snapnum125.csv`, ~127M halos down to 2 particles,
  subhalos included. The default.
- `mvir12` — `data/mdpl2_rockstar_125_pid-1_mvir12.csv`, 4,093,751 distinct halos at
  mvir ≥ 1e12. Every result before the full catalog landed used this one.

`full` is a strict superset: filtering it at `mvir >= 1e12` with `include_subhalos=False`
reproduces `mvir12` exactly. Box coordinates are half-open, [0, BOX_SIZE): conversion folds
the ~196 halos Rockstar emits at exactly `BOX_SIZE` down to `0.0`, matching what `mvir12`
already did. The pipeline reads Parquet, not CSV — convert once with
`python -m scripts.convert_mdpl2_catalog` (~15 min, ~2.9 GB for `full`). A full-catalog
run needs ~16 GB of RAM and a couple of minutes; ask before running anything long.

Conversion makes **two** passes over the CSV. The second is the ordinary streaming write;
the first tallies the `pid` column to build `num_of_subhalos` — how many halos name this one
as their parent, which no property of a halo's own row can answer and which must therefore
be complete before the first output row is written. `mvir12` was pre-cut to `pid == -1`, so
it has no subhalos to tally and carries `NUM_OF_SUBHALOS_UNKNOWN` (-1) rather than a `0`
that would falsely claim its halos host none. Only `dvcorr.pipeline.halo_class_comparison`
reads the column, and only from `full`. Loading it is opt-in
(`load_and_carve(..., with_num_of_subhalos=True)`) because it costs ~508 MB uncarved.

## Hard rules


1. **Frozen conventions live in `conventions.py`, nowhere else.** The pair orientation
   `r = s_V − s_T`, the cosine `µ = n̂_T · r̂`, the CMB frame, and the observer position are
   defined once. Never redefine them locally, never flip a sign at the point of use to make
   a plot look right. If a convention genuinely needs to change, change `conventions.py`, say
   so explicitly, and re-run the sign gate.

2. **The sign of the dipole is fixed by the center, and the code checks it.** Infall gives
   ξ_Tu,1 < 0 in the group-centered construction and ζ₁ > 0 in the velocity-centered one;
   the two differ by Nusser's (−1)^ℓ, never by an estimator edit. Reversing the pair
   vector or the flow axis flips every odd multipole *silently* — no exception, no NaN,
   just the wrong sign on the growth rate. The joint gate in
   `tests/test_velocity_centered_dipole.py` asserts both signs on the same toy
   (`tests/test_geometry.py` covers the group-centered half alone); a dipole of the wrong
   sign on an infall mock is an orientation bug, not a result. A PostToolUse hook
   (`.claude/hooks/post_edit.py`) re-runs the four gate files after every edit to
   `conventions.py`, `geometry.py` or `estimators/` and reports a failure immediately.

3. **Periodic boundary conditions everywhere simulation data is used.** The MDPL2 box is
   1000 h⁻¹ Mpc with PBC. Every spatial calculation — separations, KDTree queries, lines of
   sight, masks — must respect the minimum-image convention. A plain Euclidean difference
   on box coordinates is a bug. Corollary: no shell may exceed
   `conventions.MAX_ANALYSIS_RADIUS` = BOX_SIZE/2. In this codebase the rule is discharged
   once, at the carving step (see the PBC contract in `docs/architecture.md`): the
   estimators and `geometry.py` are plain Euclidean on already-unwrapped coordinates, and
   re-adding a minimum-image reduction there is a bug, not a fix. The 300 h⁻¹ Mpc Gadget-4
   snapshot is also periodic; its tighter ceiling is `SnapshotBox.max_analysis_radius`.
   The real CF4 and Tempel-SDSS data are not periodic; none of this applies to them.

4. **No bare numbers.** Any numeric literal in analysis code belongs in `conventions.py` or in a
   class attribute — never inline. For each new number, ask where it belongs before writing
   the code. (Pure-mathematics constants inside a formula, like the 1/3 from ⟨µ²⟩ over a
   uniform shell, are part of the derivation and stay in the expression, with a comment.)

5. **Never fill a missing velocity with zero.** A zero enters the count without entering
   the sum and pulls the mean toward zero. Drop the object instead. Trivially true in the
   box, where every halo has a velocity; the rule exists to catch a `fillna(0)` after a
   catalog merge on the data arm.

6. **Monopole and dipole are computed and returned together, plotted separately.** The
   monopole is the geometry diagnostic that says whether the dipole is trustworthy
   (finite-distance leakage, incomplete shells, residual bulk motion), so every estimator
   result and every normalized result carries both; a function that returns a dipole alone
   is a bug. Figures are one statistic each: the dipole figure and its monopole companion
   are produced by the same call and saved side by side, never merged into a single
   two-panel image — a dipole PNG must be sendable on its own. Note that with the
   flow-signed axis the velocity-centered monopole is |u|-weighted and sits near ⟨|u|⟩,
   not near zero; the signed leakage diagnostic is recovered from `per_center_u`.

7. **Cache-and-skip for derived columns.** Derived quantities computed once, written back
   to the catalog, existence-checked before recomputation: check column → skip if present
   → compute → save.

8. **Docstrings are the definition site; `architecture.md` is an index.** A changed
   signature, shape contract, or behavior updates the **docstring** — the file you are
   already editing — and nothing else. [docs/architecture.md](docs/architecture.md) is
   touched only when a file is **added, removed, or changes what it is responsible for**,
   and then only its one-line entry. Contracts that span modules (the PBC carving
   contract, import layering) live in that file's **Contracts** section and update when
   the contract changes. Never restate per-file detail there: it is duplication, it
   drifts, and the code wins every disagreement anyway. A change to
   conventions also updates `README.md` and this file. [docs/summary.md](docs/summary.md)
   and [docs/research_notes.md](docs/research_notes.md) exist to complete the doc structure
   but are **not actively maintained yet** — leave them until explicitly asked.

9. **Check every name for typos before writing it — even one you were given.** Any new name
   — a file, function, variable, column, class, or dict key — is spell-checked before it
   lands, *including when I asked for that exact name*. If a requested or proposed name looks
   like a typo (e.g. `veloctiy`, `seperation`, `analyis`, `dipoel`), do not silently adopt
   it: flag it, ask whether it is intentional, and if it is not, fix it to the correct
   spelling. Assume a misspelling is accidental unless I confirm otherwise. A wrong name
   cached to a catalog column (rule 7) or frozen into a public signature is expensive to
   rename later, so catch it at the point of writing.

10. **Default separation range and binning: r ∈ [0, 64] h⁻¹ Mpc, log bins of ratio √2
    from 0.5.** The default ladder is the inner bin [0, 0.5) followed by 14 geometric
    shells 0.5 · √2ⁿ ending exactly at 64 — fifteen shells in all — built by
    `ShellConfig(min_radius=0.5, max_radius=64.0, spacing=SPACING_LOG, n_bins=14,
    include_zero_bin=True)`. Any run that departs from this binning — **including one the
    user asked for** — is confirmed with the user before it is executed: state the edges
    you are about to use and wait for a yes. Figures draw one point per shell at r_eff
    joined by a line (never steps), with the x-axis from r = 0, and each gets a binned
    companion; the full rendering rules are in
    [.claude/rules/plotting.md](.claude/rules/plotting.md), which loads automatically
    whenever pipeline code, scripts or notebooks are read.

11. **A changed notebook is re-executed, top to bottom, before the work is called done.**
    A notebook's saved outputs are the figures the user reads, so an edit that leaves old
    outputs in place — or a library change that alters what a notebook's cells would now
    produce — leaves a notebook that lies. After editing a notebook's cells, and after
    changing any `dvcorr` code a notebook calls (a plotting helper, a stage function, a
    default), re-run every affected notebook headlessly and save the outputs in place:

    ```bash
    jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
    ```

    A run on the `full` catalog is long (see *Running*), so the confirmation there still
    applies: say which notebooks need re-running and why, ask before starting, then run
    them. What is never acceptable is silently skipping the re-run: a notebook that was
    touched but not re-executed is reported as stale in the final message, by name, so the
    user knows its outputs are not to be trusted until it is. A Stop hook
    (`.claude/hooks/stop_stale_notebooks.py`) backstops this: it traces which notebooks
    import an edited `dvcorr` module and, once per edit, lists those whose outputs
    predate it.

## Answering questions

Every answer to a question — anything asked with a question mark, or a "why / what /
how / which / should I" in prose, including one asked mid-task — is written in three
labeled layers, in this order:

1. **Bottom line:** the answer itself in a phrase or one short sentence. No hedging, no
   context; if the honest answer is "it depends", say what it depends on and which way it
   goes in the common case.
2. **One line:** one sentence giving the reason or mechanism behind the bottom line.
3. **Long answer:** the regular full response — derivation, evidence, caveats, code.

Example:

> Q: why did the man go to the supermarket?
>
> **Bottom line:** to get milk.
> **One line:** the milk in his house ran out, so he went to the supermarket to get more.
> **Long answer:** (the regular full response)

The first two layers are read on their own, so they must stand without the third: the
bottom line is the sentence that gets quoted, the one line is what makes it believable.
The long answer never contradicts them; if writing it changes the conclusion, rewrite the
bottom line rather than burying the correction below. A request to build, change, or run
something is not a question — its deliverable is the work and the ordinary report format
stands — but a question embedded in such a request still gets its three layers, at the top
of the reply.

## Coding conventions

- **Prefer functions over class methods.** Default to module-level functions for anything
  that transforms inputs to outputs. Reach for a class only to *hold* things — a catalog,
  a KDTree, a run's worth of state, or stable data like configs, settings, and numbers (see
  the two bullets below) — never to house behavior that could stand alone as a function.
  When a piece of logic doesn't need `self`, it is a function, not a method. This is the
  general default; the more specific rules that follow are instances of it.
- **Geometry and estimator primitives are pure free functions.** `geometry.py` and the
  estimator cores are stateless, array-vectorized, and fully unit-testable — that is what
  makes the sign gate possible. Do not wrap them in classes.
- **Stateful pipeline stages are classes.** Anything holding a catalog, a KDTree, a
  configuration, or a run's worth of intermediate state gets encapsulated (cf. the old
  repo's `MaskMaker`, `OverdensityCalculator`).
- **Numbers and stable configuration are stored in classes.** Parameters and settings
  that are not expected to change often live as class attributes (or a small config
  dataclass, cf. the old repo's `src/config/`), not as inline literals or scattered
  module globals. This is the storage counterpart to keeping *compute* primitives free
  functions, and the mechanism behind hard rule 4. The frozen module-level constants in
  `conventions.py` are the deliberate exception: single-source-of-truth conventions, not
  tunables.
- **Explicit array-shape contracts in every docstring.** `(3,)` for one vector, `(N, 3)` for
  many, `(N,)` for scalars-per-object. Shapes are stable: a function documented as returning
  `(N,)` returns `(N,)` even for N = 1.
- Type hints throughout; `from __future__ import annotations` at the top of each module.

## Notation

Fixed project-wide; see the README table and `conventions.py`.

- `L_ell` — Legendre polynomials. **`P` is reserved for the matter power spectrum `P_m(k)`**
  and is never used for Legendre polynomials.
- `r = s_V − s_T`, `r = |r|`, `µ = n̂_T · r̂`, `R = |s_T|`, `u = v · n̂_V` (signed)
- `ẑ = sign(u) · n̂_V` — the polar axis of a velocity-object shell expansion follows the
  object's **motion**, not its line of sight: a halo approaching the observer has `ẑ = −r̂`.
  The companion weight is the **speed** `|u|`, never the signed `u` — the sign lives in the
  axis, and entering it twice cancels the statistic. One definition site,
  `geometry.radial_flow_axis`; the observer-free variant is the same statement with the full
  3-vector, `ẑ = v̂`, weight `|v|`.
- `ξ_Tu,ℓ` — density–velocity multipoles; `Ψ∥`, `Ψ⊥` — Górski velocity–velocity functions
- Nusser (2017) centers on the velocity object, so his multipoles relate to ours by
  `(−1)^ℓ` — see `conventions.nusser_multipole_sign`. Apply the factor explicitly when comparing;
  never absorb it into an estimator. `dvcorr.theory` speaks Nusser's orientation throughout
  (its separation is spelled `y`), so its dipole is positive for infall.
- **Two normalizations, never mixed.** The velocity-centered ζ̂₁ divides by the *expected*
  uniform occupancy n̄V_b (Nusser eq. 24) and carries no (2ℓ+1): its linear-theory
  expectation is C(r)/3 = aH f b A(r)/3, the ⟨µ²⟩ = 1/3 of a uniform shell. The
  group-centered ξ_Tu,ℓ is written with (2ℓ+1)·Σ/count. Dividing by the *realized* pair
  count instead of n̄V_b gives Nusser's mass-weighted ζ^M, lower by (1+ξ(r)).

## Units

- Positions: comoving h⁻¹ Mpc (periodic box coordinates)
- Velocities: km/s; masses: `mvir` in h⁻¹ M☉
- MDPL2 cosmology: H0 = 67.77, Ωm = 0.307115, σ8 = 0.8228; box 1000 h⁻¹ Mpc
- MDPL2 particle mass: 1.5054e9 h⁻¹ M☉ (`conventions.PARTICLE_MASS`). `mvir` is exactly
  quantized in multiples of it, so `mvir / PARTICLE_MASS` is an exact particle count.
  Resolution thresholds: `RESOLVED_PARTICLE_COUNT` = 20, `CONVERGED_PARTICLE_COUNT` = 100
- Small-scale velocity noise: σ* ≈ 250 km/s (`ShellConfig.sigma_star`) — **not read by any
  code yet**; a placeholder for the error model, and measured on the mvir ≥ 1e12 population,
  so it needs re-measuring before first use on the full catalog
- Gadget-4 snapshot (`data/DM_simulation.hdf5`): 300 h⁻¹ Mpc periodic box, 256³ particles,
  z = 0, h = 0.678, Ωm = 0.308, a constrained realization (2MRS initial conditions).
  Units are read and checked from the file (`pipeline.dm_snapshot`); stored velocities
  carry Gadget's √a factor, exactly 1 here.

## Agent workflow

Delegate only through the two agents in `.claude/agents/`, never a bare `general-purpose`
spawn — their definitions pin the model, so routing is enforced by config:

- **`implementer`** (Sonnet) — writing or editing code. It exists to move code-writing
  onto a cheaper model, so delegate only when the task needs *discovery* (files the main
  session has not read) **and** the main session is not itself on Sonnet. Otherwise work
  inline: a subagent starts cold and re-reads what this context already holds.
- **`reviewer`** (Opus) — fresh eyes on a substantial change the main agent wrote. What it
  buys is a reader without the author's assumptions, not a cheaper model; skip it for a
  small diff or for code the main agent did not write.

Brief a subagent with the file paths it needs and point it at the module docstrings (the
authority on any function it touches); [docs/architecture.md](docs/architecture.md) is the
cheap index for finding files. Each agent file carries its own reading discipline and
when-not-to-use rules.
