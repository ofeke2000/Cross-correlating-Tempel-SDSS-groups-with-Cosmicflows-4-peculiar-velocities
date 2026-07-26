# CLAUDE.md

Research code measuring the **density–velocity cross-correlation dipole** ξ_Tu,1 between
Tempel et al. (2017) SDSS groups and Cosmicflows-4 peculiar velocities, following
Nusser (2017) in a group-centered rather than velocity-centered construction.

The current phase is the **simulation-validation arm**: reproduce the estimator inside the
MDPL2 box, where halo peculiar velocities and the halo density field are both known exactly
and the answer can be checked before it is trusted on data.

Read these for context before non-trivial work:

- [docs/summary.md](docs/summary.md) — science context, conventions, pipeline (best overview)
- [docs/architecture.md](docs/architecture.md) — per-file responsibilities; the live map,
  kept in sync with the code
- [docs/research_notes.md](docs/research_notes.md) — current goals; what we are working on now
- [README.md](README.md) — science goals, frozen conventions table, full pipeline, null tests
- [literature/tempel_cf4_velocity_correlations.pdf](literature/tempel_cf4_velocity_correlations.pdf)
  — the project's methodological note; **the authority on every convention and sign**
- [literature/1703.05324v2.pdf](literature/1703.05324v2.pdf) — Nusser (2017), the estimator
  being reproduced (his eq. 23 is the shell dipole)
- [literature/1988ApJ...332L...7G.pdf](literature/1988ApJ...332L...7G.pdf) — Górski (1988),
  the Ψ∥/Ψ⊥ velocity-correlation formalism used in the later CF4×CF4 analysis
- [literature/1704.04477v1.pdf](literature/1704.04477v1.pdf) — Tempel et al. (2017), the
  group catalog

## Code layout

```
config.py            frozen conventions — sign, pair orientation, observer, box, columns
settings.py          tunable settings in dataclasses — paths, cosmology, shell binning, selection
geometry.py          pure geometry primitives (unit_vector, pair_separation, mu_cosine)
estimators/          shell_dipole.py — monopole + dipole accumulators per radial shell,
                     group-centered (xi_Tu) and velocity-centered (zeta, Nusser eq. 23-24)
selection/           masks, selection functions, random catalogs (empty)
mocks/               MDPL2 observers, halo selection, mock covariance (empty)
tests/               unit tests, including the spherical-infall sign gate
notebooks/           exploration only — nothing load-bearing
scripts/             runnable, load-bearing scripts (e.g. plot_velocity_centered_dipole.py)
data/                input catalogs (gitignored, see README)
literature/          papers and the methodological note
Imports from old repo/   reference code from the bulk-flow project; not importable, read only
```

`Imports from old repo/` is a reference dump, not a package. Copy and adapt from it
deliberately — do not wire imports into it. `vector3d.py`, `overdensity.py` (periodic
KDTree overdensity), `masks.py` (CF4-like selection), and `data_loader.py` are the pieces
most likely to be worth porting.

## Running

Work inside the project virtual environment — it isolates the pinned dependencies in
`requirements.txt` from the system Python. Create it once, then activate it for every
session; don't `pip install` into the system interpreter.

```bash
python -m venv .venv                    # once
source .venv/bin/activate               # every session
pip install -r requirements.txt        # once, inside the .venv

pytest                                 # geometry and sign-convention tests
```

The sign-gate tests in [tests/test_geometry.py](tests/test_geometry.py) are currently
`xfail(strict=True)` because the physics is unimplemented. When a stub lands, remove its
marker. **Never weaken an assertion to make a test pass**; if an assertion looks wrong,
that is a conversation about conventions, not an edit.

`data/mdpl2_rockstar_125_pid-1_mvir12.csv` is ~4M halos. Don't load it fully unless the
task needs it, and ask before running anything long.

## Hard rules

0. **Check `Imports from old repo/` before starting any task.** Before writing new code —
   an estimator, a mask, an overdensity calculation, a data loader, geometry helpers — read
   the corresponding reference code there first. It is the bulk-flow project's working
   implementation and often already solves the problem (or shows the sign/PBC pitfalls).
   Copy and adapt deliberately; never wire live imports into it. The folder is gitignored
   and not a package.

1. **Frozen conventions live in `config.py`, nowhere else.** The pair orientation
   `r = s_V − s_T`, the cosine `µ = n̂_T · r̂`, the CMB frame, and the observer position are
   defined once. Never redefine them locally, never flip a sign at the point of use to make
   a plot look right. If a convention genuinely needs to change, change `config.py`, say so
   explicitly, and re-run the sign gate.

2. **The dipole is negative for infall.** This follows from rule 1 and is asserted by
   `tests/test_geometry.py`. Reversing the pair vector flips all odd multipoles *silently* —
   no exception, no NaN, just the wrong sign on the growth rate. Treat any positive dipole
   from an infall mock as an orientation bug, not a result.

3. **Periodic boundary conditions everywhere.** The MDPL2 box is 1000 h⁻¹ Mpc with PBC.
   Every spatial calculation — separations, KDTree queries, lines of sight, masks — uses the
   minimum-image convention. A plain Euclidean difference on box coordinates is a bug.
   Corollary: no shell may exceed `config.MAX_ANALYSIS_RADIUS` = BOX_SIZE/2.

4. **No bare numbers.** Any numeric literal in analysis code belongs in `config.py` or in a
   class attribute — never inline. For each new number, ask where it belongs before writing
   the code. (Pure-mathematics constants inside a formula, like the 1/3 from ⟨µ²⟩ over a
   uniform shell, are part of the derivation and stay in the expression, with a comment.)

5. **Missing velocity is missing data.** An object without a peculiar-velocity measurement
   is dropped from the pair count. It is never entered as u = 0 — that would bias the mean
   toward zero and dilute the dipole.

6. **Monopole and dipole are reported together.** The monopole is the geometry diagnostic
   that says whether the dipole is trustworthy (finite-distance leakage, incomplete shells,
   residual bulk motion). Never return or plot a dipole alone.

7. **Cache-and-skip for derived columns.** Derived quantities computed once, written back to
   the catalog, existence-checked before recomputation: check column → skip if present →
   compute → save.

8. **Keep docs in sync.** Any change to code structure — a new module, a moved
   responsibility, a changed public signature — updates [docs/architecture.md](docs/architecture.md)
   in the same task; it is the live per-file map and must never lag the code. A change to
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
  `config.py` are the deliberate exception: single-source-of-truth conventions, not
  tunables.
- **Explicit array-shape contracts in every docstring.** `(3,)` for one vector, `(N, 3)` for
  many, `(N,)` for scalars-per-object. Shapes are stable: a function documented as returning
  `(N,)` returns `(N,)` even for N = 1.
- Type hints throughout; `from __future__ import annotations` at the top of each module.

## Notation

Fixed project-wide; see the README table and `config.py`.

- `L_ell` — Legendre polynomials. **`P` is reserved for the matter power spectrum `P_m(k)`**
  and is never used for Legendre polynomials.
- `r = s_V − s_T`, `r = |r|`, `µ = n̂_T · r̂`, `R = |s_T|`, `u = v · n̂_V`
- `ξ_Tu,ℓ` — density–velocity multipoles; `Ψ∥`, `Ψ⊥` — Górski velocity–velocity functions
- Nusser (2017) centers on the velocity object, so his multipoles relate to ours by
  `(−1)^ℓ` — see `config.nusser_multipole_sign`. Apply the factor explicitly when comparing;
  never absorb it into an estimator.

## Units

- Positions: comoving h⁻¹ Mpc (periodic box coordinates)
- Velocities: km/s; masses: `mvir` in h⁻¹ M☉
- MDPL2 cosmology: H0 = 67.77, Ωm = 0.307115, σ8 = 0.8228; box 1000 h⁻¹ Mpc
- Small-scale velocity noise: σ* ≈ 250 km/s

## Agent workflow

- Delegate implementation work (writing/editing code) to a Sonnet subagent.
- Delegate code review to an Opus subagent.
