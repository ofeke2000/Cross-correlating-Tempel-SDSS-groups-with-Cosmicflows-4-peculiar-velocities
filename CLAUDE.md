# CLAUDE.md

Research code measuring the **density–velocity cross-correlation dipole** ξ_Tu,1 between
Tempel et al. (2017) SDSS groups and Cosmicflows-4 peculiar velocities, following
Nusser (2017) in a group-centred rather than velocity-centred construction.

The current phase is the **simulation-validation arm**: reproduce the estimator inside the
MDPL2 box, where halo peculiar velocities and the halo density field are both known exactly
and the answer can be checked before it is trusted on data.

Read these for context before non-trivial work:

- [README.md](README.md) — science goals, frozen conventions table, full pipeline, null tests
- [literature/tempel_cf4_velocity_correlations.pdf](literature/tempel_cf4_velocity_correlations.pdf)
  — the project's methodological note; **the authority on every convention and sign**
- [literature/1703.05324v2.pdf](literature/1703.05324v2.pdf) — Nusser (2017), the estimator
  being reproduced (his eq. 23 is the shell dipole)
- [literature/1988ApJ...332L...7G.pdf](literature/1988ApJ...332L...7G.pdf) — Górski (1988),
  the Ψ∥/Ψ⊥ velocity-correlation formalism used in the later CF4×CF4 analysis
- [literature/1704.04477v1.pdf](literature/1704.04477v1.pdf) — Tempel et al. (2017), the
  group catalogue

## Code layout

```
config.py            frozen conventions — sign, pair orientation, observer, box, columns
geometry.py          pure geometry primitives (unit_vector, pair_separation, mu_cosine)
estimators/          shell_dipole.py — monopole + dipole accumulators per radial shell
selection/           masks, selection functions, random catalogues (empty)
mocks/               MDPL2 observers, halo selection, mock covariance (empty)
tests/               unit tests, including the spherical-infall sign gate
notebooks/           exploration only — nothing load-bearing
data/                input catalogues (gitignored, see README)
literature/          papers and the methodological note
Imports from old repo/   reference code from the bulk-flow project; not importable, read only
```

`Imports from old repo/` is a reference dump, not a package. Copy and adapt from it
deliberately — do not wire imports into it. `vector3d.py`, `overdensity.py` (periodic
KDTree overdensity), `masks.py` (CF4-like selection), and `data_loader.py` are the pieces
most likely to be worth porting.

## Running

```bash
pytest    # geometry and sign-convention tests
```

The sign-gate tests in [tests/test_geometry.py](tests/test_geometry.py) are currently
`xfail(strict=True)` because the physics is unimplemented. When a stub lands, remove its
marker. **Never weaken an assertion to make a test pass**; if an assertion looks wrong,
that is a conversation about conventions, not an edit.

`data/mdpl2_rockstar_125_pid-1_mvir12.csv` is ~4M halos. Don't load it fully unless the
task needs it, and ask before running anything long.

## Hard rules

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
   the catalogue, existence-checked before recomputation: check column → skip if present →
   compute → save.

8. **Keep docs in sync.** A change to structure or conventions updates `README.md` and this
   file in the same task.

## Coding conventions

- **Geometry and estimator primitives are pure free functions.** `geometry.py` and the
  estimator cores are stateless, array-vectorised, and fully unit-testable — that is what
  makes the sign gate possible. Do not wrap them in classes.
- **Stateful pipeline stages are classes.** Anything holding a catalogue, a KDTree, a
  configuration, or a run's worth of intermediate state gets encapsulated (cf. the old
  repo's `MaskMaker`, `OverdensityCalculator`).
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
- Nusser (2017) centres on the velocity object, so his multipoles relate to ours by
  `(−1)^ℓ` — see `config.nusser_multipole_sign`. Apply the factor explicitly when comparing;
  never absorb it into an estimator.

## Units

- Positions: comoving h⁻¹ Mpc (periodic box coordinates)
- Velocities: km/s; masses: `mvir` in h⁻¹ M☉
- MDPL2 cosmology: H0 = 67.77, Ωm = 0.307115, σ8 = 0.8228; box 1000 h⁻¹ Mpc
- Small-scale velocity noise: σ* ≈ 250 km/s
