# Research notes — current goals

> **Maintenance status:** **not actively maintained yet.** This file exists so the
> three-document structure (summary / architecture / research notes) is in place; right now
> only [architecture.md](architecture.md) is kept in sync as we work. Update this file only
> when asked. Treat anything below as a starting scaffold, not a current record.

## What we're working on now

The **simulation-validation arm**: get the group-centered density–velocity dipole estimator
working and sign-verified inside the MDPL2 box before it touches data.

## Done so far

- Frozen conventions pinned in `config.py`.
- Geometry primitives (`unit_vector`, `pair_separation`, `mu_cosine`) with the sign gate.
- Shell estimator (`shell_dipole`) returning raw, un-normalized per-shell monopole and
  dipole; estimator-level sign gate and null tests passing.

## Open threads / next steps

- **Sub-volume carving + PBC.** The primitives assume neighbors are already unwrapped into
  each center's continuous frame. The carving/minimum-image step that guarantees this is not
  yet written; it is the correct home for periodic handling (see the PBC note in
  architecture.md and CLAUDE.md hard rule 3).
- **Selection** (`selection/`): CF4-like masks, φ(r), randoms.
- **Mocks** (`mocks/`): MDPL2 observers, survey-matched halo selection, mock covariance.
- **Loading MDPL2**: `data/mdpl2_rockstar_125_pid-1_mvir12.csv` is ~4M halos — port a loader
  from the old repo's `data_loader.py`; don't load fully without need.

## Questions to resolve

- Exact normalization and random-subtraction scheme for `ξ_Tu,1` downstream of the raw sums.
- Scale cut and binning for the linear-theory comparison (first pass r ≳ 20 h⁻¹ Mpc).
