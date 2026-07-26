# tempel-cf4-xcorr

**Density–velocity cross-correlations from Tempel et al. (2017) SDSS groups and Cosmicflows-4 peculiar velocities.**

This repository measures the group-centered density–velocity correlation multipoles $\xi_{Tu,0}$ and $\xi_{Tu,1}$ between the unique Tempel SDSS DR12 group catalog and Cosmicflows-4 (CF4) radial peculiar velocities, together with the complementary Górski velocity–velocity correlations $\Psi_\parallel$ and $\Psi_\perp$ from CF4–CF4 pairs. The methodology follows Nusser (2017), adapted from a velocity-centered to a group-centered construction, with all conventions fixed by the accompanying methodological note.

The primary science target is the dipole $\xi_{Tu,1}$, whose large-scale amplitude scales as $b_T f P_m$ and constrains the growth rate combination $f\sigma_8$ (equivalently $\beta = f/b$). The monopole $\xi_{Tu,0}$ serves as a geometry and calibration diagnostic; $\Psi_\parallel$ and $\Psi_\perp$ (scaling as $f^2 P_m$) probe the coherence of the velocity field itself.

## Scientific background

- Nusser, A. 2017, MNRAS 470, 445 — the u–δ shell-dipole correlation method this project reproduces and extends (there: CF3 × 2MRS).
- Górski, K. 1988, ApJ 332, L7 — the velocity correlation tensor formalism underlying $\Psi_\parallel$ and $\Psi_\perp$.
- Turner, Blake & Ruggeri 2021, MNRAS 502, 2087 — the general pair-count galaxy–velocity correlation framework.
- Tempel et al. 2017, A&A 602, A100 — the SDSS group catalog.
- Tully et al. 2023, ApJ 944, 94 — Cosmicflows-4.

## Frozen conventions

These are fixed project-wide in `src/tempelcf4/config.py` and `geometry.py` and enforced by unit tests. Do not redefine them locally.

| Symbol | Meaning |
|---|---|
| $s_T$, $s_V$ | Tempel and CF4 **redshift-space** positions, CMB frame, one shared redshift–distance convention |
| $\mathbf{r} = s_V - s_T$ | Pair separation vector, **group → velocity object**; $r = \|\mathbf{r}\|$ |
| $\mu = \hat{n}_T \cdot \hat{r}$ | Configuration-space, group-centered angular cosine |
| $r_\parallel$, $r_\perp$ | $r\mu$ and $r\sqrt{1-\mu^2}$ |
| $R = \|s_T\|$ | Observer-to-group distance |
| $u = \mathbf{v} \cdot \hat{n}_V$ | Observer-centered radial peculiar velocity (CF4 `Vpec`) |
| $\xi_{Tu,\ell}$ | Multipoles of the Tempel-density × CF4-velocity correlation |
| $\Psi_\parallel$, $\Psi_\perp$ | Górski velocity–velocity correlation functions |
| $L_\ell$ | Legendre polynomials; $P_m(k)$ is reserved for the matter power spectrum |
| $\mu_k = \hat{k} \cdot \hat{n}$ | Fourier-space line-of-sight cosine (RSD formulae only) |

Consequences that follow from these choices and must not be violated:

- Coherent infall gives a **negative** dipole $\xi_{Tu,1}$.
- Reversing the pair orientation flips the sign of all odd multipoles.
- Distance-indicator distances are **never** used as pair-position coordinates; positions are redshift-space only.
- Missing peculiar-velocity measurements are missing data, never $u = 0$.
- Every Tempel group enters exactly once (`IDg` collapsed); using all member rows silently produces a richness-weighted statistic.

## Repository layout

```
data/raw/         EDD catalog downloads — never committed (see Data below)
data/external/    Cosmological parameter files, P_m(k) tables
data/processed/   Unique group catalog, harmonized positions — rebuilt by scripts
src/tempelcf4/
  config.py       Cosmology and frozen conventions
  geometry.py     Sky → redshift-space positions, exact curved-sky pair geometry
  io/             Catalog loaders (Tempel collapse, CF4 velocity definitions)
  selection/      Angular masks, radial selection functions, random catalogs
  estimators/     ξ_Tu,ℓ pair estimator; Ψ∥/Ψ⊥ per-bin regression; ξ_TT
  theory.py       Linear-theory predictions for ξ_Tvr, Ψ∥, Ψ⊥
  mocks/          MDPL2 halo selection, mock observers, mock covariance
  errors.py       Monte Carlo propagation of distance-modulus errors
  nulltests.py    Shuffle, sign, duplicate, random-catalog, overlap, velocity-definition tests
scripts/          Numbered pipeline stages (00–07), one per analysis step
notebooks/        Exploration only (footprint maps, depth checks) — nothing load-bearing
tests/            Unit tests, including the spherical-infall sign gate
results/          Final figures and products
```

## Pipeline

Run the numbered scripts in order; each writes to `data/processed/` or `results/` and each is re-runnable from scratch.

1. `00_download_data.py` — fetch the EDD tables (`ktempel17`, `kcf4allvel`) into `data/raw/`.
2. `01_build_unique_groups.py` — collapse Tempel rows to one row per `IDg`; choose and record the group-center definition (default: `Rg = 1` member; robustness alternatives logged).
3. `02_harmonize_positions.py` — CMB-frame redshift-space positions for both catalogs under one stated convention.
4. `03_build_randoms.py` — random catalogs matching angular mask and radial selection for every richness/redshift cut; emits data-vs-random n(z) and healpy footprint diagnostics.
5. `04_measure_xi_tu.py` — $\xi_{Tu,0}$ and $\xi_{Tu,1}$ simultaneously, exact pair geometry, binned in $r$ (and group distance $R$ for the monopole-leakage check).
6. `05_measure_psi.py` — $\Psi_\parallel$ and $\Psi_\perp$ from CF4–CF4 pairs via the exact curved-sky two-parameter regression.
7. `06_mock_covariance.py` — mock-based covariance from MDPL2 observer light-cones (primary); spatial jackknife as a secondary check only.
8. `07_fit_model.py` — joint fit of $\xi_{TT}$, $\xi_{Tu,1}$, $\Psi_\parallel$, $\Psi_\perp$ for bias, growth, and nuisance dispersion.

## Null tests

Required before any measurement is treated as a result (`nulltests.py`):

velocity shuffle (cross-correlation must vanish), spherical-infall sign test (recovered dipole must be negative), duplicate test (one row per `IDg` / CF4 identifier), random-catalog consistency (angular and n(z) match after every cut), overlap removal (repeat without exact Tempel–CF4 identity matches and smallest bins), and velocity-definition robustness (`Vpds`, `Vpwf` vs `Vpec`).

## Error model

Independent-pair error bars are invalid: pairs share objects and large-scale velocities are strongly correlated. The covariance is mock-based, built from MDPL2/Rockstar light-cones with halo selection matched to the survey selection function $\phi(r)$, realistic redshift-space distortions, and Monte Carlo velocity errors drawn from the CF4 distance-modulus uncertainties. A conservative first linear-theory scale cut is $r \gtrsim 20\ h^{-1}\,\mathrm{Mpc}$.

## Installation

```bash
git clone <repo-url>
cd tempel-cf4-xcorr
conda env create -f environment.yml   # or: pip install -e .
conda activate tempelcf4
pytest                                # geometry and sign-convention tests must pass
```

Core dependencies: numpy, scipy, astropy, healpy, pandas; Corrfunc for the group autocorrelation; CAMB or CLASS for $P_m(k)$; halotools for the MDPL2 side.

## Data

**No catalog data is committed to this repository.** The raw tables are obtained from the Extragalactic Distance Database via `scripts/00_download_data.py`:

- Tempel17 SDSS DR12 Clipped: https://edd.ifa.hawaii.edu/describe_columns.php?table=ktempel17
- CF4 All Group Velocities: https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allvel

These datasets are governed by their own access and citation terms (EDD / SDSS / CF4 collaborations), which are separate from and unaffected by this repository's license. If you use the data, cite the original catalog papers listed above.

## License

The **code** in this repository is released under the MIT License (see `LICENSE`). The license covers the code only — not the input catalogs, which remain subject to their original providers' terms, and not the referenced publications.

If you use this pipeline in academic work, please cite it via `CITATION.cff` along with Nusser (2017), Górski (1988), and the catalog papers.

## Acknowledgements

Methodology developed under the supervision of Prof. Adi Nusser (Technion). The estimator design follows the group-centered adaptation of Nusser (2017) documented in the project methodological note.
