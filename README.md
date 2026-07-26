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

## Status

The project is in its **simulation-validation arm**: the estimator is being reproduced inside the MDPL2 box, where halo peculiar velocities and the halo density field are both known, so every sign and normalization can be checked before the method is trusted on data. What exists today is the installable `dvcorr` library (frozen conventions, geometry primitives, the group- and velocity-centered shell-dipole estimators, and the velocity-centered pipeline), its sign-gate test suite, and one runnable driver — `scripts/plot_velocity_centered_dipole.py`. The numbered data-arm pipeline, the EDD data download, and the `io/`, `theory.py`, `errors.py`, and null-test modules described below are the **planned** end-to-end analysis, not yet implemented.

## Frozen conventions

These are fixed project-wide in `src/dvcorr/conventions.py` and `src/dvcorr/geometry.py` and enforced by unit tests. Do not redefine them locally.

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

```text
pyproject.toml    Package metadata, dependencies, pytest config — the single dependency source
src/dvcorr/       The installable package (editable install: pip install -e .)
  conventions.py    Frozen conventions — sign, pair orientation, observer, box, catalog columns
  config/           Tunable settings, one dataclass per file (paths, cosmology, shell
                    binning, selection) plus a Settings aggregator
  geometry.py       Pure geometry primitives (unit_vector, pair_separation, mu_cosine)
  estimators/       shell_dipole.py — group-centered (ξ_Tu) and velocity-centered (Nusser ζ)
                    monopole + dipole accumulators per radial shell
  pipeline/         velocity_centered.py — reusable stage functions shared by the driver
                    script and its notebook (load/carve, draw, run, normalize, plot)
  selection/        Angular masks, radial selection functions, random catalogs (planned)
  mocks/            MDPL2 halo selection, mock observers, mock covariance (planned)
scripts/          Runnable, load-bearing drivers (e.g. plot_velocity_centered_dipole.py)
notebooks/        Exploration and presentation — thin consumers of the library, nothing load-bearing
tests/            Unit tests, including the spherical-infall sign gate
data/             Input catalogs — gitignored, never committed (see Data below)
output/           Derived products and figures
literature/       Papers and the project methodological note
```

The `io/`, `theory.py`, `errors.py`, and `nulltests.py` modules of the planned data arm (see Status) do not exist yet; the layout above is the current package.

## Pipeline (planned data arm)

The numbered stages below are the **planned** end-to-end data-arm analysis — none exist yet (see Status). Once built, they run in order; each writes to `data/` or `output/` and each is re-runnable from scratch.

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
cd Cross-correlating-Tempel-SDSS-groups-with-Cosmicflows-4-peculiar-velocities
python -m venv .venv                  # once
source .venv/bin/activate             # every session
pip install -e .[dev]                 # once, inside the .venv — editable install, dvcorr + pytest
pytest                                # geometry and sign-convention tests must pass
```

Core dependencies, declared in `pyproject.toml`: numpy, pandas, scipy, astropy, healpy, matplotlib, h5py (plus pytest for the test suite). The data arm will additionally need Corrfunc (group autocorrelation), CAMB or CLASS ($P_m(k)$), and halotools (MDPL2), added when those stages are built.

## Data

**No catalog data is committed to this repository.** The raw tables will be obtained from the Extragalactic Distance Database via the planned `scripts/00_download_data.py`:

- Tempel17 SDSS DR12 Clipped: https://edd.ifa.hawaii.edu/describe_columns.php?table=ktempel17
- CF4 All Group Velocities: https://edd.ifa.hawaii.edu/describe_columns.php?table=kcf4allvel

These datasets are governed by their own access and citation terms (EDD / SDSS / CF4 collaborations), which are separate from and unaffected by this repository's license. If you use the data, cite the original catalog papers listed above.

## License

The **code** in this repository is released under the MIT License (see `LICENSE`). The license covers the code only — not the input catalogs, which remain subject to their original providers' terms, and not the referenced publications.

If you use this pipeline in academic work, please cite it via `CITATION.cff` along with Nusser (2017), Górski (1988), and the catalog papers.

## Acknowledgements

Methodology developed under the supervision of Prof. Adi Nusser (Technion). The estimator design follows the group-centered adaptation of Nusser (2017) documented in the project methodological note.
