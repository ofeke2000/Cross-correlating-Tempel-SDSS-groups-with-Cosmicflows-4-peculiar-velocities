# Roadmap: from MDPL2 to Nusser (2017) on Tempel × CF4

## Context

The simulation arm is done. The estimators, sign gates, theory (Nusser Fig. 1), the redshift-space
and frame comparisons, robustness and bias are all in `dvcorr`. The real catalogs are already in
`data/` (`CF4_Groups.csv`, `CF4_Groups_Velocities.csv`, `SDSS_Tempel.csv`); Stage 1 added the loader
(`dvcorr.pipeline.real_catalogs`). Ofek wants a guided map from "load the data" to "every figure and number in
Nusser (2017) redone with Tempel groups as the density field and CF4 as the velocities." The map
should be built **one stage at a time**, and each stage should say what a correct result looks
like, so a wrong plot is caught when it is made.

**How to use it:** build it one stage at a time. Start each stage by pasting that stage's
prompt into a new Claude Code session, and move on only when its **Done when** line holds.

Every stage has the same parts:

- **What this stage is for:** the goal in plain words.
- **Do / Expect / Plots:** the work and the intuition checks.
- **Done when:** the exit condition before you move on.
- **Prompt to start it:** paste-ready.

Numbers marked *(measured)* came from a read-only pass over the files on 2026-10-11. Numbers marked
*(measured 10-14)* came from the same kind of pass on 2026-10-14.

**Advisor update (2026-10-14).** The scope now covers two density fields and two velocity phases,
plus a Malmquist-bias correction. See [advisor_notes.md](advisor_notes.md) for what was said, how it
was read, and the questions still open. The next section sets out the plan. Every stage below
marks what it adds with **[2026-10-14]**.

---

## The advisor's plan: two density fields, two velocity phases

| Axis | Options | Order |
| --- | --- | --- |
| Density tracer | **T** = Tempel SDSS groups (≈ 7,600 deg², north) · **M** = 2MRS K ≤ 11.75 galaxies, Nusser's own catalog (`data/2MRS_K<11.75.txt`, near all-sky, ZoA excluded) | both, run separately |
| Velocity tracer | **Phase A** = CF4 groups (38,057) · **Phase B** = CF4 individual galaxies, **one run per source** (SDSS FP, 6dFGS FP, TF, SNIa, SBF, …) | A all the way through Stage 8 first, then B (Stage 8B) |
| Velocity bias | Malmquist bias corrected with a selection function for each velocity sample (Stage 4) | before any Phase B comparison |

- **Why 2MASS:** it was Nusser's own density field (2MRS). So **M × CF4 groups** is his
  measurement with CF3 swapped for CF4, a direct comparison. It also covers the southern sky, where
  6dFGS lives and where SDSS has almost no coverage. A 6dF-only run needs M.
- **Why split by source:** CF4 combines distance indicators that each have their own zero point,
  scatter and selection. If the sources agree on ζ₁(r), the combined result can be trusted. If one
  source disagrees, the advisor's prime suspect is 6dF, and that source is either dropped or
  corrected.
- **Measured 10-14 (`CF4_Groups.csv` method counts, all / cz < 10,000):** FP 27,697 / 3,196;
  TF 10,187 / 8,919; SNIa 945 / 554; SBF (optical + IR) 235 / 235; SNII 94; calibrators 337. 96%
  of groups (36,693, counting SNII) use a single method, so the *groups* already split almost
  cleanly by method.
  However, `Nfp` combines SDSS FP and 6dFGS FP. At cz < 10,000, 1,845 FP groups are at δ < 0 and
  1,351 at δ ≥ 0. Separating the two cleanly needs Phase B's individual table.

---

## What changes when you leave the box

| | MDPL2 box | Real sky |
| --- | --- | --- |
| Geometry | periodic, full volume | **not periodic** (rule 3 no longer applies), observer at origin, SDSS footprint only |
| Density tracer | every halo, n̄ known exactly | flux-limited groups: n̄_T falls with distance, so it must be modelled with randoms |
| Velocities | exact | σ_u grows roughly in proportion to cz (hundreds to thousands of km/s); non-Gaussian |
| Truth | known | unknown. Every check is internal consistency or a comparison with mocks |
| n̄ V_b normalization | `box_number_density` | **expected count from randoms in the masked shell** |

Nusser's setup, for reference (Nusser 2017, §4–6): CF3 with cz < 10,000 km/s (~4,600 objects),
weight w = 1, objects placed at **redshift** positions in the **LG frame**, 2MRS weighted by 1/φ(r),
bins 7 h⁻¹ Mpc wide (9 points to 63), 100 bootstrap draws of u for error bars, 30 mocks for cosmic
variance, fσ8 fitted only on the 7 points at r > 15.

**Same-size sample:** at cz < 10,000 there are **4,481 CF4 groups inside the Tempel footprint**
*(measured)*. That is almost exactly Nusser's ~4,600, so his error budget is a fair first guess
for yours.

---

## Stage 1: Read and audit the catalogs  → `notebooks/11_real_catalog_audit.ipynb`

**What this stage is for.** Get to know the raw data before trusting any of it. Load the three
EDD files exactly as they are and read what every column means. Confirm that the units, ranges and
counts are what you think they are. Find the traps (shared IDs, mislabeled units, columns that
mean something other than their name). Nothing physical is computed yet. The output is a loader
you trust and a notebook of sanity plots.

**Do.** Write a loader in `src/dvcorr/pipeline/real_catalogs.py`. EDD CSVs have **5 header lines**
(title, names, formats, units, descriptions): `header=1, skiprows=[2,3,4]`. Port it from
`Imports from old repo/data_loader.py::read_cf4_csv`, which already does this. Turn the
descriptions row into a column dictionary and read it. It is the documentation.

**Expect (measured):**

- CF4 velocities: 38,057 groups, no NaNs, `Vcmb` from −617 to 32,575 km/s. `CF4_Groups.csv` lists
  the same `pgc` values in the same row order, so the two files join row by row.
- Tempel is a **galaxy table with group membership**, in the words of the EDD description. It has
  305,952 rows, one per galaxy: `ID` and `pgc` are unique, and `zobs`, `zcmb`, `pE`… are
  per-galaxy values. Each row carries its group's `IDg`, `Ng`, `Rg` and `dc`. The group catalog is
  *derived* from it by collapsing rows (Stage 2). The footprint is ≈ 7,600 deg², which matches
  SDSS Legacy.
- **Trap 1: `IDg = 0` covers all 130,010 isolated galaxies.** If you collapse by `IDg` naively,
  they merge into one fake group of 130k members. A singleton's identity is its galaxy `ID`.
- **Trap 2:** 375 groups have no `Rg = 1` row, because the clipped table dropped the brightest
  member. 1.5% of groups have fewer rows than `Ng`. `Ng`'s description says "Tempel+14 group"
  while `IDg` says "Tempel+17", so `Ng` is a parent-catalog richness, not the number of rows here.
- **Trap 3, units lie:** the units row says RA is in `hour`, but values reach 359.998, so it is
  degrees. Always check the ranges, not the labels.
- **Trap 4: `gVcmb` is not a clean group velocity.** "Group velocity in CMB frame" differs between
  members of the same group in 19% of groups. `IDg = 1558` has 6,503, 7,578 and 6,441 km/s among
  its six brightest members. `dc` ("co-moving distance to group", H0 = 67.8) is constant within
  **every** group.
- CF4 is on its own H0 scale. `Vpds = fV − 74.6·D` holds exactly, and `Vcmb/D` ≈ 71.5. Distances
  `D` are in Mpc on H0 ≈ 74.6, so they are **never** used as positions.
  *Stage 1 correction:* the exact relation is `Vpds = (fV − 74.6·D)/(1 + 74.6·D/c)`; see notebook 11.

**Plots and the intuition to check them against:**

1. Mollweide sky maps of both catalogs. CF4 is all-sky with a Zone-of-Avoidance gap. Tempel
   should be essentially the contiguous northern SDSS Legacy area. If extra patches appear, find
   out why before going on.
2. n(cz) histograms. Tempel peaks around 15,000–25,000 and is cut at z = 0.1. CF4 has a nearby
   bump plus the SDSS FP population out to about 30,000.
3. `Vpec` against cz. The median should stay ≈ 0 at every depth *(measured: −13, −2, +16, +27,
   −64 km/s)*. A median that drifts linearly with cz means an H0 mismatch, since δv ≈ (δH0/H0)·cz.
4. **Open puzzle to settle here:** the naive noise σ_u = (ln10/5)·eDM·cz *exceeds* the total
   measured `Vpec` scatter by 2–2.5× at every depth *(measured: 1,466 predicted vs 670 observed at
   5–10k km/s)*. Noise cannot be larger than signal plus noise, so `eDM` is not the error on `Vpec`
   in the way the formula assumes. Possible causes: the group √N reduction, or how `Vpec` is
   defined (it equals `Vpwf` for 94% of rows and differs for 2,125 nearby ones). This must be
   resolved before Stage 8.
   *Stage 1 answer:* the published `Vpwf = fV/(1+fV/c)·log10(fV/(74.6 D))` uses log₁₀ where
   Watkins & Feldman use ln, so it (and `Vpec` beyond Vls = 3,000) is ln 10 = 2.3× smaller than v.
   With ln restored, std(v/σ_u) ≈ 0.91–0.98 beyond 5,000 km/s: eDM is a correct, slightly
   conservative error. Which velocity to correlate is a Stage 2 decision (notebook 11, §7).
5. 228 groups have `eDM = 0`, all within D < 110 Mpc. Under inverse-variance weights they would
   get infinite weight. It is harmless under Nusser's w = 1, but note it.

**Naming (rule 9):** done — the file was `SDSS_Temple.csv` and has been renamed, with the
`config/paths.py` default, to `SDSS_Tempel.csv`.

**Done when:** the loader returns the three tables with the counts above; every trap is reproduced
and explained in the notebook; and the σ_u puzzle has a written answer, or is explicitly parked
with a plan.

**Prompt to start it:**

```text
Start Stage 1 of docs/real_data_roadmap.md (read and audit the real catalogs).
Read CLAUDE.md and that stage first. Before writing code: confirm with me the rename
SDSS_Temple.csv -> SDSS_Tempel.csv (rule 9) and the module name for the loader.
Then: write the EDD loader in src/dvcorr/pipeline/real_catalogs.py (port read_cf4_csv
from "Imports from old repo/data_loader.py"), expose the column-description dictionary,
add tests on a tiny synthetic EDD-format CSV, and build notebooks/11_real_catalog_audit.ipynb
as a thin consumer: column dictionary, the four traps, the sky maps, the n(cz) histograms, Vpec vs cz,
and the eDM -> sigma_u vs observed-scatter comparison. Compare every number against the
roadmap's "Expect (measured)" list and tell me where they disagree. Investigate the sigma_u
puzzle (read the EDD column descriptions; reproduce Vpds and, if possible, Vpwf from DMzp, fV, D)
and report what you find. Explain each check so I can verify it myself.
```

### Stage 1 addendum [2026-10-14]: audit 2MASS and the CF4 sources → extends notebook 11

**What this adds.** The two new inputs get the same scrutiny as the first three files: the 2MASS
catalog and the CF4 individual-galaxy table. CF4 also gets broken down by distance source, so the
source split in Stage 8B starts from known counts and known quirks.

**Do.**

- **Get the data.** *Done for 2MRS (2026-10-14):* `data/2MRS_K<11.75.txt`, Nusser's catalog,
  loaded by `load_twomrs_galaxies` and audited in notebook 11 §9–10. *Still to do:* download
  the CF4 individual-galaxy table from EDD, add its path to `config/paths.py` and give it a
  loader in `real_catalogs.py`. Check every new filename for typos (rule 9).
- **CF4 by method, from the group table already in hand.** `CF4_Groups.csv` (the file that joins
  row by row with the velocities) has `Nc, Nsn, Nfp, Ntf, Nsbfo, Nsbfi` and per-method
  `DM_*`/`eDM_*`. Tabulate how many groups each method has and how many methods each group mixes.
- **CF4 individuals.** Read the column dictionary, find the source label (or per-source DM
  columns), and count galaxies per source. Work out how individuals map to groups (the group `pgc`
  is presumably the principal galaxy; check it). Then check that the group DM is the
  1/σ²-weighted mean of its members, as the group table's description says.

**Expect.**

- The group method counts measured 10-14 (see "The advisor's plan").
- 2MRS *(measured 10-14, notebook 11 §9)*: 43,526 galaxies (Huchra: 43,533), `K_tc` ≤ 11.75
  in every row, zero galaxies outside the |b| ≥ 5° (8° toward the bulge) mask, median
  heliocentric cz 8,401 km/s. Velocities are **heliocentric only**: Stage 3 converts them. At
  cz < 10,000 the SDSS cap holds 6,303 2MRS galaxies against 40,839 Tempel galaxies, and
  12,077 CF4 groups lie in the 2MRS mask against 4,481 in the SDSS footprint.
- 6dFGS FP: southern sky only (δ < 0). SDSS FP: inside the SDSS footprint.

**Plots and the intuition to check them against:**

1. A sky map of each CF4 source. SDSS FP should sit inside the Tempel footprint and 6dF FP in the
   south. TF and SNIa cover the whole sky.
2. n(cz) per source. FP reaches about 30,000. TF and SNIa should be concentrated at lower cz.
3. **Median `Vpec` against cz, per source.** This is the cheapest 6dF test there is. A zero-point
   offset δm in a source's distance moduli gives δv ≈ (ln10/5)·δm·cz, a median that drifts
   linearly with cz *for that source only*. (Use the ln-restored velocity from §7 of notebook 11.)
4. 2MASS sky map, n(cz), and its overlap with Tempel inside the SDSS footprint.

**Done when:** both new files load with their counts, every source has a sky map, an n(cz) and a
median-Vpec(cz) curve, and any source whose median drifts has been flagged.

**Prompt to start it:**

```text
Do the Stage 1 addendum of docs/real_data_roadmap.md (2MASS + CF4 sources audit), extending
notebooks/11_real_catalog_audit.ipynb. Read CLAUDE.md, docs/advisor_notes.md and that section first.
Before writing code: tell me which 2MASS and CF4-individual files you need, and confirm the
filenames and the 2MASS product with me. Then add loaders to src/dvcorr/pipeline/real_catalogs.py
(with tests on tiny synthetic files), tabulate the CF4 method breakdown from CF4_Groups.csv,
and add the per-source sky maps, n(cz), and median Vpec vs cz (ln-restored velocity) to notebook 11.
Re-execute the notebook (rule 11). Tell me which sources, if any, show a zero-point drift.
```

---

## Stage 2: Build the two analysis samples

**What this stage is for.** Turn the raw tables into the two lists the estimator actually uses.
The first is **one row per Tempel group**, with a single center: the density tracer. The second is
**one row per CF4 object** that passes the cuts: the velocity tracer. Every later stage reads only
these two samples, so a mistake here, such as counting a group several times or merging the
singletons, would silently corrupt everything downstream.

**Tempel groups:**

- Keep one row per group: real groups by `IDg > 0`, and singletons by `ID`. The angular center is
  the `Rg = 1` member, or the lowest `Rg` present (Trap 2). The radial position comes from `dc`,
  which is constant within each group (Trap 4), not from `gVcmb` or the member's `zcmb`.
- **Recommended primary sample:** all unique groups, singletons included. This is the closest
  analogue of Nusser's all-galaxy 2MRS field. Richness cuts (`Ng ≥ 2`, `Ng ≥ 6`) come later as
  splits.

**CF4 velocities:** keep cz < 10,000 km/s (Nusser's cut) and objects inside the Tempel footprint.
Drop missing values; never fill them (rule 5).

**Expect (measured):** 49,671 real groups + 130,010 singletons in total. At cz < 10,000 there are
6,367 groups with Ng ≥ 2, 15,278 singletons, and 4,481 CF4 objects inside the footprint.
*Stage 2 result (notebook 12):* 49,671 + 130,010 = 179,681 tracers and 4,481 CF4 groups reproduce.
The cz < 10,000 Tempel counts are **6,360 groups / 15,272 singletons**: the 6,367 / 15,278 above were
cut on `gVcmb`, which is not the Tempel group velocity (243 singletons differ from their own c·zcmb by
> 500 km/s). The group redshift is the mean member `zcmb`, which `dc` encodes. The velocity sample
is **4,480**: pgc 39346 (fV = 0, `Vpwf` a filler 0) has no defined u and is dropped (rule 5).
u = EDD's `Vpec` ramp with the natural log restored (decided 2026-10-18); primary richness = all
groups and singletons (decided 2026-10-15).

**[2026-10-14] 2MASS and the per-source samples.**

- **2MASS density sample (M):** one row per tracer. If the product is a group catalog, collapse it
  the same way as Tempel (watch for the same singleton trap). CF4 objects are then selected inside
  the **2MASS** mask, which is nearly all-sky, so far more CF4 centers survive than the 4,481 inside
  SDSS.
- **Velocity samples:** Phase A is the CF4 groups, as above. Phase B is the CF4 individuals, one
  sample per source, each tagged by its source and cut the same way (cz, mask). A galaxy measured
  by two sources appears in *each* source's sample, never twice in one sample. The duplicate test
  runs per sample.
- Every sample is a pair (density field, velocity sample), so the counts table becomes a matrix:
  {T, M} × {groups, each source}.

**Done when:** the note's duplicate test passes (one row per group and per CF4 object), the counts
match, and a sky plot of the two samples at cz < 10,000 shows CF4 centers sitting inside the
Tempel footprint. **[2026-10-14]** The same holds for M, and the counts matrix is filled in.

**Prompt to start it:**

```text
Start Stage 2 of docs/real_data_roadmap.md (build the two analysis samples).
Read CLAUDE.md and that stage first; Stage 1's loader is in
src/dvcorr/pipeline/real_catalogs.py. Add library functions that (a) collapse the Tempel
galaxy table to one row per group, keeping singletons (IDg == 0) as their own groups keyed by ID,
centering on Rg == 1 or the lowest Rg present, taking the radius from dc; and (b) select CF4
objects by a cz cut and a footprint membership test. Put the cut values in a config dataclass
(rule 4), not inline. Add tests: duplicate test, singletons kept separate, missing-Rg=1 fallback,
no velocity ever filled (rule 5). Extend notebook 11 (or start 12) with the counts table and a
sky plot of both samples. Check every count against the roadmap's "Expect (measured)" line and
explain any difference. Ask me before choosing the primary richness cut.
```

---

## Stage 3: Put both catalogs in the same redshift-space coordinate system

**What this stage is for.** Check, and make sure, that Tempel positions and CF4 positions are
**the same kind of coordinate**. That means the same frame (CMB), the same redshift-to-distance
formula, the same units (h⁻¹ Mpc), and the same observer at the origin. The pair separation
r = s_V − s_T only means something if both ends of it live in one coordinate system. A mismatch
here (Mpc against h⁻¹ Mpc, or χ(z) against cz/100) shifts every separation and produces a fake
dipole.

- Use **one** redshift–distance convention for both catalogs (note §4.1). Tempel's `dc` is already
  a cosmological comoving distance χ(z_cmb) in Mpc with H0 = 67.8, so the matching choice is the
  note's eq. 14. Tempel: s_T = `dc` × 0.678 h⁻¹ Mpc. CF4: s_V = χ(`Vcmb`/c) in the same cosmology.
  Get n̂ from (RA, DE) and keep the observer at the origin.
- *Check 1:* rebuild `dc` from each group's mean `zcmb` with your χ(z). It should agree within the
  group's velocity spread. A fixed fractional offset means your cosmology differs from Tempel's.
- *Check 2:* 25,439 Tempel galaxies are also CF4 `pgc` entries *(measured)*. For those matches,
  |s_T − s_V| should be a few h⁻¹ Mpc at most. An offset that grows with cz means the two
  catalogs use different conventions.
- **Decision for you and Adi (rule 1):** `conventions.REFERENCE_FRAME = "cmb"`, and the
  methodological note says CMB. Nusser deliberately used **LG**-frame redshifts (§4.3).
  Recommendation: CMB as primary, with LG as a Stage 12 robustness run. Changing the primary frame
  means editing `conventions.py`, so discuss it before anyone touches it.

- **[2026-10-14] 2MASS positions:** build s_M with the *same* χ(z) and frame. Whatever 2MASS
  ships (heliocentric cz, a CMB cz, or a distance), convert it explicitly. Never assume it matches
  Tempel. *Check 3:* cross-match 2MASS with Tempel inside the SDSS footprint and with CF4 by
  `pgc`. |s_M − s_T| and |s_M − s_V| for the matches should show no trend with cz.

**Done when:** both checks pass, the radius histograms of the two samples lie on the same
h⁻¹ Mpc axis, and the frame decision is written down. **[2026-10-14]** Check 3 passes as well.

**Prompt to start it:**

```text
Start Stage 3 of docs/real_data_roadmap.md (put both catalogs in the same redshift-space
coordinate system). Read CLAUDE.md and that stage first. Add a library function that builds
s_T and s_V in h^-1 Mpc, CMB frame, observer at the origin, using one comoving-distance
convention chi(z) for both catalogs (Tempel from dc, CF4 from Vcmb). Put the cosmology used
for chi(z) in src/dvcorr/config/cosmology.py (also add H0_CF4 = 74.6 there, as its docstring
asks). Run both checks from the roadmap: rebuild dc from member zcmb, and |s_T - s_V| for the
pgc identity matches as a function of cz. Show me the plots and say plainly whether the two
catalogs are in the same coordinate system. Do NOT change conventions.REFERENCE_FRAME; if you
think the frame should change, stop and ask.
```

---

## Stage 4: Selection function and randoms  → `src/dvcorr/selection/` (currently empty)

**What this stage is for.** Describe **where Tempel could have seen a group**, as opposed to where
groups actually are. The sky footprint gives the angular part, and the flux limit makes groups
rarer with distance, which gives the radial part. Both are encoded in a random catalog: points
with the same sky coverage and the same n(cz), but no clustering. Later, "density" means data
relative to randoms. Without randoms, the survey edges and the fall-off in distance would look
like a physical signal.

Recommended approach: use randoms rather than Nusser's 1/φ weights, because the angular mask
matters here and 2MRS was nearly full-sky.

- **Angular:** a HEALPix footprint built from Tempel galaxies (nside 64; check that the area is
  ≈ 7,600 deg²).
- **Radial:** draw random radii from the smoothed Tempel group n(s) ("shuffle" method), with 20–50×
  as many randoms as data.
- **Plots and what to look for:** data/random n(s) ratio flat at 1; data/random angular ratio map
  flat apart from Poisson noise; and n̄_T(s) in groups per (h⁻¹ Mpc)³. That last curve falls
  steeply because of the flux limit. If it does not fall, the sample is not what you think it is.
- Nusser's velocity-centered ζ only needs the **density** tracer's selection *for its
  normalization*. CF4 objects are the centers, and their selection only decides which centers
  exist. The mocks model that.
- **[2026-10-14] 2MASS randoms:** the same machinery applied to M. Its angular mask is the sky
  minus the Zone of Avoidance, and its radial selection is its own n(s), which is shallower than
  Tempel's.

### [2026-10-14] Velocity-tracer selection and Malmquist bias

The advisor wants the velocities corrected for Malmquist bias with a selection function. That is a
second, separate use of selection: it does not normalize ζ, it **removes a bias from u itself**.

- **The mechanism.** A distance indicator scatters log-distance by σ_d (roughly 0.2 dex for FP and
  TF). The objects that scatter one way are not balanced by those that scatter the other way, for
  two reasons. One is that the survey's selection (flux, size, depth) cuts off one side. The other
  is that the number of objects along the line of sight changes with distance (r² n(r), and the
  local density gradients). So at a fixed cz, the measured u is biased. Because the bias depends
  on the selection, **each CF4 source has its own**. An uncorrected bias that differs between
  sources would make the Stage 8B comparison fail for reasons that have nothing to do with
  cosmology.
- **Do.** Build a selection function S_k(cz, n̂) for each velocity sample k (the CF4 groups, and
  each source in Phase B) from its n(cz) and its sky coverage. Then implement one correction scheme,
  chosen with the advisor (see the open questions in the advisor notes). One option is a per-object
  correction from S_k, homogeneous or inhomogeneous with the T/M density field. The other is a
  forward model in the Stage 9 mocks.
- **Validate on MDPL2 first.** Take a carved sphere, give the halos log-normal distance errors at
  each source's σ_d, apply S_k, and run ζ₁ on the true u, the biased u and the corrected u. The
  corrected curve must return to the true one within noise, and the biased one shows the size of
  the effect.

**Done when:** both ratio plots are flat, and the random catalog is cached (rule 7) and reproducible
from a seed. **[2026-10-14]** M has its randoms as well, every velocity sample has an S_k, and the
MDPL2 Malmquist test recovers the true ζ₁.

**Prompt to start it:**

```text
Start Stage 4 of docs/real_data_roadmap.md (selection function and randoms).
Read CLAUDE.md, docs/architecture.md (the selection/ entry) and that stage first.
Implement in src/dvcorr/selection/: a HEALPix angular footprint from the Tempel galaxies, the
radial selection n(s) of the unique-group sample, and a seeded random catalog (shuffle method,
size factor in config). Give the random draw its own seed and extend the seed-allocation test.
Cache the randoms (rule 7). Notebook: footprint map and area, data/random n(s) ratio,
data/random angular ratio map, and nbar_T(s) vs distance. Tell me what each plot should look
like if the randoms are right, and whether ours do. Update the architecture.md index line.
Then do the same for the 2MASS density field. Finally, for the Malmquist part: read
docs/advisor_notes.md, propose the per-sample selection functions S_k and the correction
scheme, and wait for my choice. Implement it and run the MDPL2 Malmquist test (true vs biased
vs corrected zeta_1) before any correction touches real velocities.
```

---

## Stage 5: Move the estimator from the box to the sky  → `notebooks/13_sky_estimator_validation.ipynb`

**What this stage is for.** Make Nusser's estimator work on a **partial, non-uniform sky**, and
prove it gives the right answer **before** it touches real data. In the box, the expected number
of neighbors in a shell was n̄V_b. On the sky, the shell is cut by the footprint and thinned with
distance, so the randoms supply that expectation. The proof uses MDPL2: put a mask on the box, run
the new estimator, and check that it reproduces the full-box answer you already trust.

The masked version of Nusser's eq. 24:

  ζ^V_b = (1/N_v) Σ_α u_α · [ Σ_{β∈b} μ_αβ − α_R Σ_{q∈b} μ_αq ] / [ α_R Σ_{q∈b} 1 ],
  where α_R = N_data / N_random.

The random term removes the false dipole that a lopsided window creates, and the denominator is
the *expected* count under the selection. In the full-sky, uniform-selection limit this reduces
exactly to eq. 24. ζ^M uses the realized count in the denominator, plus the same random
μ-correction. Reuse `estimators/shell_dipole.py` (the per-center shell sums, `real_y10`, and the
r = 0 exclusion) by calling it on the data tracers and on the randoms. Do not fork it.

**Bridge test (the most important check in the roadmap):** carve MDPL2 around an observer, apply
the SDSS mask and a Tempel-like n(s) to the halos, and run the sky estimator with randoms. It must
reproduce the full-box `velocity_centered_shell_dipole` to within noise. Then turn the random
correction off and watch it fail. That failure shows you what the window does. Run the
group-centered ξ_Tu alongside; it must come out as ≈ −ζ.

**Sign gates on the sky:** a toy infall around real Tempel positions, observed at real CF4
positions, must give ζ₁ > 0 and ξ_Tu,1 < 0. A velocity shuffle must give a dipole consistent with
zero.

**Done when:** the bridge test and both sky sign gates pass as `pytest` tests, and the notebook shows
the curves with and without the random correction.

**Prompt to start it:**

```text
Start Stage 5 of docs/real_data_roadmap.md (move the estimator from the box to the sky).
Read CLAUDE.md, the docstrings of src/dvcorr/estimators/shell_dipole.py and
src/dvcorr/pipeline/velocity_centered.py, and that stage first. Before coding, write out the
masked-sky normalization you will implement and how it reduces to Nusser eq. 24 in the
full-sky limit, and wait for my OK. Then implement it by reusing the existing per-center shell
sums (data pass + random pass), never by editing signs or forking the estimator. Add tests:
the sky sign gates (zeta_1 > 0, xi_Tu,1 < 0 on a toy infall at real Tempel/CF4 positions),
the velocity-shuffle null, and the MDPL2 bridge test (masked sky estimator vs full-box result).
Notebook 13 shows: full box vs masked with randoms vs masked without randoms, dipole and
monopole as separate figures (rule 6), default binning (rule 10). Ask me before any long
full-catalog run. Use the reviewer agent on the finished diff.
```

---

## Stage 6: First measurement on real data  → `notebooks/14_first_data_dipole.ipynb`

**What this stage is for.** This is the first look at the real signal: Nusser's Fig. 2 top panel,
built from Tempel × CF4. The error bars and mock bands come later. The questions here are only
whether the curve has the right sign, the right shape and roughly the right size, and whether the
monopole looks healthy.

- ζ^V and ζ^M, each with its monopole, as separate figures (rule 6).
- **Binning (rule 10):** run the default √2 log ladder to 64 *and* Nusser's linear 7 h⁻¹ Mpc bins
  (0–63, 9 points) for the direct overlay. The Nusser binning departs from the default, so its
  edges are stated and confirmed before the run.
- **What right looks like:** a positive dipole in Nusser's orientation, rising to a peak at
  ~10–20 h⁻¹ Mpc and then declining. The y-axis of Nusser's Fig. 2 spans 0–40 km/s. Your
  amplitude should scale roughly as b_T / b_2MRS, with b_2MRS = 1.12 (Nusser §5). Overlay
  `dvcorr.theory.xi_udelta_dipole` with Planck f and σ8. At r ≳ 20 the data should follow it,
  with the redshift-space boost b → b + 3f/5.
- **What wrong looks like:** a negative dipole (orientation bug; go back to the gates); a dipole
  that grows with r (the window is not corrected); a dipole of hundreds of km/s (normalization,
  usually a missing α_R).
- The flow-signed monopole sits near ⟨|u|⟩, which is noise-dominated here, at hundreds of km/s.
  That is expected. The signed leakage from `per_center_u` should stay small. Split it by R to
  see the finite-distance term (note §7.2).
- **[2026-10-14] Both density fields, CF4 groups only (Phase A).** Run T × CF4 groups and
  M × CF4 groups. **M × groups is Nusser's measurement with CF4 in place of CF3.** On his
  binning it should look like his Fig. 2 top panel, with b_2MRS = 1.12 and no rescaling needed.
  The two fields have different biases, so compare T and M as ζ₁/b, not raw ζ₁. Apply the
  Malmquist correction from Stage 4 to the group velocities and show the curve with and without
  it.

**Done when:** both binnings are plotted, the sign and shape are right, and the amplitude
relative to theory has been written down. **[2026-10-14]** This holds for both T and M.

**Prompt to start it:**

```text
Start Stage 6 of docs/real_data_roadmap.md (first measurement on real data).
Read CLAUDE.md, .claude/rules/plotting.md and that stage first. Using only library functions
from Stages 1-5, build notebooks/14_first_data_dipole.ipynb: zeta^V and zeta^M with
monopoles (separate figures), on the default binning, for CF4 groups against both the
Tempel (T) and 2MASS (M) density fields, with and without the Malmquist correction. Then state the edges of Nusser's
linear 7 h^-1 Mpc binning and wait for my yes before running it. Overlay the linear-theory
dipole from dvcorr.theory (state the b, f, sigma8 you use). Go through the roadmap's
"what right / what wrong looks like" list one item at a time and tell me which
holds. No error bars yet.
```

---

## Stage 7: Bias of the density tracer

**What this stage is for.** Measure how strongly Tempel groups trace the matter (b_T, or σ8g as
Nusser writes it). The dipole amplitude is proportional to b_T·fσ8, so b_T has to be known to turn
the measured amplitude into a growth rate. It also tells the mocks which halos stand in for
Tempel groups.

Nusser measured σ8g from counts in 8 h⁻¹ Mpc spheres (§5). Do the same inside the footprint with a
volume-limited cut, or measure ξ_TT with randoms (Landy–Szalay) and compare it to ξ_mm.
`pipeline/galaxy_bias.py` holds the box version. *Check:* b_T rises with an `Ng` cut. If it does
not, richness is not tracking mass.

**Done when:** b_T (or σ8g) is known with an error bar for the primary sample and for one richness
split.

**Prompt to start it:**

```text
Start Stage 7 of docs/real_data_roadmap.md (bias of the Tempel group tracer).
Read CLAUDE.md, the docstring of src/dvcorr/pipeline/galaxy_bias.py and that stage first.
Propose how to measure sigma_8g / b_T on the real (non-periodic, masked) sample, either
counts-in-spheres as Nusser section 5 does or xi_TT with the Stage 4 randoms, and wait for my choice.
Then implement it as library code, check it first on a masked MDPL2 sample where b is known
from notebook 09, and only then run it on Tempel. Report b_T for the primary sample and for an
Ng cut, and say whether the trend makes sense.
```

---

## Stage 8: Observational error bars (Nusser's bootstrap)

**What this stage is for.** Put error bars on the Stage 6 points that reflect only the
**measurement noise** of CF4 velocities, the way Nusser did. Perturb every u by its own error many
times and see how much the curve moves. This is one part of the total error. Cosmic variance comes
from the mocks in Stage 9.

Draw 100 realizations of u_α + N(0, σ_u,α), using the σ_u that Stage 1 settled. The error bar is
the RMS scatter of the realizations. *Expect* these bars to be **smaller** than the mock
cosmic-variance band (Nusser §6.1). If they dominate, the σ_u model is wrong or the cz cut is too
deep.

**Done when:** Stage 6's figure carries bootstrap error bars and their covariance is saved.

**Prompt to start it:**

```text
Start Stage 8 of docs/real_data_roadmap.md (observational bootstrap error bars).
Read CLAUDE.md and that stage first, and remind me what Stage 1 concluded about sigma_u. Implement
the bootstrap as a recombination where possible (see the "null is R realizations" contract
in docs/architecture.md), seeded with its own seed. Add the error bars to the notebook 14
figures and save the bootstrap covariance. Tell me how big the bars are relative to the signal
per bin, and whether that matches Nusser's statement that velocity errors are subdominant.
```

---

## Stage 8B [2026-10-14]: CF4 individuals, source by source (Phase B)  → `notebooks/15_cf4_sources.ipynb`

**What this stage is for.** This is the advisor's consistency test. Phase A has a trusted
group-velocity result with error bars. Now take apart what goes into it: rerun ζ₁ on the CF4
**individual** galaxies, one distance source at a time, and check that every source tells the same
story. A source that disagrees has a zero point, a selection or an error model that cannot be
trusted. The advisor's suspect is 6dFGS.

**Do.**

- For each source k (SDSS FP, 6dFGS FP, TF, SNIa, SBF, and any others the Stage 1 addendum finds),
  run ζ₁ with its own Malmquist correction (S_k from Stage 4) and its own bootstrap (Stage 8).
  Sources with too few objects at cz < 10,000 (SBF, SNII) are reported as counts, not curves.
- **Pair each source with a density field that covers it.** SDSS FP goes with T and M. 6dFGS FP is
  southern, so it goes with **M only**. TF and SNIa (all-sky) go with M, and with T inside the
  footprint.
- Two comparisons, each shown as a ratio with its bootstrap band:
  1. **Individuals against groups:** all individuals combined against Phase A on the same density
     field. Grouping should change the noise, not the mean.
  2. **Source against source on the same density field**, and where two sources overlap on the
     sky, on the **same region**. That separates "6dF differs" from "the southern sky differs"
     (cosmic variance).

**What a problematic source looks like:**

- A zero-point offset shows up first in the Stage 1 addendum's median-Vpec(cz) drift. In ζ it
  appears as a signed monopole (`per_center_u`) that grows with R, and a dipole biased in
  proportion to the drift.
- A wrong error model shows up as a source whose bootstrap scatter is far below the
  source-to-source spread.
- A Malmquist bias left uncorrected shows up as a difference that grows with cz and shrinks when
  the cz cut is lowered.

**Done when:** every source with enough objects has a ζ₁ curve with bootstrap bars, both
comparisons are plotted, and each source (6dF above all) is labelled *consistent*, *inconsistent,
explained* or *inconsistent, dropped*. That verdict is written into the advisor notes. The final
significance of any disagreement uses the per-source mocks of Stage 9.

**Prompt to start it:**

```text
Start Stage 8B of docs/real_data_roadmap.md (CF4 individuals, source by source).
Read CLAUDE.md, docs/advisor_notes.md and that stage first, and remind me what the Stage 1
addendum found per source (counts, sky coverage, median-Vpec drift). Using only library
functions, build notebooks/15_cf4_sources.ipynb: zeta_1 per source with its own Malmquist
correction and bootstrap, each paired with a density field that covers it (6dF with 2MASS only).
Plot individuals vs groups and source vs source as ratios with bands, comparing on the same sky
region where sources overlap. Tell me plainly whether 6dF agrees with the others, and how sure we
can be before the Stage 9 mocks.
```

---

## Stage 9: Mock observers in MDPL2  → `src/dvcorr/mocks/` (currently empty)

**What this stage is for.** Build fake versions of the Tempel × CF4 survey inside MDPL2, with the
same sky, the same depth, the same tracer density and bias, and the same velocity errors.
Nusser's estimator then runs on each one. The spread across mocks gives **cosmic variance plus
shot noise**, which is the main error bar. The mock mean gives the ΛCDM expectation that the data
is compared against.

- Place observers in MDPL2. With R ≈ 165 h⁻¹ Mpc (100 plus the 63-bin reach), (1000/330)³ ≈ **27
  independent spheres**, about Nusser's 30.
- In each sphere: convert to redshift space (`dvcorr.redshift_space`), apply the SDSS mask,
  subsample halos above a mass threshold to match Tempel's n̄_T(s), with the threshold chosen so
  that the mock bias matches b_T from Stage 7. Then draw CF4-like velocity objects matching CF4's
  n(s) inside the footprint (port `Imports from old repo/masks.py` `MaskMaker`), and add velocity
  errors from the Stage 1 model.
- *Checks:* mock n(s) and sky maps lie on top of the data; the mock-mean ζ matches theory at large
  r; the mock spread gives the shaded band of Nusser's Fig. 2.
- Nusser's Fig. 4 (the v–δ dipole in mocks) is the existing `velocity_frame_comparison` pipeline
  run on these mocks.
- **[2026-10-14]** Each mock also gets an **M-like** density tracer (2MASS mask and n(s), matched to
  b_M) and a **CF4-like velocity sample per source**: each source's sky, n(cz), σ_d and
  selection S_k, with the Malmquist correction applied exactly as on the data. The mock spread
  per source is what makes a Stage 8B disagreement significant or not.

**Done when:** you have Nusser's Fig. 2 top panel: data points with bootstrap bars, mock mean in
redshift and real space, and the mock 1σ band.

**Prompt to start it:**

```text
Start Stage 9 of docs/real_data_roadmap.md (MDPL2 mock observers).
Read CLAUDE.md, docs/architecture.md and that stage first, plus the docstrings of
src/dvcorr/redshift_space.py and "Imports from old repo/masks.py". First propose the mock
recipe (observer placement and independence, how tracers are matched to nbar_T(s) and b_T, CF4-like
velocity sampling, velocity errors) with the numbers you would use, and wait for my OK.
Implement in src/dvcorr/mocks/ with its own seeds. Before running all mocks, show me one
mock's sky map and n(s) against the data. Ask before any long run. Final product: Nusser
Fig. 2 top panel (data + bootstrap bars, mock mean in z- and real space, mock 1-sigma band),
and the Fig. 4 analogue via velocity_frame_comparison.
```

---

## Stage 10: Significance (Nusser eq. 34)

**What this stage is for.** Answer two yes/no questions with numbers. Is the measured dipole
**different from zero**? Is it **consistent with ΛCDM**? Both are a χ² with the full covariance
(mocks plus bootstrap).

Total covariance = mock covariance + bootstrap covariance. Compute χ² against zero (Nusser: 38.7
and 34.9 for 9 points, 4.35σ and 4σ) and χ² against the mock mean. Covariance from 27 mocks with 9
bins is noisy, so apply the Hartlap correction and state it.

**Done when:** both χ² values and their significances are reported for ζ^V and ζ^M.

**Prompt to start it:**

```text
Start Stage 10 of docs/real_data_roadmap.md (significance, Nusser eq. 34).
Read CLAUDE.md and that stage first. Build the total covariance (mocks + bootstrap) with the
Hartlap correction, compute chi^2 against zero and against the mock mean for zeta^V and zeta^M
on Nusser's binning, and convert both to sigma. Show the correlation matrix. Put the numbers next
to Nusser's (38.7 / 34.9; 14.97 / 11.13) and explain what any difference means.
```

---

## Stage 11: Growth rate (Nusser eqs. 35–36, Fig. 3)

**What this stage is for.** Turn the dipole amplitude into a cosmological number, fσ8 (and
β = f/b). Fit the measured curve with templates from the mocks, which scale as fσ8·σ8g plus a
small (fσ8)² term. The fit uses only the large-scale points, where linear theory holds. First,
test that the fit recovers the known answer on the mocks themselves.

Model: ζ = fσ8·σ8g·ξ¹_uδ + (fσ8)²·ξ¹_uw, with templates from the mocks. Minimize
−2lnP = χ² + ln det C(fσ8), using only r > 15. Then β = fσ8/σ8g. *Sanity:* MDPL2's own fσ8 ≈ 0.43,
Planck ≈ 0.43, and Nusser found 0.40 ± 0.08 (ζ^V).

**Done when:** the fit recovers fσ8 ≈ 0.43 on the mocks, and the data curve gives fσ8 and β with
1σ errors. That is Nusser's Fig. 3.

**Prompt to start it:**

```text
Start Stage 11 of docs/real_data_roadmap.md (growth rate, Nusser eqs. 35-36 and Fig. 3).
Read CLAUDE.md, the relevant pages of literature/1703.05324v2 (grep the .txt for "(35)"),
and that stage first. Implement the -2lnP fit with an fsigma8-dependent covariance,
r > 15 only. First run it on each mock and show me the distribution of recovered fsigma8
(it must centre on MDPL2's value); only then fit the data. Produce the Fig. 3 analogue
and beta = fsigma8/sigma8g with the Stage 7 sigma8g.
```

---

## Stage 12: Robustness and null suite (methodological note §6.2)

**What this stage is for.** Try to break the result. Change every choice that could reasonably
have been made differently, and check that the answer moves no more than the error bars allow.
Then run the tests that must come out as zero. A result that survives this is a result.

Velocity shuffle; duplicate test; **overlap test** (remove the 25,439 identity matches, where the
velocity object sits inside its own density field at y ≈ 0); velocity definition (`Vpwf`, and
`Vpds`, which should degrade badly at large cz); center choice; richness cuts; czmax of 8k, 10k
and 15k; CMB against LG frame; and a north/south split. Report each one as a ratio against the
primary, with the bootstrap band.

**[2026-10-14]** Also: T against M as the density field; groups against individuals; each CF4
source (Stage 8B, now with the mock bands); and the Malmquist correction on against off.

**Done when:** a table of all variants exists, with each one either consistent with the primary
or explained.

**Prompt to start it:**

```text
Start Stage 12 of docs/real_data_roadmap.md (robustness and null suite).
Read CLAUDE.md, the docstring of src/dvcorr/pipeline/robustness.py and that stage first.
Run each variant in the roadmap's list through the same library pipeline (no forks), each
with its own seed where random. Produce one summary table and one ratio-to-primary figure
with the bootstrap and mock bands. For every variant that moves the result by more than 1 sigma,
propose a physical explanation before I accept it. Ask me before the LG-frame run, because it
touches the conventions.
```

---

## Stage 13 (stretch): Nusser §7, Wiener-filter reconstruction

**What this stage is for.** Nusser's second half: reconstruct the full 3D velocity field from
the radial velocities with a Wiener filter, and repeat the dipole with 3D velocities (Fig. 2
bottom, Figs. 5–7). This is a project in its own right. Decide with Adi whether it is part of
"full reproduction."

**Prompt to start it:**

```text
Start Stage 13 of docs/real_data_roadmap.md (Wiener-filter reconstruction, Nusser section 7).
Read CLAUDE.md and pages 7-9 of literature/1703.05324v2 (grep the .txt first). Plan only: lay
out what a Wiener-filter v reconstruction needs here (the data-data and data-signal matrices from
the Gorski functions in dvcorr.theory, the CF4 error matrix, the grid) and what it would take,
and wait for my decision before writing code.
```

---

## Nusser figure → our product

| Nusser | Ours | Status |
| --- | --- | --- |
| Fig. 1 theory | `intuition/nusser_dipole_real_and_redshift_space.ipynb` | done |
| Fig. 2 top (data + mocks) | Stages 6, 8, 9 (M × CF4 groups is the like-for-like version) | to do |
| — (advisor, 2026-10-14) CF4 source consistency | Stage 8B | to do |
| χ² null / ΛCDM | Stage 10 | to do |
| Fig. 3 fσ8, β | Stage 11 | to do |
| Fig. 4 v–δ in mocks | `velocity_frame_comparison` on Stage 9 mocks | machinery done |
| Fig. 2 bottom, 5–7 (WF) | Stage 13 | stretch |

## Files touched over the full roadmap (as each stage is started)

- New: `src/dvcorr/pipeline/real_catalogs.py` (loaders, unique groups, positions);
  `src/dvcorr/selection/` (footprint, n(s), randoms); the masked-sky normalization next to
  `estimators/shell_dipole.py`; `src/dvcorr/mocks/` (observers, mock selection); and the notebooks
  numbered 11 onward.
- **[2026-10-14]** New data: a 2MASS catalog and the CF4 individual-galaxy table (EDD), with
  loaders in `real_catalogs.py`. The selection functions S_k and the Malmquist correction go in
  `src/dvcorr/selection/`, and per-source mock selection goes in `src/dvcorr/mocks/`.
- Config: `config/cosmology.py` gains `H0_CF4`; `config/paths.py` gets the Tempel filename fix
  **[2026-10-14]** and the 2MASS and CF4-individual paths.
- Docs: a one-line entry in `docs/architecture.md` per new file. The "n̄ is the box mean"
  contract gains its sky counterpart (n̄V_b → α_R × random count).

## Verification (per stage)

- Each stage's **Done when** line is its acceptance test.
- `pytest` passes, including the sky sign gates and the bridge test from Stage 5, before any data
  ζ is plotted.
- Stage 11's mock-recovery test (fσ8 ≈ 0.43 on mocks) passes before the data fit is quoted.
- **[2026-10-14]** The MDPL2 Malmquist test (Stage 4) passes before any corrected real velocity is
  plotted.
- Every edited notebook is re-executed headlessly (rule 11).
