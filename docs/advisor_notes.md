# Advisor notes

A dated log of what was agreed in advisor meetings. Each entry records what was said, how it was
read, the open questions, and what changed in the plan as a result. The roadmap
([real_data_roadmap.md](real_data_roadmap.md)) is the plan itself. This file explains why it says
what it says.

---

## 2026-10-14

### What was said

1. **Add a 2MASS density field alongside SDSS.** Run the correlation with a 2MASS catalog as the
   density tracer, as well as with the Tempel SDSS groups.
2. **CF4 groups first, then individuals.** Do the correlation calculation with the CF4 *group*
   velocities first. After that, repeat it with the CF4 *individual* galaxies.
3. **Split the individuals by source.** Filter the individual galaxies by each distance source
   that CF4 combines, run each one on its own, and check that the results agree. Pay special
   attention to **6dFGS**, which may be the problematic one.
4. **Selection function for Malmquist bias.** Use selection functions to correct the velocities
   for Malmquist bias.

### How it is read (to confirm)

- "MASS2" in the meeting notes is taken to mean **2MASS**. Which 2MASS product is still open:
  - the 2MASS Redshift Survey galaxies (2MRS, Huchra et al. 2012). This was Nusser's (2017)
    density field, so 2MRS × CF4 groups is his setup with CF3 swapped for CF4.
  - a 2MRS group catalog, which would match the Tempel *groups* more closely.
  - Tully (2015) 2MASS K < 11.75 groups. CF4's group `Vcmb` is already averaged over these
    groups (see the `CF4_Groups.csv` column description).
- Why 2MASS and 6dF probably go together: 6dFGS observed only the southern sky, and the Tempel
  SDSS footprint is mostly northern. So 6dF peculiar velocities have almost no SDSS density field
  around them. A near all-sky 2MASS field is what makes a 6dF-only run possible.
- **Measured from `CF4_Groups.csv`:** the group table records which methods each group uses
  (`Nc`, `Nsn`, `Nfp`, `Ntf`, `Nsbfo`, `Nsbfi`, `DM_SNII`). 36,715 of the 38,057 groups (96%)
  use one method only (36,693 when SNII is counted as a method too). However, the FP column
  combines SDSS FP and 6dFGS FP. Telling those two
  apart needs the individual-galaxy table, or a sky split as a rough stand-in. At
  cz < 10,000 km/s, 1,845 FP groups are at δ < 0 and 1,351 at δ ≥ 0.

### Open questions

- ~~Which 2MASS catalog?~~ **Settled the same day:** the 2MRS K_s ≤ 11.75 galaxies Nusser
  used (`data/2MRS_K<11.75.txt`, 43,526 galaxies, audited in notebook 11 §9).
- Is 2MASS a **second, separate density field** compared with SDSS, or should the two be
  **combined** into one? The working assumption is separate.
- Which Malmquist correction: a per-object correction from each source's selection function,
  or modelling the bias in the mocks and fitting it forward? Homogeneous only, or inhomogeneous
  as well, using the density field?
- Where the CF4 individual-galaxy table comes from (EDD) and which source labels it carries.

### What changed in the roadmap

- Context: there are now two density fields (Tempel SDSS, 2MASS) and two velocity phases (A: CF4
  groups, B: CF4 individuals split by source).
- Stage 1 addendum: audit 2MASS and the CF4 individuals, and break CF4 down by method. This goes
  into notebook 11.
- Stages 2–4: build 2MASS samples, coordinates and randoms. Stage 4 now also covers the CF4
  per-source selection functions and the Malmquist correction, checked on MDPL2.
- Stage 6: both density fields with CF4 groups.
- New Stage 8B: CF4 individuals, source by source, with 6dF as the focus.
- Stages 9 and 12: the mocks and the robustness table take in the new axes.

### Follow-up, same day: what the 2MRS audit and the CF4 method split showed (notebook 11 §9–12)

- 2MRS matches its survey definition exactly (magnitude limit, latitude mask). Its heliocentric
  velocities agree with Tempel's galaxy by galaxy (median −0.4 km/s, 1.5 km/s scatter over 8,848
  `pgc` matches).
- With 2MRS as the density field, CF4 has **2.7× more velocity centers** at cz < 10,000
  (12,077 against 4,481 inside SDSS), including the whole 6dFGS sky.
- First look at 6dF: single-method FP groups in the south outside SDSS (a stand-in for 6dFGS)
  have median velocities about 400 km/s above the SDSS-like FP groups. The difference is
  **constant in cz, not a zero-point slope** (implied modulus offset −0.034 ± 0.027 mag). That
  fits a bulk flow seen from opposite sides of the sky as well as it fits a constant calibration
  offset. Medians cannot separate the two. Stage 8B's per-source ζ₁ can.

