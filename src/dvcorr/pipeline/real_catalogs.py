"""
real_catalogs.py
----------------
The observed catalogs -- Cosmicflows-4 groups, the Tempel et al. (2017)
SDSS galaxy table and the 2MRS K_s <= 11.75 galaxies (Huchra et al. 2012,
Nusser 2017's density field) -- read exactly as the Extragalactic Distance
Database (EDD) publishes them, plus the formulas EDD used to build CF4's
velocity columns and the 2MRS sky mask.

Stage 1 of `docs/real_data_roadmap.md`: the loaders return the tables raw,
every row and every column, so that the audit
(`dvcorr.pipeline.real_catalog_audit`, `notebooks/11_real_catalog_audit.ipynb`)
sees the files as they are. Stage 2 builds the two analysis samples from them
(see below; `notebooks/12_analysis_samples.ipynb`). Redshift-space positions
(Stage 3) are built from these samples in `dvcorr.pipeline.sky_positions`.

The EDD CSV format
------------------
Every EDD download carries FIVE header lines before the data:

    0  title         HYPERLEDA,"CF4 All Groups","CF4 All Groups",...
    1  names         pgc,DM_zp,DM_av,...
    2  formats       %7s,%6.3f,%6.3f,...      (C/FORTRAN printf formats)
    3  units         ---,mag,mag,...
    4  descriptions  "The Catalogue of Principal Galaxies (PGC) Number",...

`read_edd_csv` keeps line 1 as the column names, turns lines 2-4 into the
column dictionary (`EddCatalog.columns`), and refuses a file whose format row
does not look like printf formats -- unlike the old repo's `read_cf4_csv`, it
never falls back to a plain `read_csv`, because a silently mis-parsed header
row becomes a data row of strings and every numeric column turns to `object`.

The units row is documentation, not truth: `CF4_Groups.csv` labels RA in
`hour` while its values run to 359.998 (degrees), and Tempel labels the
normalized densities `den3`/`den10` as `Mpc^-3`. Check ranges, not labels
(`real_catalog_audit.column_range_table`).

The format row is more useful than it looks: it states the precision each
value was ROUNDED to (`%6.3f` -> 0.0005, `%5d` -> 0.5), which is exactly the
tolerance a reconstruction of one column from others must be held to
(`format_half_ulp`).

CF4's velocity columns, reverse-engineered
------------------------------------------
`CF4_Groups_Velocities.csv` carries the inputs (`Vcmb`, `Vls`, `DMzp`) and
four derived velocities. Each was reproduced from the inputs to within the
file's own rounding -- `D`, `fV`, `Vpec` in every row; `Vpds`, `Vpwf` in all
but ~0.3% of rows, which miss by ~1 km/s apart from five single rows (the
audit notebook shows the residuals) -- with
c = `conventions.SPEED_OF_LIGHT`, H0 = `CosmologyConfig.H0_CF4` = 74.6, and
D = 10^((DMzp - 25)/5) Mpc:

- `fV` = Vcmb [1 + (1-q0) z/2 - (1-q0-3q0^2+j0) z^2/6], z = Vcmb/c, with
  q0 = -0.595, j0 = 1 (flat, Omega_m = 0.27). This is H0 times the
  luminosity distance of redshift z, to second order
  (`curvature_corrected_velocity`).
- `Vpds` = (fV - H0 D) / (1 + H0 D / c) -- Davis & Scrimgeour (2014),
  c(z - z_h)/(1 + z_h) to first order in the distance offset
  (`davis_scrimgeour_velocity`). NOT `fV - H0 D`: the 1/(1 + z_h) is there.
- `Vpwf` = fV / (1 + fV/c) * **log10**(fV / (H0 D)) -- the Watkins & Feldman
  (2015) form, but with log10 where their estimator has the natural log
  (`watkins_feldman_velocity(..., as_published=True)`). The published column
  is therefore SMALLER than the peculiar velocity it estimates by exactly
  ln 10 = 2.303. Rows with fV <= 0, where the log is undefined, carry a
  filler 0 rather than a missing value.
- `Vpec` = w Vpds + (1 - w) Vpwf, w = clip(1 - Vls / 3000, 0, 1): a linear
  ramp in the Local-Sheet velocity from pure `Vpds` at Vls = 0 to pure
  `Vpwf` at Vls >= 3000 km/s, except w = 1 where fV < 0 (`published_vpec`).
  Beyond the ramp, `Vpec` inherits `Vpwf`'s ln 10 suppression.

Three independent checks say the natural-log form is the peculiar velocity
and the published `Vpwf` is not: (1) in the linear regime |v| << cz, every
estimator of v must agree with `Vpds` to first order, and the published
`Vpwf / Vpds` sits at 0.437 = 1/ln 10 instead of 1; (2) the velocity error
implied by `eDM` exceeds the total observed scatter of the published column
by 2.3x at every depth -- impossible for an error -- and matches it once the
ln 10 is restored (`velocity_error_from_modulus_error`); (3) only the ln
form has the right limit, v -> fV - H0 D as the log's argument -> 1, which
is the defining property of Watkins & Feldman's estimator. The audit
notebook runs (1) and (2). Not checked here: whether Tully et al. (2023)
write ln or log10 in the CF4 paper (it is not in `literature/`).

`eDM = 0` (228 groups, all D < 110 Mpc) marks the Cepheid/TRGB calibrator
groups: `CF4_Groups.csv` gives them `Nc >= 1` and `eDM_av = 0.1`, and its
description says the measured-moduli error EXCLUDES the calibrators. Zero is
"not given", not "exact", so `velocity_error_from_modulus_error` returns NaN
for it rather than a zero error that would carry infinite inverse-variance
weight.

2MRS
----
`load_twomrs_galaxies` reads 43,526 galaxies (Huchra et al. quote 43,533),
one row per galaxy, `pgc` and `ID_2MASXJ` unique, no missing values. Two
things differ from the other two catalogs:

- **Velocities are heliocentric only** (`Vhel`). Comparing with CF4 `Vcmb`
  or Tempel `zcmb` needs the CMB-dipole correction (up to ~370 km/s); that
  frame conversion is Stage 3's (`sky_positions.twomrs_positions`), not done here.
- **The selection is a magnitude limit plus an analytic sky mask.** Every row
  has the extinction-corrected TOTAL magnitude `K_tc` <= 11.75 (the isophotal
  `K_c` exceeds it in 200 rows -- it is not the selection magnitude), and
  every row lies inside `in_twomrs_footprint`: |b| >= 5 deg, widened to
  8 deg toward the bulge (l < 30 or l > 330 deg). The data honour the mask
  with no exception, which is the evidence that this is the survey's mask.

Stage 2: the analysis samples
-----------------------------
Cuts live in `dvcorr.config.samples.SampleConfig` (hard rule 4).

- **Density tracer** (`collapse_tempel_groups`, `select_tempel_tracers`):
  one row per Tempel group (keyed by `IDg`) and per singleton (keyed by
  its galaxy `ID` -- the shared `IDg = 0` is not an identity), centered on
  the lowest `Rg` present, at radius `dc`. Primary sample: every tracer,
  singletons included (`min_richness = 1`). No depth cut.
- **Velocity tracer** (`select_cf4_velocities`): CF4 groups inside the
  SDSS footprint (`footprint_pixels` of the Tempel galaxies,
  `in_pixel_footprint`), at every depth unless
  `SampleConfig.velocity_cz_max` is set (default: no cz cut), with u =
  EDD's `Vpec` ramp rebuilt with the natural log (`cf4_peculiar_velocity`).
  A group whose u is undefined is dropped, never given u = 0 (hard rule 5).
- `duplicate_test` (the methodological note's section 6.2) and
  `sample_counts` check the result.

The second density field, 2MRS (M; advisor meeting 2026-10-14):

- **Density tracer** (`select_twomrs_tracers`): every 2MRS galaxy, one row
  each, keyed by `pgc`. A galaxy catalog, not a group catalog, so there is
  nothing to collapse and no singleton trap.
- **Velocity tracer** (`select_cf4_velocities_in_twomrs`): the same CF4
  selection, with the analytic 2MRS mask (`in_twomrs_footprint`) in place
  of the SDSS pixels.
- `twomrs_duplicate_test` checks it; `counts_matrix` puts the T and M
  funnels side by side, the roadmap's {T, M} x {CF4 groups} matrix.

Positions on the sky
--------------------
Not periodic (CLAUDE.md hard rule 3 does not apply to real data). The
observer is at the origin and directions come from (RA, DE) in degrees via
`radec_to_unit_vector`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from dvcorr import conventions
from dvcorr.config.cosmology import CosmologyConfig
from dvcorr.config.paths import PathsConfig
from dvcorr.config.samples import (
    VALID_VELOCITY_DEFINITIONS,
    VELOCITY_VPDS,
    VELOCITY_VPEC_PUBLISHED,
    VELOCITY_VPWF_LN,
    SampleConfig,
)

#: Line indices of the five-line EDD header (see the module docstring).
EDD_TITLE_LINE: int = 0
EDD_NAMES_LINE: int = 1
EDD_FORMATS_LINE: int = 2
EDD_UNITS_LINE: int = 3
EDD_DESCRIPTIONS_LINE: int = 4
EDD_HEADER_LINES: int = 5

#: Field index of the catalog title within the title line (field 0 is the
#: database name, "HYPERLEDA"; every later field repeats the title).
_EDD_TITLE_FIELD: int = 1

#: A printf numeric/string format as EDD writes it: %7s, %6.3f, %5d.
_PRINTF_FORMAT = re.compile(r"^%(\d+)(?:\.(\d+))?([sdfe])$")

#: Column-dictionary fields, in the order the header lines give them.
COLUMN_DICTIONARY_FIELDS: tuple[str, str, str] = ("format", "unit", "description")


@dataclass(frozen=True)
class EddCatalog:
    """One EDD table and its column dictionary, as published.

    Attributes
    ----------
    path : Path
        The CSV it was read from.
    title : str
        The catalog title from header line 0 (e.g. "CF4 All Group Velocities").
    table : pandas.DataFrame
        Every data row, every column, unmodified: no row dropped, no value
        filled (CLAUDE.md hard rule 5), columns in file order.
    columns : pandas.DataFrame
        The column dictionary, indexed by column name (same order as
        `table.columns`), with string columns `format`, `unit` and
        `description` taken verbatim from header lines 2-4. EDD writes
        `---` or an empty field for a unitless column; both are kept as-is.
    """

    path: Path
    title: str
    table: pd.DataFrame
    columns: pd.DataFrame


def read_edd_header(path: str | Path) -> tuple[str, pd.DataFrame]:
    """Read the five EDD header lines into a title and a column dictionary.

    Parameters
    ----------
    path : str or Path
        An EDD CSV.

    Returns
    -------
    title : str
    columns : pandas.DataFrame
        Indexed by column name; columns `COLUMN_DICTIONARY_FIELDS`.

    Raises
    ------
    ValueError
        If the file has fewer than five lines, the four dictionary lines
        disagree in length, or any entry of the format line is not a printf
        format -- i.e. the file is not in EDD's five-line layout.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        lines = [row for _, row in zip(range(EDD_HEADER_LINES), reader)]
    if len(lines) < EDD_HEADER_LINES:
        raise ValueError(
            f"read_edd_header: {path.name} has {len(lines)} lines; an EDD "
            f"table needs {EDD_HEADER_LINES} header lines."
        )
    names = lines[EDD_NAMES_LINE]
    formats = lines[EDD_FORMATS_LINE]
    units = lines[EDD_UNITS_LINE]
    descriptions = lines[EDD_DESCRIPTIONS_LINE]
    lengths = {len(names), len(formats), len(units), len(descriptions)}
    if len(lengths) != 1:
        raise ValueError(
            f"read_edd_header: {path.name} header lines disagree in length "
            f"(names {len(names)}, formats {len(formats)}, units {len(units)}, "
            f"descriptions {len(descriptions)}); not an EDD table."
        )
    bad = [fmt for fmt in formats if not _PRINTF_FORMAT.match(fmt.strip())]
    if bad:
        raise ValueError(
            f"read_edd_header: {path.name} line {EDD_FORMATS_LINE} is not a "
            f"row of printf formats (first offender {bad[0]!r}); not an EDD table."
        )
    title_line = lines[EDD_TITLE_LINE]
    title = title_line[_EDD_TITLE_FIELD] if len(title_line) > _EDD_TITLE_FIELD else ""
    columns = pd.DataFrame(
        {
            "format": [fmt.strip() for fmt in formats],
            "unit": units,
            "description": descriptions,
        },
        index=pd.Index([name.strip() for name in names], name="column"),
    )
    return title, columns


def read_edd_csv(path: str | Path) -> EddCatalog:
    """Read an EDD CSV: header lines -> column dictionary, the rest -> table.

    Ported from the old repo's `data_loader.read_cf4_csv` (header line 1 as
    names, lines 2-4 skipped), with the header now validated rather than
    sniffed, and kept rather than discarded.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    EddCatalog

    Raises
    ------
    ValueError
        From `read_edd_header` if the file is not EDD-formatted, or if the data
        columns pandas parses differ from the names line.
    """
    path = Path(path)
    title, columns = read_edd_header(path)
    skipped = [EDD_TITLE_LINE, EDD_FORMATS_LINE, EDD_UNITS_LINE, EDD_DESCRIPTIONS_LINE]
    table = pd.read_csv(path, skiprows=skipped)
    table.columns = [str(name).strip() for name in table.columns]
    if list(table.columns) != list(columns.index):
        raise ValueError(
            f"read_edd_csv: {path.name} data columns {list(table.columns)} do "
            f"not match its names line {list(columns.index)}."
        )
    return EddCatalog(path=path, title=title, table=table, columns=columns)


def column_descriptions(catalog: EddCatalog) -> dict[str, str]:
    """Column name -> EDD description, in file order."""
    return catalog.columns["description"].to_dict()


def format_half_ulp(fmt: str) -> float:
    """Half the last printed digit of a printf format: the rounding bound.

    A value written with `%6.3f` is within 0.0005 of the number it was
    rounded from; one written with `%5d` within 0.5. This is the tolerance any
    reconstruction of a published column must be judged against.

    Parameters
    ----------
    fmt : str
        A printf format as in an EDD format line.

    Returns
    -------
    float
        Half a unit in the last place; NaN for a string format (`%7s`), which
        has no numeric precision.
    """
    match = _PRINTF_FORMAT.match(fmt.strip())
    if match is None:
        raise ValueError(f"format_half_ulp: {fmt!r} is not a printf format.")
    _, decimals, kind = match.groups()
    if kind == "s":
        return float("nan")
    # 0.5 in the last printed place: the definition of round-to-nearest.
    return 0.5 * 10.0 ** (-int(decimals or 0))


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_cf4_velocities(paths: PathsConfig | None = None) -> EddCatalog:
    """`CF4_Groups_Velocities.csv`: 38,057 CF4 groups with peculiar velocities.

    One row per group, keyed by `pgc`, in the same row order as
    `load_cf4_groups` (the audit checks this). See the module docstring for
    what `fV`, `Vpds`, `Vpwf` and `Vpec` are.
    """
    paths = paths or PathsConfig()
    return read_edd_csv(paths.cf4_velocities_catalog)


def load_cf4_groups(paths: PathsConfig | None = None) -> EddCatalog:
    """`CF4_Groups.csv`: the same 38,057 groups with per-method distance moduli.

    Per-method columns (`DM_FP`, `Ntf`, ...) are NaN where a group has no
    measurement by that method; those NaNs are left alone (hard rule 5).
    """
    paths = paths or PathsConfig()
    return read_edd_csv(paths.cf4_groups_catalog)


def load_tempel_galaxies(paths: PathsConfig | None = None) -> EddCatalog:
    """`SDSS_Tempel.csv`: Tempel et al. (2017) as a GALAXY table, 305,952 rows.

    One row per galaxy (`ID` and `pgc` unique); each row carries its group's
    `IDg`, `Ng`, `Rg`, `dc`, `gVcmb`. It is not a group catalog: `IDg = 0`
    marks every isolated galaxy, and the group catalog is derived from this
    table by collapsing rows (Stage 2).
    """
    paths = paths or PathsConfig()
    return read_edd_csv(paths.sdss_tempel_catalog)


def load_twomrs_galaxies(paths: PathsConfig | None = None) -> EddCatalog:
    """2MRS K_s <= 11.75 (Huchra et al. 2012): 43,526 galaxies, one row each.

    The redshift survey Nusser (2017) used as his density field. Positions are
    `RAJ`/`DEJ` (J2000 degrees) and `Glon`/`Glat`; the only velocity is the
    HELIOCENTRIC `Vhel` (see the module docstring). `e_V = 0` means "not
    given", not an exact redshift.
    """
    paths = paths or PathsConfig()
    return read_edd_csv(paths.twomrs_catalog)


# --------------------------------------------------------------------------
# Sky geometry (observer at the origin; not periodic)
# --------------------------------------------------------------------------


def radec_to_unit_vector(ra_deg: np.ndarray, de_deg: np.ndarray) -> np.ndarray:
    """Equatorial (RA, DE) in degrees -> unit vectors n_hat.

    Parameters
    ----------
    ra_deg, de_deg : array_like, shape (N,)

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        (cos DE cos RA, cos DE sin RA, sin DE). Shape (N, 3) even for N = 1.
    """
    ra = np.deg2rad(np.atleast_1d(np.asarray(ra_deg, dtype=float)))
    de = np.deg2rad(np.atleast_1d(np.asarray(de_deg, dtype=float)))
    cos_de = np.cos(de)
    return np.column_stack([cos_de * np.cos(ra), cos_de * np.sin(ra), np.sin(de)])


def angular_separation_deg(
    ra1_deg: np.ndarray, de1_deg: np.ndarray, ra2_deg: np.ndarray, de2_deg: np.ndarray
) -> np.ndarray:
    """Great-circle angle between paired directions, degrees.

    Uses atan2(|a x b|, a . b), accurate at both tiny and near-antipodal
    separations, where arccos(a . b) loses all precision near 0.

    Parameters
    ----------
    ra1_deg, de1_deg, ra2_deg, de2_deg : array_like, shape (N,)

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    a = radec_to_unit_vector(ra1_deg, de1_deg)
    b = radec_to_unit_vector(ra2_deg, de2_deg)
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    dot = np.einsum("ij,ij->i", a, b)
    return np.rad2deg(np.arctan2(cross, dot))


@dataclass(frozen=True)
class TwoMrsFootprint:
    """The 2MRS Galactic-latitude mask (Huchra et al. 2012, §2).

    A fact about the survey, confirmed on the file (no galaxy violates it),
    not an analysis choice.

    Attributes
    ----------
    min_abs_latitude : float
        deg. |b| below this is masked everywhere.
    bulge_min_abs_latitude : float
        deg. The wider cut toward the Galactic bulge.
    bulge_half_width : float
        deg of Galactic longitude: the bulge region is l < this or
        l > 360 - this.
    """

    min_abs_latitude: float = 5.0
    bulge_min_abs_latitude: float = 8.0
    bulge_half_width: float = 30.0


def in_twomrs_footprint(
    glon_deg: np.ndarray, glat_deg: np.ndarray, footprint: TwoMrsFootprint | None = None
) -> np.ndarray:
    """Whether each direction lies inside the 2MRS sky mask.

    Parameters
    ----------
    glon_deg, glat_deg : array_like, shape (N,)
        Galactic longitude and latitude, degrees.
    footprint : TwoMrsFootprint, optional

    Returns
    -------
    numpy.ndarray of bool, shape (N,)
    """
    f = footprint or TwoMrsFootprint()
    glon = np.mod(np.atleast_1d(np.asarray(glon_deg, dtype=float)), 360.0)
    abs_b = np.abs(np.atleast_1d(np.asarray(glat_deg, dtype=float)))
    bulge = (glon < f.bulge_half_width) | (glon > 360.0 - f.bulge_half_width)
    limit = np.where(bulge, f.bulge_min_abs_latitude, f.min_abs_latitude)
    return abs_b >= limit


# --------------------------------------------------------------------------
# CF4 velocity definitions (how EDD built the published columns)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cf4VelocityDefinitions:
    """The constants EDD used to build CF4's velocity columns.

    Facts about the published file, recovered by reproducing its columns
    (see the module docstring), not analysis choices: changing one makes the
    functions below stop describing the catalog.

    Attributes
    ----------
    h0 : float
        km/s/Mpc; `CosmologyConfig.H0_CF4`, the scale of `D`, `Vpds`, `Vpwf`.
    q0 : float
        Deceleration parameter in `fV` (flat Omega_m = 0.27: q0 = 1.5 Om - 1).
    j0 : float
        Jerk parameter in `fV` (1 for flat LambdaCDM).
    ramp_velocity : float
        km/s. `Vpec` ramps linearly in `Vls` from pure `Vpds` at 0 to pure
        `Vpwf` at this velocity and beyond.
    """

    h0: float = field(default_factory=lambda: CosmologyConfig().H0_CF4)
    q0: float = -0.595
    j0: float = 1.0
    ramp_velocity: float = 3000.0


def distance_from_modulus(modulus: np.ndarray) -> np.ndarray:
    """Distance modulus (mag) -> luminosity distance (Mpc), shape-preserving.

    DM = 5 log10(D / 10 pc) = 25 + 5 log10(D / Mpc); the 25 and 5 are that
    definition, not tunables. Note the result is on whatever H0 scale the
    modulus was calibrated to -- for CF4, `Cf4VelocityDefinitions.h0`.
    """
    return 10.0 ** ((np.asarray(modulus, dtype=float) - 25.0) / 5.0)


def curvature_corrected_velocity(
    cz: np.ndarray, definitions: Cf4VelocityDefinitions | None = None
) -> np.ndarray:
    """CF4's `fV`: cz corrected to H0 x luminosity distance, to O(z^2).

    fV = cz [1 + (1 - q0) z / 2 - (1 - q0 - 3 q0^2 + j0) z^2 / 6], z = cz/c.
    The 1/2 and 1/6 are the Taylor coefficients of d_L(z).

    Parameters
    ----------
    cz : array_like, shape (N,)
        CMB-frame velocity `Vcmb`, km/s.
    definitions : Cf4VelocityDefinitions, optional

    Returns
    -------
    numpy.ndarray, shape (N,)
        km/s.
    """
    d = definitions or Cf4VelocityDefinitions()
    cz = np.asarray(cz, dtype=float)
    z = cz / conventions.SPEED_OF_LIGHT
    first = 0.5 * (1.0 - d.q0) * z
    second = (1.0 - d.q0 - 3.0 * d.q0**2 + d.j0) * z**2 / 6.0
    return cz * (1.0 + first - second)


def davis_scrimgeour_velocity(
    fv: np.ndarray, distance: np.ndarray, definitions: Cf4VelocityDefinitions | None = None
) -> np.ndarray:
    """CF4's `Vpds`: (fV - H0 D) / (1 + H0 D / c), km/s.

    Davis & Scrimgeour (2014)'s v = c (z - z_h) / (1 + z_h) to first order in
    the offset, with c z_h ~ H0 D. Linear in the distance error, so its noise
    is non-Gaussian when the error is Gaussian in the modulus.

    Parameters
    ----------
    fv, distance : array_like, shape (N,)
        `fV` (km/s) and D (Mpc, on `definitions.h0`).
    definitions : Cf4VelocityDefinitions, optional

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    d = definitions or Cf4VelocityDefinitions()
    hubble_velocity = d.h0 * np.asarray(distance, dtype=float)
    return (np.asarray(fv, dtype=float) - hubble_velocity) / (
        1.0 + hubble_velocity / conventions.SPEED_OF_LIGHT
    )


def watkins_feldman_velocity(
    fv: np.ndarray,
    distance: np.ndarray,
    definitions: Cf4VelocityDefinitions | None = None,
    *,
    as_published: bool = False,
) -> np.ndarray:
    """Watkins & Feldman (2015): v = fV / (1 + fV/c) * ln(fV / (H0 D)), km/s.

    The log makes the error Gaussian when the modulus error is: with
    D = 10^((DM-25)/5), ln(fV / H0 D) = (ln 10 / 5)(DM_z - DM), linear in DM.

    Parameters
    ----------
    fv, distance : array_like, shape (N,)
        `fV` (km/s) and D (Mpc, on `definitions.h0`).
    definitions : Cf4VelocityDefinitions, optional
    as_published : bool, keyword-only
        False (default): the natural log -- the peculiar velocity. True:
        log10, which reproduces EDD's published `Vpwf` column and is smaller
        by exactly ln 10. Use True only to check the file, never as a signal.

    Returns
    -------
    numpy.ndarray, shape (N,)
        NaN where fV <= 0 (log undefined). The published column writes 0
        there instead; that 0 is a filler, not a velocity (hard rule 5).
    """
    d = definitions or Cf4VelocityDefinitions()
    fv = np.asarray(fv, dtype=float)
    hubble_velocity = d.h0 * np.asarray(distance, dtype=float)
    log = np.log10 if as_published else np.log
    out = np.full(np.broadcast(fv, hubble_velocity).shape, np.nan)
    ok = (fv > 0) & (hubble_velocity > 0)
    fv_ok = np.broadcast_to(fv, out.shape)[ok]
    out[ok] = (
        fv_ok
        / (1.0 + fv_ok / conventions.SPEED_OF_LIGHT)
        * log(fv_ok / np.broadcast_to(hubble_velocity, out.shape)[ok])
    )
    return out


def published_ramp_weight(
    vls: np.ndarray, definitions: Cf4VelocityDefinitions | None = None
) -> np.ndarray:
    """Weight of `Vpds` in published `Vpec`: clip(1 - Vls / ramp, 0, 1), shape (N,)."""
    d = definitions or Cf4VelocityDefinitions()
    return np.clip(1.0 - np.asarray(vls, dtype=float) / d.ramp_velocity, 0.0, 1.0)


def published_vpec(
    vpds: np.ndarray,
    vpwf: np.ndarray,
    vls: np.ndarray,
    fv: np.ndarray,
    definitions: Cf4VelocityDefinitions | None = None,
) -> np.ndarray:
    """CF4's `Vpec` = w Vpds + (1 - w) Vpwf, w = `published_ramp_weight(Vls)`.

    Reproduces the column from the published `Vpds` and `Vpwf`; it inherits
    whatever scale those carry -- with the published `Vpwf`, everything past
    the ramp is ln 10 low. Where fV < 0 (Vpwf undefined, published as a
    filler 0) EDD takes `Vpds` outright, w = 1. At fV = 0 exactly it does
    NOT, and blends with the filler 0; both behaviors are reproduced.

    Parameters
    ----------
    vpds, vpwf, vls, fv : array_like, shape (N,)

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    w = published_ramp_weight(vls, definitions)
    w = np.where(np.asarray(fv, dtype=float) < 0, 1.0, w)
    return w * np.asarray(vpds, dtype=float) + (1.0 - w) * np.asarray(vpwf, dtype=float)


def velocity_error_from_modulus_error(edm: np.ndarray, fv: np.ndarray) -> np.ndarray:
    """sigma_u implied by `eDM` for the natural-log WF velocity, km/s.

    sigma_u = (ln 10 / 5) eDM fV / (1 + fV/c): the derivative of
    `watkins_feldman_velocity` (ln form) with respect to DM, times eDM. Exact
    for that estimator to first order, at every depth; the methodological
    note's sigma_u ~ H0 D (ln 10 / 5) sigma_mu is its low-z limit. It is
    NOT the error on the published `Vpwf`, which is ln 10 smaller.

    Parameters
    ----------
    edm, fv : array_like, shape (N,)
        Modulus error (mag) and `fV` (km/s).

    Returns
    -------
    numpy.ndarray, shape (N,)
        NaN where eDM <= 0: CF4's eDM = 0 marks calibrator groups whose error
        is not given, and a zero error would carry infinite weight.
    """
    edm = np.asarray(edm, dtype=float)
    fv = np.asarray(fv, dtype=float)
    # ln(10)/5: d ln(D) / d DM, from D = 10^((DM-25)/5).
    sigma = (np.log(10.0) / 5.0) * edm * fv / (1.0 + fv / conventions.SPEED_OF_LIGHT)
    return np.where(edm > 0, sigma, np.nan)


# --------------------------------------------------------------------------
# Stage 2: the two analysis samples
# --------------------------------------------------------------------------

#: Tempel's `IDg` for an isolated galaxy. Shared by all 130,010 of them, so
#: it is NOT a group identity (Trap 1): a singleton is keyed by its `ID`.
TEMPEL_ISOLATED_IDG: int = 0
#: Tempel's `Rg` of a group's brightest member, the fiducial center. 375
#: groups lost it to the clipping (Trap 2); they center on their lowest `Rg`.
TEMPEL_BRIGHTEST_RANK: int = 1


def footprint_pixels(ra_deg: np.ndarray, de_deg: np.ndarray, nside: int) -> np.ndarray:
    """HEALPix pixels (RING, `nside`) holding at least one of the directions.

    An occupied-pixel footprint: built from Tempel galaxies it is the SDSS
    footprint, slightly generous at the edges (an edge pixel counts whole).

    Parameters
    ----------
    ra_deg, de_deg : array_like, shape (N,)
    nside : int

    Returns
    -------
    numpy.ndarray of int, shape (M,), sorted, unique.
    """
    ra = np.asarray(ra_deg, dtype=float)
    de = np.asarray(de_deg, dtype=float)
    return np.unique(hp.ang2pix(nside, ra, de, lonlat=True))


def in_pixel_footprint(
    ra_deg: np.ndarray, de_deg: np.ndarray, pixels: np.ndarray, nside: int
) -> np.ndarray:
    """Whether each direction falls in one of `pixels` (from `footprint_pixels`
    at the same `nside`).

    Returns
    -------
    numpy.ndarray of bool, shape (N,)
    """
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    de = np.atleast_1d(np.asarray(de_deg, dtype=float))
    return np.isin(hp.ang2pix(nside, ra, de, lonlat=True), pixels)


def collapse_tempel_groups(tempel: EddCatalog) -> pd.DataFrame:
    """The Tempel galaxy table collapsed to one row per density tracer.

    - **Real groups** (`IDg > 0`): one row per `IDg`. The center is the
      member with the lowest `Rg` present -- the brightest (`Rg = 1`) where
      the clipped table kept it, the next brightest in the 375 groups where it
      did not (Trap 2). A group with `Ng >= 2` but a single row left after
      clipping stays a group: rows are grouped by `IDg`, never by row count.
    - **Singletons** (`IDg = 0`): one row per galaxy `ID`, never merged into
      the one fake 130,010-member "group" a `groupby("IDg")` makes (Trap 1).

    The radius is `dc`, which is constant within every group (Trap 4); this
    raises if it is not, or if `Ng` varies within a group. `gVcmb` is not
    used: it differs between members of one group in 19% of groups, and for
    243 singletons it is more than 500 km/s from the galaxy's own c zcmb.

    Parameters
    ----------
    tempel : EddCatalog
        `load_tempel_galaxies()`.

    Returns
    -------
    pandas.DataFrame
        One row per tracer, groups first (ascending `IDg`) then singletons
        (ascending `ID`). Columns:

        - `IDg` -- the group id; `TEMPEL_ISOLATED_IDG` (0) for a singleton.
        - `ID`, `pgc` -- the CENTER galaxy's. `ID` is a unique key over all
          tracers (every galaxy belongs to exactly one tracer).
        - `is_singleton` -- bool.
        - `Ng` -- Tempel+14 parent richness (1 for a singleton).
        - `n_members` -- rows present in the clipped table (<= `Ng`).
        - `center_Rg` -- the center's `Rg` (1 unless Trap 2).
        - `RA`, `DE`, `glon`, `glat` -- the center's direction, degrees.
        - `dc` -- comoving distance, Mpc (H0 = 67.8), the radius.
        - `cz_cmb` -- c x the mean `zcmb` of the members present, km/s: the
          group redshift `dc` encodes (equal to it to ~2 km/s wherever no
          member was clipped). For a singleton, c x its own `zcmb`.
    """
    t = tempel.table
    isolated = t["IDg"] == TEMPEL_ISOLATED_IDG
    members = t[~isolated].sort_values(["IDg", "Rg", "ID"], kind="mergesort")
    by_group = members.groupby("IDg", sort=True)
    for column in ("dc", "Ng"):
        varies = by_group[column].nunique() > 1
        if varies.any():
            raise ValueError(
                f"collapse_tempel_groups: `{column}` varies within {int(varies.sum())} "
                f"groups (first IDg {varies.idxmax()}); one value per group is assumed."
            )
    centers = members.drop_duplicates("IDg", keep="first").set_index("IDg")
    groups = pd.DataFrame(
        {
            "IDg": centers.index.to_numpy(),
            "ID": centers["ID"].to_numpy(),
            "pgc": centers["pgc"].to_numpy(),
            "is_singleton": False,
            "Ng": centers["Ng"].to_numpy(),
            "n_members": by_group.size().loc[centers.index].to_numpy(),
            "center_Rg": centers["Rg"].to_numpy(),
            "RA": centers["RA"].to_numpy(float),
            "DE": centers["DE"].to_numpy(float),
            "glon": centers["glon"].to_numpy(float),
            "glat": centers["glat"].to_numpy(float),
            "dc": centers["dc"].to_numpy(float),
            "cz_cmb": by_group["zcmb"].mean().loc[centers.index].to_numpy()
            * conventions.SPEED_OF_LIGHT,
        }
    )
    single = t[isolated].sort_values("ID", kind="mergesort")
    singletons = pd.DataFrame(
        {
            "IDg": single["IDg"].to_numpy(),
            "ID": single["ID"].to_numpy(),
            "pgc": single["pgc"].to_numpy(),
            "is_singleton": True,
            "Ng": single["Ng"].to_numpy(),
            "n_members": 1,
            "center_Rg": single["Rg"].to_numpy(),
            "RA": single["RA"].to_numpy(float),
            "DE": single["DE"].to_numpy(float),
            "glon": single["glon"].to_numpy(float),
            "glat": single["glat"].to_numpy(float),
            "dc": single["dc"].to_numpy(float),
            "cz_cmb": single["zcmb"].to_numpy(float) * conventions.SPEED_OF_LIGHT,
        }
    )
    return pd.concat([groups, singletons], ignore_index=True)


def select_tempel_tracers(groups: pd.DataFrame, config: SampleConfig | None = None) -> pd.DataFrame:
    """The density-tracer sample: `collapse_tempel_groups` rows with `Ng` >=
    `config.min_richness`. No depth cut (see `SampleConfig.velocity_cz_max`).

    Returns
    -------
    pandas.DataFrame
        Same columns, index reset.
    """
    config = config or SampleConfig()
    return groups[groups["Ng"] >= config.min_richness].reset_index(drop=True)


def select_twomrs_tracers(twomrs: EddCatalog) -> pd.DataFrame:
    """The 2MRS density-tracer sample (M): every galaxy, one row each.

    Nusser (2017)'s density field. No richness, magnitude or depth cut: the
    file already is the K_tc <= 11.75 survey inside its own mask (notebook
    11 section 9), and the density tracer is never cut in depth. Galaxies
    with negative `Vhel` (Local Group neighbours, down to -300 km/s) are
    kept: `cz_hel` is a position here, not a peculiar velocity, and Stage 3
    decides what distance it maps to.

    Parameters
    ----------
    twomrs : EddCatalog
        `load_twomrs_galaxies()`.

    Returns
    -------
    pandas.DataFrame
        One row per galaxy, in file order. Columns: `pgc` (the key), `RA`,
        `DE`, `glon`, `glat` (degrees), `cz_hel` (km/s, the HELIOCENTRIC
        `Vhel`; the CMB-frame conversion is Stage 3's), `K_tc` (mag, the
        selection magnitude, for the 1/phi weights of Stage 4).

    Raises
    ------
    ValueError
        If a `pgc` appears twice.
    """
    t = twomrs.table
    if t["pgc"].duplicated().any():
        raise ValueError("select_twomrs_tracers: pgc is not unique in the 2MRS table.")
    return pd.DataFrame(
        {
            "pgc": t["pgc"].to_numpy(),
            "RA": t["RAJ"].to_numpy(float),
            "DE": t["DEJ"].to_numpy(float),
            "glon": t["Glon"].to_numpy(float),
            "glat": t["Glat"].to_numpy(float),
            "cz_hel": t["Vhel"].to_numpy(float),
            "K_tc": t["K_tc"].to_numpy(float),
        }
    )


def cf4_peculiar_velocity(
    velocities: pd.DataFrame,
    definition: str,
    definitions: Cf4VelocityDefinitions | None = None,
) -> np.ndarray:
    """The CF4 group peculiar velocity u under one of the `config.samples`
    velocity definitions, km/s.

    - `VELOCITY_VPEC_LN`: EDD's ramp with the natural log restored,
      w Vpds + (1 - w) v_WF, w = `published_ramp_weight(Vls)`, w = 1 where
      fV < 0 (EDD's rule). Vpds and v_WF are both rebuilt from `fV` and
      `DMzp` (no published rounding). Undefined (NaN) where fV = 0 exactly
      with w < 1: EDD blends a filler 0 there, which is not a velocity.
    - `VELOCITY_VPWF_LN`: v_WF with ln everywhere; NaN where fV <= 0.
    - `VELOCITY_VPEC_PUBLISHED`, `VELOCITY_VPDS`: the published columns.

    NaN marks "undefined"; it is never replaced by a number here (hard
    rule 5) -- the sample builder drops those rows.

    Parameters
    ----------
    velocities : pandas.DataFrame
        Rows of `load_cf4_velocities().table` (columns `fV`, `DMzp`, `Vls`,
        `Vpec`, `Vpds`).
    definition : str
        One of `config.samples.VALID_VELOCITY_DEFINITIONS`.
    definitions : Cf4VelocityDefinitions, optional

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    if definition not in VALID_VELOCITY_DEFINITIONS:
        raise ValueError(
            f"cf4_peculiar_velocity: definition must be one of "
            f"{sorted(VALID_VELOCITY_DEFINITIONS)}, got {definition!r}."
        )
    if definition == VELOCITY_VPEC_PUBLISHED:
        return velocities["Vpec"].to_numpy(dtype=float)
    if definition == VELOCITY_VPDS:
        return velocities["Vpds"].to_numpy(dtype=float)
    d = definitions or Cf4VelocityDefinitions()
    fv = velocities["fV"].to_numpy(dtype=float)
    distance = distance_from_modulus(velocities["DMzp"])
    wf = watkins_feldman_velocity(fv, distance, d)
    if definition == VELOCITY_VPWF_LN:
        return wf
    w = np.where(fv < 0, 1.0, published_ramp_weight(velocities["Vls"], d))
    ds = davis_scrimgeour_velocity(fv, distance, d)
    # Pure Vpds where w = 1, so an undefined v_WF there never enters as 0 * NaN.
    blended = w * ds + (1.0 - w) * wf
    return np.where(w == 1.0, ds, blended)


@dataclass(frozen=True)
class Cf4VelocitySample:
    """The CF4 velocity-tracer sample and the funnel that built it.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per CF4 group kept. Columns `pgc`; `RA`, `DE`, `glon`, `glat`
        (degrees, from `CF4_Groups.csv` -- the velocities file has pgc 51922
        41 deg off, notebook 11 section 2); `Vcmb`, `Vls`, `fV` (km/s, from
        the velocities file, which `fV` was built from); `DMzp`, `eDM` (mag);
        `u` (km/s, `velocity_definition`); `sigma_u` (km/s, the eDM error of
        the ln velocity, `velocity_error_from_modulus_error`; NaN for the
        eDM = 0 calibrator groups, whose error is not given -- they are kept,
        since Nusser's w = 1 needs no error).
    velocity_definition : str
    n_catalog : int
        Groups in the files.
    n_cz : int
        Of those, `Vcmb` < `velocity_cz_max` (all of them when the cut is
        None).
    n_footprint : int
        Of those, inside the footprint.
    n_undefined_velocity : int
        Of those, dropped because u is undefined (never filled, rule 5).
    undefined_pgc : tuple of int
        Their `pgc`, so the drop can be inspected.
    """

    table: pd.DataFrame
    velocity_definition: str
    n_catalog: int
    n_cz: int
    n_footprint: int
    n_undefined_velocity: int
    undefined_pgc: tuple[int, ...] = ()


def select_cf4_velocities(
    velocities: EddCatalog,
    groups: EddCatalog,
    pixels: np.ndarray,
    config: SampleConfig | None = None,
    definitions: Cf4VelocityDefinitions | None = None,
) -> Cf4VelocitySample:
    """The CF4 velocity-tracer sample for the Tempel field (T): groups inside
    the SDSS footprint `pixels`, at `Vcmb` < `velocity_cz_max` when that
    cut is set, with u defined.

    The two CF4 files are joined row by row and must list the same `pgc` in
    the same order (raises otherwise). The cz cut uses the velocities file's
    `Vcmb` (the one `fV` and u are built from; it differs from the groups
    file's in 7 rows); membership uses the groups file's position.

    Parameters
    ----------
    velocities, groups : EddCatalog
        `load_cf4_velocities()`, `load_cf4_groups()`.
    pixels : numpy.ndarray of int
        `footprint_pixels(...)` of the Tempel galaxies at
        `config.footprint_nside`.
    config : SampleConfig, optional
    definitions : Cf4VelocityDefinitions, optional

    Returns
    -------
    Cf4VelocitySample
    """
    config = config or SampleConfig()
    _check_cf4_row_order(velocities, groups)
    g = groups.table
    inside = in_pixel_footprint(g["RA"], g["DE"], pixels, config.footprint_nside)
    return _select_cf4_inside(velocities, groups, inside, config, definitions)


def select_cf4_velocities_in_twomrs(
    velocities: EddCatalog,
    groups: EddCatalog,
    config: SampleConfig | None = None,
    definitions: Cf4VelocityDefinitions | None = None,
    footprint: TwoMrsFootprint | None = None,
) -> Cf4VelocitySample:
    """The CF4 velocity-tracer sample for the 2MRS field (M): as
    `select_cf4_velocities`, with the analytic 2MRS latitude mask
    (`in_twomrs_footprint` of the groups file's `GLong`, `GLat`) in place of
    the SDSS pixels. `config.footprint_nside` is not used.

    Returns
    -------
    Cf4VelocitySample
        `n_footprint` counts the groups inside the 2MRS mask.
    """
    config = config or SampleConfig()
    _check_cf4_row_order(velocities, groups)
    g = groups.table
    inside = in_twomrs_footprint(g["GLong"], g["GLat"], footprint)
    return _select_cf4_inside(velocities, groups, inside, config, definitions)


def _check_cf4_row_order(velocities: EddCatalog, groups: EddCatalog) -> None:
    """Raise unless the two CF4 files list the same `pgc` in the same order."""
    v, g = velocities.table, groups.table
    if len(v) != len(g) or not np.array_equal(v["pgc"].to_numpy(), g["pgc"].to_numpy()):
        raise ValueError(
            "select_cf4_velocities: the velocities and groups files do not list the "
            "same pgc in the same row order; they cannot be joined row by row."
        )


def _select_cf4_inside(
    velocities: EddCatalog,
    groups: EddCatalog,
    inside: np.ndarray,
    config: SampleConfig,
    definitions: Cf4VelocityDefinitions | None,
) -> Cf4VelocitySample:
    """The shared CF4 selection: the cz cut, the sky membership `inside`
    ((N,) bool, one per row of the joined files), then u, dropping (never
    filling) the groups whose u is undefined."""
    v = velocities.table
    g = groups.table
    vcmb = v["Vcmb"].to_numpy(dtype=float)
    near = (
        np.ones(vcmb.size, bool) if config.velocity_cz_max is None
        else vcmb < config.velocity_cz_max
    )
    kept = near & inside
    u = cf4_peculiar_velocity(v, config.velocity_definition, definitions)
    defined = np.isfinite(u)
    rows = kept & defined
    fv = v["fV"].to_numpy(dtype=float)
    table = pd.DataFrame(
        {
            "pgc": v["pgc"].to_numpy(),
            "RA": g["RA"].to_numpy(float),
            "DE": g["DE"].to_numpy(float),
            "glon": g["GLong"].to_numpy(float),
            "glat": g["GLat"].to_numpy(float),
            "Vcmb": vcmb,
            "Vls": v["Vls"].to_numpy(float),
            "fV": fv,
            "DMzp": v["DMzp"].to_numpy(float),
            "eDM": v["eDM"].to_numpy(float),
            "u": u,
            "sigma_u": velocity_error_from_modulus_error(v["eDM"], fv),
        }
    )[rows].reset_index(drop=True)
    return Cf4VelocitySample(
        table=table,
        velocity_definition=config.velocity_definition,
        n_catalog=len(v),
        n_cz=int(near.sum()),
        n_footprint=int(kept.sum()),
        n_undefined_velocity=int((kept & ~defined).sum()),
        undefined_pgc=tuple(int(x) for x in v.loc[kept & ~defined, "pgc"]),
    )


def duplicate_test(tracers: pd.DataFrame, cf4: Cf4VelocitySample) -> pd.DataFrame:
    """The methodological note's duplicate test (section 6.2) for the Tempel
    field: one Tempel row per group identifier and one CF4 row per
    identifier.

    Returns
    -------
    pandas.DataFrame
        One row per key checked -- Tempel groups by `IDg`, Tempel singletons
        by `ID`, all Tempel tracers by center `ID`, CF4 by `pgc` -- with
        `rows`, `unique_keys` and `duplicates` (= rows - unique keys; must be
        0 everywhere).
    """
    singleton = tracers["is_singleton"].to_numpy()
    return _duplicate_table(
        {
            "Tempel groups, by IDg": tracers.loc[~singleton, "IDg"],
            "Tempel singletons, by ID": tracers.loc[singleton, "ID"],
            "Tempel tracers, by center ID": tracers["ID"],
            "CF4 groups, by pgc": cf4.table["pgc"],
        }
    )


def twomrs_duplicate_test(tracers: pd.DataFrame, cf4: Cf4VelocitySample) -> pd.DataFrame:
    """`duplicate_test` for the 2MRS field: one row per 2MRS galaxy (`pgc`)
    and one per CF4 group (`pgc`). Same columns."""
    return _duplicate_table(
        {"2MRS galaxies, by pgc": tracers["pgc"], "CF4 groups, by pgc": cf4.table["pgc"]}
    )


def _duplicate_table(checks: dict[str, pd.Series]) -> pd.DataFrame:
    """rows / unique_keys / duplicates for each named key column."""
    records = [
        {"key": name, "rows": len(keys), "unique_keys": int(keys.nunique()),
         "duplicates": len(keys) - int(keys.nunique())}
        for name, keys in checks.items()
    ]
    return pd.DataFrame.from_records(records).set_index("key")


def sample_counts(
    tracers: pd.DataFrame, cf4: Cf4VelocitySample, config: SampleConfig | None = None
) -> pd.DataFrame:
    """The Stage 2 counts table: the density-tracer and velocity-tracer funnels.

    When `config.velocity_cz_max` is set, the Tempel rows at "cz < cut" use
    `cz_cmb` (the group redshift `dc` encodes), for comparison only: the
    density sample is not cut in depth. With no cut (the default) those
    rows are left out and the CF4 rows say "all depths".

    Returns
    -------
    pandas.DataFrame
        Indexed by a description of each count; one column `count`.
    """
    config = config or SampleConfig()
    singleton = tracers["is_singleton"].to_numpy()
    counts = {
        "Tempel real groups (IDg > 0)": int((~singleton).sum()),
        "Tempel singletons (IDg = 0)": int(singleton.sum()),
        f"Tempel density tracers, Ng >= {config.min_richness}": len(tracers),
    }
    if config.velocity_cz_max is None:
        depth = "all depths"
    else:
        near = tracers["cz_cmb"].to_numpy() < config.velocity_cz_max
        depth = f"Vcmb < {config.velocity_cz_max:,.0f}"
        counts[f"Tempel groups at cz < {config.velocity_cz_max:,.0f}"] = int(
            (~singleton & near).sum()
        )
        counts[f"Tempel singletons at cz < {config.velocity_cz_max:,.0f}"] = int(
            (singleton & near).sum()
        )
    counts |= {
        "Tempel groups centered on Rg > 1 (Trap 2 fallback)": int(
            (tracers["center_Rg"] > TEMPEL_BRIGHTEST_RANK).sum()
        ),
        "CF4 groups (files)": cf4.n_catalog,
        f"CF4 groups at {depth}": cf4.n_cz,
        f"CF4 groups at {depth}, inside the SDSS footprint": cf4.n_footprint,
        f"  dropped: u undefined ({cf4.velocity_definition})": cf4.n_undefined_velocity,
        "CF4 velocity sample": len(cf4.table),
        "  of which calibrators (eDM = 0, sigma_u not given)": int(
            cf4.table["sigma_u"].isna().sum()
        ),
    }
    return pd.DataFrame({"count": counts}).rename_axis("sample")


def counts_matrix(
    pairs: dict[str, tuple[pd.DataFrame, Cf4VelocitySample]],
) -> pd.DataFrame:
    """The roadmap's Stage 2 counts matrix: one row per (density field,
    velocity sample) pair, e.g. {"T": (tempel_tracers, cf4),
    "M": (twomrs_tracers, cf4_m)}.

    Returns
    -------
    pandas.DataFrame
        Indexed by the pair's name. Columns: `density_tracers`;
        `cf4_in_footprint` (CF4 groups past the cz cut, if any, inside that
        field's sky); `cf4_undefined_u` (of those, dropped, never filled);
        `cf4_velocity_sample`; `cf4_calibrators` (eDM = 0, sigma_u not
        given).
    """
    records = [
        {
            "pair": name,
            "density_tracers": len(tracers),
            "cf4_in_footprint": cf4.n_footprint,
            "cf4_undefined_u": cf4.n_undefined_velocity,
            "cf4_velocity_sample": len(cf4.table),
            "cf4_calibrators": int(cf4.table["sigma_u"].isna().sum()),
        }
        for name, (tracers, cf4) in pairs.items()
    ]
    return pd.DataFrame.from_records(records).set_index("pair")
