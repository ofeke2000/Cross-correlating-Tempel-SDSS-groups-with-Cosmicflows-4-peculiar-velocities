"""
real_catalog_audit.py
---------------------
Stage 1 of `docs/real_data_roadmap.md`: the checks and figures that say
whether the raw EDD tables (`dvcorr.pipeline.real_catalogs`) are what we think
they are. Consumed by `notebooks/11_real_catalog_audit.ipynb`, and by
`notebooks/12_analysis_samples.ipynb` for its sky map and n(cz)
(`tempel_tracer_sky_sample`, `cf4_velocity_sky_sample`), and by
`notebooks/13_cf4_vpwf_error.ipynb`, which re-runs the peculiar-velocities
test on its own for the advisor. The only library module importing it is
`vpwf_error_figures` (notebook 13's figures).

Every check returns numbers or tables rather than printing, so the notebook
can show them next to the roadmap's expectations. The checks, and what each
one guards against:

- `column_range_table` -- unit LABELS against value RANGES (the RA "hour"
  trap): a label is documentation, a range is evidence.
- `compare_cf4_files` -- the two CF4 files are joined by `pgc`; this says
  where they disagree on a quantity both carry (Vcmb, position, eDM).
- `tempel_traps` / `group_members` -- the four ways the Tempel galaxy table
  silently breaks a naive "one row = one group" reading.
- `footprint_area_table` -- the Tempel sky area at several HEALPix
  resolutions (occupied-pixel area depends on pixel size; one number hides it).
- `reconstruct_cf4_velocities` -- each derived CF4 velocity column rebuilt
  from its inputs and judged against the file's own rounding, so a "match"
  is a statement with a tolerance, not a visual impression.
- `linear_regime_ratio` and `velocity_noise_by_depth` -- the two tests that
  settle the sigma_u puzzle (see `real_catalogs`' module docstring).
- `velocity_test_table`, `velocity_test_summary`, `velocity_test_by_depth`,
  `velocity_test_binned` -- the peculiar-velocities test: published `Vpwf`, published `Vpec`, and
  Watkins & Feldman's estimator rebuilt with ln and with log10, each put
  through identity, scale (against `Vpds`) and noise (v / sigma_u, Watkins
  & Feldman 2015 Fig. 1) checks, and shown against cz and measured distance.
  All but the table take a `labels` tuple: `VELOCITY_TEST_LABELS` (notebook
  11 §13) or `VPWF_ERROR_LABELS` (`Vpds` in place of `Vpec`, notebook 13,
  whose figures `dvcorr.pipeline.vpwf_error_figures` draws).

Stage 1 addendum (advisor meeting 2026-10-14, `docs/advisor_notes.md`):

- `twomrs_checks` / `twomrs_crossmatch` -- the 2MRS file against its survey
  definition (magnitude limit, latitude mask) and, galaxy by galaxy via
  `pgc`, against the heliocentric velocities Tempel and CF4 carry.
- `SkySample`, `sky_overlap_table` -- the three catalogs reduced to
  directions and cz, and how many of each fall inside the SDSS footprint
  (occupied HEALPix pixels of Tempel) and the 2MRS mask.
- `cf4_method_table`, `cf4_source_samples`, `velocity_drift_by_source`,
  `zero_point_drift_table` -- CF4 broken down by distance method, the
  Fundamental Plane split by sky into SDSS-like and 6dFGS-like parts, and
  the per-source median velocity against cz whose linear trend measures a
  distance-modulus zero-point offset.

Figures: Mollweide sky maps (one catalog, or all three overlaid), n(cz),
Vpec against cz, the Vpwf-against-Vpds scale test, the noise-against-scatter
comparison and the per-source velocity drift. Colors are the project's
categorical palette in fixed order, assigned per entity: CF4 slot 1 (blue),
Tempel slot 2 (orange), 2MRS slot 3 (aqua); the CF4 sources take the slots
in `_SOURCE_COLORS` order.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dvcorr import conventions
from dvcorr.config.samples import SampleConfig
from dvcorr.pipeline.real_catalogs import (
    TEMPEL_BRIGHTEST_RANK,
    TEMPEL_ISOLATED_IDG,
    Cf4VelocityDefinitions,
    Cf4VelocitySample,
    EddCatalog,
    TwoMrsFootprint,
    angular_separation_deg,
    curvature_corrected_velocity,
    davis_scrimgeour_velocity,
    distance_from_modulus,
    footprint_pixels,
    format_half_ulp,
    in_pixel_footprint,
    in_twomrs_footprint,
    published_vpec,
    velocity_error_from_modulus_error,
    watkins_feldman_velocity,
)

# Plot styling -- named here rather than inline (hard rule 4).
_COLOR_PRIMARY = "#2a78d6"
_COLOR_SECONDARY = "#eb6834"
_COLOR_TERTIARY = "#1baf7a"
#: Categorical slots 4-7 of the project palette, after the three above.
_COLOR_SLOT_4 = "#eda100"
_COLOR_SLOT_5 = "#e87ba4"
_COLOR_SLOT_6 = "#008300"
_COLOR_SLOT_7 = "#6250d6"
_COLOR_SLOT_8 = "#e34948"
_COLOR_CF4 = _COLOR_PRIMARY
_COLOR_TEMPEL = _COLOR_SECONDARY
_COLOR_TWOMRS = _COLOR_TERTIARY
_OVERLAP_POINT_SIZE = 1.2
_OVERLAP_POINT_ALPHA = 0.45
_LEGEND_MARKER_SCALE = 8.0
#: Bottom strip of an overlap map reserved for its legend (figure fraction).
_OVERLAP_LEGEND_STRIP = 0.07
_PLANE_SAMPLES = 721
_SKY_LARGE_FIGSIZE = (11.0, 6.2)
_DRIFT_FIGSIZE = (9.0, 5.5)
_ERRORBAR_CAPSIZE = 3.0
_COLOR_GUIDE = "0.45"
_LABEL_COLOR = "#111111"
_LINE_WIDTH = 2.0
_MARKER_SIZE = 6.0
_SKY_POINT_SIZE = 0.3
_SKY_POINT_ALPHA = 0.4
_HEXBIN_GRID = 120
_HEXBIN_CMAP = "Blues"
_GRID_ALPHA = 0.25
_GRID_LINE_WIDTH = 0.6
_GUIDE_LINE_WIDTH = 1.0
_HIST_ALPHA = 0.85
_SKY_FIGSIZE = (9.0, 5.0)
_SKY_TITLE_PAD = 14.0
_SKY_TICK_STEP_DEG = 30
_TICK_BACKING_ALPHA = 0.7
_TICK_BACKING_PAD = 1.0
_HIST_WIDTH = 8.0
_HIST_PANEL_HEIGHT = 2.8
_VPEC_FIGSIZE = (9.0, 5.5)
_DEFINITION_FIGSIZE = (6.5, 6.0)
_NOISE_FIGSIZE = (12.0, 4.8)


#: The 2MRS selection magnitude limit, on the total extinction-corrected
#: K_s (`K_tc`) -- Huchra et al. (2012), and the catalog's own title.
TWOMRS_K_LIMIT: float = 11.75

#: CF4 distance methods: label -> per-method modulus column in
#: `CF4_Groups.csv`. A group uses a method iff that column is present
#: (checked equal to N_method > 0 for every method that has an N column).
CF4_METHOD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("FP", "DM_FP"),
    ("TF", "DM_TFR"),
    ("SNIa", "DM_SNIa"),
    ("SNII", "DM_SNII"),
    ("SBF optical", "DM_SBFo"),
    ("SBF IR", "DM_SBFir"),
)
#: Fundamental-Plane groups split by sky -- a stand-in for the SDSS / 6dFGS
#: split, which the group table does not record (it needs the individuals).
SOURCE_FP_SDSS = "FP in SDSS footprint (~SDSS FP)"
SOURCE_FP_SOUTH = "FP, DE < 0 outside SDSS (~6dFGS FP)"
SOURCE_FP_OTHER = "FP, elsewhere"
#: Fixed color per source (entity, not rank): slots 1-7 in this order.
_SOURCE_COLORS: dict[str, str] = {
    SOURCE_FP_SDSS: _COLOR_PRIMARY,
    SOURCE_FP_SOUTH: _COLOR_SECONDARY,
    "TF": _COLOR_TERTIARY,
    "SNIa": _COLOR_SLOT_4,
    SOURCE_FP_OTHER: _COLOR_SLOT_5,
    "SNII": _COLOR_SLOT_6,
    "SBF optical": _COLOR_SLOT_7,
    "SBF IR": _COLOR_SLOT_8,
}


@dataclass(frozen=True)
class AuditConfig:
    """Binning and tolerances for the Stage 1 audit.

    Attributes
    ----------
    cz_edges : tuple of float
        km/s. Depth bins for the velocity-noise table (rows with Vcmb below
        the first edge, 45 nearby groups with negative CMB velocity, fall
        outside them).
    position_tolerance_deg : float
        Two positions of one CF4 group "agree" below this. Ten times the
        1e-4 deg the files print to.
    footprint_nsides : tuple of int
        HEALPix resolutions for the occupied-area estimate.
    histogram_bin_width : float
        km/s, for the n(cz) histograms.
    tempel_redshift_limit : float
        Tempel et al. (2017)'s z <= 0.1 sample limit, drawn on n(cz).
    linear_regime_fraction, linear_regime_floor : float
        The Vpwf/Vpds scale test uses rows with floor < |Vpds| < fraction *
        fV: small enough that ln(1+x) ~ x (so every estimator must agree),
        large enough that 1 km/s rounding does not dominate the ratio.
    example_group : int
        Tempel `IDg` shown for the gVcmb trap (the roadmap's example).
    velocity_plot_limit : float
        km/s, symmetric y-range of the velocity figures.
    overlap_nside : int
        HEALPix resolution of the SDSS footprint used for overlap counts
        (occupied Tempel pixels). Defaults to `SampleConfig.footprint_nside`
        (32), so the audit counts the same footprint the analysis sample uses.
    analysis_cz_max : float
        km/s. Nusser's cz < 10,000 cut, used for the audit's "Nusser-depth"
        overlap counts and map. The audit's own number: the analysis sample
        has no cz cut by default (`SampleConfig.velocity_cz_max` = None).
    velocity_match_tolerance : float
        km/s. A pgc cross-match whose heliocentric velocities differ by more
        than this is counted as a disagreement.
    drift_cz_edges : tuple of float
        km/s. Depth bins for the per-source median velocity.
    drift_min_count : int
        A (source, bin) cell with fewer objects is dropped from the drift
        table: its median error would be dominated by small-number noise.
    mask_area_nside : int
        HEALPix resolution for the analytic-mask area (`twomrs_mask_area`).
        The mask edges are thin latitude bands, so pixel centers must be fine:
        nside 128 is 0.13% off the analytic band area, nside 512 (0.11 deg
        pixels) within 0.1%.
    noise_dominated_sigma : float
        km/s. The peculiar-velocities test (`velocity_test_summary`) takes
        std(v / sigma_u) over objects with sigma_u above this, where the
        cosmic signal (~300 km/s) is negligible and v / sigma_u must be unit
        Gaussian for the right velocity. Watkins & Feldman (2015) use the
        same 1,000 km/s cut for their Fig. 1.
    normalized_velocity_limit : float
        Symmetric range of the v / sigma_u histograms.
    normalized_velocity_bins : int
        Bins across that range.
    distance_bin_width : float
        Mpc (CF4's H0 = 74.6 scale). Bins of measured distance D for
        `velocity_test_binned(..., VELOCITY_TEST_X_DISTANCE)`.
    distance_plot_max : float
        Mpc. Last bin edge and x-range of the velocity-against-distance
        figure; 99% of CF4 groups lie closer.
    redshift_bin_width : float
        km/s. Bins of CMB-frame cz for
        `velocity_test_binned(..., VELOCITY_TEST_X_REDSHIFT)`.
    redshift_plot_max : float
        km/s. Last bin edge and x-range of the velocity-against-cz figure;
        above CF4's largest Vcmb (32,575 km/s).
    scatter_velocity_limit : float
        km/s, symmetric y-range of the velocity-against-x figures. Wider
        than `velocity_plot_limit` because the ln-scale velocity is 2.3x
        wider than the published columns; this holds 99% of it.
    scatter_min_count : int
        An x bin with fewer groups is left out of the binned curves.
    """

    cz_edges: tuple[float, ...] = (0.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 35000.0)
    position_tolerance_deg: float = 1e-3
    footprint_nsides: tuple[int, ...] = (32, 64, 128)
    histogram_bin_width: float = 1000.0
    tempel_redshift_limit: float = 0.1
    linear_regime_fraction: float = 0.1
    linear_regime_floor: float = 200.0
    example_group: int = 1558
    velocity_plot_limit: float = 6000.0
    overlap_nside: int = field(default_factory=lambda: SampleConfig().footprint_nside)
    analysis_cz_max: float = 10000.0
    velocity_match_tolerance: float = 100.0
    drift_cz_edges: tuple[float, ...] = (
        0.0, 2500.0, 5000.0, 7500.0, 10000.0, 15000.0, 20000.0, 25000.0, 35000.0
    )
    drift_min_count: int = 20
    mask_area_nside: int = 512
    noise_dominated_sigma: float = 1000.0
    normalized_velocity_limit: float = 5.0
    normalized_velocity_bins: int = 50
    distance_bin_width: float = 25.0
    distance_plot_max: float = 600.0
    redshift_bin_width: float = 1500.0
    redshift_plot_max: float = 33000.0
    scatter_velocity_limit: float = 12000.0
    scatter_min_count: int = 20


# --------------------------------------------------------------------------
# Units against ranges
# --------------------------------------------------------------------------


def column_range_table(catalog: EddCatalog) -> pd.DataFrame:
    """The column dictionary with each column's realized range beside it.

    Parameters
    ----------
    catalog : EddCatalog

    Returns
    -------
    pandas.DataFrame
        Indexed by column; `unit`, `format`, `min`, `max`, `n_missing`,
        `description`. A unit that cannot hold the range (RA in `hour`
        reaching 360) is visible on one row.
    """
    table = catalog.table
    numeric = table.select_dtypes("number")
    out = catalog.columns[["unit", "format"]].copy()
    out["min"] = numeric.min().reindex(out.index)
    out["max"] = numeric.max().reindex(out.index)
    out["n_missing"] = table.isna().sum().reindex(out.index)
    out["description"] = catalog.columns["description"]
    return out


# --------------------------------------------------------------------------
# The two CF4 files against each other
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cf4FileComparison:
    """Where `CF4_Groups_Velocities.csv` and `CF4_Groups.csv` disagree.

    Attributes
    ----------
    n_velocities, n_groups, n_joined : int
        Row counts of each file and of their one-to-one join on `pgc`.
    same_row_order : bool
        Whether the two files list the same `pgc` in the same order.
    vcmb_mismatch : pandas.DataFrame
        Groups whose `Vcmb` differs between the files.
    position_mismatch : pandas.DataFrame
        Groups whose (RA, DE) differ by more than the tolerance, with the
        separation in degrees.
    edm_mismatch : pandas.DataFrame
        Groups whose velocities-file `eDM` differs from the groups-file
        `eDM_av`, with `Nc` (Cepheid/TRGB calibrator count) beside them.
    """

    n_velocities: int
    n_groups: int
    n_joined: int
    same_row_order: bool
    vcmb_mismatch: pd.DataFrame
    position_mismatch: pd.DataFrame
    edm_mismatch: pd.DataFrame


def compare_cf4_files(
    velocities: EddCatalog, groups: EddCatalog, config: AuditConfig | None = None
) -> Cf4FileComparison:
    """Join the two CF4 files on `pgc` and list every disagreement.

    The join is on the key, not on row position, so a reordering would show
    up as `same_row_order = False` rather than as a silent mismatch.
    `pandas.merge(..., validate="one_to_one")` raises if `pgc` repeats.
    """
    config = config or AuditConfig()
    v = velocities.table
    g = groups.table
    joined = v[["pgc", "Vcmb", "RA", "DE", "eDM", "D"]].merge(
        g[["pgc", "Vcmb", "RA", "DE", "eDM_av", "Nc"]],
        on="pgc",
        suffixes=("_velocities", "_groups"),
        validate="one_to_one",
    )
    same_order = len(v) == len(g) and bool(np.array_equal(v["pgc"].to_numpy(), g["pgc"].to_numpy()))

    vcmb = joined[joined["Vcmb_velocities"] != joined["Vcmb_groups"]].copy()
    vcmb["difference"] = vcmb["Vcmb_velocities"] - vcmb["Vcmb_groups"]

    separation = angular_separation_deg(
        joined["RA_velocities"], joined["DE_velocities"], joined["RA_groups"], joined["DE_groups"]
    )
    position = joined.assign(separation_deg=separation)
    position = position[position["separation_deg"] > config.position_tolerance_deg]

    edm_tol = format_half_ulp(velocities.columns.loc["eDM", "format"])
    edm = joined[np.abs(joined["eDM"] - joined["eDM_av"]) > edm_tol]

    return Cf4FileComparison(
        n_velocities=len(v),
        n_groups=len(g),
        n_joined=len(joined),
        same_row_order=same_order,
        vcmb_mismatch=vcmb[["pgc", "Vcmb_velocities", "Vcmb_groups", "difference"]],
        position_mismatch=position[
            ["pgc", "RA_velocities", "DE_velocities", "RA_groups", "DE_groups", "separation_deg"]
        ],
        edm_mismatch=edm[["pgc", "eDM", "eDM_av", "Nc", "D"]],
    )


# --------------------------------------------------------------------------
# Tempel: the galaxy table and its traps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TempelTraps:
    """Counts behind the four Tempel traps of the roadmap.

    Attributes
    ----------
    n_galaxies : int
        Rows (one per galaxy).
    id_unique, pgc_unique : bool
        Whether galaxy `ID` and `pgc` are unique keys.
    n_isolated : int
        Trap 1: rows with `IDg = 0`. All are isolated galaxies.
    isolated_ng_values, isolated_rg_values : tuple of int
        Distinct `Ng` and `Rg` among them (expected (1,) each).
    naive_largest_group : int
        Size of the largest "group" a naive `groupby("IDg")` produces -- the
        fake group the isolated galaxies merge into.
    n_groups : int
        Distinct `IDg > 0`.
    n_group_rows : int
        Rows belonging to them.
    groups_without_brightest : int
        Trap 2: groups with no `Rg = 1` row.
    groups_fewer_rows_than_ng : int
        Trap 2: groups listing fewer rows than their `Ng`.
    groups_more_rows_than_ng : int
        Groups listing more rows than `Ng` (expected 0).
    groups_single_row : int
        Groups with `Ng >= 2` represented by one row only.
    groups_ng_varies : int
        Groups whose `Ng` differs between members (expected 0).
    groups_gvcmb_varies : int
        Trap 4: groups whose `gVcmb` differs between members.
    groups_dc_varies : int
        Groups whose `dc` differs between members (expected 0).
    """

    n_galaxies: int
    id_unique: bool
    pgc_unique: bool
    n_isolated: int
    isolated_ng_values: tuple[int, ...]
    isolated_rg_values: tuple[int, ...]
    naive_largest_group: int
    n_groups: int
    n_group_rows: int
    groups_without_brightest: int
    groups_fewer_rows_than_ng: int
    groups_more_rows_than_ng: int
    groups_single_row: int
    groups_ng_varies: int
    groups_gvcmb_varies: int
    groups_dc_varies: int


def tempel_traps(tempel: EddCatalog) -> TempelTraps:
    """Count every Tempel trap on the raw galaxy table (see `TempelTraps`)."""
    t = tempel.table
    isolated = t["IDg"] == TEMPEL_ISOLATED_IDG
    members = t[~isolated].groupby("IDg")
    rows = members.size()
    ng = members["Ng"].first()
    return TempelTraps(
        n_galaxies=len(t),
        id_unique=bool(t["ID"].is_unique),
        pgc_unique=bool(t["pgc"].is_unique),
        n_isolated=int(isolated.sum()),
        isolated_ng_values=tuple(int(x) for x in np.unique(t.loc[isolated, "Ng"])),
        isolated_rg_values=tuple(int(x) for x in np.unique(t.loc[isolated, "Rg"])),
        naive_largest_group=int(t.groupby("IDg").size().max()),
        n_groups=int(rows.size),
        n_group_rows=int(rows.sum()),
        groups_without_brightest=int((members["Rg"].min() != TEMPEL_BRIGHTEST_RANK).sum()),
        groups_fewer_rows_than_ng=int((rows < ng).sum()),
        groups_more_rows_than_ng=int((rows > ng).sum()),
        groups_single_row=int((rows == 1).sum()),
        groups_ng_varies=int((members["Ng"].nunique() > 1).sum()),
        groups_gvcmb_varies=int((members["gVcmb"].nunique() > 1).sum()),
        groups_dc_varies=int((members["dc"].nunique() > 1).sum()),
    )


def group_members(tempel: EddCatalog, idg: int) -> pd.DataFrame:
    """The rows of one Tempel group, brightest first, with redshift as cz.

    Returns
    -------
    pandas.DataFrame
        `Rg`, `ID`, `pgc`, `Ng`, `gVcmb`, `cz_cmb` (= zcmb * c, km/s), `dc`,
        sorted by `Rg`.
    """
    t = tempel.table
    rows = t.loc[t["IDg"] == idg, ["Rg", "ID", "pgc", "Ng", "gVcmb", "zcmb", "dc"]]
    rows = rows.assign(cz_cmb=rows["zcmb"] * conventions.SPEED_OF_LIGHT).drop(columns="zcmb")
    return rows.sort_values("Rg").reset_index(drop=True)


def footprint_area_table(
    ra_deg: np.ndarray, de_deg: np.ndarray, config: AuditConfig | None = None
) -> pd.DataFrame:
    """Occupied-HEALPix-pixel area of a point set at several resolutions.

    Coarse pixels overcount (an edge pixel counts whole); fine pixels
    undercount (sparse interior pixels come out empty). The true footprint
    area lies between, which is why one nside alone is not the answer.

    Parameters
    ----------
    ra_deg, de_deg : array_like, shape (N,)

    Returns
    -------
    pandas.DataFrame
        One row per nside: `pixel_area_deg2`, `occupied_pixels`, `area_deg2`.
    """
    config = config or AuditConfig()
    records = []
    for nside in config.footprint_nsides:
        pixels = hp.ang2pix(nside, np.asarray(ra_deg), np.asarray(de_deg), lonlat=True)
        occupied = np.unique(pixels).size
        pixel_area = hp.nside2pixarea(nside, degrees=True)
        records.append(
            {
                "nside": nside,
                "pixel_area_deg2": pixel_area,
                "occupied_pixels": occupied,
                "area_deg2": occupied * pixel_area,
            }
        )
    return pd.DataFrame.from_records(records).set_index("nside")


# --------------------------------------------------------------------------
# CF4 velocity columns, rebuilt and judged against rounding
# --------------------------------------------------------------------------


def rounding_envelope(
    func: Callable[..., np.ndarray],
    inputs: dict[str, np.ndarray],
    half_ulps: dict[str, float],
    output_half_ulp: float,
) -> np.ndarray:
    """Largest change in `func` from rounding its inputs, plus output rounding.

    Evaluates `func` at every corner of the box [x - h, x + h] over its inputs
    (2^k evaluations for k inputs) and takes the largest deviation from the
    center. Exact for a function monotone in each input over the box, which
    every formula here is at the 1e-3 relative scale of the rounding.

    Parameters
    ----------
    func : callable
        Called with the inputs as keyword arguments; returns shape (N,).
    inputs : dict of str -> array, shape (N,)
    half_ulps : dict of str -> float
        Rounding half-width per input (`format_half_ulp` of its format).
    output_half_ulp : float
        Rounding half-width of the published output column.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The tolerance a reconstruction residual must stay within, km/s.
    """
    center = func(**inputs)
    names = list(inputs)
    deviation = np.zeros_like(center, dtype=float)
    for signs in itertools.product((-1.0, 1.0), repeat=len(names)):
        shifted = {n: inputs[n] + s * half_ulps[n] for n, s in zip(names, signs)}
        deviation = np.fmax(deviation, np.abs(func(**shifted) - center))
    return deviation + output_half_ulp


@dataclass(frozen=True)
class VelocityReconstruction:
    """Each derived CF4 column rebuilt from its inputs.

    Attributes
    ----------
    summary : pandas.DataFrame
        One row per rebuilt column: `formula`, `n_compared`, `rms_residual`,
        `max_abs_residual`, `n_outside_rounding`, `fraction_within_rounding`.
    residuals : pandas.DataFrame
        Per row of the catalog: `pgc` and, for each rebuilt column,
        `<col>_residual` (published - rebuilt) and `<col>_envelope`.
        NaN where a column is not compared (Vpwf at fV <= 0).
    """

    summary: pd.DataFrame
    residuals: pd.DataFrame


def reconstruct_cf4_velocities(
    velocities: EddCatalog, definitions: Cf4VelocityDefinitions | None = None
) -> VelocityReconstruction:
    """Rebuild `D`, `fV`, `Vpds`, `Vpwf` and `Vpec` and judge each by rounding.

    Each column is rebuilt only from the columns it is derived from (not from
    other rebuilt values), so a residual blames exactly one formula:
    D from DMzp; fV from Vcmb; Vpds and Vpwf from fV and DMzp; Vpec from the
    published Vpds, Vpwf, Vls (and the sign of fV). A row counts as reproduced when its residual
    is within `rounding_envelope` of its inputs' and output's printed
    precision. Vpwf is compared only where fV > 0; the file writes 0 where
    the log is undefined.
    """
    d = definitions or Cf4VelocityDefinitions()
    t = velocities.table
    ulp = {name: format_half_ulp(f) for name, f in velocities.columns["format"].items()}
    used = ("DMzp", "D", "Vcmb", "fV", "Vls", "Vpds", "Vpwf", "Vpec")
    data = {name: t[name].to_numpy(dtype=float) for name in used}

    def _d(DMzp):
        return distance_from_modulus(DMzp)

    def _fv(Vcmb):
        return curvature_corrected_velocity(Vcmb, d)

    def _vpds(fV, DMzp):
        return davis_scrimgeour_velocity(fV, distance_from_modulus(DMzp), d)

    def _vpwf(fV, DMzp):
        return watkins_feldman_velocity(fV, distance_from_modulus(DMzp), d, as_published=True)

    def _vpec(Vpds, Vpwf, Vls):
        return published_vpec(Vpds, Vpwf, Vls, data["fV"], d)

    everything = np.ones(len(t), dtype=bool)
    checks = (
        ("D", "10^((DMzp - 25)/5)", _d, ("DMzp",), everything),
        ("fV", "Vcmb[1 + (1-q0)z/2 - (1-q0-3q0^2+j0)z^2/6]", _fv, ("Vcmb",), everything),
        ("Vpds", "(fV - H0 D)/(1 + H0 D/c)", _vpds, ("fV", "DMzp"), everything),
        ("Vpwf", "fV/(1 + fV/c) log10(fV/(H0 D))", _vpwf, ("fV", "DMzp"), data["fV"] > 0),
        ("Vpec", "w Vpds + (1-w) Vpwf, w = clip(1 - Vls/ramp, 0, 1); w = 1 if fV < 0", _vpec,
         ("Vpds", "Vpwf", "Vls"), everything),
    )
    residuals = pd.DataFrame({"pgc": t["pgc"].to_numpy()})
    records = []
    for column, formula, func, args, mask in checks:
        kwargs = {a: data[a] for a in args}
        rebuilt = func(**kwargs)
        envelope = rounding_envelope(func, kwargs, {a: ulp[a] for a in args}, ulp[column])
        residual = np.where(mask, data[column] - rebuilt, np.nan)
        residuals[f"{column}_residual"] = residual
        residuals[f"{column}_envelope"] = np.where(mask, envelope, np.nan)
        compared = residual[mask]
        within = np.abs(compared) <= envelope[mask]
        records.append(
            {
                "column": column,
                "formula": formula,
                "n_compared": int(mask.sum()),
                "rms_residual": float(np.sqrt(np.mean(compared**2))),
                "max_abs_residual": float(np.max(np.abs(compared))),
                "n_outside_rounding": int((~within).sum()),
                "fraction_within_rounding": float(within.mean()),
            }
        )
    summary = pd.DataFrame.from_records(records).set_index("column")
    return VelocityReconstruction(summary=summary, residuals=residuals)


@dataclass(frozen=True)
class LinearRegimeRatio:
    """Published Vpwf against Vpds where every estimator must agree.

    Attributes
    ----------
    n_rows : int
        Rows in the linear regime (see `AuditConfig.linear_regime_fraction`).
    median_published_ratio : float
        median(Vpwf_published / Vpds). 1 if Vpwf estimates v; 1/ln 10 =
        0.434 if it carries log10.
    median_natural_log_ratio : float
        The same with Vpwf rebuilt in the natural log: the control.
    """

    n_rows: int
    median_published_ratio: float
    median_natural_log_ratio: float


def _linear_regime_rows(fv: np.ndarray, vpds: np.ndarray, config: AuditConfig) -> np.ndarray:
    """Rows with floor < |Vpds| < fraction * fV (see `AuditConfig`), shape (N,) bool."""
    return (
        (fv > 0)
        & (np.abs(vpds) < config.linear_regime_fraction * fv)
        & (np.abs(vpds) > config.linear_regime_floor)
    )


def linear_regime_ratio(
    velocities: EddCatalog,
    definitions: Cf4VelocityDefinitions | None = None,
    config: AuditConfig | None = None,
) -> LinearRegimeRatio:
    """The scale test: Vpwf / Vpds where |v| << cz.

    For x = (fV - H0 D)/(H0 D) small, ln(1 + x) = x + O(x^2), so the WF and DS
    estimators agree to first order -- they are two forms of one quantity.
    A median ratio far from 1 is a scale error in one column, independent of
    any error model.
    """
    d = definitions or Cf4VelocityDefinitions()
    config = config or AuditConfig()
    t = velocities.table
    fv = t["fV"].to_numpy(dtype=float)
    vpds = t["Vpds"].to_numpy(dtype=float)
    rows = _linear_regime_rows(fv, vpds, config)
    natural = watkins_feldman_velocity(fv, distance_from_modulus(t["DMzp"]), d)
    return LinearRegimeRatio(
        n_rows=int(rows.sum()),
        median_published_ratio=float(np.median(t["Vpwf"].to_numpy()[rows] / vpds[rows])),
        median_natural_log_ratio=float(np.median(natural[rows] / vpds[rows])),
    )


def velocity_noise_by_depth(
    velocities: EddCatalog,
    definitions: Cf4VelocityDefinitions | None = None,
    config: AuditConfig | None = None,
) -> pd.DataFrame:
    """Predicted velocity noise against observed scatter, per cz bin.

    The total scatter of a velocity column is signal plus noise, so it can
    never be much smaller than the noise. This table compares, per depth:

    - `sigma_u_rms`: rms of the eDM-predicted error of the natural-log WF
      velocity (`velocity_error_from_modulus_error`);
    - `naive_sigma_u_mean`: the roadmap's (ln 10/5) eDM cz, for reference;
    - `std_*`: observed standard deviation of published `Vpec`, of the
      natural-log WF velocity rebuilt from fV and DMzp (`vpwf_ln`), and of
      published `Vpds`;
    - `zstd_*`: std of velocity / sigma_u per object. ~1 means eDM explains
      the scatter; ~1/ln 10 means the column is ln 10 too small;
    - `implied_signal_rms`: sqrt(var(vpwf_ln) - mean sigma_u^2), signed --
      the cosmic velocity dispersion left after the noise; negative means
      the noise alone overshoots (eDM conservative).

    Calibrator groups (eDM = 0, no error given) are counted in
    `n_calibrators` and excluded from every column.

    Returns
    -------
    pandas.DataFrame
        One row per bin of `AuditConfig.cz_edges`.
    """
    d = definitions or Cf4VelocityDefinitions()
    config = config or AuditConfig()
    t = velocities.table
    cz = t["Vcmb"].to_numpy(dtype=float)
    fv = t["fV"].to_numpy(dtype=float)
    edm = t["eDM"].to_numpy(dtype=float)
    sigma = velocity_error_from_modulus_error(edm, fv)
    vpwf_ln = watkins_feldman_velocity(fv, distance_from_modulus(t["DMzp"]), d)
    columns = {
        "vpec_published": t["Vpec"].to_numpy(dtype=float),
        "vpwf_ln": vpwf_ln,
        "vpds": t["Vpds"].to_numpy(dtype=float),
    }
    edges = np.asarray(config.cz_edges)
    records = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (cz >= lo) & (cz < hi)
        use = in_bin & np.isfinite(sigma) & np.isfinite(vpwf_ln)
        record = {
            "cz_lo": lo,
            "cz_hi": hi,
            "cz_median": float(np.median(cz[use])),
            "n": int(use.sum()),
            "n_calibrators": int((in_bin & (edm <= 0)).sum()),
            "vpec_median": float(np.median(columns["vpec_published"][use])),
            "sigma_u_rms": float(np.sqrt(np.mean(sigma[use] ** 2))),
            # (ln 10)/5 = d ln D / d DM, as in velocity_error_from_modulus_error.
            "naive_sigma_u_mean": float(np.mean(np.log(10.0) / 5.0 * edm[use] * cz[use])),
        }
        for name, values in columns.items():
            record[f"std_{name}"] = float(np.std(values[use]))
        for name, values in columns.items():
            record[f"zstd_{name}"] = float(np.std(values[use] / sigma[use]))
        excess = np.var(vpwf_ln[use]) - np.mean(sigma[use] ** 2)
        record["implied_signal_rms"] = float(np.sign(excess) * np.sqrt(abs(excess)))
        records.append(record)
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# The peculiar-velocities test: four candidate velocities, three checks
# --------------------------------------------------------------------------

#: The four velocities of the peculiar-velocities test (notebook 11 §13),
#: in display order. Each is a column of `velocity_test_table`.
VELOCITY_VPWF_PUBLISHED = "Vpwf (published)"
VELOCITY_VPEC_PUBLISHED = "Vpec (published)"
VELOCITY_VPDS_PUBLISHED = "Vpds (published)"
VELOCITY_VPWF_LN = "Vpwf rebuilt, ln"
VELOCITY_VPWF_LOG10 = "Vpwf rebuilt, log10"
VELOCITY_TEST_LABELS: tuple[str, ...] = (
    VELOCITY_VPWF_PUBLISHED, VELOCITY_VPEC_PUBLISHED, VELOCITY_VPWF_LN, VELOCITY_VPWF_LOG10,
)
#: The same test with `Vpds` in place of `Vpec` (notebook 13): the
#: Davis-Scrimgeour estimator, which carries no log and so is on the right
#: scale by construction -- the reference the other three are held against.
VPWF_ERROR_LABELS: tuple[str, ...] = (
    VELOCITY_VPDS_PUBLISHED, VELOCITY_VPWF_PUBLISHED, VELOCITY_VPWF_LN, VELOCITY_VPWF_LOG10,
)
_VELOCITY_TEST_COLORS: dict[str, str] = {
    VELOCITY_VPWF_PUBLISHED: _COLOR_SLOT_4,
    VELOCITY_VPEC_PUBLISHED: _COLOR_SECONDARY,
    VELOCITY_VPDS_PUBLISHED: _COLOR_TERTIARY,
    VELOCITY_VPWF_LN: _COLOR_PRIMARY,
    VELOCITY_VPWF_LOG10: _COLOR_SLOT_7,
}
#: The rebuilt log10 column lies on the published one by construction;
#: dashed with open markers, it stays visible on top of it.
_VELOCITY_TEST_DASHED = frozenset({VELOCITY_VPWF_LOG10})
_VELOCITY_HIST_FIGSIZE = (11.0, 7.5)
_GAUSSIAN_SAMPLES = 401
#: Grey, so each panel's colored curves read on top of the point density.
_SCATTER_HEXBIN_CMAP = "Greys"

#: x-axes of the velocity-against-x figures: the `velocity_test_table`
#: column `velocity_test_binned` bins on.
VELOCITY_TEST_X_DISTANCE = "D"
VELOCITY_TEST_X_REDSHIFT = "Vcmb"
_VELOCITY_TEST_X_AXIS_LABELS: dict[str, str] = {
    VELOCITY_TEST_X_DISTANCE: r"measured distance $D$ [Mpc, $H_0$ = 74.6]",
    VELOCITY_TEST_X_REDSHIFT: r"$cz_{\rm CMB}$ = Vcmb [km/s]",
}
_VELOCITY_TEST_X_NAMES: dict[str, str] = {
    VELOCITY_TEST_X_DISTANCE: "measured distance",
    VELOCITY_TEST_X_REDSHIFT: "redshift (cz)",
}


def velocity_test_table(
    velocities: EddCatalog, definitions: Cf4VelocityDefinitions | None = None
) -> pd.DataFrame:
    """The candidate peculiar velocities side by side, one row per group.

    The three published columns as EDD ships them, and Watkins & Feldman's
    estimator rebuilt from `fV` and `DMzp` twice: with the natural log their
    eq. 7 needs, and with the log10 that reproduces EDD's `Vpwf`.

    Rows are DROPPED, never filled (hard rule 5), where any of the five
    quantities is undefined: fV <= 0 (log undefined; the published Vpwf
    carries a filler 0 there) and eDM = 0 (calibrator groups, sigma_u not
    given).

    Returns
    -------
    pandas.DataFrame
        Columns `pgc`, `Vcmb`, `fV`, `D` (Mpc, from `DMzp`, on CF4's
        H0 = 74.6 scale), `Vpds` and `sigma_u` (the eDM error of the
        natural-log velocity, `velocity_error_from_modulus_error`), then one
        column per label of `VELOCITY_TEST_LABELS` and `VPWF_ERROR_LABELS`
        (`VELOCITY_VPDS_PUBLISHED` repeats `Vpds` under its label), km/s.
    """
    d = definitions or Cf4VelocityDefinitions()
    t = velocities.table
    fv = t["fV"].to_numpy(dtype=float)
    distance = distance_from_modulus(t["DMzp"])
    sigma = velocity_error_from_modulus_error(t["eDM"], fv)
    table = pd.DataFrame(
        {
            "pgc": t["pgc"].to_numpy(),
            "Vcmb": t["Vcmb"].to_numpy(dtype=float),
            "fV": fv,
            "D": distance,
            "Vpds": t["Vpds"].to_numpy(dtype=float),
            "sigma_u": sigma,
            VELOCITY_VPWF_PUBLISHED: t["Vpwf"].to_numpy(dtype=float),
            VELOCITY_VPEC_PUBLISHED: t["Vpec"].to_numpy(dtype=float),
            VELOCITY_VPDS_PUBLISHED: t["Vpds"].to_numpy(dtype=float),
            VELOCITY_VPWF_LN: watkins_feldman_velocity(fv, distance, d),
            VELOCITY_VPWF_LOG10: watkins_feldman_velocity(fv, distance, d, as_published=True),
        }
    )
    defined = (fv > 0) & np.isfinite(sigma)
    return table[defined].reset_index(drop=True)


def velocity_test_summary(
    table: pd.DataFrame,
    config: AuditConfig | None = None,
    labels: tuple[str, ...] = VELOCITY_TEST_LABELS,
) -> pd.DataFrame:
    """Three checks on each of the velocities `labels` of `velocity_test_table`.

    - `median_ratio_to_published_vpwf`: identity. Over rows with
      |published Vpwf| > `linear_regime_floor` (so 1 km/s rounding does not
      dominate). The log10 rebuild must give 1, the ln rebuild ln 10 = 2.303.
    - `max_abs_difference_from_published_vpwf`: the same, as the worst row
      in km/s. A few km/s is rounding.
    - `median_ratio_to_vpds_linear`: scale. Where |v| << cz every estimator
      of v agrees with `Vpds` to first order (`linear_regime_ratio`), so
      the right velocity gives 1 and a log10 one 1/ln 10 = 0.434.
    - `std_normalized_noise_dominated`: noise. std(v / sigma_u) over objects
      with sigma_u > `noise_dominated_sigma`. There the measurement error
      dominates, so v / sigma_u is unit Gaussian for the velocity sigma_u is
      the error of (Watkins & Feldman 2015, Fig. 1). 1/ln 10 means the
      column is ln 10 smaller than its own error says.

    Returns
    -------
    pandas.DataFrame
        One row per `labels` entry, plus the row counts
        `n_identity`, `n_linear` and `n_noise_dominated`.
    """
    config = config or AuditConfig()
    published = table[VELOCITY_VPWF_PUBLISHED].to_numpy()
    identity = np.abs(published) > config.linear_regime_floor
    vpds = table["Vpds"].to_numpy()
    linear = _linear_regime_rows(table["fV"].to_numpy(), vpds, config)
    sigma = table["sigma_u"].to_numpy()
    noisy = sigma > config.noise_dominated_sigma
    records = []
    for label in labels:
        v = table[label].to_numpy()
        records.append(
            {
                "velocity": label,
                "median_ratio_to_published_vpwf": float(np.median(v[identity] / published[identity])),
                "max_abs_difference_from_published_vpwf": float(np.max(np.abs(v - published))),
                "median_ratio_to_vpds_linear": float(np.median(v[linear] / vpds[linear])),
                "std_normalized_noise_dominated": float(np.std(v[noisy] / sigma[noisy])),
                "n_identity": int(identity.sum()),
                "n_linear": int(linear.sum()),
                "n_noise_dominated": int(noisy.sum()),
            }
        )
    return pd.DataFrame.from_records(records).set_index("velocity")


def velocity_test_by_depth(
    table: pd.DataFrame,
    config: AuditConfig | None = None,
    labels: tuple[str, ...] = VELOCITY_TEST_LABELS,
) -> pd.DataFrame:
    """std(v / sigma_u) per cz bin for each of the velocities `labels`.

    Unlike `std_normalized_noise_dominated`, every object is used, so the
    nearby bins carry cosmic signal on top of the noise and sit above 1 for
    the right velocity. Past ~5,000 km/s noise dominates and the right
    velocity sits at ~1, a log10 one at ~1/ln 10.

    Returns
    -------
    pandas.DataFrame
        One row per bin of `AuditConfig.cz_edges`: `cz_lo`, `cz_hi`,
        `cz_median`, `n`, and one column per `labels` entry.
    """
    config = config or AuditConfig()
    cz = table["Vcmb"].to_numpy()
    sigma = table["sigma_u"].to_numpy()
    edges = np.asarray(config.cz_edges)
    records = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = (cz >= lo) & (cz < hi)
        record = {"cz_lo": lo, "cz_hi": hi, "cz_median": float(np.median(cz[rows])),
                  "n": int(rows.sum())}
        for label in labels:
            record[label] = float(np.std(table[label].to_numpy()[rows] / sigma[rows]))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _velocity_test_edges(x_column: str, config: AuditConfig) -> np.ndarray:
    """Bin edges from 0 for one of the `VELOCITY_TEST_X_*` axes, shape (M,)."""
    width, top = {
        VELOCITY_TEST_X_DISTANCE: (config.distance_bin_width, config.distance_plot_max),
        VELOCITY_TEST_X_REDSHIFT: (config.redshift_bin_width, config.redshift_plot_max),
    }[x_column]
    return np.arange(0.0, top + width, width)


def velocity_test_binned(
    table: pd.DataFrame,
    x_column: str,
    config: AuditConfig | None = None,
    labels: tuple[str, ...] = VELOCITY_TEST_LABELS,
) -> pd.DataFrame:
    """Median and spread of each of the velocities `labels` in bins of `x_column`.

    Parameters
    ----------
    table : pandas.DataFrame
        `velocity_test_table` output.
    x_column : str
        `VELOCITY_TEST_X_REDSHIFT` (cz = Vcmb, bins of `redshift_bin_width`
        up to `redshift_plot_max`) or `VELOCITY_TEST_X_DISTANCE` (measured
        D, bins of `distance_bin_width` up to `distance_plot_max`). Bins with
        fewer than `scatter_min_count` groups are left out.
    config : AuditConfig, optional
    labels : tuple of str, optional
        `velocity_test_table` columns to bin; default `VELOCITY_TEST_LABELS`.

    The two axes answer different questions:

    - **Binned on cz**, the distance error enters v only, so the median is
      the mean flow plus any zero-point offset (near 0 for CF4) and the
      spread is the clean noise test at every depth.
    - **Binned on measured D**, the distance error enters both axes: at
      fixed D, v ~ cz ln(cz / H0 D) is set by where each group's cz lies
      relative to H0 D, i.e. by the sample's n(cz), not by the flow. Up to
      the SDSS FP peak (cz ~ 24,000 km/s) a bin is dominated by groups
      scattered inward and the median is positive; past the peak and the
      cz ~ 30,000 km/s survey edge every group was scattered outward and the
      median plunges negative while the spread shrinks below sigma_u (cz is
      truncated). This is inhomogeneous Malmquist bias -- a property of the
      selection, identical in shape for all four columns -- so the width
      test there is clean only below ~200 Mpc.

    Returns
    -------
    pandas.DataFrame
        Long format, one row per (bin, velocity): `velocity`, `x_lo`,
        `x_hi`, `x_median`, `n`, `median`, `std` (km/s), and `sigma_u_rms`
        (the rms eDM-predicted error of the ln velocity in that bin).
    """
    config = config or AuditConfig()
    x = table[x_column].to_numpy()
    sigma = table["sigma_u"].to_numpy()
    edges = _velocity_test_edges(x_column, config)
    records = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = (x >= lo) & (x < hi)
        n = int(rows.sum())
        if n < config.scatter_min_count:
            continue
        for label in labels:
            v = table[label].to_numpy()[rows]
            records.append(
                {
                    "velocity": label,
                    "x_lo": lo,
                    "x_hi": hi,
                    "x_median": float(np.median(x[rows])),
                    "n": n,
                    "median": float(np.median(v)),
                    "std": float(np.std(v)),
                    "sigma_u_rms": float(np.sqrt(np.mean(sigma[rows] ** 2))),
                }
            )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# 2MRS: the file against its survey definition and against the other two
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoMrsChecks:
    """2MRS against its own definition (Huchra et al. 2012).

    Attributes
    ----------
    n_galaxies : int
    pgc_unique, id_unique : bool
        Whether `pgc` and the 2MASX identifier are unique keys.
    n_missing : int
        Missing values anywhere in the table (expected 0).
    max_total_magnitude : float
        Largest `K_tc` (expected <= `TWOMRS_K_LIMIT`).
    n_isophotal_beyond_limit : int
        Rows with the isophotal `K_c` > the limit -- allowed, since the
        selection is on the total magnitude.
    n_outside_mask : int
        Rows outside `in_twomrs_footprint` (expected 0).
    n_nonpositive_vhel : int
        Blueshifted (or zero) heliocentric velocities: Local Group members
        and Virgo galaxies falling toward us, not errors.
    n_velocity_error_not_given : int
        Rows with `e_V = 0`.
    median_vhel : float
        km/s.
    """

    n_galaxies: int
    pgc_unique: bool
    id_unique: bool
    n_missing: int
    max_total_magnitude: float
    n_isophotal_beyond_limit: int
    n_outside_mask: int
    n_nonpositive_vhel: int
    n_velocity_error_not_given: int
    median_vhel: float


def twomrs_checks(twomrs: EddCatalog, footprint: TwoMrsFootprint | None = None) -> TwoMrsChecks:
    """Count every way the 2MRS table could differ from its survey definition."""
    t = twomrs.table
    return TwoMrsChecks(
        n_galaxies=len(t),
        pgc_unique=bool(t["pgc"].is_unique),
        id_unique=bool(t["ID_2MASXJ"].is_unique),
        n_missing=int(t.isna().sum().sum()),
        max_total_magnitude=float(t["K_tc"].max()),
        n_isophotal_beyond_limit=int((t["K_c"] > TWOMRS_K_LIMIT).sum()),
        n_outside_mask=int((~in_twomrs_footprint(t["Glon"], t["Glat"], footprint)).sum()),
        n_nonpositive_vhel=int((t["Vhel"] <= 0).sum()),
        n_velocity_error_not_given=int((t["e_V"] == 0).sum()),
        median_vhel=float(t["Vhel"].median()),
    )


def twomrs_mask_area(
    config: AuditConfig | None = None, footprint: TwoMrsFootprint | None = None
) -> float:
    """Sky area inside the 2MRS latitude mask, deg^2.

    The fraction of HEALPix pixel centers (nside `config.mask_area_nside`)
    inside `in_twomrs_footprint`, times the full sky. Unlike the occupied-pixel area of the galaxies themselves
    (`footprint_area_table`), this does not depend on how densely the survey
    samples the sky.
    """
    config = config or AuditConfig()
    nside = config.mask_area_nside
    glon, glat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    full_sky = 4.0 * np.pi * np.rad2deg(1.0) ** 2
    return float(in_twomrs_footprint(glon, glat, footprint).mean() * full_sky)


def twomrs_crossmatch(
    twomrs: EddCatalog,
    tempel: EddCatalog,
    cf4_velocities: EddCatalog,
    config: AuditConfig | None = None,
) -> pd.DataFrame:
    """2MRS galaxies matched by `pgc` to Tempel and CF4, heliocentric v compared.

    Heliocentric against heliocentric, so no frame conversion enters:
    2MRS `Vhel` against Tempel c * `zobs` (a per-galaxy redshift) and against
    CF4 `Vhel`. The CF4 value is the GROUP's velocity, keyed by the group's
    `pgc`, so a spread there is group velocity dispersion, not disagreement.

    Returns
    -------
    pandas.DataFrame
        One row per partner: `n_matched`, `median_dv` and `robust_sigma_dv`
        (1.4826 x MAD, the Gaussian-equivalent sigma) of 2MRS minus partner,
        `n_beyond_tolerance` (|dv| > `config.velocity_match_tolerance`).
    """
    config = config or AuditConfig()
    m = twomrs.table[["pgc", "Vhel"]]
    partners = {
        "Tempel galaxies (c zobs)": tempel.table[["pgc"]].assign(
            v_partner=tempel.table["zobs"] * conventions.SPEED_OF_LIGHT
        ),
        "CF4 groups (group Vhel)": cf4_velocities.table[["pgc"]].assign(
            v_partner=cf4_velocities.table["Vhel"].astype(float)
        ),
    }
    records = []
    for name, partner in partners.items():
        joined = m.merge(partner, on="pgc", how="inner", validate="one_to_one")
        dv = joined["Vhel"] - joined["v_partner"]
        median = float(dv.median())
        # 1.4826: MAD -> sigma for a Gaussian (1 / Phi^-1(3/4)).
        records.append(
            {
                "partner": name,
                "n_matched": len(joined),
                "median_dv": median,
                "robust_sigma_dv": 1.4826 * float((dv - median).abs().median()),
                "n_beyond_tolerance": int((dv.abs() > config.velocity_match_tolerance).sum()),
            }
        )
    return pd.DataFrame.from_records(records).set_index("partner")


# --------------------------------------------------------------------------
# The three catalogs on one sky
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SkySample:
    """One catalog reduced to what the overlap counts and figures need.

    Attributes
    ----------
    name : str
    unit : str
        What one row is ("groups", "galaxies"), for axis labels.
    ra, de, glon, glat : numpy.ndarray, shape (N,)
        Degrees.
    cz : numpy.ndarray, shape (N,)
        km/s, in the frame `cz_frame` names.
    cz_frame : str
        "CMB" or "heliocentric". The two differ by up to ~370 km/s, so a
        cz cut applied to both is approximate at that level.
    color : str
    cz_limit : float or None
        A survey redshift limit (km/s) to mark on n(cz).
    """

    name: str
    unit: str
    ra: np.ndarray
    de: np.ndarray
    glon: np.ndarray
    glat: np.ndarray
    cz: np.ndarray
    cz_frame: str
    color: str
    cz_limit: float | None = None


def cf4_sky_sample(cf4_groups: EddCatalog) -> SkySample:
    """CF4 groups from `CF4_Groups.csv` -- the file whose positions are right
    for pgc 51922 (notebook 11, section 2) -- at CMB-frame `Vcmb`."""
    t = cf4_groups.table
    return SkySample(
        name="CF4 groups", unit="groups",
        ra=t["RA"].to_numpy(float), de=t["DE"].to_numpy(float),
        glon=t["GLong"].to_numpy(float), glat=t["GLat"].to_numpy(float),
        cz=t["Vcmb"].to_numpy(float), cz_frame="CMB", color=_COLOR_CF4,
    )


def tempel_sky_sample(tempel: EddCatalog, config: AuditConfig | None = None) -> SkySample:
    """Tempel galaxies (every row, not yet collapsed to groups) at c * `zcmb`."""
    config = config or AuditConfig()
    t = tempel.table
    return SkySample(
        name="Tempel galaxies", unit="galaxies",
        ra=t["RA"].to_numpy(float), de=t["DE"].to_numpy(float),
        glon=t["glon"].to_numpy(float), glat=t["glat"].to_numpy(float),
        cz=t["zcmb"].to_numpy(float) * conventions.SPEED_OF_LIGHT, cz_frame="CMB",
        color=_COLOR_TEMPEL,
        cz_limit=config.tempel_redshift_limit * conventions.SPEED_OF_LIGHT,
    )


def twomrs_sky_sample(twomrs: EddCatalog) -> SkySample:
    """2MRS galaxies at HELIOCENTRIC `Vhel` (the only velocity it carries)."""
    t = twomrs.table
    return SkySample(
        name="2MRS galaxies", unit="galaxies",
        ra=t["RAJ"].to_numpy(float), de=t["DEJ"].to_numpy(float),
        glon=t["Glon"].to_numpy(float), glat=t["Glat"].to_numpy(float),
        cz=t["Vhel"].to_numpy(float), cz_frame="heliocentric", color=_COLOR_TWOMRS,
    )


def tempel_tracer_sky_sample(tracers: pd.DataFrame) -> SkySample:
    """The Stage 2 density tracers (`real_catalogs.select_tempel_tracers`) at
    their centers' directions and the group redshift c x mean `zcmb`."""
    return SkySample(
        name="Tempel tracers", unit="tracers",
        ra=tracers["RA"].to_numpy(float), de=tracers["DE"].to_numpy(float),
        glon=tracers["glon"].to_numpy(float), glat=tracers["glat"].to_numpy(float),
        cz=tracers["cz_cmb"].to_numpy(float), cz_frame="CMB", color=_COLOR_TEMPEL,
    )


def twomrs_tracer_sky_sample(tracers: pd.DataFrame) -> SkySample:
    """The Stage 2 2MRS density tracers (`real_catalogs.select_twomrs_tracers`)
    at HELIOCENTRIC `cz_hel`."""
    return SkySample(
        name="2MRS tracers", unit="galaxies",
        ra=tracers["RA"].to_numpy(float), de=tracers["DE"].to_numpy(float),
        glon=tracers["glon"].to_numpy(float), glat=tracers["glat"].to_numpy(float),
        cz=tracers["cz_hel"].to_numpy(float), cz_frame="heliocentric", color=_COLOR_TWOMRS,
    )


def cf4_velocity_sky_sample(cf4: Cf4VelocitySample) -> SkySample:
    """The Stage 2 velocity tracers (`real_catalogs.select_cf4_velocities`) at
    CMB-frame `Vcmb`."""
    t = cf4.table
    return SkySample(
        name="CF4 velocity sample", unit="groups",
        ra=t["RA"].to_numpy(float), de=t["DE"].to_numpy(float),
        glon=t["glon"].to_numpy(float), glat=t["glat"].to_numpy(float),
        cz=t["Vcmb"].to_numpy(float), cz_frame="CMB", color=_COLOR_CF4,
    )


def sdss_footprint_pixels(tempel: EddCatalog, config: AuditConfig | None = None) -> np.ndarray:
    """HEALPix pixels (nside `config.overlap_nside`, RING) holding a Tempel galaxy:
    `real_catalogs.footprint_pixels` of every Tempel row. Stage 4 builds the
    real angular mask.

    Returns
    -------
    numpy.ndarray of int, shape (M,), sorted, unique.
    """
    config = config or AuditConfig()
    t = tempel.table
    return footprint_pixels(t["RA"], t["DE"], config.overlap_nside)


def in_sdss_footprint(
    ra_deg: np.ndarray, de_deg: np.ndarray, pixels: np.ndarray, config: AuditConfig | None = None
) -> np.ndarray:
    """Whether each direction falls in one of `pixels` (`sdss_footprint_pixels`):
    `real_catalogs.in_pixel_footprint` at `config.overlap_nside`.

    Returns
    -------
    numpy.ndarray of bool, shape (N,)
    """
    config = config or AuditConfig()
    return in_pixel_footprint(ra_deg, de_deg, pixels, config.overlap_nside)


def sky_overlap_table(
    samples: list[SkySample],
    sdss_pixels: np.ndarray,
    config: AuditConfig | None = None,
    *,
    cz_max: float | None = None,
) -> pd.DataFrame:
    """How much of each catalog lies inside the SDSS footprint and the 2MRS mask.

    Parameters
    ----------
    samples : list of SkySample
    sdss_pixels : numpy.ndarray of int
        `sdss_footprint_pixels(tempel)`.
    config : AuditConfig, optional
    cz_max : float, keyword-only, optional
        km/s. Count only rows with cz below it (each in its own frame).

    Returns
    -------
    pandas.DataFrame
        One row per sample: `n`, `in_sdss`, `in_twomrs_mask`, `in_both`, and
        the three as fractions of `n`.
    """
    config = config or AuditConfig()
    records = []
    for s in samples:
        keep = np.ones(s.cz.size, bool) if cz_max is None else s.cz < cz_max
        sdss = in_sdss_footprint(s.ra[keep], s.de[keep], sdss_pixels, config)
        mrs = in_twomrs_footprint(s.glon[keep], s.glat[keep])
        n = int(keep.sum())
        records.append(
            {
                "catalog": s.name,
                "n": n,
                "in_sdss": int(sdss.sum()),
                "in_twomrs_mask": int(mrs.sum()),
                "in_both": int((sdss & mrs).sum()),
                "frac_sdss": sdss.mean() if n else np.nan,
                "frac_twomrs_mask": mrs.mean() if n else np.nan,
                "frac_both": (sdss & mrs).mean() if n else np.nan,
            }
        )
    return pd.DataFrame.from_records(records).set_index("catalog")


# --------------------------------------------------------------------------
# CF4 by distance method: counts and zero-point drift
# --------------------------------------------------------------------------


def cf4_method_table(cf4_groups: EddCatalog, config: AuditConfig | None = None) -> pd.DataFrame:
    """Groups per CF4 distance method, all depths and below `analysis_cz_max`.

    Returns
    -------
    pandas.DataFrame
        One row per method in `CF4_METHOD_COLUMNS`: `n_groups`,
        `n_single_method` (groups using this method only), and the same two
        for cz < `config.analysis_cz_max`; plus a "calibrators (Nc > 0)" row.
    """
    config = config or AuditConfig()
    t = cf4_groups.table
    uses = pd.DataFrame({label: t[column].notna() for label, column in CF4_METHOD_COLUMNS})
    single = uses.sum(axis=1) == 1
    near = t["Vcmb"] < config.analysis_cz_max
    records = [
        {
            "method": label,
            "n_groups": int(uses[label].sum()),
            "n_single_method": int((uses[label] & single).sum()),
            "n_groups_near": int((uses[label] & near).sum()),
            "n_single_method_near": int((uses[label] & single & near).sum()),
        }
        for label in uses.columns
    ]
    calibrators = t["Nc"] > 0
    records.append(
        {
            "method": "calibrators (Nc > 0)",
            "n_groups": int(calibrators.sum()),
            "n_single_method": np.nan,
            "n_groups_near": int((calibrators & near).sum()),
            "n_single_method_near": np.nan,
        }
    )
    return pd.DataFrame.from_records(records).set_index("method")


def cf4_methods_per_group(cf4_groups: EddCatalog) -> pd.Series:
    """How many groups combine 0, 1, 2, ... distance methods.

    0 means calibrator-only (Cepheid/TRGB, none of the six methods).
    """
    t = cf4_groups.table
    n_methods = sum(t[column].notna().astype(int) for _, column in CF4_METHOD_COLUMNS)
    return n_methods.value_counts().sort_index().rename("groups")


def cf4_source_samples(
    cf4_groups: EddCatalog,
    cf4_velocities: EddCatalog,
    sdss_pixels: np.ndarray,
    config: AuditConfig | None = None,
    definitions: Cf4VelocityDefinitions | None = None,
) -> pd.DataFrame:
    """Single-method CF4 groups, labelled by source, with the ln-scale velocity.

    Only groups measured by ONE method, so a group's velocity can be blamed
    on one source. The FP groups are split by sky, the stand-in for the
    SDSS / 6dFGS split the group table does not record: inside the SDSS
    footprint (`SOURCE_FP_SDSS`), outside it at DE < 0 (`SOURCE_FP_SOUTH`;
    6dFGS observed only the south), and the rest (`SOURCE_FP_OTHER`).

    The velocity is `watkins_feldman_velocity` with the natural log (the
    peculiar velocity; the published `Vpwf` is ln 10 too small), from the
    velocities file's `fV` and `DMzp`; rows where it is undefined (fV <= 0)
    are dropped, never filled (hard rule 5). Positions and `Vcmb` come from
    `CF4_Groups.csv`.

    Returns
    -------
    pandas.DataFrame
        `pgc`, `source`, `Vcmb`, `v` (km/s), `RA`, `DE`.
    """
    config = config or AuditConfig()
    g = cf4_groups.table
    uses = pd.DataFrame({label: g[column].notna() for label, column in CF4_METHOD_COLUMNS})
    single = uses.sum(axis=1) == 1
    source = uses.idxmax(axis=1).where(single)
    fp = source == "FP"
    sdss = in_sdss_footprint(g["RA"], g["DE"], sdss_pixels, config)
    source = source.mask(fp & sdss, SOURCE_FP_SDSS)
    source = source.mask(fp & ~sdss & (g["DE"] < 0), SOURCE_FP_SOUTH)
    source = source.mask(fp & ~sdss & (g["DE"] >= 0), SOURCE_FP_OTHER)
    groups = g[["pgc", "Vcmb", "RA", "DE"]].assign(source=source)[single]
    v = cf4_velocities.table
    velocity = v[["pgc"]].assign(
        v=watkins_feldman_velocity(v["fV"], distance_from_modulus(v["DMzp"]), definitions)
    )
    out = groups.merge(velocity, on="pgc", how="inner", validate="one_to_one")
    out = out[np.isfinite(out["v"])]
    return out[["pgc", "source", "Vcmb", "v", "RA", "DE"]].reset_index(drop=True)


def velocity_drift_by_source(
    sources: pd.DataFrame, config: AuditConfig | None = None
) -> pd.DataFrame:
    """Median velocity per source per cz bin -- the zero-point drift test.

    A distance-modulus offset dm in one source shifts its log-distance
    velocities by dv = -(ln 10 / 5) dm cz: a median that drifts LINEARLY with
    cz, for that source only. A bulk flow seen over the source's patch of sky
    instead shifts the median by a roughly constant amount.

    Parameters
    ----------
    sources : pandas.DataFrame
        `cf4_source_samples` output.

    Returns
    -------
    pandas.DataFrame
        One row per (source, bin) with at least `config.drift_min_count`
        objects: `cz_lo`, `cz_hi`, `n`, `cz_median`, `v_median`, and
        `v_median_err` = sqrt(pi/2) std / sqrt(n), the large-n standard error
        of a median for a Gaussian.
    """
    config = config or AuditConfig()
    edges = np.asarray(config.drift_cz_edges)
    records = []
    for name, rows in sources.groupby("source", sort=False):
        bins = np.digitize(rows["Vcmb"], edges) - 1
        for i in range(edges.size - 1):
            cell = rows[bins == i]
            if len(cell) < config.drift_min_count:
                continue
            records.append(
                {
                    "source": name,
                    "cz_lo": edges[i],
                    "cz_hi": edges[i + 1],
                    "n": len(cell),
                    "cz_median": float(cell["Vcmb"].median()),
                    "v_median": float(cell["v"].median()),
                    "v_median_err": float(np.sqrt(np.pi / 2.0) * cell["v"].std(ddof=1)
                                          / np.sqrt(len(cell))),
                }
            )
    return pd.DataFrame.from_records(records)


def zero_point_drift_table(drift: pd.DataFrame) -> pd.DataFrame:
    """Per source, fit median v = a + s cz to the drift bins (weights 1/err^2).

    `a` absorbs a bulk-flow-like constant offset over the source's sky; `s`
    is the linear drift, converted to the distance-modulus offset it implies,
    dm = -(5 / ln 10) s (positive dm: the source puts galaxies too far away).

    Parameters
    ----------
    drift : pandas.DataFrame
        `velocity_drift_by_source` output.

    Returns
    -------
    pandas.DataFrame
        One row per source with >= 3 bins: `n_bins`, `n_objects`, `offset`
        and `offset_err` (km/s), `slope` and `slope_err` (per km/s),
        `modulus_offset` and `modulus_offset_err` (mag), `chi2`, `dof`.
    """
    records = []
    for name, rows in drift.groupby("source", sort=False):
        # Two parameters plus at least one degree of freedom.
        if len(rows) < 3:
            continue
        x = rows["cz_median"].to_numpy()
        y = rows["v_median"].to_numpy()
        w = 1.0 / rows["v_median_err"].to_numpy() ** 2
        design = np.column_stack([np.ones_like(x), x])
        fisher = design.T @ (w[:, None] * design)
        covariance = np.linalg.inv(fisher)
        (offset, slope) = covariance @ (design.T @ (w * y))
        residual = y - design @ np.array([offset, slope])
        # 5 / ln 10: d DM / d ln D, from D = 10^((DM - 25) / 5).
        to_modulus = -5.0 / np.log(10.0)
        records.append(
            {
                "source": name,
                "n_bins": len(rows),
                "n_objects": int(rows["n"].sum()),
                "offset": offset,
                "offset_err": float(np.sqrt(covariance[0, 0])),
                "slope": slope,
                "slope_err": float(np.sqrt(covariance[1, 1])),
                "modulus_offset": to_modulus * slope,
                "modulus_offset_err": abs(to_modulus) * float(np.sqrt(covariance[1, 1])),
                "chi2": float(np.sum(w * residual**2)),
                "dof": len(rows) - 2,
            }
        )
    return pd.DataFrame.from_records(records).set_index("source")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _finish(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, color=_LABEL_COLOR)
    ax.set_ylabel(ylabel, color=_LABEL_COLOR)
    ax.set_title(title, color=_LABEL_COLOR)
    ax.grid(alpha=_GRID_ALPHA, lw=_GRID_LINE_WIDTH)


def _sky_xy(ra_deg: np.ndarray, de_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(RA, DE) degrees -> Mollweide (x, y) radians, RA growing to the LEFT."""
    ra = np.asarray(ra_deg, dtype=float)
    # Wrap RA to (-180, 180] and negate so RA grows leftwards on the plot.
    x = -np.deg2rad(np.mod(ra + 180.0, 360.0) - 180.0)
    return x, np.deg2rad(np.asarray(de_deg, dtype=float))


def _sky_axes(fig: plt.Figure) -> plt.Axes:
    """A Mollweide axis with RA tick labels in degrees, sky convention."""
    ax = fig.add_subplot(111, projection="mollweide")
    # Every _SKY_TICK_STEP_DEG of RA, short of the +-180 edges.
    ticks = np.arange(-180 + _SKY_TICK_STEP_DEG, 180, _SKY_TICK_STEP_DEG)
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels(
        [f"{int(np.mod(-t, 360))}°" for t in ticks],
        color=_LABEL_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": _TICK_BACKING_ALPHA, "pad": _TICK_BACKING_PAD},
    )
    ax.grid(alpha=_GRID_ALPHA, lw=_GRID_LINE_WIDTH)
    return ax


def _draw_galactic_latitude(ax: plt.Axes, latitude_deg: float, **style) -> None:
    """Draw the curve b = `latitude_deg` in equatorial Mollweide coordinates,
    broken where it wraps across the RA = 180 deg edge."""
    glon = np.linspace(0.0, 360.0, _PLANE_SAMPLES)
    ra, de = hp.Rotator(coord=["G", "C"])(glon, np.full_like(glon, latitude_deg), lonlat=True)
    x, y = _sky_xy(ra, de)
    # A jump of more than half the plot width is the wrap, not the curve.
    breaks = np.flatnonzero(np.abs(np.diff(x)) > np.pi) + 1
    for xs, ys in zip(np.split(x, breaks), np.split(y, breaks)):
        ax.plot(xs, ys, **style)


def make_sky_map(
    ra_deg: np.ndarray, de_deg: np.ndarray, *, title: str, color: str = _COLOR_PRIMARY
) -> plt.Figure:
    """Equatorial Mollweide map of a point set, RA increasing to the LEFT.

    Sky convention (as seen from inside the sphere): RA = 0 at the center,
    180 deg at both edges. Tick labels are RA in degrees.

    Parameters
    ----------
    ra_deg, de_deg : array_like, shape (N,)
    title : str, keyword-only
    color : str, keyword-only
    """
    x, y = _sky_xy(ra_deg, de_deg)
    fig = plt.figure(figsize=_SKY_FIGSIZE)
    ax = _sky_axes(fig)
    ax.scatter(x, y, s=_SKY_POINT_SIZE, color=color, alpha=_SKY_POINT_ALPHA,
               lw=0, rasterized=True)
    ax.set_title(f"{title} (N = {x.size:,})", color=_LABEL_COLOR, pad=_SKY_TITLE_PAD)
    fig.tight_layout()
    return fig


def make_overlap_sky_map(
    samples: list[SkySample],
    *,
    title: str,
    cz_max: float | None = None,
    footprint: TwoMrsFootprint | None = None,
) -> plt.Figure:
    """Several catalogs on one equatorial Mollweide map, to see their overlap.

    Drawn in list order, so the first sample is at the bottom: pass the
    densest one first. Guides: the Galactic plane (b = 0, solid) and the 2MRS
    mask edge at |b| = `footprint.min_abs_latitude` (dashed; the wider bulge
    cut is not drawn).

    Parameters
    ----------
    samples : list of SkySample
    title : str, keyword-only
    cz_max : float, keyword-only, optional
        km/s. Plot only rows with cz below it, each in its own frame.
    footprint : TwoMrsFootprint, keyword-only, optional
    """
    f = footprint or TwoMrsFootprint()
    fig = plt.figure(figsize=_SKY_LARGE_FIGSIZE)
    ax = _sky_axes(fig)
    for s in samples:
        keep = np.ones(s.cz.size, bool) if cz_max is None else s.cz < cz_max
        x, y = _sky_xy(s.ra[keep], s.de[keep])
        ax.scatter(x, y, s=_OVERLAP_POINT_SIZE, color=s.color, alpha=_OVERLAP_POINT_ALPHA,
                   lw=0, rasterized=True, label=f"{s.name} (N = {x.size:,})")
    _draw_galactic_latitude(ax, 0.0, color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH)
    for sign in (1.0, -1.0):
        _draw_galactic_latitude(ax, sign * f.min_abs_latitude, color=_COLOR_GUIDE,
                                lw=_GUIDE_LINE_WIDTH, ls="--")
    legend = fig.legend(loc="lower center", ncol=len(samples), frameon=False,
                        markerscale=_LEGEND_MARKER_SCALE)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    ax.set_title(title, color=_LABEL_COLOR, pad=_SKY_TITLE_PAD)
    fig.tight_layout(rect=(0.0, _OVERLAP_LEGEND_STRIP, 1.0, 1.0))
    return fig


def make_cz_histogram_figure(
    samples: list[SkySample], config: AuditConfig | None = None
) -> plt.Figure:
    """n(cz) of each sample in its own panel, on a shared cz axis.

    Separate panels with their own count axes (not one dual-axis plot): the
    catalogs differ several-fold in size, and the question is each one's
    shape. Each panel's x label states its velocity frame; a sample's
    `cz_limit` is drawn as a dashed guide.
    """
    config = config or AuditConfig()
    width = config.histogram_bin_width
    lo = width * np.floor(min(s.cz.min() for s in samples) / width)
    hi = width * np.ceil(max(s.cz.max() for s in samples) / width)
    bins = np.arange(lo, hi + width, width)
    fig, axes = plt.subplots(len(samples), 1, figsize=(_HIST_WIDTH, _HIST_PANEL_HEIGHT * len(samples)),
                             sharex=True, squeeze=False)
    for ax, s in zip(axes[:, 0], samples):
        ax.hist(s.cz, bins=bins, color=s.color, alpha=_HIST_ALPHA)
        if s.cz_limit is not None:
            ax.axvline(s.cz_limit, color=_COLOR_GUIDE, ls="--", lw=_GUIDE_LINE_WIDTH,
                       label=f"survey limit, cz = {s.cz_limit:,.0f} km/s")
            ax.legend(frameon=False)
        _finish(ax, "", f"{s.unit} per bin", f"{s.name} (N = {s.cz.size:,}; cz {s.cz_frame})")
    axes[-1, 0].set_xlabel("cz [km/s]", color=_LABEL_COLOR)
    axes[-1, 0].set_xlim(left=bins[0])
    fig.tight_layout()
    return fig


def make_vpec_vs_cz_figure(
    velocities: EddCatalog, noise: pd.DataFrame, config: AuditConfig | None = None
) -> plt.Figure:
    """Published Vpec against cz, with the binned median, the observed ±1σ,
    and the ±σ_u that eDM predicts for a velocity on the natural-log scale.

    Parameters
    ----------
    velocities : EddCatalog
    noise : pandas.DataFrame
        `velocity_noise_by_depth` output (gives the per-bin curves).
    config : AuditConfig, optional
    """
    config = config or AuditConfig()
    t = velocities.table
    fig, ax = plt.subplots(figsize=_VPEC_FIGSIZE)
    hb = ax.hexbin(t["Vcmb"], t["Vpec"], gridsize=_HEXBIN_GRID, bins="log", cmap=_HEXBIN_CMAP,
                   mincnt=1, extent=(t["Vcmb"].min(), t["Vcmb"].max(),
                                     -config.velocity_plot_limit, config.velocity_plot_limit))
    fig.colorbar(hb, ax=ax, label="groups per cell")
    x = noise["cz_median"]
    ax.plot(x, noise["vpec_median"], color=_COLOR_SECONDARY, lw=_LINE_WIDTH, marker="o",
            ms=_MARKER_SIZE, label="median Vpec")
    for sign, label in ((1, "observed ±1σ of Vpec"), (-1, None)):
        ax.plot(x, sign * noise["std_vpec_published"], color=_COLOR_SECONDARY, lw=_LINE_WIDTH,
                ls="--", label=label)
    for sign, label in ((1, r"±$\sigma_u$ predicted from eDM"), (-1, None)):
        ax.plot(x, sign * noise["sigma_u_rms"], color=_COLOR_TERTIARY, lw=_LINE_WIDTH,
                ls=":", label=label)
    ax.axhline(0.0, color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH)
    ax.set_ylim(-config.velocity_plot_limit, config.velocity_plot_limit)
    ax.legend(frameon=False, loc="lower left")
    _finish(ax, r"$cz_{\rm CMB}$ = Vcmb [km/s]", "Vpec (published) [km/s]",
            "CF4 peculiar velocity against depth")
    fig.tight_layout()
    return fig


def make_velocity_definition_figure(
    velocities: EddCatalog, config: AuditConfig | None = None
) -> plt.Figure:
    """Published Vpwf against Vpds: the slope says what scale Vpwf is on.

    Two estimators of one velocity must lie on the slope-1 line where |v| is
    small. Lines at slope 1 and slope 1/ln 10 show which one the data follow.
    """
    config = config or AuditConfig()
    t = velocities.table
    lim = config.velocity_plot_limit
    fig, ax = plt.subplots(figsize=_DEFINITION_FIGSIZE)
    hb = ax.hexbin(t["Vpds"], t["Vpwf"], gridsize=_HEXBIN_GRID, bins="log", cmap=_HEXBIN_CMAP,
                   mincnt=1, extent=(-lim, lim, -lim, lim))
    fig.colorbar(hb, ax=ax, label="groups per cell")
    line = np.array([-lim, lim])
    ax.plot(line, line, color=_COLOR_SECONDARY, lw=_LINE_WIDTH, label="slope 1 (same velocity)")
    ax.plot(line, line / np.log(10.0), color=_COLOR_TERTIARY, lw=_LINE_WIDTH, ls="--",
            label=r"slope 1/ln 10 = 0.434")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(frameon=False, loc="upper left")
    _finish(ax, "Vpds (published) [km/s]", "Vpwf (published) [km/s]",
            "Two estimators of one velocity")
    fig.tight_layout()
    return fig


def make_noise_comparison_figure(noise: pd.DataFrame) -> plt.Figure:
    """The sigma_u puzzle in one figure: left, scatter and predicted noise in
    km/s; right, per-object std(v / σ_u), with guides at 1 and 1/ln 10.
    """
    x = noise["cz_median"]
    series = (
        ("std_vpec_published", "Vpec (published)", _COLOR_SECONDARY, "o"),
        ("std_vpwf_ln", "Vpwf rebuilt with ln", _COLOR_PRIMARY, "s"),
        ("std_vpds", "Vpds (published)", _COLOR_TERTIARY, "^"),
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=_NOISE_FIGSIZE)
    left.plot(x, noise["sigma_u_rms"], color=_LABEL_COLOR, lw=_LINE_WIDTH, ls=":", marker="D",
              ms=_MARKER_SIZE, label=r"predicted rms $\sigma_u$ (eDM)")
    for column, label, color, marker in series:
        left.plot(x, noise[column], color=color, lw=_LINE_WIDTH, marker=marker,
                  ms=_MARKER_SIZE, label=f"observed std, {label}")
    left.set_xlim(left=0)
    left.set_ylim(bottom=0)
    left.legend(frameon=False)
    _finish(left, r"median $cz_{\rm CMB}$ of bin [km/s]", "km/s", "Scatter against predicted noise")
    for column, label, color, marker in series:
        right.plot(x, noise[column.replace("std_", "zstd_")], color=color, lw=_LINE_WIDTH,
                   marker=marker, ms=_MARKER_SIZE, label=label)
    right.axhline(1.0, color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH, label="1: eDM explains the scatter")
    right.axhline(1.0 / np.log(10.0), color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH, ls="--",
                  label="1/ln 10: column ln 10 too small")
    right.set_xlim(left=0)
    right.set_ylim(bottom=0)
    right.legend(frameon=False)
    _finish(right, r"median $cz_{\rm CMB}$ of bin [km/s]", r"std($v\,/\,\sigma_u$)",
            "Per-object normalized scatter")
    fig.tight_layout()
    return fig


def make_velocity_test_histogram_figure(
    table: pd.DataFrame,
    config: AuditConfig | None = None,
    labels: tuple[str, ...] = VELOCITY_TEST_LABELS,
) -> plt.Figure:
    """Watkins & Feldman's Fig. 1 test for each of the four velocities `labels`.

    One panel per `labels` entry, on a 2 x 2 grid: the histogram of v / sigma_u
    over noise-dominated objects (sigma_u > `noise_dominated_sigma`),
    normalized to unit area, against the unit Gaussian (solid) the right
    velocity must follow and the width-1/ln 10 Gaussian (dashed) a log10
    velocity follows. Each panel title gives its std.

    Parameters
    ----------
    table : pandas.DataFrame
        `velocity_test_table` output.
    config : AuditConfig, optional
    labels : tuple of four str, optional
        Default `VELOCITY_TEST_LABELS`.
    """
    config = config or AuditConfig()
    noisy = table["sigma_u"].to_numpy() > config.noise_dominated_sigma
    sigma = table["sigma_u"].to_numpy()[noisy]
    lim = config.normalized_velocity_limit
    bins = np.linspace(-lim, lim, config.normalized_velocity_bins + 1)
    x = np.linspace(-lim, lim, _GAUSSIAN_SAMPLES)
    narrow = 1.0 / np.log(10.0)
    # Unit-area Gaussians: exp(-x^2 / 2 s^2) / (s sqrt(2 pi)).
    unit = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    log10_scale = np.exp(-0.5 * (x / narrow) ** 2) / (narrow * np.sqrt(2.0 * np.pi))
    fig, axes = plt.subplots(2, 2, figsize=_VELOCITY_HIST_FIGSIZE, sharex=True, sharey=True)
    for ax, label in zip(axes.flat, labels, strict=True):
        z = table[label].to_numpy()[noisy] / sigma
        ax.hist(z, bins=bins, density=True, color=_VELOCITY_TEST_COLORS[label], alpha=_HIST_ALPHA)
        ax.plot(x, unit, color=_LABEL_COLOR, lw=_LINE_WIDTH, label="unit Gaussian")
        ax.plot(x, log10_scale, color=_COLOR_GUIDE, lw=_LINE_WIDTH, ls="--",
                label="Gaussian, width 1/ln 10")
        _finish(ax, "", "", f"{label}: std = {np.std(z):.3f}")
    for ax in axes[-1]:
        ax.set_xlabel(r"$v\,/\,\sigma_u$", color=_LABEL_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("probability density", color=_LABEL_COLOR)
    axes[0, 0].legend(frameon=False, fontsize="small")
    fig.suptitle(rf"CF4 groups with $\sigma_u$ > {config.noise_dominated_sigma:,.0f} km/s "
                 f"(N = {int(noisy.sum()):,})", color=_LABEL_COLOR)
    fig.tight_layout()
    return fig


def make_velocity_test_depth_figure(
    by_depth: pd.DataFrame, labels: tuple[str, ...] = VELOCITY_TEST_LABELS
) -> plt.Figure:
    """std(v / sigma_u) against depth for the velocities `labels`, guides at 1
    and 1/ln 10.

    Parameters
    ----------
    by_depth : pandas.DataFrame
        `velocity_test_by_depth(..., labels=labels)` output.
    labels : tuple of str, optional
        Default `VELOCITY_TEST_LABELS`.
    """
    x = by_depth["cz_median"]
    fig, ax = plt.subplots(figsize=_VPEC_FIGSIZE)
    for label in labels:
        color = _VELOCITY_TEST_COLORS[label]
        dashed = label in _VELOCITY_TEST_DASHED
        ax.plot(x, by_depth[label], color=color, lw=_LINE_WIDTH, ls="--" if dashed else "-",
                marker="o", ms=_MARKER_SIZE, mfc="none" if dashed else color, label=label)
    ax.axhline(1.0, color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH, label="1: eDM explains the scatter")
    ax.axhline(1.0 / np.log(10.0), color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH, ls="--",
               label="1/ln 10: column ln 10 too small")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    _finish(ax, r"median $cz_{\rm CMB}$ of bin [km/s]", r"std($v\,/\,\sigma_u$)",
            "Normalized scatter of the four velocities")
    fig.tight_layout()
    return fig


def make_velocity_test_scatter_figure(
    table: pd.DataFrame,
    binned: pd.DataFrame,
    x_column: str,
    config: AuditConfig | None = None,
    labels: tuple[str, ...] = VELOCITY_TEST_LABELS,
) -> plt.Figure:
    """The four velocities `labels` against cz or measured distance, one panel each.

    Each panel: a hexbin of every group (grey), the binned median (solid)
    and median ± observed σ (dashed) in the velocity's color, and median ±
    the σ_u that eDM predicts for the ln velocity (black dotted). Both
    bands are centered on the median, so their widths compare apart from
    where the median goes (see `velocity_test_binned` for why it moves on
    the distance axis). Shared axes: for the right velocity the dashed and
    dotted bands coincide; for a log10 one the dashed band sits ln 10 inside.

    Parameters
    ----------
    table : pandas.DataFrame
        `velocity_test_table` output.
    binned : pandas.DataFrame
        `velocity_test_binned(table, x_column, config, labels)` output.
    x_column : str
        `VELOCITY_TEST_X_REDSHIFT` or `VELOCITY_TEST_X_DISTANCE`.
    config : AuditConfig, optional
    labels : tuple of four str, optional
        Default `VELOCITY_TEST_LABELS`; filled into a 2 x 2 grid.
    """
    config = config or AuditConfig()
    lim = config.scatter_velocity_limit
    x_max = _velocity_test_edges(x_column, config)[-1]
    extent = (0.0, x_max, -lim, lim)
    fig, axes = plt.subplots(2, 2, figsize=_VELOCITY_HIST_FIGSIZE, sharex=True, sharey=True)
    for ax, label in zip(axes.flat, labels, strict=True):
        color = _VELOCITY_TEST_COLORS[label]
        ax.hexbin(table[x_column], table[label], gridsize=_HEXBIN_GRID, bins="log",
                  cmap=_SCATTER_HEXBIN_CMAP, mincnt=1, extent=extent)
        rows = binned[binned["velocity"] == label]
        x = rows["x_median"]
        ax.plot(x, rows["median"], color=color, lw=_LINE_WIDTH, label="median")
        for sign, name in ((1, "median ± observed σ"), (-1, None)):
            ax.plot(x, rows["median"] + sign * rows["std"], color=color, lw=_LINE_WIDTH, ls="--",
                    label=name)
        for sign, name in ((1, r"median ± $\sigma_u$ predicted (eDM)"), (-1, None)):
            ax.plot(x, rows["median"] + sign * rows["sigma_u_rms"], color=_LABEL_COLOR,
                    lw=_LINE_WIDTH, ls=":", label=name)
        ax.axhline(0.0, color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH)
        _finish(ax, "", "", label)
    for ax in axes[-1]:
        ax.set_xlabel(_VELOCITY_TEST_X_AXIS_LABELS[x_column], color=_LABEL_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("peculiar velocity [km/s]", color=_LABEL_COLOR)
    axes[0, 0].set_xlim(0.0, x_max)
    axes[0, 0].set_ylim(-lim, lim)
    axes[0, 0].legend(frameon=False, fontsize="small", loc="lower left")
    fig.suptitle(f"CF4 groups: the four velocities against {_VELOCITY_TEST_X_NAMES[x_column]} "
                 f"(N = {len(table):,})", color=_LABEL_COLOR)
    fig.tight_layout()
    return fig


def make_source_drift_figure(drift: pd.DataFrame, fits: pd.DataFrame) -> plt.Figure:
    """Median ln-scale peculiar velocity against cz, one line per CF4 source.

    Points at each bin's median cz with sqrt(pi/2) sigma/sqrt(n) error bars;
    the dashed line of the same color is the `zero_point_drift_table` fit
    a + s cz. A source whose line tilts while the others stay flat has a
    zero-point offset of its own.

    Parameters
    ----------
    drift : pandas.DataFrame
        `velocity_drift_by_source` output.
    fits : pandas.DataFrame
        `zero_point_drift_table` output.
    """
    fig, ax = plt.subplots(figsize=_DRIFT_FIGSIZE)
    for name, rows in drift.groupby("source", sort=False):
        color = _SOURCE_COLORS[name]
        ax.errorbar(rows["cz_median"], rows["v_median"], yerr=rows["v_median_err"], color=color,
                    lw=_LINE_WIDTH, marker="o", ms=_MARKER_SIZE, capsize=_ERRORBAR_CAPSIZE,
                    label=f"{name} ({int(rows['n'].sum()):,})")
        if name in fits.index:
            fit = fits.loc[name]
            x = np.array([0.0, rows["cz_hi"].max()])
            ax.plot(x, fit["offset"] + fit["slope"] * x, color=color, lw=_GUIDE_LINE_WIDTH, ls="--")
    ax.axhline(0.0, color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize="small")
    _finish(ax, r"$cz_{\rm CMB}$ = Vcmb [km/s]", "median peculiar velocity (ln form) [km/s]",
            "CF4 single-method groups: median velocity by source")
    fig.tight_layout()
    return fig
