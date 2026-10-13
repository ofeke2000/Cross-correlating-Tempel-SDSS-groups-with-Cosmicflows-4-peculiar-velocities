"""
real_catalog_audit.py
---------------------
Stage 1 of `docs/real_data_roadmap.md`: the checks and figures that say
whether the raw EDD tables (`dvcorr.pipeline.real_catalogs`) are what we think
they are. Consumed by `notebooks/11_real_catalog_audit.ipynb`; nothing
downstream imports it.

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

Figures: Mollweide sky maps, n(cz), Vpec against cz, the Vpwf-against-Vpds
scale test and the noise-against-scatter comparison. Colors are categorical
slots 1-3 of the project's palette (the same blue/orange as
`mass_diagnostics`), in fixed order.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dvcorr import conventions
from dvcorr.pipeline.real_catalogs import (
    Cf4VelocityDefinitions,
    EddCatalog,
    angular_separation_deg,
    curvature_corrected_velocity,
    davis_scrimgeour_velocity,
    distance_from_modulus,
    format_half_ulp,
    published_vpec,
    velocity_error_from_modulus_error,
    watkins_feldman_velocity,
)

# Plot styling -- named here rather than inline (hard rule 4).
_COLOR_PRIMARY = "#2a78d6"
_COLOR_SECONDARY = "#eb6834"
_COLOR_TERTIARY = "#1baf7a"
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
_HIST_FIGSIZE = (8.0, 6.0)
_VPEC_FIGSIZE = (9.0, 5.5)
_DEFINITION_FIGSIZE = (6.5, 6.0)
_NOISE_FIGSIZE = (12.0, 4.8)

#: A missing Tempel group id: every isolated galaxy carries IDg = 0.
TEMPEL_ISOLATED_IDG: int = 0
#: The brightest-member rank in Tempel's `Rg` (the fiducial group center).
TEMPEL_BRIGHTEST_RANK: int = 1


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
    rows = (
        (fv > 0)
        & (np.abs(vpds) < config.linear_regime_fraction * fv)
        & (np.abs(vpds) > config.linear_regime_floor)
    )
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
# Figures
# --------------------------------------------------------------------------


def _finish(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, color=_LABEL_COLOR)
    ax.set_ylabel(ylabel, color=_LABEL_COLOR)
    ax.set_title(title, color=_LABEL_COLOR)
    ax.grid(alpha=_GRID_ALPHA, lw=_GRID_LINE_WIDTH)


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
    ra = np.asarray(ra_deg, dtype=float)
    # Wrap RA to (-180, 180] and negate so RA grows leftwards on the plot.
    x = -np.deg2rad(np.mod(ra + 180.0, 360.0) - 180.0)
    y = np.deg2rad(np.asarray(de_deg, dtype=float))
    fig = plt.figure(figsize=_SKY_FIGSIZE)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.scatter(x, y, s=_SKY_POINT_SIZE, color=color, alpha=_SKY_POINT_ALPHA,
               lw=0, rasterized=True)
    # Every _SKY_TICK_STEP_DEG of RA, short of the +-180 edges.
    ticks = np.arange(-180 + _SKY_TICK_STEP_DEG, 180, _SKY_TICK_STEP_DEG)
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels(
        [f"{int(np.mod(-t, 360))}°" for t in ticks],
        color=_LABEL_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": _TICK_BACKING_ALPHA, "pad": _TICK_BACKING_PAD},
    )
    ax.grid(alpha=_GRID_ALPHA, lw=_GRID_LINE_WIDTH)
    ax.set_title(f"{title} (N = {ra.size:,})", color=_LABEL_COLOR, pad=_SKY_TITLE_PAD)
    fig.tight_layout()
    return fig


def make_cz_histogram_figure(
    tempel: EddCatalog, cf4_velocities: EddCatalog, config: AuditConfig | None = None
) -> plt.Figure:
    """n(cz) of Tempel galaxies (zcmb * c) and CF4 groups (Vcmb), two panels.

    Separate panels with their own count axes (not one dual-axis plot): the
    catalogs differ eightfold in size, and the question is each one's shape.
    """
    config = config or AuditConfig()
    tempel_cz = tempel.table["zcmb"].to_numpy() * conventions.SPEED_OF_LIGHT
    cf4_cz = cf4_velocities.table["Vcmb"].to_numpy(dtype=float)
    lo = config.histogram_bin_width * np.floor(min(tempel_cz.min(), cf4_cz.min()) / config.histogram_bin_width)
    hi = config.histogram_bin_width * np.ceil(max(tempel_cz.max(), cf4_cz.max()) / config.histogram_bin_width)
    bins = np.arange(lo, hi + config.histogram_bin_width, config.histogram_bin_width)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=_HIST_FIGSIZE, sharex=True)
    top.hist(tempel_cz, bins=bins, color=_COLOR_PRIMARY, alpha=_HIST_ALPHA)
    top.axvline(config.tempel_redshift_limit * conventions.SPEED_OF_LIGHT, color=_COLOR_GUIDE,
                ls="--", lw=_GUIDE_LINE_WIDTH, label=f"z = {config.tempel_redshift_limit}")
    top.legend(frameon=False)
    _finish(top, "", "galaxies per bin", f"Tempel galaxies (N = {tempel_cz.size:,})")
    bottom.hist(cf4_cz, bins=bins, color=_COLOR_SECONDARY, alpha=_HIST_ALPHA)
    _finish(bottom, r"$cz_{\rm CMB}$ [km/s]", "groups per bin",
            f"CF4 groups (N = {cf4_cz.size:,})")
    bottom.set_xlim(left=bins[0])
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
