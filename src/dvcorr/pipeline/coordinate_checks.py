"""
coordinate_checks.py
--------------------
Stage 3's checks (`docs/real_data_roadmap.md`) that Tempel, CF4 and 2MRS
positions from `dvcorr.pipeline.sky_positions` are one coordinate system:
the tables and the figures. Computes nothing the analysis uses -- the
positions are `sky_positions`' -- and only notebook 14 imports it, as only
the audit notebooks import `real_catalog_audit`.

- **Check 0** (`apex_fit_table`, `apex_agreement`,
  `apex_by_redshift_table`): the dipole behind Tempel's `zcmb - zobs` and
  CF4's `Vcmb - Vhel`, in the additive and the exact form, over all z and
  refitted per cz bin; the two apexes must agree.
- **Check 1** (`rebuild_tempel_dc`, `dc_rebuild_summary`): `dc` rebuilt as
  the mean member chi(`zcmb`) in Mpc. Agreement to rounding wherever every
  member row is present; a fixed fractional offset would mean another h or
  Omega_m.
- **Check 2** (`tempel_cf4_matches`): Tempel galaxies matched by `pgc` to CF4
  groups; residuals at galaxy level (Tempel's own galaxy redshift) and group
  level (s_T from `dc` against s_V).
- **Check 3** (`twomrs_tempel_matches`, `twomrs_cf4_matches`): s_M against
  both.

Every match table carries `cz` (km/s, CMB) and `cos_apex` (cosine to the
Tempel apex), the two axes a trend is looked for along: a residual growing
with cz is a distance-convention mismatch, one growing with cz x cos_apex a
CMB-formula or apex mismatch. `binned_residual` and `match_residual_summary`
measure the trends; `residual_trend_slope` fits them.

Residual signs: the catalog being checked minus its reference -- CF4 minus
Tempel, 2MRS minus Tempel or CF4. Velocities in km/s, distances in h^-1 Mpc.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dvcorr import conventions
from dvcorr.config.cosmology import (
    CosmologyConfig,
    comoving_distance_derivative_h_units,
    comoving_distance_h_units,
)
from dvcorr.pipeline.real_catalogs import TEMPEL_ISOLATED_IDG, EddCatalog, format_half_ulp
from dvcorr.pipeline.sky_positions import (
    CMB_FORM_ADDITIVE,
    CMB_FORM_EXACT,
    VelocityDipoleFit,
    angle_between_deg,
    cmb_correction_velocity,
    comoving_distance_or_nan,
    fit_cf4_cmb_apex,
    fit_tempel_cmb_apex,
    fit_velocity_dipole,
    galactic_unit_vector,
)

# Figure style, matching `real_catalog_audit`'s palette.
_COLOR_PRIMARY = "#2a78d6"
_COLOR_SECONDARY = "#eb6834"
_COLOR_TERTIARY = "#1baf7a"
_COLOR_QUATERNARY = "#8a5cc2"
_COLOR_GUIDE = "0.45"
_LABEL_COLOR = "#111111"
_HEXBIN_CMAP = "Greys"
_HEXBIN_GRID = 90
_GRID_ALPHA = 0.25
_GRID_LINE_WIDTH = 0.6
_GUIDE_LINE_WIDTH = 1.0
_LINE_WIDTH = 1.8
_MARKER_SIZE = 5.0
_ERRORBAR_CAPSIZE = 3.0
_BAND_ALPHA = 0.2
_HIST_LINE_WIDTH = 1.6
_PANEL_WIDTH = 5.2
_PANEL_HEIGHT = 3.6

#: 1.4826: MAD -> sigma for a Gaussian (1 / Phi^-1(3/4)).
_MAD_TO_SIGMA = 1.4826
#: sqrt(pi / 2): the standard error of a median is this times sigma / sqrt(n)
#: for a Gaussian parent.
_MEDIAN_ERROR_FACTOR = float(np.sqrt(np.pi / 2.0))


@dataclass(frozen=True)
class CoordinateCheckConfig:
    """Tolerances and binning for the Stage 3 checks.

    Attributes
    ----------
    cf4_apex_min_cz : float
        km/s. Below this heliocentric cz CF4's `Vcmb - Vhel` is not a pure
        dipole (rms 50-76 km/s per bin); above it, it is one to integer
        rounding. Check 0 fits CF4 both over every group and above this.
    apex_angle_tolerance_deg : float
        Check 0 passes when the Tempel and CF4 apexes are closer than this
        (the roadmap's "within about 1 deg").
    apex_amplitude_tolerance : float
        km/s; and their amplitudes closer than this ("a few km/s").
    cz_edges : tuple of float
        km/s. Bins of CMB cz for the residual trends; the last edge is past
        CF4's deepest group and Tempel's z = 0.2 limit.
    cos_bins : int
        Equal-width bins of cos_apex over [-1, 1].
    min_bin_count : int
        A bin with fewer objects is left out of the trends.
    cz_slope_unit : float
        km/s. Trend slopes against cz are quoted per this much cz.
    richness_edges : tuple of int
        `Ng` class lower edges for the group-level scatter by richness; the
        last class is open.
    plot_percentile : float
        Scatter panels show the central this-percent of the residual
        (symmetric about zero).
    histogram_bins : int
        Bins of the radius and |dc residual| histograms.
    dc_bins : int
        Bins of `dc` for the Check 1 residual-against-distance panel.
    """

    cf4_apex_min_cz: float = 5000.0
    apex_angle_tolerance_deg: float = 1.0
    apex_amplitude_tolerance: float = 5.0
    cz_edges: tuple[float, ...] = (
        0.0, 2500.0, 5000.0, 7500.0, 10000.0, 12500.0, 15000.0, 17500.0,
        20000.0, 22500.0, 25000.0, 27500.0, 30000.0, 32500.0, 60000.0,
    )
    cos_bins: int = 10
    min_bin_count: int = 20
    cz_slope_unit: float = 10000.0
    richness_edges: tuple[int, ...] = (1, 2, 6, 20)
    plot_percentile: float = 98.0
    histogram_bins: int = 60
    dc_bins: int = 15

    @property
    def cos_edges(self) -> np.ndarray:
        """`cos_bins` + 1 equal-width edges over [-1, 1]."""
        return np.linspace(-1.0, 1.0, self.cos_bins + 1)


# --------------------------------------------------------------------------
# Check 0: the CMB apex
# --------------------------------------------------------------------------


#: `apex_fit_table` key of the CF4 fit beyond `cf4_apex_min_cz`.
CF4_DEEP_ADDITIVE = "CF4, additive, beyond cf4_apex_min_cz"
CF4_DEEP_EXACT = "CF4, exact, beyond cf4_apex_min_cz"


def apex_fit_table(
    tempel: EddCatalog, cf4_velocities: EddCatalog, config: CoordinateCheckConfig | None = None
) -> tuple[pd.DataFrame, dict]:
    """The dipole fits of Check 0 and the fits themselves.

    Returns
    -------
    table : pandas.DataFrame
        Indexed by fit name ("Tempel, additive", "Tempel, exact", "CF4,
        additive", "CF4, exact", and the two CF4 fits beyond
        `config.cf4_apex_min_cz`, `CF4_DEEP_ADDITIVE` and `CF4_DEEP_EXACT`);
        columns `amplitude` (km/s), `glon`, `glat` (deg), `rms` (km/s),
        `n_objects`, `angle_to_tempel` (deg, to the Tempel additive apex).
    fits : dict of str -> VelocityDipoleFit
        Same keys.
    """
    config = config or CoordinateCheckConfig()
    deep = config.cf4_apex_min_cz
    fits = {
        "Tempel, additive": fit_tempel_cmb_apex(tempel, CMB_FORM_ADDITIVE),
        "Tempel, exact": fit_tempel_cmb_apex(tempel, CMB_FORM_EXACT),
        "CF4, additive": fit_cf4_cmb_apex(cf4_velocities, CMB_FORM_ADDITIVE),
        "CF4, exact": fit_cf4_cmb_apex(cf4_velocities, CMB_FORM_EXACT),
        CF4_DEEP_ADDITIVE: fit_cf4_cmb_apex(cf4_velocities, CMB_FORM_ADDITIVE, min_cz=deep),
        CF4_DEEP_EXACT: fit_cf4_cmb_apex(cf4_velocities, CMB_FORM_EXACT, min_cz=deep),
    }
    reference = fits["Tempel, additive"].vector
    table = pd.DataFrame.from_records(
        [
            {
                "fit": name,
                "amplitude": f.amplitude,
                "glon": f.glon,
                "glat": f.glat,
                "rms": f.rms,
                "n_objects": f.n_objects,
                "angle_to_tempel": angle_between_deg(f.vector, reference),
            }
            for name, f in fits.items()
        ]
    ).set_index("fit")
    return table, fits


def dipole_amplitude_by_redshift(
    z_hel: np.ndarray,
    z_cmb: np.ndarray,
    glon_deg: np.ndarray,
    glat_deg: np.ndarray,
    form: str,
    edges: np.ndarray,
    min_count: int,
) -> pd.DataFrame:
    """The CMB-correction dipole refitted in bins of redshift.

    Under the form that built the catalog the amplitude is the same in every
    bin. Under the other it drifts by ~ z V, since the two forms differ by a
    factor (1 + z): the trend with z that a single all-z fit hides by
    averaging it into one amplitude (and leaves in its residual as
    ~ (z - <z>) V.n, zero on average in every bin).

    Parameters
    ----------
    z_hel, z_cmb, glon_deg, glat_deg : array_like, shape (N,)
    form : str
        `CMB_FORM_ADDITIVE` or `CMB_FORM_EXACT`.
    edges : array_like, shape (B + 1,)
        Redshift bin edges (bins on `z_hel`).
    min_count : int
        Bins with fewer objects are dropped.

    Returns
    -------
    pandas.DataFrame
        One row per kept bin: `lo`, `hi`, `z_median`, `n`, `amplitude`,
        `glon`, `glat`, `rms`.
    """
    z_hel = np.asarray(z_hel, dtype=float)
    dv = cmb_correction_velocity(z_hel, z_cmb, form)
    n_hat = galactic_unit_vector(glon_deg, glat_deg)
    edges = np.asarray(edges, dtype=float)
    index = np.digitize(z_hel, edges) - 1
    records = []
    for b in range(edges.size - 1):
        inside = index == b
        if inside.sum() < min_count:
            continue
        fit = fit_velocity_dipole(n_hat[inside], dv[inside])
        records.append(
            {
                "lo": edges[b], "hi": edges[b + 1], "z_median": float(np.median(z_hel[inside])),
                "n": fit.n_objects, "amplitude": fit.amplitude, "glon": fit.glon,
                "glat": fit.glat, "rms": fit.rms,
            }
        )
    return pd.DataFrame.from_records(records)


def apex_agreement(
    a: VelocityDipoleFit, b: VelocityDipoleFit, config: CoordinateCheckConfig | None = None
) -> dict:
    """Whether two apexes agree within the Check 0 tolerances.

    Returns
    -------
    dict
        `angle_deg`, `amplitude_difference` (km/s, a - b), `passes` (bool).
    """
    config = config or CoordinateCheckConfig()
    angle = angle_between_deg(a.vector, b.vector)
    d_amp = a.amplitude - b.amplitude
    return {
        "angle_deg": angle,
        "amplitude_difference": d_amp,
        "passes": bool(
            angle < config.apex_angle_tolerance_deg and abs(d_amp) < config.apex_amplitude_tolerance
        ),
    }


# --------------------------------------------------------------------------
# Check 1: dc rebuilt
# --------------------------------------------------------------------------


def rebuild_tempel_dc(
    tempel: EddCatalog,
    cosmology: CosmologyConfig | None = None,
    *,
    h: float | None = None,
    om0: float | None = None,
) -> pd.DataFrame:
    """Tempel's `dc` rebuilt as the mean over member rows of chi(`zcmb`), Mpc.

    chi in Mpc is chi_h / h. Defaults are Tempel's (`h_tempel`,
    `Om0_TEMPEL`); pass `h` or `om0` to see what another cosmology does.

    Parameters
    ----------
    tempel : EddCatalog
        `load_tempel_galaxies()`.
    cosmology : CosmologyConfig, optional
    h, om0 : float, optional

    Returns
    -------
    pandas.DataFrame
        One row per tracer, keyed as `collapse_tempel_groups` keys them:
        `IDg` (0 for a singleton), `ID` (a singleton's galaxy; for a group
        its lowest-`Rg` member), `Ng`, `n_members` (rows present),
        `complete` (`n_members` == `Ng`), `dc` (published, Mpc),
        `dc_rebuilt`, `residual` (rebuilt - published, Mpc), `allowance`
        (Mpc: `dc`'s half-ulp plus d chi/dz times `zcmb`'s half-ulp -- a mean
        of rounded values is off by no more than one rounding).
    """
    cosmology = cosmology or CosmologyConfig()
    h = cosmology.h_tempel if h is None else h
    om0 = cosmology.Om0_TEMPEL if om0 is None else om0
    t = tempel.table.sort_values(["IDg", "Rg", "ID"], kind="mergesort")
    chi_mpc = comoving_distance_h_units(t["zcmb"].to_numpy(float), om0) / h
    rows = t[["IDg", "ID", "Ng", "dc", "zcmb"]].assign(chi=chi_mpc)
    isolated = rows["IDg"] == TEMPEL_ISOLATED_IDG
    by_group = rows[~isolated].groupby("IDg", sort=True)
    groups = pd.DataFrame(
        {
            "IDg": by_group["IDg"].first(),
            "ID": by_group["ID"].first(),
            "Ng": by_group["Ng"].first(),
            "n_members": by_group.size(),
            "dc": by_group["dc"].first(),
            "dc_rebuilt": by_group["chi"].mean(),
            "zcmb": by_group["zcmb"].mean(),
        }
    ).reset_index(drop=True)
    single = rows[isolated].sort_values("ID", kind="mergesort")
    singletons = pd.DataFrame(
        {
            "IDg": single["IDg"].to_numpy(),
            "ID": single["ID"].to_numpy(),
            "Ng": single["Ng"].to_numpy(),
            "n_members": 1,
            "dc": single["dc"].to_numpy(float),
            "dc_rebuilt": single["chi"].to_numpy(float),
            "zcmb": single["zcmb"].to_numpy(float),
        }
    )
    out = pd.concat([groups, singletons], ignore_index=True)
    out["complete"] = out["n_members"] == out["Ng"]
    out["residual"] = out["dc_rebuilt"] - out["dc"]
    formats = tempel.columns["format"]
    slope_mpc = comoving_distance_derivative_h_units(out["zcmb"].to_numpy(float), om0) / h
    out["allowance"] = format_half_ulp(formats["dc"]) + slope_mpc * format_half_ulp(formats["zcmb"])
    return out.drop(columns="zcmb")


def dc_rebuild_summary(rebuilt: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check 1's numbers, one row per (variant, complete / incomplete).

    Parameters
    ----------
    rebuilt : dict of str -> DataFrame
        Variant name -> `rebuild_tempel_dc(...)`.

    Returns
    -------
    pandas.DataFrame
        Columns `variant`, `rows` ("complete" / "incomplete"), `n`,
        `median_abs` (Mpc), `p90_abs`, `max_abs`, `fraction_within_rounding`
        (|residual| <= allowance), `median_fractional` (residual / dc).
    """
    records = []
    for variant, frame in rebuilt.items():
        for label, mask in (("complete", frame["complete"]), ("incomplete", ~frame["complete"])):
            sub = frame[mask]
            absolute = sub["residual"].abs()
            records.append(
                {
                    "variant": variant,
                    "rows": label,
                    "n": len(sub),
                    "median_abs": float(absolute.median()),
                    "p90_abs": float(absolute.quantile(0.9)),
                    "max_abs": float(absolute.max()),
                    "fraction_within_rounding": float((absolute <= sub["allowance"]).mean()),
                    "median_fractional": float((sub["residual"] / sub["dc"]).median()),
                }
            )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Checks 2 and 3: pgc matches
# --------------------------------------------------------------------------


def _cos_apex(glon: pd.Series, glat: pd.Series, apex: VelocityDipoleFit) -> np.ndarray:
    return galactic_unit_vector(glon, glat) @ apex.unit_vector


def _tempel_reference(tempel: EddCatalog, cosmology: CosmologyConfig) -> pd.DataFrame:
    """Per Tempel galaxy: c zobs, c zcmb, its own chi(zcmb) and its group's s_T."""
    t = tempel.table
    c = conventions.SPEED_OF_LIGHT
    return pd.DataFrame(
        {
            "pgc": t["pgc"].to_numpy(),
            "IDg": t["IDg"].to_numpy(),
            "Ng": t["Ng"].to_numpy(),
            "cz_hel_T": t["zobs"].to_numpy(float) * c,
            "cz_cmb_T": t["zcmb"].to_numpy(float) * c,
            "s_galaxy_T": comoving_distance_or_nan(t["zcmb"].to_numpy(float), cosmology.Om0_TEMPEL),
            "s_T": t["dc"].to_numpy(float) * cosmology.h_tempel,
        }
    )


def tempel_cf4_matches(
    tempel: EddCatalog,
    cf4_velocities: EddCatalog,
    apex: VelocityDipoleFit,
    cosmology: CosmologyConfig | None = None,
) -> pd.DataFrame:
    """Check 2: Tempel galaxies whose `pgc` is a CF4 group's, CF4 minus Tempel.

    CF4 `Vcmb` and `Vhel` are averages over the CF4 (Tully 2015) group named
    by that `pgc`, so even the galaxy-level residual carries CF4's group
    averaging; `d_frame` does not (it differences two CMB corrections, both
    smooth on the sky) and is the sharpest of them.

    Returns
    -------
    pandas.DataFrame
        One row per match: `pgc`, `IDg`, `Ng`, `cz` (Tempel c zcmb, km/s),
        `cos_apex` (Tempel direction), and the residuals
        `d_cz_hel` = Vhel - c zobs, `d_cz_cmb` = Vcmb - c zcmb,
        `d_frame` = (Vcmb - Vhel) - c (zcmb - zobs) (km/s);
        `d_s_galaxy` = s_V - chi(zcmb), `d_s_group` = s_V - s_T (h^-1 Mpc).
    """
    cosmology = cosmology or CosmologyConfig()
    c = conventions.SPEED_OF_LIGHT
    ref = _tempel_reference(tempel, cosmology).assign(
        cos_apex=_cos_apex(tempel.table["glon"], tempel.table["glat"], apex)
    )
    v = cf4_velocities.table
    cf4 = pd.DataFrame(
        {
            "pgc": v["pgc"].to_numpy(),
            "Vhel": v["Vhel"].to_numpy(float),
            "Vcmb": v["Vcmb"].to_numpy(float),
            "s_V": comoving_distance_or_nan(v["Vcmb"].to_numpy(float) / c, cosmology.Om0_TEMPEL),
        }
    )
    j = ref.merge(cf4, on="pgc", how="inner", validate="one_to_one")
    return pd.DataFrame(
        {
            "pgc": j["pgc"],
            "IDg": j["IDg"],
            "Ng": j["Ng"],
            "cz": j["cz_cmb_T"],
            "cos_apex": j["cos_apex"],
            "d_cz_hel": j["Vhel"] - j["cz_hel_T"],
            "d_cz_cmb": j["Vcmb"] - j["cz_cmb_T"],
            "d_frame": (j["Vcmb"] - j["Vhel"]) - (j["cz_cmb_T"] - j["cz_hel_T"]),
            "d_s_galaxy": j["s_V"] - j["s_galaxy_T"],
            "d_s_group": j["s_V"] - j["s_T"],
        }
    )


def twomrs_tempel_matches(
    twomrs_positioned: pd.DataFrame,
    tempel: EddCatalog,
    apex: VelocityDipoleFit,
    cosmology: CosmologyConfig | None = None,
) -> pd.DataFrame:
    """Check 3, Tempel side: 2MRS galaxies matched by `pgc`, 2MRS minus Tempel.

    Galaxy against galaxy, so `d_frame` (2MRS's built correction against
    Tempel's published one) isolates the conversion: it must be rounding.

    Parameters
    ----------
    twomrs_positioned : pandas.DataFrame
        `sky_positions.twomrs_positions(...)`.

    Returns
    -------
    pandas.DataFrame
        `pgc`, `IDg`, `Ng`, `cz` (Tempel c zcmb), `cos_apex` (2MRS direction),
        `d_cz_hel` = Vhel - c zobs, `d_cz_cmb` = cz_cmb - c zcmb, `d_frame` =
        (cz_cmb - Vhel) - c (zcmb - zobs) (km/s); `d_s_galaxy` =
        s_M - chi(zcmb), `d_s_group` = s_M - s_T (h^-1 Mpc).
    """
    cosmology = cosmology or CosmologyConfig()
    ref = _tempel_reference(tempel, cosmology)
    m = twomrs_positioned.assign(
        cos_apex=_cos_apex(twomrs_positioned["glon"], twomrs_positioned["glat"], apex)
    )[["pgc", "cz_hel", "cz_cmb", "s", "cos_apex"]]
    j = m.merge(ref, on="pgc", how="inner", validate="one_to_one")
    return pd.DataFrame(
        {
            "pgc": j["pgc"],
            "IDg": j["IDg"],
            "Ng": j["Ng"],
            "cz": j["cz_cmb_T"],
            "cos_apex": j["cos_apex"],
            "d_cz_hel": j["cz_hel"] - j["cz_hel_T"],
            "d_cz_cmb": j["cz_cmb"] - j["cz_cmb_T"],
            "d_frame": (j["cz_cmb"] - j["cz_hel"]) - (j["cz_cmb_T"] - j["cz_hel_T"]),
            "d_s_galaxy": j["s"] - j["s_galaxy_T"],
            "d_s_group": j["s"] - j["s_T"],
        }
    )


def twomrs_cf4_matches(
    twomrs_positioned: pd.DataFrame,
    cf4_velocities: EddCatalog,
    apex: VelocityDipoleFit,
    cosmology: CosmologyConfig | None = None,
) -> pd.DataFrame:
    """Check 3, CF4 side: 2MRS galaxies whose `pgc` is a CF4 group's,
    2MRS minus CF4. CF4 is group-averaged, so there is no galaxy level.

    Returns
    -------
    pandas.DataFrame
        `pgc`, `cz` (CF4 `Vcmb`), `cos_apex` (2MRS direction), `d_cz_hel` =
        Vhel_M - Vhel_CF4, `d_cz_cmb` = cz_cmb_M - Vcmb, `d_frame` =
        (cz_cmb_M - Vhel_M) - (Vcmb - Vhel_CF4) (km/s); `d_s_group` =
        s_M - s_V (h^-1 Mpc).
    """
    cosmology = cosmology or CosmologyConfig()
    c = conventions.SPEED_OF_LIGHT
    m = twomrs_positioned.assign(
        cos_apex=_cos_apex(twomrs_positioned["glon"], twomrs_positioned["glat"], apex)
    )[["pgc", "cz_hel", "cz_cmb", "s", "cos_apex"]]
    v = cf4_velocities.table
    cf4 = pd.DataFrame(
        {
            "pgc": v["pgc"].to_numpy(),
            "Vhel": v["Vhel"].to_numpy(float),
            "Vcmb": v["Vcmb"].to_numpy(float),
            "s_V": comoving_distance_or_nan(v["Vcmb"].to_numpy(float) / c, cosmology.Om0_TEMPEL),
        }
    )
    j = m.merge(cf4, on="pgc", how="inner", validate="one_to_one")
    return pd.DataFrame(
        {
            "pgc": j["pgc"],
            "cz": j["Vcmb"],
            "cos_apex": j["cos_apex"],
            "d_cz_hel": j["cz_hel"] - j["Vhel"],
            "d_cz_cmb": j["cz_cmb"] - j["Vcmb"],
            "d_frame": (j["cz_cmb"] - j["cz_hel"]) - (j["Vcmb"] - j["Vhel"]),
            "d_s_group": j["s"] - j["s_V"],
        }
    )


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 x MAD: the Gaussian-equivalent sigma, blind to the tails."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(_MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def binned_residual(
    x: np.ndarray, residual: np.ndarray, edges: np.ndarray, min_count: int
) -> pd.DataFrame:
    """Median residual in bins of x, with its standard error.

    Parameters
    ----------
    x, residual : array_like, shape (N,)
        NaN rows are ignored.
    edges : array_like, shape (B + 1,)
    min_count : int
        Bins with fewer objects are dropped.

    Returns
    -------
    pandas.DataFrame
        One row per kept bin: `lo`, `hi`, `x_median`, `n`, `median`,
        `error` (sqrt(pi/2) x robust sigma / sqrt(n)), `robust_sigma`.
    """
    x = np.asarray(x, dtype=float)
    residual = np.asarray(residual, dtype=float)
    ok = np.isfinite(x) & np.isfinite(residual)
    x, residual = x[ok], residual[ok]
    edges = np.asarray(edges, dtype=float)
    index = np.digitize(x, edges) - 1
    records = []
    for b in range(edges.size - 1):
        inside = index == b
        n = int(inside.sum())
        if n < min_count:
            continue
        r = residual[inside]
        sigma = robust_sigma(r)
        records.append(
            {
                "lo": edges[b],
                "hi": edges[b + 1],
                "x_median": float(np.median(x[inside])),
                "n": n,
                "median": float(np.median(r)),
                "error": _MEDIAN_ERROR_FACTOR * sigma / np.sqrt(n),
                "robust_sigma": sigma,
            }
        )
    return pd.DataFrame.from_records(records)


def residual_trend_slope(binned: pd.DataFrame) -> tuple[float, float]:
    """Weighted least-squares slope of the binned medians against `x_median`.

    Returns
    -------
    (slope, error) : tuple of float
        Residual units per unit x.
    """
    x = binned["x_median"].to_numpy(float)
    y = binned["median"].to_numpy(float)
    w = 1.0 / binned["error"].to_numpy(float) ** 2
    design = np.column_stack([np.ones_like(x), x])
    normal = design.T @ (w[:, None] * design)
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ (design.T @ (w * y))
    return float(coefficients[1]), float(np.sqrt(covariance[1, 1]))


def match_residual_summary(
    matches: pd.DataFrame, columns: dict[str, str], config: CoordinateCheckConfig | None = None
) -> pd.DataFrame:
    """Median, scatter and the two trends of each residual of a match table.

    Parameters
    ----------
    matches : pandas.DataFrame
        A `*_matches` table.
    columns : dict of str -> str
        Residual column -> its unit, for the table.

    Returns
    -------
    pandas.DataFrame
        One row per residual: `unit`, `n`, `median`, `robust_sigma`,
        `slope_cz` / `slope_cz_error` (per `config.cz_slope_unit` km/s),
        `slope_cos` / `slope_cos_error` (per unit cos_apex), and each slope's
        significance `slope_cz_sigmas`, `slope_cos_sigmas`.
    """
    config = config or CoordinateCheckConfig()
    records = []
    for column, unit in columns.items():
        r = matches[column].to_numpy(float)
        by_cz = binned_residual(matches["cz"], r, np.asarray(config.cz_edges), config.min_bin_count)
        by_cos = binned_residual(matches["cos_apex"], r, config.cos_edges, config.min_bin_count)
        slope_cz, err_cz = residual_trend_slope(by_cz)
        slope_cos, err_cos = residual_trend_slope(by_cos)
        finite = r[np.isfinite(r)]
        records.append(
            {
                "residual": column,
                "unit": unit,
                "n": finite.size,
                "median": float(np.median(finite)),
                "robust_sigma": robust_sigma(finite),
                "slope_cz": slope_cz * config.cz_slope_unit,
                "slope_cz_error": err_cz * config.cz_slope_unit,
                "slope_cz_sigmas": slope_cz / err_cz,
                "slope_cos": slope_cos,
                "slope_cos_error": err_cos,
                "slope_cos_sigmas": slope_cos / err_cos,
            }
        )
    return pd.DataFrame.from_records(records).set_index("residual")


def residual_by_richness(
    matches: pd.DataFrame, column: str, config: CoordinateCheckConfig | None = None
) -> pd.DataFrame:
    """Median and robust sigma of one residual by Tempel `Ng` class.

    Returns
    -------
    pandas.DataFrame
        One row per class: `Ng` (label), `n`, `median`, `robust_sigma`.
    """
    config = config or CoordinateCheckConfig()
    edges = list(config.richness_edges)
    records = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        inside = matches["Ng"] >= lo if hi is None else (matches["Ng"] >= lo) & (matches["Ng"] < hi)
        label = f"Ng >= {lo}" if hi is None else (f"Ng = {lo}" if hi == lo + 1 else f"{lo} <= Ng < {hi}")
        r = matches.loc[inside, column].to_numpy(float)
        r = r[np.isfinite(r)]
        records.append(
            {"Ng": label, "n": r.size, "median": float(np.median(r)), "robust_sigma": robust_sigma(r)}
        )
    return pd.DataFrame.from_records(records).set_index("Ng")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _finish(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, color=_LABEL_COLOR)
    ax.set_ylabel(ylabel, color=_LABEL_COLOR)
    ax.set_title(title, color=_LABEL_COLOR)
    ax.grid(alpha=_GRID_ALPHA, lw=_GRID_LINE_WIDTH)


def _draw_binned(ax: plt.Axes, binned: pd.DataFrame, color: str, label: str | None = None) -> None:
    ax.errorbar(
        binned["x_median"], binned["median"], yerr=binned["error"], color=color, lw=_LINE_WIDTH,
        marker="o", ms=_MARKER_SIZE, capsize=_ERRORBAR_CAPSIZE, label=label,
    )


def _symmetric_limit(values: np.ndarray, percentile: float) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile))


def apex_by_redshift_table(
    tempel: EddCatalog, cf4_velocities: EddCatalog, config: CoordinateCheckConfig | None = None
) -> pd.DataFrame:
    """`dipole_amplitude_by_redshift` for both catalogs and both forms, in
    the `config.cz_edges` bins of heliocentric cz.

    Returns
    -------
    pandas.DataFrame
        The per-bin rows with a `fit` column ("Tempel, additive", ...).
    """
    config = config or CoordinateCheckConfig()
    c = conventions.SPEED_OF_LIGHT
    edges = np.asarray(config.cz_edges) / c
    t, v = tempel.table, cf4_velocities.table
    inputs = {
        "Tempel": (t["zobs"], t["zcmb"], t["glon"], t["glat"]),
        "CF4": (v["Vhel"].to_numpy(float) / c, v["Vcmb"].to_numpy(float) / c, v["glon"], v["glat"]),
    }
    frames = []
    for catalog, (z_hel, z_cmb, glon, glat) in inputs.items():
        for form in (CMB_FORM_ADDITIVE, CMB_FORM_EXACT):
            frames.append(
                dipole_amplitude_by_redshift(z_hel, z_cmb, glon, glat, form, edges, config.min_bin_count)
                .assign(fit=f"{catalog}, {form}")
            )
    return pd.concat(frames, ignore_index=True)


def make_apex_by_redshift_figure(by_redshift: pd.DataFrame) -> plt.Figure:
    """Check 0: the CMB-correction dipole amplitude refitted per cz bin.

    The form a catalog was built with is flat in cz; the other drifts by
    ~ z V. Tempel's additive form is flat at the rounding level, so it is the
    one Tempel used. CF4 is flat too beyond cz ~ 5,000 km/s, at a different
    amplitude; below that its correction is not a pure dipole.

    Parameters
    ----------
    by_redshift : pandas.DataFrame
        `apex_by_redshift_table(...)`.
    """
    c = conventions.SPEED_OF_LIGHT
    fig, ax = plt.subplots(figsize=(1.6 * _PANEL_WIDTH, _PANEL_HEIGHT))
    styles = {
        "Tempel, additive": (_COLOR_PRIMARY, "-"),
        "Tempel, exact": (_COLOR_PRIMARY, "--"),
        "CF4, additive": (_COLOR_SECONDARY, "-"),
        "CF4, exact": (_COLOR_SECONDARY, "--"),
    }
    for name, (color, style) in styles.items():
        rows = by_redshift[by_redshift["fit"] == name]
        ax.plot(rows["z_median"] * c, rows["amplitude"], color=color, ls=style, marker="o",
                ms=_MARKER_SIZE, lw=_LINE_WIDTH, label=name)
    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, fontsize="small")
    _finish(ax, r"heliocentric $cz$ [km/s]", r"dipole amplitude $|\mathbf{V}|$ [km/s]",
            "CMB-correction dipole refitted per cz bin")
    fig.tight_layout()
    return fig


def make_dc_rebuild_figure(
    rebuilt: dict[str, pd.DataFrame], primary: str, config: CoordinateCheckConfig | None = None
) -> plt.Figure:
    """Check 1: the `dc` residual against `dc`, and |residual| histograms.

    Left: the median residual per `dc` bin for every variant (Tempel's must sit
    inside the rounding allowance; another h or Omega_m shows as a ramp),
    complete groups only. Right: |residual| of `primary` for complete and
    incomplete tracers on a log axis.
    """
    config = config or CoordinateCheckConfig()
    fig, (left, right) = plt.subplots(1, 2, figsize=(2 * _PANEL_WIDTH, _PANEL_HEIGHT))
    colors = (_COLOR_PRIMARY, _COLOR_SECONDARY, _COLOR_TERTIARY, _COLOR_QUATERNARY)
    reference = rebuilt[primary]
    complete = reference[reference["complete"]]
    edges = np.linspace(0.0, float(complete["dc"].max()), config.dc_bins + 1)
    allowance = binned_residual(complete["dc"], complete["allowance"], edges, config.min_bin_count)
    left.fill_between(
        allowance["x_median"], -allowance["median"], allowance["median"], color=_COLOR_GUIDE,
        alpha=_BAND_ALPHA, lw=0, label="rounding allowance",
    )
    for color, (name, frame) in zip(colors, rebuilt.items()):
        sub = frame[frame["complete"]]
        _draw_binned(left, binned_residual(sub["dc"], sub["residual"], edges, config.min_bin_count), color, name)
    left.axhline(0.0, color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH)
    left.set_xlim(left=0.0)
    left.legend(frameon=False, fontsize="small", loc="upper left")
    _finish(left, r"$d_c$ [Mpc]", r"median $d_c^{\rm rebuilt} - d_c$ [Mpc]", "complete tracers")

    absolute = reference["residual"].abs()
    positive = absolute[absolute > 0]
    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), config.histogram_bins)
    for mask, color, label in (
        (reference["complete"], _COLOR_PRIMARY, "every row present"),
        (~reference["complete"], _COLOR_SECONDARY, "fewer rows than Ng"),
    ):
        right.hist(absolute[mask], bins=bins, histtype="step", lw=_HIST_LINE_WIDTH, color=color,
                   label=f"{label} ({int(mask.sum()):,})")
    right.set_xscale("log")
    right.set_yscale("log")
    right.legend(frameon=False, fontsize="small")
    _finish(right, r"$|d_c^{\rm rebuilt} - d_c|$ [Mpc]", "tracers", primary)
    fig.tight_layout()
    return fig


def make_residual_trend_figure(
    matches: pd.DataFrame,
    column: str,
    *,
    title: str,
    ylabel: str,
    config: CoordinateCheckConfig | None = None,
) -> plt.Figure:
    """One residual of a match table against cz and against cos_apex.

    Top row: every match (hexbin) with the binned medians. Bottom row: the
    binned medians alone, zoomed, so a trend far below the scatter is
    visible.
    """
    config = config or CoordinateCheckConfig()
    r = matches[column].to_numpy(float)
    limit = _symmetric_limit(r, config.plot_percentile)
    fig, axes = plt.subplots(2, 2, figsize=(2 * _PANEL_WIDTH, 2 * _PANEL_HEIGHT), sharex="col")
    for col, (x_name, edges, xlabel) in enumerate(
        (
            ("cz", np.asarray(config.cz_edges), r"$cz_{\rm cmb}$ [km/s]"),
            ("cos_apex", config.cos_edges, r"$\cos\theta_{\rm apex}$"),
        )
    ):
        x = matches[x_name].to_numpy(float)
        binned = binned_residual(x, r, edges, config.min_bin_count)
        top, bottom = axes[0, col], axes[1, col]
        ok = np.isfinite(x) & np.isfinite(r)
        top.hexbin(x[ok], np.clip(r[ok], -limit, limit), gridsize=_HEXBIN_GRID, cmap=_HEXBIN_CMAP,
                   bins="log", mincnt=1, extent=(x[ok].min(), x[ok].max(), -limit, limit))
        _draw_binned(top, binned, _COLOR_SECONDARY)
        top.set_ylim(-limit, limit)
        _finish(top, "", ylabel, f"{title}" if col == 0 else f"n = {int(ok.sum()):,}")
        _draw_binned(bottom, binned, _COLOR_SECONDARY)
        bottom.axhline(0.0, color=_COLOR_GUIDE, lw=_GUIDE_LINE_WIDTH)
        _finish(bottom, xlabel, f"median {ylabel}", "binned medians ± error")
        if x_name == "cz":
            bottom.set_xlim(left=0.0)
    fig.tight_layout()
    return fig


def make_radius_histogram_figure(
    radii: dict[str, np.ndarray], config: CoordinateCheckConfig | None = None
) -> plt.Figure:
    """The samples' redshift-space radii on one h^-1 Mpc axis (Stage 3's
    "Done when"), each normalized to unit area."""
    config = config or CoordinateCheckConfig()
    finite = {name: np.asarray(s, float)[np.isfinite(s)] for name, s in radii.items()}
    top = max(float(s.max()) for s in finite.values())
    bins = np.linspace(0.0, top, config.histogram_bins + 1)
    fig, ax = plt.subplots(figsize=(1.6 * _PANEL_WIDTH, _PANEL_HEIGHT))
    colors = (_COLOR_SECONDARY, _COLOR_PRIMARY, _COLOR_TERTIARY, _COLOR_QUATERNARY)
    for color, (name, s) in zip(colors, finite.items()):
        ax.hist(s, bins=bins, density=True, histtype="step", lw=_HIST_LINE_WIDTH, color=color,
                label=f"{name} ({s.size:,})")
    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, fontsize="small")
    _finish(ax, r"$s$ [$h^{-1}$ Mpc]", "density", "redshift-space radius, CMB frame, one χ(z)")
    fig.tight_layout()
    return fig
