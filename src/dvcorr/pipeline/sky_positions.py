"""
sky_positions.py
----------------
Stage 3 of `docs/real_data_roadmap.md`: the three observed catalogs placed in
ONE redshift-space coordinate system -- CMB frame, h^-1 Mpc, one chi(z), the
observer at the origin -- so that the pair separation r = s_V - s_T means the
same thing at both ends.

The one chi(z)
--------------
`dvcorr.config.cosmology.comoving_distance_h_units` at
`CosmologyConfig.Om0_TEMPEL` (0.308), flat. In h^-1 Mpc the Hubble constant
cancels, so chi depends on Omega_m alone; Tempel's Omega_m is used so that his
`dc` is rebuilt to rounding (`coordinate_checks.rebuild_tempel_dc`).

The three positions
-------------------
- **Tempel (T)**, `tempel_positions`: s_T = `dc` x `CosmologyConfig.h_tempel`
  (0.678). `dc` is Tempel's mean of the members' chi(z_cmb) in Mpc at
  H0 = 67.8 (all members, including those the EDD table clipped), so it is
  taken as published; never multiplied by `conventions.HUBBLE_PARAM`.
- **CF4 (V)**, `cf4_positions`: s_V = chi(`Vcmb` / c). `Vcmb` is c z_cmb.
  Never `fV` (a curvature-adjusted velocity, not a redshift) and never `D`
  (a measured distance on CF4's H0 = 74.6 scale).
- **2MRS (M)**, `twomrs_positions`: s_M = chi(z_cmb), z_cmb built from the
  heliocentric `Vhel` with Tempel's additive formula (`cmb_redshift_additive`)
  and the apex fitted to Tempel's own correction (`fit_tempel_cmb_apex`), so
  that s_M lands in Tempel's system rather than assuming it.

Each gets n_hat from (RA, DE) via `real_catalogs.radec_to_unit_vector` and
the position s = s n_hat, observer at the origin, equatorial Cartesian axes.
Not periodic: hard rule 3 does not apply to the sky.

A redshift <= 0 has no redshift-space distance. Such rows get NaN position
and `has_position = False`; they are flagged, never given a made-up radius,
and never silently dropped here (the sample builders downstream decide). On
the real files: no Tempel tracer, 1 CF4 group of the T sample, 43 of the M
sample and 22 2MRS galaxies (notebook 14).

The CMB correction
------------------
Tempel et al. (2017, footnote 3) use the additive z_cmb = z_obs - v_p / c,
v_p the observer's motion along the line of sight, not the exact
(1 + z_cmb) = (1 + z_obs) / (1 + v_p / c). The paper names no dipole, so it is
FITTED: `fit_velocity_dipole` of c (z_cmb - z_obs) on the Galactic unit
vectors gives the apex vector V (km/s, toward the observer's motion), and the
additive formula is z_cmb = z_hel + V . n_hat / c, i.e. v_p = -V . n_hat. The
sign is carried by the fitted vector, not typed in. `CMB_FORM_EXACT` exists
only to show that Tempel did not use it (Check 0).

The dipole fits and the CMB correction work on Galactic unit vectors
(`galactic_unit_vector`), since the apex is quoted in (l, b); the positions
use equatorial ones. The two never mix: a dot product with the apex is a
scalar.

Positions are built in the CMB frame only (`conventions.REFERENCE_FRAME`);
every builder raises if that convention changes, rather than returning CMB
positions under another frame's name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dvcorr import conventions
from dvcorr.config.cosmology import CosmologyConfig, comoving_distance_h_units
from dvcorr.pipeline.real_catalogs import EddCatalog, radec_to_unit_vector

#: The frame every position here is built in.
_POSITION_FRAME: str = "cmb"

#: Tempel's additive CMB correction, z_cmb = z_obs - v_p / c.
CMB_FORM_ADDITIVE = "additive"
#: The exact (1 + z_cmb) = (1 + z_obs) / (1 + v_p / c), for comparison only.
CMB_FORM_EXACT = "exact"
VALID_CMB_FORMS = frozenset({CMB_FORM_ADDITIVE, CMB_FORM_EXACT})

#: Position columns every builder adds, h^-1 Mpc, equatorial Cartesian axes.
POSITION_COLUMNS: tuple[str, str, str] = ("s_x", "s_y", "s_z")


# --------------------------------------------------------------------------
# The CMB apex
# --------------------------------------------------------------------------


def galactic_unit_vector(glon_deg: np.ndarray, glat_deg: np.ndarray) -> np.ndarray:
    """Galactic (l, b) in degrees -> unit vectors in Galactic Cartesian axes.

    The same longitude/latitude-to-Cartesian map as `radec_to_unit_vector`,
    applied to Galactic angles.

    Parameters
    ----------
    glon_deg, glat_deg : array_like, shape (N,)

    Returns
    -------
    numpy.ndarray, shape (N, 3)
    """
    return radec_to_unit_vector(glon_deg, glat_deg)


@dataclass(frozen=True)
class VelocityDipoleFit:
    """A least-squares fit dv = V . n_hat (no monopole).

    Attributes
    ----------
    vector : numpy.ndarray, shape (3,)
        V, km/s, Galactic Cartesian axes.
    amplitude : float
        |V|, km/s.
    glon, glat : float
        Direction of V, degrees.
    rms : float
        km/s, root-mean-square residual.
    n_objects : int
    residual : numpy.ndarray, shape (N,)
        dv - V . n_hat, km/s, in input order.
    """

    vector: np.ndarray
    amplitude: float
    glon: float
    glat: float
    rms: float
    n_objects: int
    residual: np.ndarray

    @property
    def unit_vector(self) -> np.ndarray:
        """V / |V|, shape (3,), Galactic Cartesian."""
        return self.vector / self.amplitude


def fit_velocity_dipole(n_hat: np.ndarray, dv: np.ndarray) -> VelocityDipoleFit:
    """Least-squares dipole dv = V . n_hat, no monopole term.

    Parameters
    ----------
    n_hat : numpy.ndarray, shape (N, 3)
        Galactic unit vectors (`galactic_unit_vector`).
    dv : array_like, shape (N,)
        km/s.

    Returns
    -------
    VelocityDipoleFit
    """
    n_hat = np.asarray(n_hat, dtype=float)
    dv = np.asarray(dv, dtype=float)
    vector, *_ = np.linalg.lstsq(n_hat, dv, rcond=None)
    residual = dv - n_hat @ vector
    amplitude = float(np.linalg.norm(vector))
    return VelocityDipoleFit(
        vector=vector,
        amplitude=amplitude,
        glon=float(np.rad2deg(np.arctan2(vector[1], vector[0])) % 360.0),
        glat=float(np.rad2deg(np.arcsin(vector[2] / amplitude))),
        rms=float(np.sqrt(np.mean(residual**2))),
        n_objects=int(dv.size),
        residual=residual,
    )


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two (3,) vectors, degrees (atan2 form, exact near 0)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def _check_form(form: str) -> None:
    if form not in VALID_CMB_FORMS:
        raise ValueError(f"CMB correction form must be one of {sorted(VALID_CMB_FORMS)}, got {form!r}.")


def cmb_correction_velocity(z_hel: np.ndarray, z_cmb: np.ndarray, form: str) -> np.ndarray:
    """The observer-motion projection V . n_hat that turns z_hel into z_cmb, km/s.

    - `CMB_FORM_ADDITIVE`: c (z_cmb - z_hel).
    - `CMB_FORM_EXACT`: inverting (1 + z_cmb) = (1 + z_hel) / (1 - V.n/c),
      c (z_cmb - z_hel) / (1 + z_cmb).

    Whichever form produced the catalog is the one whose V . n_hat is a pure
    dipole (`fit_velocity_dipole` leaves only rounding).

    Parameters
    ----------
    z_hel, z_cmb : array_like, shape (N,)
    form : str
        `CMB_FORM_ADDITIVE` or `CMB_FORM_EXACT`.

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    _check_form(form)
    z_hel = np.atleast_1d(np.asarray(z_hel, dtype=float))
    z_cmb = np.atleast_1d(np.asarray(z_cmb, dtype=float))
    dv = conventions.SPEED_OF_LIGHT * (z_cmb - z_hel)
    return dv if form == CMB_FORM_ADDITIVE else dv / (1.0 + z_cmb)


def fit_tempel_cmb_apex(tempel: EddCatalog, form: str = CMB_FORM_ADDITIVE) -> VelocityDipoleFit:
    """The CMB apex Tempel's `zcmb` was built with: a dipole fit to the
    correction of every galaxy (`zobs` -> `zcmb`), Galactic `glon`/`glat`.

    With the additive form (Tempel's) the residual is the rounding of the two
    printed redshifts alone (Check 0).

    Parameters
    ----------
    tempel : EddCatalog
        `load_tempel_galaxies()`.
    form : str

    Returns
    -------
    VelocityDipoleFit
    """
    t = tempel.table
    dv = cmb_correction_velocity(t["zobs"], t["zcmb"], form)
    return fit_velocity_dipole(galactic_unit_vector(t["glon"], t["glat"]), dv)


def fit_cf4_cmb_apex(
    velocities: EddCatalog, form: str = CMB_FORM_ADDITIVE, min_cz: float | None = None
) -> VelocityDipoleFit:
    """The CMB apex behind CF4's `Vcmb`: a dipole fit to `Vcmb` - `Vhel`
    (as c z, `form` as in `cmb_correction_velocity`).

    Over every group the residual is tens of km/s, from the nearby groups
    (heliocentric cz below ~5,000 km/s), whose correction is not a pure
    dipole. Beyond them (`min_cz`) the additive form fits to the integer
    rounding of the two columns -- 371.0 km/s toward (264.15, 48.26) deg,
    rms 0.29 km/s = 1/sqrt(12) -- and the exact form drifts with z (Check 0,
    notebook 14).

    Parameters
    ----------
    velocities : EddCatalog
        `load_cf4_velocities()`.
    form : str
    min_cz : float, optional
        km/s. Fit only groups with `Vhel` above this.

    Returns
    -------
    VelocityDipoleFit
    """
    v = velocities.table
    if min_cz is not None:
        v = v[v["Vhel"] > min_cz]
    c = conventions.SPEED_OF_LIGHT
    dv = cmb_correction_velocity(v["Vhel"].to_numpy(float) / c, v["Vcmb"].to_numpy(float) / c, form)
    return fit_velocity_dipole(galactic_unit_vector(v["glon"], v["glat"]), dv)


def cmb_redshift_additive(
    z_hel: np.ndarray, n_hat_galactic: np.ndarray, apex: VelocityDipoleFit
) -> np.ndarray:
    """Tempel's additive CMB correction, z_cmb = z_hel + V . n_hat / c.

    Tempel's z_cmb = z_obs - v_p / c with v_p = -V . n_hat, V the fitted apex
    vector (`fit_tempel_cmb_apex`); the sign is the fit's.

    Parameters
    ----------
    z_hel : array_like, shape (N,)
        Heliocentric redshift.
    n_hat_galactic : numpy.ndarray, shape (N, 3)
        `galactic_unit_vector` of the objects.
    apex : VelocityDipoleFit

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    z_hel = np.atleast_1d(np.asarray(z_hel, dtype=float))
    return z_hel + (np.asarray(n_hat_galactic, dtype=float) @ apex.vector) / conventions.SPEED_OF_LIGHT


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------


def _require_cmb_frame() -> None:
    if conventions.REFERENCE_FRAME != _POSITION_FRAME:
        raise NotImplementedError(
            f"sky_positions builds CMB-frame positions only, but "
            f"conventions.REFERENCE_FRAME = {conventions.REFERENCE_FRAME!r}. An LG-frame "
            "run needs its own correction here, not CMB positions under the LG name."
        )


def comoving_distance_or_nan(z: np.ndarray, om0: float) -> np.ndarray:
    """chi(z) in h^-1 Mpc where z > 0; NaN where z <= 0 (no distance).

    Parameters
    ----------
    z : array_like, shape (N,)
    om0 : float

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.full(z.shape, np.nan)
    defined = np.isfinite(z) & (z > 0)
    out[defined] = comoving_distance_h_units(z[defined], om0)
    return out


def sky_position(radius: np.ndarray, ra_deg: np.ndarray, de_deg: np.ndarray) -> np.ndarray:
    """s = radius x n_hat(RA, DE), observer at the origin.

    Parameters
    ----------
    radius : array_like, shape (N,)
        h^-1 Mpc. NaN propagates to a NaN row.
    ra_deg, de_deg : array_like, shape (N,)

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        h^-1 Mpc, equatorial Cartesian axes.
    """
    radius = np.atleast_1d(np.asarray(radius, dtype=float))
    return radius[:, None] * radec_to_unit_vector(ra_deg, de_deg)


def _with_position(table: pd.DataFrame, radius: np.ndarray, **extra: np.ndarray) -> pd.DataFrame:
    """A copy of `table` with `extra` columns, `s`, the `POSITION_COLUMNS`
    and `has_position` appended."""
    out = table.copy()
    for name, values in extra.items():
        out[name] = values
    out["s"] = radius
    position = sky_position(radius, table["RA"], table["DE"])
    for i, name in enumerate(POSITION_COLUMNS):
        out[name] = position[:, i]
    out["has_position"] = np.isfinite(radius)
    return out


def position_array(table: pd.DataFrame) -> np.ndarray:
    """The `POSITION_COLUMNS` of a positioned table as an (N, 3) array, h^-1 Mpc."""
    return table[list(POSITION_COLUMNS)].to_numpy(dtype=float)


def tempel_positions(tracers: pd.DataFrame, cosmology: CosmologyConfig | None = None) -> pd.DataFrame:
    """s_T for the Tempel density tracers: `dc` x `h_tempel`, h^-1 Mpc.

    Parameters
    ----------
    tracers : pandas.DataFrame
        `real_catalogs.collapse_tempel_groups` / `select_tempel_tracers` rows
        (columns `RA`, `DE`, `dc`).
    cosmology : CosmologyConfig, optional

    Returns
    -------
    pandas.DataFrame
        A copy with `s` (h^-1 Mpc), `s_x`, `s_y`, `s_z` and `has_position`
        (`dc` > 0; all rows on the real file).
    """
    _require_cmb_frame()
    cosmology = cosmology or CosmologyConfig()
    dc = tracers["dc"].to_numpy(dtype=float)
    radius = np.where(dc > 0, dc * cosmology.h_tempel, np.nan)
    return _with_position(tracers, radius)


def cf4_positions(table: pd.DataFrame, cosmology: CosmologyConfig | None = None) -> pd.DataFrame:
    """s_V for CF4 groups: chi(`Vcmb` / c) at `Om0_TEMPEL`, h^-1 Mpc.

    Parameters
    ----------
    table : pandas.DataFrame
        `Cf4VelocitySample.table` (columns `RA`, `DE`, `Vcmb`), or the raw
        velocities table.
    cosmology : CosmologyConfig, optional

    Returns
    -------
    pandas.DataFrame
        A copy with `z_cmb`, `s`, `s_x`, `s_y`, `s_z` and `has_position`
        (False where `Vcmb` <= 0).
    """
    _require_cmb_frame()
    cosmology = cosmology or CosmologyConfig()
    z_cmb = table["Vcmb"].to_numpy(dtype=float) / conventions.SPEED_OF_LIGHT
    radius = comoving_distance_or_nan(z_cmb, cosmology.Om0_TEMPEL)
    return _with_position(table, radius, z_cmb=z_cmb)


def twomrs_positions(
    tracers: pd.DataFrame, apex: VelocityDipoleFit, cosmology: CosmologyConfig | None = None
) -> pd.DataFrame:
    """s_M for the 2MRS galaxies: heliocentric `cz_hel` to the CMB frame with
    Tempel's additive formula and `apex`, then chi(z_cmb), h^-1 Mpc.

    Parameters
    ----------
    tracers : pandas.DataFrame
        `real_catalogs.select_twomrs_tracers` rows (columns `RA`, `DE`,
        `glon`, `glat`, `cz_hel`).
    apex : VelocityDipoleFit
        `fit_tempel_cmb_apex(tempel)`, so that s_M is in Tempel's system.
    cosmology : CosmologyConfig, optional

    Returns
    -------
    pandas.DataFrame
        A copy with `cz_cmb` (km/s), `s`, `s_x`, `s_y`, `s_z` and
        `has_position` (False where `cz_cmb` <= 0).
    """
    _require_cmb_frame()
    cosmology = cosmology or CosmologyConfig()
    c = conventions.SPEED_OF_LIGHT
    n_gal = galactic_unit_vector(tracers["glon"], tracers["glat"])
    z_cmb = cmb_redshift_additive(tracers["cz_hel"].to_numpy(dtype=float) / c, n_gal, apex)
    radius = comoving_distance_or_nan(z_cmb, cosmology.Om0_TEMPEL)
    return _with_position(tracers, radius, cz_cmb=z_cmb * c)
