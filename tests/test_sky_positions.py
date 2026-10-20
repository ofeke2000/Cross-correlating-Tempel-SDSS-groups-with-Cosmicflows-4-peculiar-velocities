"""
test_sky_positions.py
---------------------
Stage 3: the one chi(z) (`dvcorr.config.cosmology.comoving_distance_h_units`),
the three position builders and the CMB apex machinery of
`dvcorr.pipeline.sky_positions`, and the Check 1 rebuild of
`dvcorr.pipeline.coordinate_checks`.

Everything except `TestRealFiles` runs on synthetic tables. `TestRealFiles`
pins the roadmap's Stage 3 numbers on the real catalogs and skips when they
are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from dvcorr import conventions
from dvcorr.config import CosmologyConfig
from dvcorr.config.cosmology import (
    comoving_distance_derivative_h_units,
    comoving_distance_h_units,
    flat_lcdm_expansion_rate,
)
from dvcorr.config.paths import PathsConfig
from dvcorr.pipeline import coordinate_checks as cc
from dvcorr.pipeline import sky_positions as sp
from dvcorr.pipeline.real_catalogs import (
    EddCatalog,
    load_cf4_velocities,
    load_tempel_galaxies,
    load_twomrs_galaxies,
    radec_to_unit_vector,
    select_twomrs_tracers,
)

C = conventions.SPEED_OF_LIGHT
COSMO = CosmologyConfig()
#: Hubble distance in h^-1 Mpc: c / (100 km/s per h^-1 Mpc).
HUBBLE_DISTANCE = C / 100.0


# --------------------------------------------------------------------------
# chi(z)
# --------------------------------------------------------------------------


class TestComovingDistance:
    def test_zero_at_zero(self) -> None:
        assert comoving_distance_h_units(0.0, 0.308)[0] == 0.0

    def test_matches_adaptive_quadrature(self) -> None:
        z = np.array([1e-4, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
        for om0 in (0.308, 0.307115, 0.1, 1.0):
            expected = [
                HUBBLE_DISTANCE * quad(lambda x: 1.0 / flat_lcdm_expansion_rate(x, om0)[0], 0.0, zi,
                                       epsabs=0, epsrel=1e-13)[0]
                for zi in z
            ]
            np.testing.assert_allclose(comoving_distance_h_units(z, om0), expected, rtol=1e-12)

    def test_einstein_de_sitter_closed_form(self) -> None:
        # Om = 1: chi = 2 (c/100) (1 - 1/sqrt(1+z)).
        z = np.linspace(0.0, 0.3, 7)
        expected = 2.0 * HUBBLE_DISTANCE * (1.0 - 1.0 / np.sqrt(1.0 + z))
        np.testing.assert_allclose(comoving_distance_h_units(z, 1.0), expected, rtol=1e-13, atol=1e-12)

    def test_hubble_law_at_low_z_and_q0_at_second_order(self) -> None:
        # chi = (c/100) [z - (1 + q0) z^2 / 2 + O(z^3)], q0 = 3 Om / 2 - 1.
        z, om0 = 1e-3, 0.308
        q0 = 1.5 * om0 - 1.0
        expected = HUBBLE_DISTANCE * (z - 0.5 * (1.0 + q0) * z**2)
        assert comoving_distance_h_units(z, om0)[0] == pytest.approx(expected, rel=1e-6)

    def test_seven_mpc_below_cz_over_100_at_z_tenth(self) -> None:
        """The roadmap's warning: cz/100 and chi differ by ~7 h^-1 Mpc (2.4%)."""
        chi = comoving_distance_h_units(0.1, COSMO.Om0_TEMPEL)[0]
        assert 0.1 * HUBBLE_DISTANCE - chi == pytest.approx(7.05, abs=0.05)

    def test_shape_is_n_for_scalar(self) -> None:
        assert comoving_distance_h_units(0.05, 0.3).shape == (1,)
        assert comoving_distance_h_units(np.array([0.05]), 0.3).shape == (1,)

    @pytest.mark.parametrize("bad", [-1e-3, np.nan, np.inf])
    def test_refuses_negative_or_missing_redshift(self, bad) -> None:
        with pytest.raises(ValueError):
            comoving_distance_h_units(np.array([0.01, bad]), 0.3)

    def test_derivative_matches_finite_difference(self) -> None:
        z, dz = np.array([0.02, 0.1, 0.19]), 1e-6
        numeric = (comoving_distance_h_units(z + dz, 0.308) - comoving_distance_h_units(z - dz, 0.308)) / (2 * dz)
        np.testing.assert_allclose(comoving_distance_derivative_h_units(z, 0.308), numeric, rtol=1e-8)


class TestTempelMetadata:
    def test_values(self) -> None:
        assert COSMO.H0_TEMPEL == 67.8
        assert COSMO.Om0_TEMPEL == 0.308
        assert COSMO.h_tempel == pytest.approx(0.678, abs=1e-15)

    def test_mdpl2_h_untouched(self) -> None:
        assert COSMO.h == pytest.approx(conventions.HUBBLE_PARAM, abs=1e-15)
        assert COSMO.h_tempel != COSMO.h


# --------------------------------------------------------------------------
# The CMB apex
# --------------------------------------------------------------------------

#: A planted apex near the real one, km/s and degrees.
APEX_SPEED, APEX_L, APEX_B = 369.0, 264.0, 48.0


def _sky(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    glon = rng.uniform(0.0, 360.0, n)
    glat = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return glon, glat


def _planted_apex() -> sp.VelocityDipoleFit:
    vector = APEX_SPEED * sp.galactic_unit_vector(APEX_L, APEX_B)[0]
    glon, glat = _sky()
    return sp.fit_velocity_dipole(sp.galactic_unit_vector(glon, glat), sp.galactic_unit_vector(glon, glat) @ vector)


class TestApex:
    def test_fit_recovers_a_planted_dipole_exactly(self) -> None:
        fit = _planted_apex()
        assert fit.amplitude == pytest.approx(APEX_SPEED, rel=1e-12)
        assert fit.glon == pytest.approx(APEX_L, abs=1e-9)
        assert fit.glat == pytest.approx(APEX_B, abs=1e-9)
        assert fit.rms < 1e-9
        assert fit.residual.shape == (fit.n_objects,)

    def test_the_form_that_built_the_catalog_fits_and_the_other_trends(self) -> None:
        glon, glat = _sky()
        n_gal = sp.galactic_unit_vector(glon, glat)
        z_hel = np.random.default_rng(4).uniform(0.0, 0.2, glon.size)
        z_cmb = sp.cmb_redshift_additive(z_hel, n_gal, _planted_apex())
        additive = sp.fit_velocity_dipole(n_gal, sp.cmb_correction_velocity(z_hel, z_cmb, sp.CMB_FORM_ADDITIVE))
        exact = sp.fit_velocity_dipole(n_gal, sp.cmb_correction_velocity(z_hel, z_cmb, sp.CMB_FORM_EXACT))
        assert additive.rms < 1e-6
        assert exact.rms > 10.0  # ~ (z - <z>) V.n, tens of km/s
        # The wrong form's dipole drifts with z (by ~ z V); the right one's does not.
        edges = np.array([0.0, 0.05, 0.15, 0.2])
        by_z = {form: cc.dipole_amplitude_by_redshift(z_hel, z_cmb, glon, glat, form, edges, min_count=100)
                for form in (sp.CMB_FORM_ADDITIVE, sp.CMB_FORM_EXACT)}
        np.testing.assert_allclose(by_z[sp.CMB_FORM_ADDITIVE]["amplitude"], APEX_SPEED, rtol=1e-9)
        drift = by_z[sp.CMB_FORM_EXACT]["amplitude"].to_numpy()
        assert drift[0] - drift[-1] > 0.1 * APEX_SPEED

    def test_correction_sign_follows_the_fit(self) -> None:
        """Toward the apex the observer moves toward the source: blueshift,
        so z_cmb > z_hel there and < z_hel opposite."""
        apex = _planted_apex()
        n = sp.galactic_unit_vector([APEX_L, APEX_L + 180.0], [APEX_B, -APEX_B])
        z_cmb = sp.cmb_redshift_additive(np.array([0.05, 0.05]), n, apex)
        assert z_cmb[0] - 0.05 == pytest.approx(APEX_SPEED / C, rel=1e-12)
        assert z_cmb[1] - 0.05 == pytest.approx(-APEX_SPEED / C, rel=1e-12)

    def test_unknown_form_refused(self) -> None:
        with pytest.raises(ValueError):
            sp.cmb_correction_velocity([0.1], [0.1], "relativistic")

    def test_agreement_tolerances(self) -> None:
        a = _planted_apex()
        tilted = sp.fit_velocity_dipole(np.eye(3), (APEX_SPEED + 2.0) * sp.galactic_unit_vector(APEX_L + 0.5, APEX_B)[0])
        assert cc.apex_agreement(a, tilted)["passes"]
        far = sp.fit_velocity_dipole(np.eye(3), APEX_SPEED * sp.galactic_unit_vector(APEX_L + 3.0, APEX_B)[0])
        assert not cc.apex_agreement(a, far)["passes"]


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------


def _directions(n=5):
    ra = np.linspace(10.0, 300.0, n)
    de = np.linspace(-40.0, 70.0, n)
    return ra, de


class TestPositions:
    def test_tempel_uses_tempels_h_not_mdpl2s(self) -> None:
        ra, de = _directions()
        dc = np.array([10.0, 100.0, 200.0, 300.0, 430.0])
        out = sp.tempel_positions(pd.DataFrame({"RA": ra, "DE": de, "dc": dc}))
        np.testing.assert_allclose(out["s"], dc * 0.678, rtol=1e-15)
        assert not np.allclose(out["s"], dc * conventions.HUBBLE_PARAM, rtol=1e-6)
        assert out["has_position"].all()

    def test_position_is_radius_times_direction_from_the_origin(self) -> None:
        ra, de = _directions()
        out = sp.tempel_positions(pd.DataFrame({"RA": ra, "DE": de, "dc": np.full(5, 100.0)}))
        s = sp.position_array(out)
        assert s.shape == (5, 3)
        np.testing.assert_allclose(np.linalg.norm(s, axis=1), out["s"], rtol=1e-14)
        np.testing.assert_allclose(s / out["s"].to_numpy()[:, None], radec_to_unit_vector(ra, de), atol=1e-15)

    def test_cf4_is_chi_of_vcmb_and_ignores_fv_and_d(self) -> None:
        ra, de = _directions()
        vcmb = np.array([1000.0, 5000.0, 10000.0, 20000.0, 30000.0])
        table = pd.DataFrame({"RA": ra, "DE": de, "Vcmb": vcmb, "fV": vcmb * 1.03, "D": vcmb / 74.6})
        out = sp.cf4_positions(table)
        np.testing.assert_allclose(out["s"], comoving_distance_h_units(vcmb / C, 0.308), rtol=1e-14)
        scrambled = sp.cf4_positions(table.assign(fV=-1.0, D=1e6))
        np.testing.assert_array_equal(out["s"], scrambled["s"])

    def test_cf4_without_a_positive_redshift_has_no_position(self) -> None:
        ra, de = _directions(3)
        out = sp.cf4_positions(pd.DataFrame({"RA": ra, "DE": de, "Vcmb": [-61.0, 0.0, 500.0]}))
        assert list(out["has_position"]) == [False, False, True]
        assert np.isnan(sp.position_array(out)[:2]).all()
        assert len(out) == 3  # flagged, not dropped

    def test_twomrs_lands_on_the_apex_corrected_redshift(self) -> None:
        apex = _planted_apex()
        glon, glat = np.array([APEX_L, APEX_L + 180.0, 10.0]), np.array([APEX_B, -APEX_B, 0.0])
        cz_hel = np.array([5000.0, 5000.0, 200.0])
        tracers = pd.DataFrame({"RA": glon, "DE": glat, "glon": glon, "glat": glat, "cz_hel": cz_hel})
        out = sp.twomrs_positions(tracers, apex)
        expected_cz = cz_hel + sp.galactic_unit_vector(glon, glat) @ apex.vector
        np.testing.assert_allclose(out["cz_cmb"], expected_cz, rtol=1e-13)
        np.testing.assert_allclose(out["s"], comoving_distance_h_units(expected_cz / C, 0.308), rtol=1e-13)
        assert out["cz_cmb"][0] == pytest.approx(5000.0 + APEX_SPEED)

    def test_twomrs_pushed_below_zero_has_no_position(self) -> None:
        apex = _planted_apex()
        anti_l, anti_b = APEX_L - 180.0, -APEX_B
        tracers = pd.DataFrame({"RA": [0.0], "DE": [0.0], "glon": [anti_l], "glat": [anti_b], "cz_hel": [100.0]})
        out = sp.twomrs_positions(tracers, apex)
        assert out["cz_cmb"][0] < 0 and not out["has_position"][0]

    def test_builders_refuse_a_non_cmb_frame(self, monkeypatch) -> None:
        monkeypatch.setattr(conventions, "REFERENCE_FRAME", "lg")
        with pytest.raises(NotImplementedError):
            sp.tempel_positions(pd.DataFrame({"RA": [0.0], "DE": [0.0], "dc": [10.0]}))
        with pytest.raises(NotImplementedError):
            sp.cf4_positions(pd.DataFrame({"RA": [0.0], "DE": [0.0], "Vcmb": [1000.0]}))


# --------------------------------------------------------------------------
# Check 1 on a synthetic Tempel table
# --------------------------------------------------------------------------


def _tempel_catalog():
    """Group 7: two rows of Ng = 2 (complete). Group 8: Ng = 3, two rows
    (one clipped). Two singletons. `dc` built Tempel's way, as the mean of
    the members' chi(zcmb) in Mpc -- for group 8 over three members, one of
    which is not in the table."""
    h = COSMO.h_tempel
    chi = lambda z: comoving_distance_h_units(np.asarray(z), COSMO.Om0_TEMPEL) / h  # noqa: E731
    z7, z8, z8_clipped = [0.030, 0.031], [0.050, 0.052], 0.090
    rows = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "IDg": [7, 7, 8, 8, 0, 0],
            "Ng": [2, 2, 3, 3, 1, 1],
            "Rg": [1, 2, 1, 2, 1, 1],
            "zcmb": z7 + z8 + [0.010, 0.020],
        }
    )
    dc = np.concatenate([
        np.full(2, chi(z7).mean()),
        np.full(2, chi(z8 + [z8_clipped]).mean()),
        chi([0.010, 0.020]),
    ])
    rows["dc"] = dc
    columns = pd.DataFrame({"format": {"dc": "%6.2f", "zcmb": "%7.5f", "ID": "%6d", "IDg": "%5d",
                                       "Ng": "%3d", "Rg": "%3d"}, "unit": "", "description": ""})
    return EddCatalog(path=None, title="synthetic", table=rows, columns=columns.loc[rows.columns])


class TestDcRebuild:
    def test_complete_tracers_rebuild_exactly_and_the_clipped_group_does_not(self) -> None:
        out = cc.rebuild_tempel_dc(_tempel_catalog())
        assert len(out) == 4  # two groups, two singletons
        complete = out[out["complete"]]
        np.testing.assert_allclose(complete["residual"], 0.0, atol=1e-10)
        clipped = out[~out["complete"]]
        assert list(clipped["IDg"]) == [8]
        assert (clipped["residual"].abs() > clipped["allowance"]).all()

    def test_another_h_shows_as_a_fixed_fractional_offset(self) -> None:
        out = cc.rebuild_tempel_dc(_tempel_catalog(), h=conventions.HUBBLE_PARAM)
        complete = out[out["complete"]]
        np.testing.assert_allclose(complete["residual"] / complete["dc"],
                                   COSMO.h_tempel / conventions.HUBBLE_PARAM - 1.0, rtol=1e-9)


# --------------------------------------------------------------------------
# Real files
# --------------------------------------------------------------------------

_PATHS = PathsConfig()
_HAVE_REAL = all(
    p.exists() for p in (_PATHS.cf4_velocities_catalog, _PATHS.sdss_tempel_catalog, _PATHS.twomrs_catalog)
)


@pytest.mark.skipif(not _HAVE_REAL, reason="real catalogs not in data/")
class TestRealFiles:
    @pytest.fixture(scope="class")
    @classmethod
    def tempel(cls):
        return load_tempel_galaxies()

    @pytest.fixture(scope="class")
    @classmethod
    def velocities(cls):
        return load_cf4_velocities()

    def test_check0_apexes(self, tempel, velocities) -> None:
        table, fits = cc.apex_fit_table(tempel, velocities)
        t = table.loc["Tempel, additive"]
        assert t["amplitude"] == pytest.approx(368.9, abs=0.1)
        assert (t["glon"], t["glat"]) == (pytest.approx(263.3, abs=0.1), pytest.approx(47.9, abs=0.1))
        assert t["rms"] < 1.3  # the rounding of zcmb and zobs
        assert table.loc["Tempel, exact", "rms"] > 4 * t["rms"]
        cf4 = table.loc["CF4, additive"]
        assert cf4["amplitude"] == pytest.approx(367.8, abs=0.1)
        assert (cf4["glon"], cf4["glat"]) == (pytest.approx(263.8, abs=0.1), pytest.approx(48.6, abs=0.1))
        assert cc.apex_agreement(fits["Tempel, additive"], fits["CF4, additive"])["passes"]

    def test_check1_dc_rebuilt_to_rounding(self, tempel) -> None:
        out = cc.rebuild_tempel_dc(tempel)
        complete = out[out["complete"]]
        assert len(complete) == 178_929
        assert (complete["residual"].abs() <= complete["allowance"]).all()
        assert (~out["complete"]).sum() == 752
        groups = complete[complete["IDg"] > 0]["residual"].abs()
        assert groups.median() == pytest.approx(0.006, abs=0.001)
        assert groups.quantile(0.9) == pytest.approx(0.014, abs=0.001)

    def test_check3_twomrs_conversion_reproduces_tempels(self, tempel) -> None:
        apex = sp.fit_tempel_cmb_apex(tempel)
        positioned = sp.twomrs_positions(select_twomrs_tracers(load_twomrs_galaxies()), apex)
        matches = cc.twomrs_tempel_matches(positioned, tempel, apex)
        assert len(matches) == 8_848
        assert abs(matches["d_frame"].median()) < 0.1
        assert cc.robust_sigma(matches["d_frame"]) < 1.5  # rounding of zcmb and zobs

    def test_check0_cf4_beyond_the_nearby_groups_is_an_exact_additive_dipole(self, velocities) -> None:
        """Beyond cz ~ 5,000 km/s CF4's Vcmb - Vhel is a pure additive dipole to
        the integer rounding of the columns (rms 1/sqrt(12)); the exact form is not."""
        table, _ = cc.apex_fit_table(load_tempel_galaxies(), velocities)
        deep = table.loc[cc.CF4_DEEP_ADDITIVE]
        assert deep["amplitude"] == pytest.approx(371.0, abs=0.05)
        assert (deep["glon"], deep["glat"]) == (pytest.approx(264.15, abs=0.02), pytest.approx(48.26, abs=0.02))
        assert deep["rms"] < 0.3
        assert table.loc[cc.CF4_DEEP_EXACT, "rms"] > 3 * deep["rms"]
        assert deep["angle_to_tempel"] < cc.CoordinateCheckConfig().apex_angle_tolerance_deg
