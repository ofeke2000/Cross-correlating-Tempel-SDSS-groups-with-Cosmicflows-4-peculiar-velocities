"""
test_real_catalogs.py
---------------------
Tests for `dvcorr.pipeline.real_catalogs` (the EDD loader and the CF4
velocity definitions) and `dvcorr.pipeline.real_catalog_audit`.

Everything except `TestRealFiles` runs on tiny synthetic EDD-format CSVs
written to `tmp_path`: a five-line header, quoted descriptions with commas in
them, values printed with each column's own printf format (so the rounding
the audit judges against is real rounding). `TestRealFiles` pins the Stage 1
numbers of `docs/real_data_roadmap.md` on the real catalogs and skips when
they are absent.
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from dvcorr import conventions
from dvcorr.config.paths import PathsConfig
from dvcorr.pipeline import real_catalog_audit as audit
from dvcorr.pipeline.real_catalogs import (
    Cf4VelocityDefinitions,
    EddCatalog,
    angular_separation_deg,
    column_descriptions,
    curvature_corrected_velocity,
    davis_scrimgeour_velocity,
    distance_from_modulus,
    format_half_ulp,
    in_twomrs_footprint,
    load_cf4_velocities,
    load_tempel_galaxies,
    load_twomrs_galaxies,
    published_ramp_weight,
    published_vpec,
    radec_to_unit_vector,
    read_edd_csv,
    velocity_error_from_modulus_error,
    watkins_feldman_velocity,
)

LN10 = np.log(10.0)
DEFS = Cf4VelocityDefinitions()


def _write_edd(path, title, columns, frame):
    """Write `frame` as an EDD table. `columns`: name -> (format, unit, description).

    Each value is printed with its column's printf format, as EDD does; NaN
    is written as an empty field.
    """
    names = list(columns)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["HYPERLEDA"] + [title] * (len(names) - 1))
        writer.writerow(names)
        writer.writerow([columns[n][0] for n in names])
        writer.writerow([columns[n][1] for n in names])
        writer.writerow([columns[n][2] for n in names])
        for _, row in frame[names].iterrows():
            fields = []
            for n in names:
                value = row[n]
                fmt = columns[n][0]
                if pd.isna(value):
                    fields.append("")
                elif fmt.endswith("d"):
                    fields.append((fmt % round(value)).strip())
                elif fmt.endswith("s"):
                    fields.append((fmt % int(value)).strip())
                else:
                    fields.append((fmt % value).strip())
            writer.writerow(fields)


_SMALL_COLUMNS = {
    "pgc": ("%7s", "---", "The Catalogue of Principal Galaxies (PGC) Number"),
    "DMzp": ("%6.3f", "mag", "Distance modulus, averaged; scale set by calibrators, in mag"),
    "RA": ("%8.4f", "hour", "Right ascension J2000 in decimal degrees"),
    "Nfp": ("%3d", "", "Number of galaxies with FP measures in group"),
}


def _small_catalog(tmp_path, name="small.csv"):
    frame = pd.DataFrame(
        {
            "pgc": [12, 14, 16],
            "DMzp": [34.984, 36.084, 34.644],
            "RA": [0.036, 359.9805, 120.5],
            "Nfp": [1, np.nan, 3],
        }
    )
    path = tmp_path / name
    _write_edd(path, "Small Test Table", _SMALL_COLUMNS, frame)
    return path, frame


# --------------------------------------------------------------------------
# The EDD format
# --------------------------------------------------------------------------


class TestReadEdd:
    def test_table_and_dictionary_round_trip(self, tmp_path) -> None:
        path, frame = _small_catalog(tmp_path)
        catalog = read_edd_csv(path)
        assert catalog.title == "Small Test Table"
        assert list(catalog.table.columns) == list(_SMALL_COLUMNS)
        assert list(catalog.columns.index) == list(_SMALL_COLUMNS)
        assert len(catalog.table) == len(frame)
        np.testing.assert_allclose(catalog.table["DMzp"], frame["DMzp"])
        for name, (fmt, unit, description) in _SMALL_COLUMNS.items():
            assert catalog.columns.loc[name, "format"] == fmt
            assert catalog.columns.loc[name, "unit"] == unit
            assert catalog.columns.loc[name, "description"] == description

    def test_descriptions_with_commas_survive(self, tmp_path) -> None:
        """A comma inside a quoted description must not split the field."""
        path, _ = _small_catalog(tmp_path)
        descriptions = column_descriptions(read_edd_csv(path))
        assert list(descriptions) == list(_SMALL_COLUMNS)
        assert descriptions["DMzp"] == _SMALL_COLUMNS["DMzp"][2]

    def test_numeric_columns_are_numeric(self, tmp_path) -> None:
        """A header line parsed as data would turn every column to object."""
        path, _ = _small_catalog(tmp_path)
        table = read_edd_csv(path).table
        assert all(pd.api.types.is_numeric_dtype(table[c]) for c in table.columns)

    def test_missing_value_stays_missing(self, tmp_path) -> None:
        """Hard rule 5: an empty field is NaN, never filled with 0."""
        path, _ = _small_catalog(tmp_path)
        table = read_edd_csv(path).table
        assert np.isnan(table.loc[1, "Nfp"])
        assert table["Nfp"].isna().sum() == 1

    def test_plain_csv_is_refused(self, tmp_path) -> None:
        path = tmp_path / "plain.csv"
        pd.DataFrame({"pgc": [1, 2], "x": [0.1, 0.2]}).to_csv(path, index=False)
        with pytest.raises(ValueError):
            read_edd_csv(path)

    def test_ragged_header_is_refused(self, tmp_path) -> None:
        path = tmp_path / "ragged.csv"
        path.write_text("HYPERLEDA,T,T\npgc,a,b\n%7s,%5d\n---,,\nd1,d2,d3\n1,2,3\n")
        with pytest.raises(ValueError):
            read_edd_csv(path)

    def test_format_half_ulp(self) -> None:
        assert format_half_ulp("%6.3f") == pytest.approx(5e-4)
        assert format_half_ulp("%5d") == 0.5
        assert format_half_ulp("%5.1f") == pytest.approx(0.05)
        assert np.isnan(format_half_ulp("%7s"))
        with pytest.raises(ValueError):
            format_half_ulp("hour")

    def test_loaders_resolve_through_paths_config(self, tmp_path) -> None:
        paths = PathsConfig(data_dir=tmp_path)
        assert paths.sdss_tempel_catalog.name == "SDSS_Tempel.csv"
        _small_catalog(tmp_path, name=paths.sdss_tempel_catalog.name)
        _small_catalog(tmp_path, name=paths.cf4_velocities_catalog.name)
        _small_catalog(tmp_path, name=paths.twomrs_catalog.name)
        assert len(load_tempel_galaxies(paths).table) == 3
        assert len(load_cf4_velocities(paths).table) == 3
        assert len(load_twomrs_galaxies(paths).table) == 3


# --------------------------------------------------------------------------
# Sky geometry
# --------------------------------------------------------------------------


class TestSkyGeometry:
    def test_unit_vector_shape_and_norm(self) -> None:
        one = radec_to_unit_vector(10.0, 20.0)
        assert one.shape == (1, 3)
        many = radec_to_unit_vector(np.linspace(0, 359, 7), np.linspace(-80, 80, 7))
        np.testing.assert_allclose(np.linalg.norm(many, axis=1), 1.0)

    def test_cardinal_directions(self) -> None:
        n = radec_to_unit_vector([0.0, 90.0, 0.0], [0.0, 0.0, 90.0])
        np.testing.assert_allclose(n, np.eye(3), atol=1e-15)

    def test_separation_accurate_when_tiny_and_antipodal(self) -> None:
        tiny = angular_separation_deg([10.0], [0.0], [10.0 + 1e-6], [0.0])
        assert tiny[0] == pytest.approx(1e-6, rel=1e-6)
        anti = angular_separation_deg([0.0], [30.0], [180.0], [-30.0])
        assert anti[0] == pytest.approx(180.0)

    def test_twomrs_mask_edges(self) -> None:
        # |b| >= 5 away from the bulge, >= 8 within 30 deg of l = 0; l wraps.
        glon = np.array([180.0, 180.0, 10.0, 10.0, 350.0, 329.9, 29.9, -10.0])
        glat = np.array([5.0, -4.99, 7.9, -8.0, 8.1, 5.0, 5.0, 7.0])
        expected = [True, False, False, True, True, True, False, False]
        assert in_twomrs_footprint(glon, glat).tolist() == expected
        assert in_twomrs_footprint(180.0, 30.0).shape == (1,)

    def test_separation_across_ra_wrap(self) -> None:
        sep = angular_separation_deg([359.9], [0.0], [0.1], [0.0])
        assert sep[0] == pytest.approx(0.2)


# --------------------------------------------------------------------------
# CF4 velocity definitions
# --------------------------------------------------------------------------


class TestVelocityDefinitions:
    def test_distance_modulus_definition(self) -> None:
        np.testing.assert_allclose(distance_from_modulus([25.0, 30.0, 35.0]), [1.0, 10.0, 100.0])

    def test_fv_low_redshift_limit(self) -> None:
        """fV/cz - 1 -> (1 - q0) z / 2 as z -> 0."""
        cz = np.array([30.0, 300.0])
        z = cz / conventions.SPEED_OF_LIGHT
        np.testing.assert_allclose(
            curvature_corrected_velocity(cz) / cz - 1.0, 0.5 * (1.0 - DEFS.q0) * z, rtol=1e-3
        )

    def test_vpds_zero_on_the_hubble_line(self) -> None:
        distance = np.array([10.0, 100.0, 300.0])
        assert np.allclose(davis_scrimgeour_velocity(DEFS.h0 * distance, distance), 0.0)

    def test_vpds_is_not_the_bare_difference(self) -> None:
        """The 1/(1 + H0 D/c) the roadmap's fV - H0 D left out."""
        fv, distance = 9000.0, 100.0
        bare = fv - DEFS.h0 * distance
        expected = bare / (1.0 + DEFS.h0 * distance / conventions.SPEED_OF_LIGHT)
        assert davis_scrimgeour_velocity(fv, distance) == pytest.approx(expected)
        assert abs(davis_scrimgeour_velocity(fv, distance) - bare) > 30.0

    def test_natural_log_agrees_with_vpds_to_first_order(self) -> None:
        """Two estimators of one velocity: ratio = 1 + O(x), x = v / (H0 D).

        ln(1 + x) = x - x^2/2 + ..., so the ratio departs from 1 by ~x/2:
        bounded by |x|, and shrinking in proportion as x does.
        """
        distance = np.array([100.0, 100.0])
        hubble_velocity = DEFS.h0 * distance
        departures = []
        for offset in (200.0, 20.0):  # km/s, |v| << H0 D
            fv = hubble_velocity + np.array([-offset, offset])
            ratio = watkins_feldman_velocity(fv, distance) / davis_scrimgeour_velocity(fv, distance)
            x = offset / hubble_velocity
            assert np.all(np.abs(ratio - 1.0) < x)
            departures.append(np.abs(ratio - 1.0).max())
        assert departures[1] < departures[0] / 5.0  # tenfold smaller x -> ~tenfold smaller gap

    def test_published_form_is_ln10_smaller(self) -> None:
        fv = np.array([5000.0, 12000.0, 25000.0])
        distance = np.array([60.0, 170.0, 330.0])
        np.testing.assert_allclose(
            watkins_feldman_velocity(fv, distance, as_published=True) * LN10,
            watkins_feldman_velocity(fv, distance),
        )

    def test_undefined_log_is_nan_not_zero(self) -> None:
        """Hard rule 5: the published file writes 0 here; we must not."""
        out = watkins_feldman_velocity(np.array([-50.0, 0.0, 500.0]), np.array([5.0, 5.0, 5.0]))
        assert np.isnan(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])

    def test_ramp_endpoints(self) -> None:
        w = published_ramp_weight([0.0, DEFS.ramp_velocity / 2, DEFS.ramp_velocity, 9000.0])
        np.testing.assert_allclose(w, [1.0, 0.5, 0.0, 0.0])

    def test_vpec_blend_and_negative_fv(self) -> None:
        vpds = np.array([100.0, 100.0, 100.0, -300.0, -300.0])
        vpwf = np.array([40.0, 40.0, 40.0, 0.0, 0.0])
        vls = np.array([0.0, DEFS.ramp_velocity / 2, 5000.0, 600.0, 600.0])
        fv = np.array([100.0, 1500.0, 5000.0, -10.0, 0.0])
        out = published_vpec(vpds, vpwf, vls, fv)
        # fV < 0 takes Vpds outright; fV = 0 blends with the filler 0.
        np.testing.assert_allclose(out, [100.0, 70.0, 40.0, -300.0, -300.0 * 0.8])

    def test_error_is_the_derivative_of_the_ln_velocity(self) -> None:
        fv, modulus, edm, step = np.array([8000.0]), 35.0, np.array([0.4]), 1e-4
        up = watkins_feldman_velocity(fv, distance_from_modulus(np.array([modulus + step])))
        down = watkins_feldman_velocity(fv, distance_from_modulus(np.array([modulus - step])))
        slope = np.abs(up - down) / (2 * step)
        np.testing.assert_allclose(velocity_error_from_modulus_error(edm, fv), slope * edm, rtol=1e-6)

    def test_zero_edm_gives_nan(self) -> None:
        out = velocity_error_from_modulus_error(np.array([0.0, 0.3]), np.array([800.0, 800.0]))
        assert np.isnan(out[0]) and out[1] > 0


# --------------------------------------------------------------------------
# A synthetic CF4 velocities file, built the way EDD builds it
# --------------------------------------------------------------------------

_VELOCITY_COLUMNS = {
    "pgc": ("%7s", "---", "PGC"),
    "DMzp": ("%6.3f", "mag", "Distance modulus"),
    "eDM": ("%5.3f", "mag", "Error assigned to distance modulus"),
    "D": ("%5.1f", "Mpc", "Distance"),
    "Vls": ("%5d", "km/s", "Local Sheet velocity"),
    "Vcmb": ("%5d", "km/s", "CMB velocity"),
    "fV": ("%5d", "km/s", "CMB velocity with curvature adjustment"),
    "Vpds": ("%6d", "km/s", "Peculiar velocity following Davis & Scrimgeour (2014)"),
    "Vpwf": ("%6d", "km/s", "Peculiar velocity following Watkins & Feldman (2015)"),
    "Vpec": ("%6d", "km/s", "Peculiar velocity wih ramp between Vpds and Vpwf"),
    "RA": ("%8.4f", "deg", "RA"),
    "DE": ("%8.4f", "deg", "DE"),
}


def _synthetic_velocities(tmp_path, n=4000, seed=1, true_rms=0.0, edm=0.4):
    """Groups with a known true velocity and Gaussian modulus noise, written
    with EDD's own formulas -- log10 in Vpwf included -- and EDD rounding."""
    rng = np.random.default_rng(seed)
    cz = rng.uniform(4000.0, 30000.0, n)
    fv = curvature_corrected_velocity(cz)
    v_true = rng.normal(0.0, true_rms, n) if true_rms else np.zeros(n)
    # Distance such that the ln-form velocity equals v_true exactly, then
    # Gaussian scatter of edm magnitudes in the modulus.
    factor = (1.0 + fv / conventions.SPEED_OF_LIGHT) / fv
    hubble_velocity = fv / np.exp(v_true * factor)
    modulus = 25.0 + 5.0 * np.log10(hubble_velocity / DEFS.h0) + rng.normal(0.0, edm, n)
    modulus = np.round(modulus, 3)
    distance = distance_from_modulus(modulus)
    fv = np.round(fv)
    vpds = davis_scrimgeour_velocity(fv, distance)
    vpwf = watkins_feldman_velocity(fv, distance, as_published=True)
    vls = cz + rng.normal(0.0, 300.0, n)
    frame = pd.DataFrame(
        {
            "pgc": np.arange(n) + 10,
            "DMzp": modulus,
            "eDM": np.full(n, edm),
            "D": distance,
            "Vls": vls,
            "Vcmb": cz,
            "fV": fv,
            "Vpds": np.round(vpds),
            "Vpwf": np.round(vpwf),
            "Vpec": published_vpec(np.round(vpds), np.round(vpwf), np.round(vls), fv),
            "RA": rng.uniform(0.0, 360.0, n),
            "DE": np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n))),
        }
    )
    path = tmp_path / "velocities.csv"
    _write_edd(path, "Synthetic CF4 Velocities", _VELOCITY_COLUMNS, frame)
    return read_edd_csv(path)


class TestReconstruction:
    def test_synthetic_file_reproduced_within_rounding(self, tmp_path) -> None:
        catalog = _synthetic_velocities(tmp_path, n=500)
        summary = audit.reconstruct_cf4_velocities(catalog).summary
        for column in ("D", "fV", "Vpds", "Vpwf", "Vpec"):
            assert summary.loc[column, "fraction_within_rounding"] == 1.0, column

    def test_a_natural_log_vpwf_row_is_caught(self, tmp_path) -> None:
        """The envelope is tight enough to see a factor ln 10 in one row."""
        catalog = _synthetic_velocities(tmp_path, n=500)
        table = catalog.table.copy()
        table.loc[0, "Vpwf"] = round(table.loc[0, "Vpwf"] * LN10)
        planted = EddCatalog(catalog.path, catalog.title, table, catalog.columns)
        residuals = audit.reconstruct_cf4_velocities(planted).residuals
        assert abs(residuals.loc[0, "Vpwf_residual"]) > residuals.loc[0, "Vpwf_envelope"]

    def test_linear_regime_ratio_reads_the_scale(self, tmp_path) -> None:
        catalog = _synthetic_velocities(tmp_path, edm=0.05)
        ratio = audit.linear_regime_ratio(catalog)
        assert ratio.n_rows > 100
        assert ratio.median_published_ratio == pytest.approx(1.0 / LN10, abs=0.01)
        assert ratio.median_natural_log_ratio == pytest.approx(1.0, abs=0.02)

    def test_noise_table_resolves_the_puzzle(self, tmp_path) -> None:
        """Pure noise (v_true = 0): eDM must explain the ln-form scatter
        exactly (z-std 1) and over-explain the published one by ln 10."""
        catalog = _synthetic_velocities(tmp_path, n=6000, true_rms=0.0)
        config = audit.AuditConfig(cz_edges=(4000.0, 15000.0, 30000.0))
        noise = audit.velocity_noise_by_depth(catalog, config=config)
        np.testing.assert_allclose(noise["zstd_vpwf_ln"], 1.0, atol=0.05)
        np.testing.assert_allclose(noise["zstd_vpec_published"], 1.0 / LN10, atol=0.03)

    def test_signal_shows_up_as_excess_scatter(self, tmp_path) -> None:
        catalog = _synthetic_velocities(tmp_path, n=6000, true_rms=600.0, edm=0.1)
        config = audit.AuditConfig(cz_edges=(4000.0, 10000.0))
        noise = audit.velocity_noise_by_depth(catalog, config=config)
        assert noise.loc[0, "implied_signal_rms"] == pytest.approx(600.0, rel=0.15)


# --------------------------------------------------------------------------
# Audit checks on synthetic tables
# --------------------------------------------------------------------------


def _catalog(table):
    columns = pd.DataFrame(
        {"format": "%5d", "unit": "", "description": ""}, index=pd.Index(table.columns)
    )
    return EddCatalog(path=None, title="synthetic", table=table, columns=columns)


class TestTempelTraps:
    @pytest.fixture
    def tempel(self):
        # Two isolated galaxies; group 7 complete; group 8 lost its Rg = 1 row
        # and has fewer rows than Ng; group 9 has disagreeing gVcmb.
        table = pd.DataFrame(
            {
                "ID": [1, 2, 3, 4, 5, 6, 7, 8],
                "pgc": [11, 12, 13, 14, 15, 16, 17, 18],
                "IDg": [0, 0, 7, 7, 8, 8, 9, 9],
                "Ng": [1, 1, 2, 2, 4, 4, 2, 2],
                "Rg": [1, 1, 1, 2, 2, 3, 1, 2],
                "dc": [50.0, 60.0, 70.0, 70.0, 80.0, 80.0, 90.0, 90.0],
                "gVcmb": [3000, 4000, 5000, 5000, 6000, 6000, 7000, 7100],
                "zcmb": [0.01, 0.013, 0.017, 0.0168, 0.02, 0.0201, 0.023, 0.0237],
            }
        )
        return _catalog(table)

    def test_counts(self, tempel) -> None:
        traps = audit.tempel_traps(tempel)
        assert traps.n_isolated == 2
        assert traps.naive_largest_group == 2  # the fake "group" IDg = 0
        assert traps.n_groups == 3
        assert traps.groups_without_brightest == 1
        assert traps.groups_fewer_rows_than_ng == 1
        assert traps.groups_gvcmb_varies == 1
        assert traps.groups_dc_varies == 0
        assert traps.isolated_ng_values == (1,)

    def test_group_members_sorted_by_rank(self, tempel) -> None:
        rows = audit.group_members(tempel, 8)
        assert list(rows["Rg"]) == [2, 3]
        assert "cz_cmb" in rows


class TestCf4FileComparison:
    def test_planted_disagreements_are_found(self) -> None:
        pgc = np.array([1, 2, 3, 4])
        velocities = pd.DataFrame(
            {"pgc": pgc, "Vcmb": [100, 200, 300, 400], "RA": [10.0, 20.0, 30.0, 40.0],
             "DE": [0.0, 0.0, 0.0, 0.0], "eDM": [0.0, 0.3, 0.3, 0.3], "D": [1.0, 2.0, 3.0, 4.0]}
        )
        groups = pd.DataFrame(
            {"pgc": pgc, "Vcmb": [100, 250, 300, 400], "RA": [10.0, 20.0, 30.0, 70.0],
             "DE": [0.0, 0.0, 0.0, 0.0], "eDM_av": [0.1, 0.3, 0.3, 0.3], "Nc": [1.0, np.nan, np.nan, np.nan]}
        )
        cv = _catalog(velocities)
        cv.columns.loc["eDM", "format"] = "%5.3f"
        result = audit.compare_cf4_files(cv, _catalog(groups))
        assert result.same_row_order
        assert list(result.vcmb_mismatch["pgc"]) == [2]
        assert list(result.position_mismatch["pgc"]) == [4]
        assert result.position_mismatch["separation_deg"].iloc[0] == pytest.approx(30.0)
        assert list(result.edm_mismatch["pgc"]) == [1]


def test_footprint_area_of_a_full_sky_is_four_pi() -> None:
    rng = np.random.default_rng(3)
    n = 400_000
    ra = rng.uniform(0.0, 360.0, n)
    de = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    table = audit.footprint_area_table(ra, de, audit.AuditConfig(footprint_nsides=(8,)))
    full_sky = 4.0 * np.pi * np.rad2deg(1.0) ** 2
    assert table.loc[8, "area_deg2"] == pytest.approx(full_sky)


def test_figures_build(tmp_path) -> None:
    catalog = _synthetic_velocities(tmp_path, n=300)
    noise = audit.velocity_noise_by_depth(
        catalog, config=audit.AuditConfig(cz_edges=(4000.0, 15000.0, 30000.0))
    )
    sample = _sky_sample(np.array([10.0, 200.0, 300.0]), np.array([0.0, 40.0, -30.0]),
                         cz_limit=29_979.0)
    drift = pd.DataFrame({"source": ["TF"] * 3, "cz_lo": [0.0, 5e3, 1e4], "cz_hi": [5e3, 1e4, 2e4],
                          "n": [30, 30, 30], "cz_median": [2e3, 7e3, 15e3],
                          "v_median": [0.0, 10.0, 20.0], "v_median_err": [5.0, 5.0, 5.0]})
    figures = [
        audit.make_sky_map(catalog.table["RA"], catalog.table["DE"], title="synthetic"),
        audit.make_cz_histogram_figure([sample, sample]),
        audit.make_overlap_sky_map([sample, sample], title="overlap", cz_max=10_000.0),
        audit.make_source_drift_figure(drift, audit.zero_point_drift_table(drift)),
        audit.make_vpec_vs_cz_figure(catalog, noise),
        audit.make_velocity_definition_figure(catalog),
        audit.make_noise_comparison_figure(noise),
    ]
    assert all(fig.axes for fig in figures)
    matplotlib.pyplot.close("all")


# --------------------------------------------------------------------------
# Stage 1 addendum: 2MRS, the three-catalog overlap, CF4 by source
# --------------------------------------------------------------------------


def _sky_sample(ra, de, *, glat=None, cz=None, cz_limit=None):
    n = ra.size
    return audit.SkySample(
        name="synthetic", unit="objects", ra=ra, de=de, glon=np.full(n, 180.0),
        glat=np.full(n, 45.0) if glat is None else glat,
        cz=np.full(n, 5000.0) if cz is None else cz, cz_frame="CMB", color="#2a78d6",
        cz_limit=cz_limit,
    )


def test_twomrs_checks_find_planted_defects() -> None:
    table = pd.DataFrame(
        {
            "pgc": [1, 2, 3, 3],
            "ID_2MASXJ": ["a", "b", "c", "d"],
            "K_c": [11.0, 11.8, 11.2, 11.5],
            "K_tc": [10.9, 11.7, 11.75, 11.4],
            "Glon": [100.0, 10.0, 200.0, 250.0],
            "Glat": [30.0, 6.0, -20.0, 60.0],
            "Vhel": [-100, 5000, 8000, 12000],
            "e_V": [0, 30, 40, 0],
        }
    )
    checks = audit.twomrs_checks(_catalog(table))
    assert checks.pgc_unique is False and checks.id_unique is True
    assert checks.n_isophotal_beyond_limit == 1
    assert checks.n_outside_mask == 1  # l = 10 (bulge) at b = 6 < 8
    assert checks.n_nonpositive_vhel == 1
    assert checks.n_velocity_error_not_given == 2
    assert checks.max_total_magnitude == 11.75


def test_twomrs_mask_area_matches_the_spherical_band_formula() -> None:
    # Area with |b| < b0 is 4 pi sin(b0); the bulge widens the cut over 60/360 of l.
    sin5, sin8 = np.sin(np.deg2rad(5.0)), np.sin(np.deg2rad(8.0))
    full_sky = 4.0 * np.pi * np.rad2deg(1.0) ** 2
    expected = full_sky * (1.0 - sin5 - (60.0 / 360.0) * (sin8 - sin5))
    assert audit.twomrs_mask_area() == pytest.approx(expected, rel=1e-3)


def test_twomrs_crossmatch_compares_heliocentric_velocities() -> None:
    c = conventions.SPEED_OF_LIGHT
    twomrs = _catalog(pd.DataFrame({"pgc": [1, 2, 3], "Vhel": [3000, 6000, 9000]}))
    tempel = _catalog(pd.DataFrame({"pgc": [1, 2, 9], "zobs": [3000 / c, 6500 / c, 0.01]}))
    cf4 = _catalog(pd.DataFrame({"pgc": [3], "Vhel": [9000]}))
    table = audit.twomrs_crossmatch(twomrs, tempel, cf4)
    tempel_row = table.iloc[0]
    assert tempel_row["n_matched"] == 2
    assert tempel_row["n_beyond_tolerance"] == 1
    assert table.iloc[1]["n_matched"] == 1 and table.iloc[1]["median_dv"] == 0.0


def test_sky_overlap_counts_footprint_and_mask() -> None:
    config = audit.AuditConfig(overlap_nside=8)
    tempel = _catalog(pd.DataFrame({"RA": [180.0], "DE": [30.0]}))
    pixels = audit.sdss_footprint_pixels(tempel, config)
    # In the footprint and the mask; in the mask only; in neither (b = 2);
    # in the footprint but too deep for the cz cut.
    sample = _sky_sample(
        np.array([180.0, 0.0, 0.0, 180.0]), np.array([30.0, -60.0, -60.0, 30.0]),
        glat=np.array([45.0, 45.0, 2.0, 45.0]), cz=np.array([5e3, 5e3, 5e3, 2e4]),
    )
    table = audit.sky_overlap_table([sample], pixels, config, cz_max=1e4)
    row = table.loc["synthetic"]
    assert (row["n"], row["in_sdss"], row["in_twomrs_mask"], row["in_both"]) == (3, 1, 2, 1)


def _cf4_groups_table():
    # Five groups: FP inside the footprint, FP in the south outside it,
    # TF, FP+TF (two methods: excluded from sources), a calibrator only.
    nan = np.nan
    return pd.DataFrame(
        {
            "pgc": [1, 2, 3, 4, 5],
            "RA": [180.0, 0.0, 90.0, 180.0, 10.0],
            "DE": [30.0, -40.0, 20.0, 30.0, 50.0],
            "Vcmb": [5000, 6000, 3000, 12000, 500],
            "Nc": [0, 0, 0, 0, 1],
            "DM_FP": [34.0, 34.5, nan, 35.0, nan],
            "DM_TFR": [nan, nan, 33.0, 35.1, nan],
            "DM_SNIa": [nan] * 5,
            "DM_SNII": [nan] * 5,
            "DM_SBFo": [nan] * 5,
            "DM_SBFir": [nan] * 5,
        }
    )


def test_cf4_method_table_and_methods_per_group() -> None:
    groups = _catalog(_cf4_groups_table())
    table = audit.cf4_method_table(groups, audit.AuditConfig(analysis_cz_max=10_000.0))
    assert table.loc["FP", ["n_groups", "n_single_method", "n_groups_near"]].tolist() == [3, 2, 2]
    assert table.loc["TF", "n_single_method"] == 1
    assert table.loc["calibrators (Nc > 0)", "n_groups"] == 1
    assert audit.cf4_methods_per_group(groups).to_dict() == {0: 1, 1: 3, 2: 1}


def test_cf4_source_samples_split_fp_by_sky_and_drop_undefined_velocity() -> None:
    config = audit.AuditConfig(overlap_nside=8)
    g = _cf4_groups_table()
    pixels = audit.sdss_footprint_pixels(_catalog(pd.DataFrame({"RA": [180.0], "DE": [30.0]})), config)
    v = pd.DataFrame({"pgc": [1, 2, 3, 4, 5], "fV": [5100.0, 6100.0, -5.0, 12300.0, 500.0],
                      "DMzp": [34.0, 34.5, 33.0, 35.0, 30.0]})
    sources = audit.cf4_source_samples(_catalog(g), _catalog(v), pixels, config)
    labels = dict(zip(sources["pgc"], sources["source"]))
    # pgc 3 (TF) has fV < 0: velocity undefined, dropped rather than filled.
    assert labels == {1: audit.SOURCE_FP_SDSS, 2: audit.SOURCE_FP_SOUTH}
    assert np.isfinite(sources["v"]).all()


def test_velocity_drift_drops_small_cells() -> None:
    rows = pd.DataFrame({"source": ["TF"] * 5, "Vcmb": [1000, 1500, 2000, 6000, 6100],
                         "v": [10.0, 20.0, 30.0, 0.0, 0.0]})
    drift = audit.velocity_drift_by_source(
        rows, audit.AuditConfig(drift_cz_edges=(0.0, 5000.0, 10000.0), drift_min_count=3)
    )
    assert drift["cz_lo"].tolist() == [0.0]
    assert drift.loc[0, "v_median"] == 20.0


def test_zero_point_fit_recovers_a_planted_modulus_offset() -> None:
    # A modulus offset dm gives dv = -(ln 10 / 5) dm cz, on top of a bulk offset.
    dm, bulk = 0.05, 120.0
    cz = np.array([2000.0, 5000.0, 9000.0, 14000.0, 20000.0])
    drift = pd.DataFrame({"source": "SNIa", "cz_lo": cz - 1, "cz_hi": cz + 1, "n": 50,
                          "cz_median": cz, "v_median": bulk - LN10 / 5.0 * dm * cz,
                          "v_median_err": 30.0})
    fit = audit.zero_point_drift_table(drift).loc["SNIa"]
    assert fit["modulus_offset"] == pytest.approx(dm)
    assert fit["offset"] == pytest.approx(bulk)
    assert fit["chi2"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# The real files: the roadmap's Stage 1 numbers
# --------------------------------------------------------------------------

_PATHS = PathsConfig()
_HAVE_REAL = all(
    p.exists()
    for p in (_PATHS.cf4_velocities_catalog, _PATHS.cf4_groups_catalog, _PATHS.sdss_tempel_catalog)
)


@pytest.mark.skipif(not _HAVE_REAL, reason="real EDD catalogs not in data/")
class TestRealFiles:
    @pytest.fixture(scope="class")
    @classmethod
    def velocities(cls):
        return load_cf4_velocities()

    def test_counts(self, velocities) -> None:
        assert len(velocities.table) == 38_057
        assert velocities.table.isna().sum().sum() == 0
        assert len(load_tempel_galaxies().table) == 305_952

    def test_velocity_columns_reproduced(self, velocities) -> None:
        summary = audit.reconstruct_cf4_velocities(velocities).summary
        assert summary.loc["D", "fraction_within_rounding"] == 1.0
        assert summary.loc["fV", "fraction_within_rounding"] == 1.0
        assert summary.loc["Vpec", "fraction_within_rounding"] == 1.0
        assert summary.loc["Vpds", "fraction_within_rounding"] > 0.99
        assert summary.loc["Vpwf", "fraction_within_rounding"] > 0.99

    def test_published_vpwf_is_ln10_low(self, velocities) -> None:
        ratio = audit.linear_regime_ratio(velocities)
        assert ratio.median_published_ratio == pytest.approx(1.0 / LN10, abs=0.01)

    def test_tempel_traps(self) -> None:
        traps = audit.tempel_traps(load_tempel_galaxies())
        assert (traps.n_isolated, traps.n_groups) == (130_010, 49_671)
        assert traps.groups_without_brightest == 375
        assert traps.groups_dc_varies == 0


@pytest.mark.skipif(not _PATHS.twomrs_catalog.exists(), reason="2MRS catalog not in data/")
def test_real_twomrs_matches_its_survey_definition() -> None:
    checks = audit.twomrs_checks(load_twomrs_galaxies())
    assert checks.n_galaxies == 43_526
    assert checks.pgc_unique and checks.id_unique and checks.n_missing == 0
    assert checks.max_total_magnitude <= audit.TWOMRS_K_LIMIT
    assert checks.n_outside_mask == 0
