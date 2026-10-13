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
    load_cf4_velocities,
    load_tempel_galaxies,
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
        assert len(load_tempel_galaxies(paths).table) == 3
        assert len(load_cf4_velocities(paths).table) == 3


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
    tempel = _catalog(pd.DataFrame({"zcmb": [0.02, 0.05, 0.09]}))
    figures = [
        audit.make_sky_map(catalog.table["RA"], catalog.table["DE"], title="synthetic"),
        audit.make_cz_histogram_figure(tempel, catalog),
        audit.make_vpec_vs_cz_figure(catalog, noise),
        audit.make_velocity_definition_figure(catalog),
        audit.make_noise_comparison_figure(noise),
    ]
    assert all(fig.axes for fig in figures)
    matplotlib.pyplot.close("all")


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
