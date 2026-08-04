"""
test_settings.py
-----------------
Unit tests for the dvcorr.config dataclasses: PathsConfig, CosmologyConfig,
ShellConfig, SelectionConfig, and the Settings aggregator.

These are ordinary dataclass/validation tests, not part of the sign gate
(tests/test_geometry.py); nothing here touches conventions.py's frozen conventions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import integrate

from dvcorr import conventions
from dvcorr.estimators.shell_dipole import shell_dipole
from dvcorr.config import (
    CATALOG_FULL,
    CATALOG_MVIR12,
    SPACING_LINEAR,
    SPACING_LOG,
    VALID_CATALOGS,
    CatalogConfig,
    CosmologyConfig,
    PathsConfig,
    SelectionConfig,
    Settings,
    ShellConfig,
    default_settings,
    volume_weighted_shell_radii,
)


# ---------------------------------------------------------------------------
# PathsConfig
# ---------------------------------------------------------------------------


class TestPathsConfig:
    def test_derived_paths_land_under_project_root(self) -> None:
        paths = PathsConfig()
        assert paths.data_dir == paths.project_root / "data"
        assert paths.output_dir == paths.project_root / "output"

    def test_catalog_filenames(self) -> None:
        paths = PathsConfig()
        assert paths.mdpl2_catalog_full.name == "mdpl2_rockstar_snapnum125.csv"
        assert paths.mdpl2_catalog_mvir12.name == "mdpl2_rockstar_125_pid-1_mvir12.csv"
        assert paths.cf4_groups_catalog.name == "CF4_Groups.csv"
        assert paths.cf4_velocities_catalog.name == "CF4_Groups_Velocities.csv"
        assert paths.sdss_tempel_catalog.name == "SDSS_Temple.csv"
        # And they sit under data_dir specifically.
        assert paths.mdpl2_catalog_full.parent == paths.data_dir
        assert paths.mdpl2_catalog_mvir12.parent == paths.data_dir
        assert paths.cf4_groups_catalog.parent == paths.data_dir
        assert paths.cf4_velocities_catalog.parent == paths.data_dir
        assert paths.sdss_tempel_catalog.parent == paths.data_dir

    def test_halo_catalog_resolves_names_to_both_catalogs(self) -> None:
        paths = PathsConfig()
        for name in VALID_CATALOGS:
            csv_path = paths.halo_catalog(name, parquet=False)
            parquet_path = paths.halo_catalog(name)
            assert csv_path.suffix == ".csv"
            assert parquet_path.suffix == ".parquet"
            # Same catalog, two encodings -- not two different catalogs.
            assert parquet_path.stem == csv_path.stem
            assert parquet_path.parent == paths.data_dir
        # The two names must not collide onto one file, which would make a
        # catalog comparison silently compare a catalog with itself.
        assert paths.halo_catalog(CATALOG_FULL) != paths.halo_catalog(CATALOG_MVIR12)

    def test_halo_catalog_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError):
            PathsConfig().halo_catalog("ful")

    def test_ensure_output_dir_creates_directory(self, tmp_path) -> None:
        # Point output_dir at a tmp directory so the test never touches the
        # real repo's output/ folder.
        target = tmp_path / "settings_test_output"
        paths = PathsConfig(output_dir=target)
        assert not target.exists()
        paths.ensure_output_dir()
        assert target.is_dir()

    def test_explicit_overrides_are_respected(self, tmp_path) -> None:
        paths = PathsConfig(project_root=tmp_path)
        assert paths.data_dir == tmp_path / "data"
        assert paths.mdpl2_catalog_mvir12 == tmp_path / "data" / "mdpl2_rockstar_125_pid-1_mvir12.csv"


# ---------------------------------------------------------------------------
# CosmologyConfig
# ---------------------------------------------------------------------------


class TestCosmologyConfig:
    def test_h_matches_config_hubble_param(self) -> None:
        cosmology = CosmologyConfig()
        assert cosmology.h == pytest.approx(conventions.HUBBLE_PARAM, abs=1e-9)

    def test_to_colossus_dict_keys(self) -> None:
        cosmology = CosmologyConfig()
        d = cosmology.to_colossus_dict()
        assert set(d.keys()) == {"flat", "H0", "Om0", "Ode0", "Ob0", "sigma8", "ns"}
        assert d["H0"] == cosmology.H0
        assert d["sigma8"] == cosmology.sigma8

    def test_is_frozen(self) -> None:
        cosmology = CosmologyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cosmology.H0 = 70.0

    def test_inconsistent_h0_raises(self) -> None:
        # H0 = 70.0 gives h = 0.70, which does not match conventions.HUBBLE_PARAM
        # (0.6777); __post_init__ must catch this rather than silently
        # accepting a cosmology that disagrees with the frozen convention.
        with pytest.raises(ValueError):
            CosmologyConfig(H0=70.0)


# ---------------------------------------------------------------------------
# ShellConfig
# ---------------------------------------------------------------------------

# Named constants for the log-spacing tests below, following this module's
# existing style of naming edge-generating parameters rather than inlining
# them (CLAUDE.md hard rule 4).
_LOG_MIN_RADIUS = 1.0
_LOG_MAX_RADIUS = 64.0
_LOG_N_BINS = 12
_TYPO_SPACING = "lgo"  # plausible typo of SPACING_LOG


class TestShellConfig:
    def test_shell_edges_shape_and_monotonicity(self) -> None:
        shells = ShellConfig()
        edges = shells.shell_edges
        assert edges.ndim == 1
        assert np.all(np.diff(edges) > 0.0)
        n_bins = edges.size - 1
        assert n_bins > 0

    def test_shell_edges_respect_max_analysis_radius(self) -> None:
        shells = ShellConfig()
        assert shells.shell_edges[-1] <= conventions.MAX_ANALYSIS_RADIUS

    def test_shell_centers_shape_and_bounds(self) -> None:
        shells = ShellConfig()
        edges = shells.shell_edges
        centers = shells.shell_centers
        n_bins = edges.size - 1
        assert centers.shape == (n_bins,)
        assert np.all(centers > edges[:-1])
        assert np.all(centers < edges[1:])

    def test_shell_edges_strictly_increasing_under_float_accumulation(self) -> None:
        # np.arange(0.0, 0.3, 0.1) accumulates to a final element that lands
        # on (or a hair past) 0.3 due to floating-point error; naively
        # appending max_radius on top of that would duplicate the outer edge
        # and violate strict monotonicity (shell_dipole's own validation would
        # then reject it). Guard against that regression explicitly.
        shells = ShellConfig(min_radius=0.0, max_radius=0.3, radii_step=0.1)
        edges = shells.shell_edges
        assert np.all(np.diff(edges) > 0.0)
        assert edges[-1] == pytest.approx(0.3)

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError):
            ShellConfig(min_radius=100.0, max_radius=50.0)

    def test_negative_step_raises(self) -> None:
        with pytest.raises(ValueError):
            ShellConfig(radii_step=-5.0)

    def test_zero_step_raises(self) -> None:
        with pytest.raises(ValueError):
            ShellConfig(radii_step=0.0)

    def test_negative_min_radius_raises(self) -> None:
        with pytest.raises(ValueError):
            ShellConfig(min_radius=-10.0)

    def test_max_radius_beyond_max_analysis_radius_raises(self) -> None:
        with pytest.raises(ValueError):
            ShellConfig(max_radius=conventions.MAX_ANALYSIS_RADIUS + 1.0)

    def test_shell_edges_feed_shell_dipole(self) -> None:
        # A tiny toy: one central object, a couple of neighbors, using the
        # edges ShellConfig produces. This is only exercising that the
        # produced edges are well-formed inputs to shell_dipole, not the
        # sign gate (see tests/test_geometry.py, tests/test_shell_dipole.py).
        shells = ShellConfig(min_radius=0.0, max_radius=20.0, radii_step=10.0)
        s_center = np.array([0.0, 0.0, 0.0])
        s_neighbors = np.array(
            [
                [5.0, 0.0, 0.0],
                [0.0, 15.0, 0.0],
            ]
        )
        result = shell_dipole(s_center, s_neighbors, shells.shell_edges)
        n_bins = shells.shell_edges.size - 1
        assert result.pair_count.shape == (n_bins,)
        assert result.monopole.shape == (n_bins,)
        assert result.dipole.shape == (n_bins,)
        assert result.pair_count.sum() == pytest.approx(2.0)

    def test_positional_construction_is_unchanged(self) -> None:
        """Backward-compatibility pin.

        `spacing` and `n_bins` were appended AFTER `sigma_star` in the field
        order specifically so that every existing positional call site
        (`ShellConfig(min_radius, max_radius, radii_step)`) keeps binding to
        the same three fields. This test fails LOUDLY if a future edit ever
        inserts a new field earlier in the dataclass -- positional arguments
        would then silently land on the wrong keyword with no error, only a
        wrong binning.
        """
        shells = ShellConfig(20.0, 150.0, 10.0)
        assert shells.spacing == SPACING_LINEAR
        np.testing.assert_array_equal(shells.shell_edges, np.arange(20.0, 160.0, 10.0))

    def test_log_shell_edges_are_geometric(self) -> None:
        shells = ShellConfig(
            _LOG_MIN_RADIUS, _LOG_MAX_RADIUS, spacing=SPACING_LOG, n_bins=_LOG_N_BINS
        )
        edges = shells.shell_edges

        assert edges.size == _LOG_N_BINS + 1
        # np.geomspace pins BOTH endpoints exactly -- unlike
        # linear_shell_edges's drop-then-append dance (needed because
        # np.arange can float-accumulate past max_radius), log_shell_edges
        # has no drift to defend against, so an exact == is the honest
        # assertion here; pytest.approx would stop testing that property.
        assert edges[0] == _LOG_MIN_RADIUS
        assert edges[-1] == _LOG_MAX_RADIUS
        np.testing.assert_allclose(np.diff(np.log(edges)), np.log(2.0) / 2.0)
        assert np.all(np.diff(edges) > 0.0)

    def test_log_spacing_requires_positive_min_radius(self) -> None:
        """min_radius <= 0 is invalid ONLY under log spacing (np.geomspace is
        undefined at zero); the identical min_radius=0.0 must remain valid
        under linear spacing, which is what keeps the existing
        `ShellConfig(0.0, 0.3, 0.1)` case (test_shell_edges_strictly_
        increasing_under_float_accumulation, above) working."""
        with pytest.raises(ValueError):
            ShellConfig(0.0, 64.0, spacing=SPACING_LOG, n_bins=12)

        linear = ShellConfig(0.0, 64.0, spacing=SPACING_LINEAR)
        assert linear.min_radius == 0.0

    def test_unknown_spacing_raises(self) -> None:
        """A plausible typo ('lgo') raises, and the message names the valid values."""
        with pytest.raises(ValueError) as exc_info:
            ShellConfig(spacing=_TYPO_SPACING)
        assert SPACING_LINEAR in str(exc_info.value)
        assert SPACING_LOG in str(exc_info.value)

    def test_n_bins_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ShellConfig(n_bins=0)
        with pytest.raises(ValueError):
            ShellConfig(n_bins=-3)

    def test_log_shell_edges_respect_max_analysis_radius(self) -> None:
        with pytest.raises(ValueError):
            ShellConfig(
                _LOG_MIN_RADIUS,
                conventions.MAX_ANALYSIS_RADIUS + 1.0,
                spacing=SPACING_LOG,
                n_bins=_LOG_N_BINS,
            )

        at_ceiling = ShellConfig(
            _LOG_MIN_RADIUS,
            conventions.MAX_ANALYSIS_RADIUS,
            spacing=SPACING_LOG,
            n_bins=_LOG_N_BINS,
        )
        assert at_ceiling.shell_edges[-1] <= conventions.MAX_ANALYSIS_RADIUS

    def test_shell_centers_are_still_midpoints_under_log_spacing(self) -> None:
        """Anti-regression pin for the deliberate decision NOT to redefine
        `shell_centers` under log spacing.

        Three existing binning-correctness tests in
        tests/test_velocity_centered_dipole.py assert
        `shell_centers == [5, 15, 25]` on LINEAR edges; this is the companion
        pin that a future edit does not quietly make `shell_centers` track
        `shell_effective_radii` instead, which would break those tests
        silently the day log spacing becomes the default.
        """
        shells = ShellConfig(
            _LOG_MIN_RADIUS, _LOG_MAX_RADIUS, spacing=SPACING_LOG, n_bins=_LOG_N_BINS
        )
        edges = shells.shell_edges
        centers = shells.shell_centers
        midpoint = 0.5 * (edges[:-1] + edges[1:])
        geometric_mean = np.sqrt(edges[:-1] * edges[1:])

        np.testing.assert_allclose(centers, midpoint)
        assert np.all(centers > geometric_mean)
        assert np.all(centers < shells.shell_effective_radii)

    def test_zero_bin_is_prepended_without_disturbing_the_log_ladder(self) -> None:
        """`include_zero_bin` is purely ADDITIVE.

        `min_radius` keeps meaning "innermost LOG edge", so the geometric
        ladder above it must come out bit-for-bit identical to the same
        config with the flag off -- the new bin is one extra entry at the
        front, not a re-spacing of the existing ones.
        """
        kwargs = dict(spacing=SPACING_LOG, n_bins=_LOG_N_BINS)
        plain = ShellConfig(_LOG_MIN_RADIUS, _LOG_MAX_RADIUS, **kwargs)
        with_zero = ShellConfig(
            _LOG_MIN_RADIUS, _LOG_MAX_RADIUS, include_zero_bin=True, **kwargs
        )
        edges = with_zero.shell_edges

        assert edges.size == plain.shell_edges.size + 1
        assert edges[0] == 0.0
        assert edges[1] == _LOG_MIN_RADIUS
        np.testing.assert_array_equal(edges[1:], plain.shell_edges)
        assert np.all(np.diff(edges) > 0.0)

    def test_zero_bin_defaults_off_and_is_inert_under_linear_spacing(self) -> None:
        """The flag defaults off (so no existing config's binning moves), and
        under `SPACING_LINEAR` it is inert -- a linear binning already reaches
        zero by setting `min_radius = 0.0`, so there is nothing to add."""
        assert ShellConfig().include_zero_bin is False

        linear = ShellConfig(20.0, 150.0, 10.0, include_zero_bin=True)
        np.testing.assert_array_equal(
            linear.shell_edges, ShellConfig(20.0, 150.0, 10.0).shell_edges
        )

    def test_zero_bin_effective_radius_is_the_full_sphere_closed_form(self) -> None:
        """The [0, min_radius) shell's abscissa is 0.75 * min_radius, not
        min_radius/2: it is volume-weighted like every other shell, and for a
        full sphere `volume_weighted_shell_radii` reduces to 3R/4."""
        shells = ShellConfig(
            _LOG_MIN_RADIUS,
            _LOG_MAX_RADIUS,
            spacing=SPACING_LOG,
            n_bins=_LOG_N_BINS,
            include_zero_bin=True,
        )
        assert shells.shell_effective_radii[0] == pytest.approx(0.75 * _LOG_MIN_RADIUS)
        assert shells.shell_centers[0] == pytest.approx(0.5 * _LOG_MIN_RADIUS)

    def test_include_zero_bin_must_be_a_bool(self) -> None:
        with pytest.raises(ValueError):
            ShellConfig(spacing=SPACING_LINEAR, include_zero_bin="yes")


# ---------------------------------------------------------------------------
# volume_weighted_shell_radii
# ---------------------------------------------------------------------------

_QUAD_ATOL = 1e-8       # numerical-quadrature agreement tolerance
_SCALE_INVARIANT_Q_SQRT2 = 1.01943414   # r_eff/midpoint at bin ratio q = sqrt(2)


class TestVolumeWeightedShellRadii:
    def test_full_sphere_gives_the_closed_form(self) -> None:
        """edges[0] == 0 gives exactly 0.75 * r2 -- the full-sphere closed form."""
        r_eff = volume_weighted_shell_radii(np.array([0.0, 4.0]))
        assert r_eff[0] == 3.0

    def test_thin_shell_limit_matches_the_midpoint(self) -> None:
        # rtol=1e-5, not the old 1e-3: true and midpoint differ by only
        # ~1.7e-7 relative on this shell, so 1e-3 passed for anything within
        # +/-1% and gated almost nothing. 1e-5 still holds with ~60x margin.
        r_eff = volume_weighted_shell_radii(np.array([10.0, 10.01]))
        np.testing.assert_allclose(r_eff, [10.005], rtol=1e-5)

    def test_ordering_on_a_wide_shell(self) -> None:
        """[1, 2]: sqrt(2) < 1.5 < r_eff < 2.0, and r_eff == 45/28 exactly."""
        r_eff = volume_weighted_shell_radii(np.array([1.0, 2.0]))

        assert np.sqrt(2.0) < 1.5 < r_eff[0] < 2.0
        assert r_eff[0] == pytest.approx(1.6071428571, abs=1e-10)

    def test_matches_dense_numerical_quadrature(self) -> None:
        """Catches an algebra slip in the quartic closed form: compare each
        bin's r_eff against int(r * r**2 dr) / int(r**2 dr), evaluated with
        `scipy.integrate.quad` directly on that bin -- the same weight the
        closed form claims to be the first moment of, computed a completely
        independent way.
        """
        edges = np.geomspace(_LOG_MIN_RADIUS, _LOG_MAX_RADIUS, _LOG_N_BINS + 1)
        r_eff = volume_weighted_shell_radii(edges)

        for b in range(edges.size - 1):
            r1, r2 = edges[b], edges[b + 1]
            numerator, _ = integrate.quad(lambda r: r * r**2, r1, r2)
            denominator, _ = integrate.quad(lambda r: r**2, r1, r2)
            expected = numerator / denominator
            assert r_eff[b] == pytest.approx(expected, abs=_QUAD_ATOL)

    def test_scale_invariance_under_fixed_ratio_log_binning(self) -> None:
        """r_eff / midpoint depends only on the bin ratio q = r2/r1, so for a
        fixed-ratio (log) binning it is ONE constant across every bin."""
        edges = np.geomspace(_LOG_MIN_RADIUS, _LOG_MAX_RADIUS, _LOG_N_BINS + 1)
        r_eff = volume_weighted_shell_radii(edges)
        midpoint = 0.5 * (edges[:-1] + edges[1:])
        ratio = r_eff / midpoint

        np.testing.assert_allclose(ratio, ratio[0])
        assert ratio[0] == pytest.approx(_SCALE_INVARIANT_Q_SQRT2, abs=1e-6)

    def test_non_monotonic_edges_raises(self) -> None:
        with pytest.raises(ValueError):
            volume_weighted_shell_radii(np.array([1.0, 3.0, 2.0]))

    def test_single_element_array_raises(self) -> None:
        with pytest.raises(ValueError):
            volume_weighted_shell_radii(np.array([5.0]))


# ---------------------------------------------------------------------------
# SelectionConfig
# ---------------------------------------------------------------------------


class TestSelectionConfig:
    def test_valid_construction(self) -> None:
        selection = SelectionConfig()
        assert selection.number_of_observers == 1000
        assert selection.observer_selection == "random"

    def test_no_longer_carries_halo_mass_knobs(self) -> None:
        # mass_min/mass_max moved to CatalogConfig, which is the one that is
        # actually read. A second, unread copy here is exactly the trap that
        # made the original pair dead for as long as it existed.
        fields = {f.name for f in dataclasses.fields(SelectionConfig)}
        assert "mass_min" not in fields
        assert "mass_max" not in fields

    def test_valid_construction_with_virgo(self) -> None:
        selection = SelectionConfig(observer_selection="virgo")
        assert selection.observer_selection == "virgo"

    def test_number_of_observers_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            SelectionConfig(number_of_observers=0)

    def test_invalid_observer_selection_raises(self) -> None:
        with pytest.raises(ValueError):
            SelectionConfig(observer_selection="not_a_real_strategy")


# ---------------------------------------------------------------------------
# CatalogConfig
# ---------------------------------------------------------------------------


class TestCatalogConfig:
    def test_default_is_the_full_catalog_uncut(self) -> None:
        catalog = CatalogConfig()
        assert catalog.name == CATALOG_FULL
        # None, not 0.0: "no floor" is a distinct state from "a floor at zero",
        # and only None disables the Parquet row-group filter entirely.
        assert catalog.mass_min is None
        assert catalog.mass_max is None
        assert catalog.include_subhalos is True

    def test_valid_catalogs_are_exactly_the_two_named_constants(self) -> None:
        assert VALID_CATALOGS == {CATALOG_FULL, CATALOG_MVIR12}

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError):
            CatalogConfig(name="ful")

    def test_non_positive_mass_bounds_raise(self) -> None:
        with pytest.raises(ValueError):
            CatalogConfig(mass_min=0.0)
        with pytest.raises(ValueError):
            CatalogConfig(mass_min=-1e12)
        with pytest.raises(ValueError):
            CatalogConfig(mass_max=0.0)

    def test_mass_max_not_greater_than_mass_min_raises(self) -> None:
        with pytest.raises(ValueError):
            CatalogConfig(mass_min=1e13, mass_max=1e12)
        with pytest.raises(ValueError):
            CatalogConfig(mass_min=1e12, mass_max=1e12)

    def test_one_sided_bounds_are_allowed(self) -> None:
        assert CatalogConfig(mass_min=1e12).mass_max is None
        assert CatalogConfig(mass_max=1e12).mass_min is None

    def test_describe_cuts_reports_the_selection(self) -> None:
        assert CatalogConfig().describe_cuts() == "full, no mass cut, subhalos included"
        described = CatalogConfig(
            name=CATALOG_MVIR12, mass_min=1e12, include_subhalos=False
        ).describe_cuts()
        assert CATALOG_MVIR12 in described
        assert "distinct halos only" in described
        assert "1e+12" in described


# ---------------------------------------------------------------------------
# Settings / default_settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_default_settings_returns_settings_instance(self) -> None:
        settings = default_settings()
        assert isinstance(settings, Settings)
        assert isinstance(settings.paths, PathsConfig)
        assert isinstance(settings.cosmology, CosmologyConfig)
        assert isinstance(settings.shells, ShellConfig)
        assert isinstance(settings.selection, SelectionConfig)

    def test_independent_sub_configs_across_instances(self) -> None:
        first = default_settings()
        second = default_settings()

        first.shells.max_radius = 300.0
        assert second.shells.max_radius != 300.0
        assert second.shells.max_radius == ShellConfig().max_radius

        first.selection.number_of_observers = 5
        assert second.selection.number_of_observers == 1000
