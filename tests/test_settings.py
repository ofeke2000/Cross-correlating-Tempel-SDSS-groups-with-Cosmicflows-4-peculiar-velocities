"""
test_settings.py
-----------------
Unit tests for settings.py: PathsConfig, CosmologyConfig, ShellConfig,
SelectionConfig, and the Settings aggregator.

These are ordinary dataclass/validation tests, not part of the sign gate
(tests/test_geometry.py); nothing here touches config.py's frozen conventions.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import config
from estimators.shell_dipole import shell_dipole
from settings import (
    CosmologyConfig,
    PathsConfig,
    SelectionConfig,
    Settings,
    ShellConfig,
    default_settings,
)


# ---------------------------------------------------------------------------
# PathsConfig
# ---------------------------------------------------------------------------


class TestPathsConfig:
    def test_derived_paths_land_under_project_root(self) -> None:
        paths = PathsConfig()
        assert paths.data_dir == paths.project_root / "data"
        assert paths.output_dir == paths.project_root / "output"

    def test_catalogue_filenames(self) -> None:
        paths = PathsConfig()
        assert paths.mdpl2_catalog.name == "mdpl2_rockstar_125_pid-1_mvir12.csv"
        assert paths.cf4_groups_catalog.name == "CF4_Groups.csv"
        assert paths.cf4_velocities_catalog.name == "CF4_Groups_Velocities.csv"
        assert paths.sdss_tempel_catalog.name == "SDSS_Temple.csv"
        # And they sit under data_dir specifically.
        assert paths.mdpl2_catalog.parent == paths.data_dir
        assert paths.cf4_groups_catalog.parent == paths.data_dir
        assert paths.cf4_velocities_catalog.parent == paths.data_dir
        assert paths.sdss_tempel_catalog.parent == paths.data_dir

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
        assert paths.mdpl2_catalog == tmp_path / "data" / "mdpl2_rockstar_125_pid-1_mvir12.csv"


# ---------------------------------------------------------------------------
# CosmologyConfig
# ---------------------------------------------------------------------------


class TestCosmologyConfig:
    def test_h_matches_config_hubble_param(self) -> None:
        cosmology = CosmologyConfig()
        assert cosmology.h == pytest.approx(config.HUBBLE_PARAM, abs=1e-9)

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
        # H0 = 70.0 gives h = 0.70, which does not match config.HUBBLE_PARAM
        # (0.6777); __post_init__ must catch this rather than silently
        # accepting a cosmology that disagrees with the frozen convention.
        with pytest.raises(ValueError):
            CosmologyConfig(H0=70.0)


# ---------------------------------------------------------------------------
# ShellConfig
# ---------------------------------------------------------------------------


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
        assert shells.shell_edges[-1] <= config.MAX_ANALYSIS_RADIUS

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
            ShellConfig(max_radius=config.MAX_ANALYSIS_RADIUS + 1.0)

    def test_shell_edges_feed_shell_dipole(self) -> None:
        # A tiny toy: one central object, a couple of neighbours, using the
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


# ---------------------------------------------------------------------------
# SelectionConfig
# ---------------------------------------------------------------------------


class TestSelectionConfig:
    def test_valid_construction(self) -> None:
        selection = SelectionConfig()
        assert selection.mass_min == 1e12
        assert selection.mass_max is None
        assert selection.number_of_observers == 1000
        assert selection.observer_selection == "random"

    def test_valid_construction_with_virgo(self) -> None:
        selection = SelectionConfig(observer_selection="virgo")
        assert selection.observer_selection == "virgo"

    def test_non_positive_mass_min_raises(self) -> None:
        with pytest.raises(ValueError):
            SelectionConfig(mass_min=0.0)
        with pytest.raises(ValueError):
            SelectionConfig(mass_min=-1e12)

    def test_mass_max_not_greater_than_mass_min_raises(self) -> None:
        with pytest.raises(ValueError):
            SelectionConfig(mass_min=1e13, mass_max=1e12)
        with pytest.raises(ValueError):
            SelectionConfig(mass_min=1e12, mass_max=1e12)

    def test_number_of_observers_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            SelectionConfig(number_of_observers=0)

    def test_invalid_observer_selection_raises(self) -> None:
        with pytest.raises(ValueError):
            SelectionConfig(observer_selection="not_a_real_strategy")


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
