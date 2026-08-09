"""
test_plot_wiring.py
--------------------
Pins the figure axes built by `dvcorr.pipeline.velocity_centered.make_figure`
as CARTESIAN -- linear x and linear y -- for BOTH shell spacings. The binning
is what may be logarithmic (`ShellConfig.spacing == SPACING_LOG`, radii
geometrically spaced); the AXES it is drawn on are not. An earlier version of
this pipeline coupled the two, putting the radial axis in log scale and the
dipole/monopole axes in symlog whenever the binning was logarithmic; these
tests exist so that coupling cannot come back silently.

Both spacings are checked because the failure mode is spacing-conditional: a
reintroduced `set_xscale("log")` / `set_yscale("symlog")` would fire only on
the `SPACING_LOG` path and leave the `SPACING_LINEAR` figures looking correct.

Backend discipline: this test module calls `matplotlib.use("Agg")` itself,
before importing `dvcorr.pipeline.velocity_centered` (which imports
`matplotlib.pyplot` at module level but never selects a backend) -- the same
division of responsibility documented in that module's own docstring and
followed by every `scripts/plot_*.py` driver: the CONSUMER selects the
backend, the library never does.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.config import SPACING_LINEAR, SPACING_LOG, ShellConfig
from dvcorr.estimators.shell_dipole import velocity_centered_shell_dipole
from dvcorr.pipeline.velocity_centered import (
    NormalizedDipole,
    RunConfig,
    VelocityCenteredShellDipoleResult,
    box_number_density,
    make_figure,
    normalize_result,
)

_N_BAR_ARBITRARY = 1.0e-3  # tracers per (h^-1 Mpc)^3 -- values are irrelevant to this test


def _tiny_result_and_normalized(
    shells: ShellConfig,
) -> tuple[VelocityCenteredShellDipoleResult, NormalizedDipole]:
    """One center, zero tracers -- shape-correct synthetic input; VALUES don't
    matter here, only that `make_figure` runs end to end and its axes come out
    cartesian. Mirrors the empty-shell contract already pinned in
    `test_velocity_centered_dipole.py::test_zero_tracers_gives_all_zero_shells`.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    s_center = (observer + np.array([100.0, 0.0, 0.0]))[None, :]
    v_center = np.array([[300.0, 0.0, 0.0]])
    s_tracers = np.empty((0, 3))

    result = velocity_centered_shell_dipole(
        s_centers=s_center,
        v_centers=v_center,
        s_tracers=s_tracers,
        shell_edges=shells.shell_edges,
        sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS,
    )
    n_bar = box_number_density(1)
    normalized = normalize_result(
        result, n_bar, shuffle_seed=1, gaussian_null_seed=2, n_realizations=4
    )
    return result, normalized


def _log_shells() -> ShellConfig:
    return ShellConfig(min_radius=1.0, max_radius=64.0, spacing=SPACING_LOG, n_bins=12)


def _linear_shells() -> ShellConfig:
    return ShellConfig(min_radius=20.0, max_radius=150.0, radii_step=10.0, spacing=SPACING_LINEAR)


def _zero_bin_shells() -> ShellConfig:
    """The production default: the log ladder plus the [0, 1) inner bin."""
    return ShellConfig(
        min_radius=1.0,
        max_radius=64.0,
        spacing=SPACING_LOG,
        n_bins=12,
        include_zero_bin=True,
    )


def test_make_figure_log_shells_gives_linear_radial_axis():
    shells = _log_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    fig = make_figure(cfg, result, normalized)
    ax_mono = fig.axes[-1]  # bottom axis: the shared-x one carrying set_xlabel

    assert ax_mono.get_xscale() == "linear"


def test_make_figure_linear_shells_gives_linear_radial_axis():
    shells = _linear_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    fig = make_figure(cfg, result, normalized)
    ax_mono = fig.axes[-1]

    assert ax_mono.get_xscale() == "linear"


def test_make_figure_log_shells_gives_linear_dipole_and_monopole_yaxes():
    shells = _log_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    fig = make_figure(cfg, result, normalized)
    ax_dipole, ax_mono = fig.axes

    assert ax_dipole.get_yscale() == "linear"
    assert ax_mono.get_yscale() == "linear"


def test_make_figure_linear_shells_gives_linear_dipole_and_monopole_yaxes():
    shells = _linear_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    fig = make_figure(cfg, result, normalized)
    ax_dipole, ax_mono = fig.axes

    assert ax_dipole.get_yscale() == "linear"
    assert ax_mono.get_yscale() == "linear"


def test_make_figure_plots_every_shell_including_the_zero_bin():
    """A zero innermost edge must be plottable.

    `shell_edges[0] == 0` is the one thing a log AXIS could not have
    accommodated; on the cartesian axes this pipeline actually uses it is
    ordinary, and the [0, 1) bin's abscissa is the finite, strictly positive
    `volume_weighted_shell_radii` value 0.75. This test pins that the figure
    carries a point for it -- i.e. that B = n_bins + 1 shells reach the plot,
    not the n_bins the config's `n_bins` field names.
    """
    shells = _zero_bin_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    assert result.shell_edges[0] == 0.0
    assert result.shell_edges.size == shells.n_bins + 2

    fig = make_figure(cfg, result, normalized)
    ax_dipole, ax_mono = fig.axes

    assert ax_dipole.get_xscale() == "linear"
    x_plotted = ax_dipole.lines[0].get_xdata()
    assert len(x_plotted) == shells.n_bins + 1
    assert x_plotted[0] == pytest.approx(0.75 * shells.min_radius)
