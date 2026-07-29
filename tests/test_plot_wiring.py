"""
test_plot_wiring.py
--------------------
Pins the log/linear radial x-axis wiring `dvcorr.pipeline.velocity_centered
.apply_radial_axis_scale` adds to `make_figure`: `ShellConfig.spacing ==
SPACING_LOG` must put the shared x-axis in log scale with its limits pinned
to the outer shell edges (CLAUDE.md hard rule 3's ceiling and the log-mode
autoscale trap `apply_radial_axis_scale`'s docstring documents); `spacing ==
SPACING_LINEAR` must leave the axis exactly as it was before that function
existed -- the stated backward-compatibility guarantee.

Also pins the companion y-axis wiring, `apply_dipole_axis_scale`: under
`SPACING_LOG` BOTH panels of `make_figure` (dipole and monopole) must come
out `symlog` (not plain `log` -- the shuffle null crosses zero and a plain
log axis would silently drop those points); under `SPACING_LINEAR` both must
stay `linear`, the same backward-compatibility guarantee as the x-axis case.

This is the only flag path in the log-spacing task otherwise unpinned by any
test: everything else (edge construction, normalization under non-uniform
bins) is covered in `test_velocity_centered_dipole.py`, but nothing there
builds an actual `Figure` and reads its axis back.

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

from dvcorr import conventions
from dvcorr.config import SPACING_LINEAR, SPACING_LOG, ShellConfig
from dvcorr.estimators.shell_dipole import velocity_centered_shell_dipole
from dvcorr.pipeline.velocity_centered import (
    NormalizedDipole,
    RunConfig,
    VelocityCenteredShellDipoleResult,
    global_number_density,
    make_figure,
    normalize_result,
)

_N_BAR_ARBITRARY = 1.0e-3  # tracers per (h^-1 Mpc)^3 -- values are irrelevant to this test


def _tiny_result_and_normalized(
    shells: ShellConfig,
) -> tuple[VelocityCenteredShellDipoleResult, NormalizedDipole]:
    """One center, zero tracers -- shape-correct synthetic input; VALUES don't
    matter here, only that `make_figure` runs end to end and the x-axis comes
    out wired to `shells.spacing`. Mirrors the empty-shell contract already
    pinned in `test_velocity_centered_dipole.py::test_zero_tracers_gives_all_zero_shells`.
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
    n_bar = global_number_density(1, conventions.MAX_ANALYSIS_RADIUS)
    normalized = normalize_result(result, n_bar, shuffle_seed=1)
    return result, normalized


def test_make_figure_log_shells_gives_log_axis_pinned_to_outer_edges():
    shells = ShellConfig(min_radius=1.0, max_radius=64.0, spacing=SPACING_LOG, n_bins=12)
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    fig = make_figure(cfg, result, normalized)
    ax_mono = fig.axes[-1]  # bottom axis: the one apply_radial_axis_scale is called on

    assert ax_mono.get_xscale() == "log"
    edges = shells.shell_edges
    np.testing.assert_allclose(ax_mono.get_xlim(), (edges[0], edges[-1]))


def test_make_figure_linear_shells_gives_linear_axis():
    shells = ShellConfig(min_radius=20.0, max_radius=150.0, radii_step=10.0, spacing=SPACING_LINEAR)
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    fig = make_figure(cfg, result, normalized)
    ax_mono = fig.axes[-1]

    assert ax_mono.get_xscale() == "linear"


def test_make_figure_log_shells_gives_symlog_dipole_and_monopole_yaxes():
    shells = ShellConfig(min_radius=1.0, max_radius=64.0, spacing=SPACING_LOG, n_bins=12)
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    fig = make_figure(cfg, result, normalized)
    ax_dipole, ax_mono = fig.axes

    assert ax_dipole.get_yscale() == "symlog"
    assert ax_mono.get_yscale() == "symlog"


def test_make_figure_linear_shells_gives_linear_dipole_and_monopole_yaxes():
    shells = ShellConfig(min_radius=20.0, max_radius=150.0, radii_step=10.0, spacing=SPACING_LINEAR)
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    fig = make_figure(cfg, result, normalized)
    ax_dipole, ax_mono = fig.axes

    assert ax_dipole.get_yscale() == "linear"
    assert ax_mono.get_yscale() == "linear"
