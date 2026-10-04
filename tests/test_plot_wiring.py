"""
test_plot_wiring.py
--------------------
Pins the figures built by `dvcorr.pipeline.velocity_centered.make_figure`
as CARTESIAN -- linear x and linear y -- for BOTH shell spacings. The binning
is what may be logarithmic (`ShellConfig.spacing == SPACING_LOG`, radii
geometrically spaced); the AXES it is drawn on are not. An earlier version of
this pipeline coupled the two, putting the radial axis in log scale and the
dipole/monopole axes in symlog whenever the binning was logarithmic; these
tests exist so that coupling cannot come back silently.

Both spacings are checked because the failure mode is spacing-conditional: a
reintroduced `set_xscale("log")` / `set_yscale("symlog")` would fire only on
the `SPACING_LOG` path and leave the `SPACING_LINEAR` figures looking correct.

Also pins CLAUDE.md hard rules 6 and 10: `make_figure` returns a
`DipoleFigures(dipole, monopole)` -- two separate `Figure` objects, never one
merged two-panel image (rule 6) -- and every curve is a LINE through one
point per shell at the volume-weighted radius
(`volume_weighted_shell_radii`), never a step held flat across the bin, on
an x-axis pinned to start at r = 0 (rule 10).

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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dvcorr import conventions
from dvcorr.config import SPACING_LINEAR, SPACING_LOG, ShellConfig, volume_weighted_shell_radii
from dvcorr.estimators.shell_dipole import velocity_centered_shell_dipole
from dvcorr.pipeline.velocity_centered import (
    DipoleFigures,
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
    """A log ladder plus the [0, 1) inner bin -- the shape the production
    default has (CLAUDE.md hard rule 10), built here with its own explicit
    numbers rather than importing `RunConfig`'s default so this test stays
    stable if that default is retuned again (see `tests/test_default_shells.py`
    for the default itself).
    """
    return ShellConfig(
        min_radius=1.0,
        max_radius=64.0,
        spacing=SPACING_LOG,
        n_bins=12,
        include_zero_bin=True,
    )


def test_make_figure_returns_two_separate_figures_never_one_merged_image():
    """CLAUDE.md hard rule 6: dipole and monopole are two Figures, not two
    panels of one -- a dipole PNG must be sendable on its own.
    """
    shells = _log_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    figures = make_figure(cfg, result, normalized)

    assert isinstance(figures, DipoleFigures)
    assert figures.dipole is not figures.monopole
    assert len(figures.dipole.axes) == 1
    assert len(figures.monopole.axes) == 1


def test_make_figure_log_shells_gives_linear_radial_axis():
    shells = _log_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    figures = make_figure(cfg, result, normalized)

    assert figures.dipole.axes[0].get_xscale() == "linear"
    assert figures.monopole.axes[0].get_xscale() == "linear"


def test_make_figure_linear_shells_gives_linear_radial_axis():
    shells = _linear_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    figures = make_figure(cfg, result, normalized)

    assert figures.dipole.axes[0].get_xscale() == "linear"
    assert figures.monopole.axes[0].get_xscale() == "linear"


def test_make_figure_log_shells_gives_linear_dipole_and_monopole_yaxes():
    shells = _log_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    figures = make_figure(cfg, result, normalized)

    assert figures.dipole.axes[0].get_yscale() == "linear"
    assert figures.monopole.axes[0].get_yscale() == "linear"


def test_make_figure_linear_shells_gives_linear_dipole_and_monopole_yaxes():
    shells = _linear_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    figures = make_figure(cfg, result, normalized)

    assert figures.dipole.axes[0].get_yscale() == "linear"
    assert figures.monopole.axes[0].get_yscale() == "linear"


def test_make_figure_plots_every_shell_including_the_zero_bin():
    """A zero innermost edge must be plottable.

    `shell_edges[0] == 0` is the one thing a log AXIS could not have
    accommodated; on the cartesian axes this pipeline actually uses it is
    ordinary, and the [0, 1) bin's abscissa is the finite, strictly positive
    `volume_weighted_shell_radii` value 0.75 (3R/4 for a full sphere). This
    test pins that the figure carries a point for it -- i.e. that
    B = n_bins + 1 shells reach the plot, not the n_bins the config's
    `n_bins` field names.
    """
    shells = _zero_bin_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    assert result.shell_edges[0] == 0.0
    assert result.shell_edges.size == shells.n_bins + 2

    figures = make_figure(cfg, result, normalized)
    ax_dipole = figures.dipole.axes[0]

    assert ax_dipole.get_xscale() == "linear"
    x_plotted = ax_dipole.lines[0].get_xdata()
    assert len(x_plotted) == shells.n_bins + 1
    assert x_plotted[0] == pytest.approx(0.75 * shells.min_radius)
    np.testing.assert_allclose(x_plotted, volume_weighted_shell_radii(result.shell_edges))


# ---------------------------------------------------------------------------
# Every figure starts at r = 0 (CLAUDE.md hard rule 10)
# ---------------------------------------------------------------------------


def test_dipole_and_monopole_curves_are_lines_at_r_eff_on_axes_that_start_at_zero():
    """CLAUDE.md hard rule 10, both halves. The binning is how the estimator
    accumulates, not how the figure is drawn: each curve is a `Line2D`
    through one point per shell at the volume-weighted radius, never a step
    (`ax.stairs`, a `StepPatch`) held flat across the bin. Independently of
    where that first abscissa falls (0.75 * min_radius here, strictly
    positive), both figures' x-axes are pinned to start at r = 0.
    """
    shells = _zero_bin_shells()
    cfg = RunConfig(sub_volume_radius=conventions.MAX_ANALYSIS_RADIUS, shells=shells)
    result, normalized = _tiny_result_and_normalized(shells)

    figures = make_figure(cfg, result, normalized)
    ax_dipole = figures.dipole.axes[0]
    ax_mono = figures.monopole.axes[0]

    # No step artists anywhere: a stairs rendering is the regression this pins.
    assert not ax_dipole.patches
    assert not ax_mono.patches

    r_eff = volume_weighted_shell_radii(result.shell_edges)
    for ax in (ax_dipole, ax_mono):
        curves = [line for line in ax.get_lines() if len(line.get_xdata()) == r_eff.size]
        assert curves, "expected at least one curve with one point per shell"
        for line in curves:
            np.testing.assert_allclose(line.get_xdata(), r_eff)

    # The x-axis lower limit is pinned to 0 explicitly, independent of the
    # innermost abscissa being strictly positive.
    assert r_eff[0] > 0.0
    assert ax_dipole.get_xlim()[0] == pytest.approx(0.0)
    assert ax_mono.get_xlim()[0] == pytest.approx(0.0)
