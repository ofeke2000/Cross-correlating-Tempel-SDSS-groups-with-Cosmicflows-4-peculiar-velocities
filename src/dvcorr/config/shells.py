"""
shells.py
---------
Radial shell binning for `dvcorr.estimators.shell_dipole.shell_dipole`.

Part of the `dvcorr.config` package -- the tunable-settings counterpart to
`dvcorr.conventions`'s frozen conventions. Anything with a sign or a
normalization ceiling (the shell-edge ceiling `MAX_ANALYSIS_RADIUS`) is
imported from `dvcorr.conventions`, never restated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dvcorr import conventions


@dataclass
class ShellConfig:
    """Radial shell binning for `dvcorr.estimators.shell_dipole.shell_dipole`.

    First-pass analysis knobs: a plausible starting binning for the
    simulation-validation arm, not a derived or frozen quantity. Expected to
    be swept/tuned once real shell-dipole runs start.

    Attributes
    ----------
    min_radius : float
        Innermost shell edge, h^-1 Mpc.
    max_radius : float
        Outermost shell edge, h^-1 Mpc. Must not exceed
        `dvcorr.conventions.MAX_ANALYSIS_RADIUS` (PBC minimum-image ceiling,
        hard rule 3).
    radii_step : float
        Shell width, h^-1 Mpc.
    sigma_star : float
        Small-scale velocity noise, km/s (CLAUDE.md Units: sigma* ~ 250 km/s).
    """

    min_radius: float = 20.0
    max_radius: float = 150.0
    radii_step: float = 10.0
    sigma_star: float = 250.0

    def __post_init__(self) -> None:
        if self.min_radius < 0.0:
            raise ValueError(f"ShellConfig: min_radius must be >= 0, got {self.min_radius}.")
        if self.max_radius <= self.min_radius:
            raise ValueError(
                f"ShellConfig: max_radius ({self.max_radius}) must be greater "
                f"than min_radius ({self.min_radius})."
            )
        if self.max_radius > conventions.MAX_ANALYSIS_RADIUS:
            raise ValueError(
                f"ShellConfig: max_radius ({self.max_radius}) exceeds "
                f"dvcorr.conventions.MAX_ANALYSIS_RADIUS ({conventions.MAX_ANALYSIS_RADIUS}); "
                "minimum-image geometry is not well defined beyond it "
                "(CLAUDE.md hard rule 3)."
            )
        if self.radii_step <= 0.0:
            raise ValueError(f"ShellConfig: radii_step must be > 0, got {self.radii_step}.")

    @property
    def shell_edges(self) -> np.ndarray:
        """Strictly increasing radial shell edges, shape (B + 1,).

        Feeds directly into `dvcorr.estimators.shell_dipole.shell_dipole`'s
        `shell_edges` argument. Built as `min_radius` to `max_radius` in steps
        of `radii_step`, then `max_radius` appended exactly so the outer edge
        is exact and, via `__post_init__`, guaranteed not to exceed
        `dvcorr.conventions.MAX_ANALYSIS_RADIUS`.

        `np.arange` with a float step can, through floating-point
        accumulation, emit a final element equal to (or slightly above)
        `max_radius` even though the stop is exclusive; appending `max_radius`
        on top of that would duplicate the outer edge and break strict
        monotonicity. Any interior edge >= `max_radius` is therefore dropped
        before the append, so the result is always strictly increasing. The
        final `np.minimum` is a belt-and-braces clip against the same ceiling.
        """
        edges = np.arange(self.min_radius, self.max_radius, self.radii_step)
        edges = edges[edges < self.max_radius]
        edges = np.append(edges, self.max_radius)
        return np.minimum(edges, conventions.MAX_ANALYSIS_RADIUS)

    @property
    def shell_centers(self) -> np.ndarray:
        """Shell midpoints, shape (B,); `0.5 * (edge_low + edge_high)` per shell."""
        edges = self.shell_edges
        return 0.5 * (edges[:-1] + edges[1:])
