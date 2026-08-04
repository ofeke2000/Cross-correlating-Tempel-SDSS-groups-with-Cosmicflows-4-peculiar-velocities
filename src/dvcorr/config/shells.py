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

#: Linear shell spacing: `min_radius` to `max_radius` in fixed steps of
#: `radii_step` (`linear_shell_edges`).
SPACING_LINEAR: str = "linear"
#: Logarithmic shell spacing: `n_bins` shells geometrically spaced between
#: `min_radius` and `max_radius` (`log_shell_edges`).
SPACING_LOG: str = "log"
#: Every value `ShellConfig.spacing` is allowed to take. Named constants
#: rather than bare string literals so a typo'd spacing (e.g. "lgo") raises in
#: `__post_init__` instead of silently falling through (CLAUDE.md hard rules
#: 4 and 9).
_VALID_SPACINGS: tuple[str, str] = (SPACING_LINEAR, SPACING_LOG)


def linear_shell_edges(min_radius: float, max_radius: float, radii_step: float) -> np.ndarray:
    """Strictly increasing radial shell edges, linear spacing, shape (B + 1,).

    Feeds directly into `dvcorr.estimators.shell_dipole.shell_dipole`'s
    `shell_edges` argument. Built as `min_radius` to `max_radius` in steps of
    `radii_step`, then `max_radius` appended exactly so the outer edge is
    exact and, via `ShellConfig.__post_init__`, guaranteed not to exceed
    `dvcorr.conventions.MAX_ANALYSIS_RADIUS`.

    `np.arange` with a float step can, through floating-point accumulation,
    emit a final element equal to (or slightly above) `max_radius` even
    though the stop is exclusive; appending `max_radius` on top of that would
    duplicate the outer edge and break strict monotonicity. Any interior edge
    >= `max_radius` is therefore dropped before the append, so the result is
    always strictly increasing. The final `np.minimum` is a belt-and-braces
    clip against the same ceiling.
    """
    edges = np.arange(min_radius, max_radius, radii_step)
    edges = edges[edges < max_radius]
    edges = np.append(edges, max_radius)
    return np.minimum(edges, conventions.MAX_ANALYSIS_RADIUS)


def log_shell_edges(
    min_radius: float,
    max_radius: float,
    n_bins: int,
    include_zero_bin: bool = False,
) -> np.ndarray:
    """Strictly increasing radial shell edges, log spacing, shape (B + 1,).

    `n_bins` shells geometrically spaced between `min_radius` and
    `max_radius` via `np.geomspace`, which pins BOTH endpoints exactly -- so,
    unlike `linear_shell_edges`, there is no float-accumulation drift to
    defend against and no drop-then-append dance is needed: the result is
    strictly increasing by construction. The `np.minimum` clip against
    `dvcorr.conventions.MAX_ANALYSIS_RADIUS` is kept anyway, belt-and-braces,
    matching `linear_shell_edges`.

    Caller obligation: `min_radius` must be strictly positive -- `geomspace`
    is undefined at zero. Enforced at `ShellConfig.__post_init__`, not here.

    Parameters
    ----------
    min_radius, max_radius : float
        Innermost and outermost LOG edge, h^-1 Mpc. `min_radius > 0`.
    n_bins : int
        Number of geometrically spaced shells between them.
    include_zero_bin : bool, default False
        Prepend a single linear inner bin `[0, min_radius)`, giving
        `n_bins + 2` edges (`B = n_bins + 1` shells) with `edges[0] == 0.0`
        exactly. `min_radius` keeps its meaning -- the innermost LOG edge --
        so the geometric ladder above it is bit-for-bit unchanged and the
        zero bin is purely additive.

        A log ladder cannot reach zero (that is the whole content of the
        `min_radius > 0` requirement above), so the r < `min_radius` region
        is otherwise not measured at all. This one bin covers it, at the
        price of being the only non-geometric shell in the array: it is
        `min_radius` wide where its neighbor is `min_radius * (q - 1)` wide,
        and on a shared plot it is the one point whose horizontal extent does
        not scale with r. Nothing downstream cares -- every consumer takes
        the edge ARRAY, never a ratio or a width (see `ShellConfig`) -- but a
        reader of the figure should know.

        Caller obligation, and the reason this is opt-in rather than always
        on: `edges[0] == 0` re-admits the r = 0 COINCIDENT PAIR, which the
        estimators exclude explicitly on `r_mag > 0` (see
        `dvcorr.geometry.unit_vector`'s Notes -- a zero-length separation has
        no direction and would otherwise enter the monopole as a pure
        self-correlation). That filter lives in the estimators, not here.

    Returns
    -------
    ndarray, shape (n_bins + 1,), or (n_bins + 2,) with `include_zero_bin`
    """
    edges = np.geomspace(min_radius, max_radius, n_bins + 1)
    if include_zero_bin:
        edges = np.concatenate(([0.0], edges))
    return np.minimum(edges, conventions.MAX_ANALYSIS_RADIUS)


def volume_weighted_shell_radii(shell_edges: np.ndarray) -> np.ndarray:
    """Volume-weighted shell radii, shape (B,) -- the plotting abscissa.

    r_eff = (3/4) * (r2**4 - r1**4) / (r2**3 - r1**3)

    This is the FIRST MOMENT of the same `r**2 dr` weight that the
    estimator's normalization already divides by:
    `dvcorr.estimators.shell_dipole.expected_shell_occupancy` uses
    `n_bar * (4*pi/3) * (r2**3 - r1**3)`, the UNIFORM-field shell volume, so
    the plotted ordinate at bin b is the `r**2 dr`-weighted mean of the
    statistic over the shell, and `r_eff = int(r * r**2 dr) / int(r**2 dr)`
    is the matching abscissa. Ordinate and abscissa agree. The `4*pi/3`
    cancels in the ratio, so no constant is duplicated from
    `expected_shell_occupancy`.

    Not the geometric mean `sqrt(r1 * r2)`: that is the right abscissa only
    if the weight were uniform in log r, which it is not -- the weight is
    uniform in r**2 dr.

    Not a pair-count-weighted mean r: it would pick up the clustering factor
    `1 + xi(r)` that the denominator does not carry, would be
    realization-dependent, and would give the two curves on a comparison
    figure DIFFERENT x-coordinates.

    Honest sizing: `r_eff / midpoint` depends only on the ratio `q = r2/r1`,
    so for a fixed-ratio (logarithmic) binning it is ONE constant for every
    bin -- +1.9% at q = sqrt(2), a rigid 0.008 dex shift on a log axis. By
    contrast the current LINEAR binning has a bin-dependent offset: +7.1% at
    [5, 10] falling to +0.13% at [55, 60]. Nobody should mistake this for a
    large correction.

    `ShellConfig.shell_centers` (the midpoint) is deliberately NOT redefined
    here; this is a separate, additional quantity -- see
    `ShellConfig.shell_effective_radii`.

    `edges[0] == 0` is fine and gives exactly `0.75 * r2` (the full-sphere
    closed form); no guard is needed against it.

    Parameters
    ----------
    shell_edges : ndarray, shape (B + 1,)
        Strictly increasing radial shell boundaries, h^-1 Mpc.

    Returns
    -------
    ndarray, shape (B,)
        Volume-weighted radius per shell, h^-1 Mpc.
    """
    edges = np.asarray(shell_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("shell_edges must be 1-D with at least two entries.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("shell_edges must be strictly increasing.")
    r1 = edges[:-1]
    r2 = edges[1:]
    # 3/4: first moment of the r**2 dr weight, int(r**3 dr) / int(r**2 dr).
    return (3.0 / 4.0) * (r2**4 - r1**4) / (r2**3 - r1**3)


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
        Shell width, h^-1 Mpc. Only read when `spacing == SPACING_LINEAR`;
        inert (but still validated as well-formed) under `SPACING_LOG`.
    sigma_star : float
        Small-scale velocity noise, km/s (CLAUDE.md Units: sigma* ~ 250 km/s).
        NOT READ BY ANYTHING YET -- a placeholder for the error model, carried
        so the number has one home when that lands. It does not enter the
        estimator, the normalization, or the error bars, so changing it
        changes no output. Said explicitly because the field previously read
        as though it were live; `SelectionConfig.mass_min` sat unread in the
        same way and quietly became wrong. Note also that 250 km/s describes
        the mvir >= 1e12 population; the full catalog's low-mass halos have a
        larger effective dispersion, so this value needs re-measuring before
        it is first used, not merely wiring up.
    spacing : str
        Shell-edge spacing, one of `_VALID_SPACINGS`
        (`SPACING_LINEAR`, `SPACING_LOG`). Defaults to `SPACING_LINEAR`, so
        `Settings()` / `default_settings()` and every existing linear-binning
        caller are unaffected.
    n_bins : int
        Number of shells for `SPACING_LOG`. Only read when
        `spacing == SPACING_LOG`; inert (but still validated as well-formed)
        under `SPACING_LINEAR`.
    include_zero_bin : bool
        Prepend the inner bin `[0, min_radius)` to the log ladder, so
        `shell_edges` has `n_bins + 2` entries and `B = n_bins + 1` shells;
        see `log_shell_edges`. Only read when `spacing == SPACING_LOG`, and
        inert under `SPACING_LINEAR` for the same reason `n_bins` is: a
        linear binning can already start at zero by setting
        `min_radius = 0.0`, so there is nothing for the flag to add there.

        Note that `min_radius` then no longer means "innermost edge" -- it
        means "innermost LOG edge", and `shell_edges[0]` is 0.0. Anything
        reading `shell_edges[0]` as the analysis floor (figure titles, the
        estimators' `in_range` masks) sees zero and must be able to cope
        with it; `volume_weighted_shell_radii` already can (it gives the
        full-sphere `0.75 * min_radius` for that bin), and the estimators
        exclude the r = 0 coincident pair explicitly rather than relying on
        a positive innermost edge to do it for them.
    """

    min_radius: float = 20.0
    max_radius: float = 150.0
    radii_step: float = 10.0
    sigma_star: float = 250.0
    spacing: str = SPACING_LINEAR
    n_bins: int = 12
    include_zero_bin: bool = False

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
        if self.spacing not in _VALID_SPACINGS:
            raise ValueError(
                f"ShellConfig: spacing must be one of {_VALID_SPACINGS}, got "
                f"{self.spacing!r}."
            )
        n_bins_is_integral = isinstance(self.n_bins, int) and not isinstance(self.n_bins, bool)
        if not n_bins_is_integral or self.n_bins < 1:
            raise ValueError(
                f"ShellConfig: n_bins must be a positive integer, got {self.n_bins!r}."
            )
        if not isinstance(self.include_zero_bin, bool):
            raise ValueError(
                f"ShellConfig: include_zero_bin must be a bool, got "
                f"{self.include_zero_bin!r}."
            )
        if self.spacing == SPACING_LOG and self.min_radius <= 0.0:
            raise ValueError(
                f"ShellConfig: min_radius must be > 0 for spacing={SPACING_LOG!r}, "
                f"got {self.min_radius}; log spacing uses np.geomspace, which is "
                "undefined at zero."
            )

    @property
    def shell_edges(self) -> np.ndarray:
        """Strictly increasing radial shell edges, shape (B + 1,).

        Feeds directly into `dvcorr.estimators.shell_dipole.shell_dipole`'s
        `shell_edges` argument. Dispatches on `spacing` to
        `linear_shell_edges` or `log_shell_edges`; see those docstrings for
        the construction details.

        B is `n_bins` under `SPACING_LOG`, or `n_bins + 1` when
        `include_zero_bin` adds the `[0, min_radius)` inner shell.
        """
        if self.spacing == SPACING_LOG:
            return log_shell_edges(
                self.min_radius,
                self.max_radius,
                self.n_bins,
                include_zero_bin=self.include_zero_bin,
            )
        return linear_shell_edges(self.min_radius, self.max_radius, self.radii_step)

    @property
    def shell_centers(self) -> np.ndarray:
        """Shell midpoints, shape (B,); `0.5 * (edge_low + edge_high)` per shell.

        Deliberately the plain midpoint, not volume-weighted; see
        `shell_effective_radii` for the volume-weighted abscissa used for
        plotting.
        """
        edges = self.shell_edges
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def shell_effective_radii(self) -> np.ndarray:
        """Volume-weighted shell radii, shape (B,); see `volume_weighted_shell_radii`."""
        return volume_weighted_shell_radii(self.shell_edges)
