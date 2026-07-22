"""
settings.py
------------
Tunable settings, stored in classes -- the counterpart to `config.py`'s frozen
conventions.

`config.py` is the single source of truth for anything that carries a sign or
a normalisation (pair orientation, the observer, the box, catalogue column
names): change it and the science answer changes, silently, if you are not
careful. Nothing in *this* module may redefine or restate one of those
conventions; where a setting here needs one (e.g. the shell-edge ceiling, or
the cosmology's self-consistency check against `config.HUBBLE_PARAM`), it is
imported from `config`, never re-typed.

Everything in this module is a first-pass, tunable knob: file locations, the
MDPL2 cosmology parameters (fixed by the simulation, but not sign-bearing),
the radial binning used by `estimators.shell_dipole.shell_dipole`, and the
not-yet-built mocks/selection arm's placeholder parameters. These are exactly
the "numbers and stable configuration ... stored in classes" that CLAUDE.md's
coding conventions ask for, and the mechanism behind hard rule 4 (no bare
numbers): every literal below lives on a dataclass field, not inline in
analysis code.

Four small dataclasses, one aggregator:

    PathsConfig      -- catalogue and output file locations, repo-root-relative
    CosmologyConfig  -- MDPL2 cosmology (frozen/immutable, but not a `config.py`
                        convention -- it is simulation metadata, not a sign)
    ShellConfig      -- radial shell binning for the shell-dipole estimator
    SelectionConfig  -- placeholder knobs for the not-yet-built mocks/ and
                        selection/ arms

    Settings         -- aggregates one instance of each; use `default_settings()`
                        to obtain a fresh, independent instance rather than a
                        shared module-level singleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

# --------------------------------------------------------------------------
# PathsConfig
# --------------------------------------------------------------------------


@dataclass
class PathsConfig:
    """Catalogue and output file locations, derived from the repo root.

    Unlike the old bulk-flow repo's `paths_config.py`, nothing here is a
    hardcoded absolute path: every field is derived from `project_root` so the
    settings are portable across machines and checkouts. Only `project_root`
    itself has a real default (the directory this file lives in); every other
    field defaults to `None` and is resolved relative to `project_root` /
    `data_dir` in `__post_init__`, since one dataclass default cannot cleanly
    reference another.

    The MDPL2 halo catalogue (`mdpl2_catalog`) is ~4M rows / ~600 MB and is
    gitignored; this class only names its path, it does not load it -- see
    CLAUDE.md's "don't load it fully unless the task needs it".

    Attributes
    ----------
    project_root : Path
        Repository root, i.e. the directory containing this file.
    data_dir : Path
        Input catalogue directory, default `project_root / "data"`.
    output_dir : Path
        Output directory for derived products, default `project_root / "output"`.
    mdpl2_catalog : Path
        MDPL2/Rockstar halo catalogue, default
        `data_dir / "mdpl2_rockstar_125_pid-1_mvir12.csv"`.
    cf4_groups_catalog : Path
        Cosmicflows-4 group catalogue, default `data_dir / "CF4_Groups.csv"`.
    cf4_velocities_catalog : Path
        Cosmicflows-4 group peculiar velocities, default
        `data_dir / "CF4_Groups_Velocities.csv"`.
    sdss_tempel_catalog : Path
        Tempel et al. (2017) SDSS group catalogue, default
        `data_dir / "SDSS_Temple.csv"`.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path | None = None
    output_dir: Path | None = None
    mdpl2_catalog: Path | None = None
    cf4_groups_catalog: Path | None = None
    cf4_velocities_catalog: Path | None = None
    sdss_tempel_catalog: Path | None = None

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        if self.mdpl2_catalog is None:
            self.mdpl2_catalog = self.data_dir / "mdpl2_rockstar_125_pid-1_mvir12.csv"
        if self.cf4_groups_catalog is None:
            self.cf4_groups_catalog = self.data_dir / "CF4_Groups.csv"
        if self.cf4_velocities_catalog is None:
            self.cf4_velocities_catalog = self.data_dir / "CF4_Groups_Velocities.csv"
        if self.sdss_tempel_catalog is None:
            self.sdss_tempel_catalog = self.data_dir / "SDSS_Temple.csv"

    def ensure_output_dir(self) -> None:
        """Create `output_dir` (and any missing parents) if it does not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# CosmologyConfig
# --------------------------------------------------------------------------

#: Tolerance for the CosmologyConfig <-> config.HUBBLE_PARAM consistency
#: check. The two are computed from the same literal (H0 = 67.77) via two
#: different paths (a stored constant vs. H0/100), so any real disagreement is
#: an editing mistake, not floating-point noise; this only needs to be looser
#: than double-precision round-off.
_H_CONSISTENCY_TOL: float = 1e-9


@dataclass(frozen=True)
class CosmologyConfig:
    """MDPL2 simulation cosmology.

    Fixed by the simulation the halo catalogue was drawn from, not a
    user-editable analysis knob -- hence `frozen=True` rather than a plain
    dataclass. It is simulation *metadata*, not a `config.py`-style sign or
    normalisation convention, which is why it lives here rather than in
    `config.py`; but `h` (see below) must still agree with
    `config.HUBBLE_PARAM`, so `__post_init__` checks that explicitly.

    Values from CLAUDE.md's Units section / MDPL2 (Klypin et al. 2016):
    H0 = 67.77, Om0 = 0.307115, sigma8 = 0.8228.

    Attributes
    ----------
    H0 : float
        Hubble constant, km/s/Mpc.
    Om0 : float
        Total matter density parameter today.
    Ode0 : float
        Dark energy density parameter today.
    Ob0 : float
        Baryon density parameter today.
    sigma8 : float
        RMS matter fluctuation in 8 h^-1 Mpc spheres.
    ns : float
        Scalar spectral index.
    growth_index : float
        Growth-rate index gamma, used for f(z) = Om(z)^gamma.
    flat : bool
        Whether the cosmology is spatially flat.

    Notes
    -----
    CF4's own distance-ladder H0 (Tully et al. 2023, ~74.6 km/s/Mpc) is
    deliberately not included here -- it is out of scope for the simulation-
    validation arm. Add an `H0_CF4` field (see the old repo's
    `cosmology_config.py`) when the CF4 data arm starts.
    """

    H0: float = 67.77
    Om0: float = 0.307115
    Ode0: float = 0.692885
    Ob0: float = 0.048206
    sigma8: float = 0.8228
    ns: float = 0.96
    growth_index: float = 0.55
    flat: bool = True

    def __post_init__(self) -> None:
        if abs(self.h - config.HUBBLE_PARAM) > _H_CONSISTENCY_TOL:
            raise ValueError(
                f"CosmologyConfig.h = {self.h} (from H0 = {self.H0}) does not "
                f"match config.HUBBLE_PARAM = {config.HUBBLE_PARAM}. h and "
                "config.HUBBLE_PARAM are the same reduced Hubble parameter "
                "and must agree; if the cosmology genuinely changed, update "
                "config.HUBBLE_PARAM explicitly (see CLAUDE.md hard rule 1) "
                "rather than letting the two drift apart."
            )

    @property
    def h(self) -> float:
        """Reduced Hubble parameter, h = H0 / 100.

        h == 100 is the *definition* of the reduced Hubble parameter; the
        literal 100 here is that definition, not a tunable constant (CLAUDE.md
        hard rule 4's pure-mathematics exemption).
        """
        return self.H0 / 100.0

    def to_colossus_dict(self) -> dict:
        """Return the cosmology as a flat dict in colossus's constructor shape."""
        return {
            "flat": self.flat,
            "H0": self.H0,
            "Om0": self.Om0,
            "Ode0": self.Ode0,
            "Ob0": self.Ob0,
            "sigma8": self.sigma8,
            "ns": self.ns,
        }


# --------------------------------------------------------------------------
# ShellConfig
# --------------------------------------------------------------------------


@dataclass
class ShellConfig:
    """Radial shell binning for `estimators.shell_dipole.shell_dipole`.

    First-pass analysis knobs: a plausible starting binning for the
    simulation-validation arm, not a derived or frozen quantity. Expected to
    be swept/tuned once real shell-dipole runs start.

    Attributes
    ----------
    min_radius : float
        Innermost shell edge, h^-1 Mpc.
    max_radius : float
        Outermost shell edge, h^-1 Mpc. Must not exceed
        `config.MAX_ANALYSIS_RADIUS` (PBC minimum-image ceiling, hard rule 3).
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
        if self.max_radius > config.MAX_ANALYSIS_RADIUS:
            raise ValueError(
                f"ShellConfig: max_radius ({self.max_radius}) exceeds "
                f"config.MAX_ANALYSIS_RADIUS ({config.MAX_ANALYSIS_RADIUS}); "
                "minimum-image geometry is not well defined beyond it "
                "(CLAUDE.md hard rule 3)."
            )
        if self.radii_step <= 0.0:
            raise ValueError(f"ShellConfig: radii_step must be > 0, got {self.radii_step}.")

    @property
    def shell_edges(self) -> np.ndarray:
        """Strictly increasing radial shell edges, shape (B + 1,).

        Feeds directly into `estimators.shell_dipole.shell_dipole`'s
        `shell_edges` argument. Built as `min_radius` to `max_radius` in steps
        of `radii_step`, then `max_radius` appended exactly so the outer edge
        is exact and, via `__post_init__`, guaranteed not to exceed
        `config.MAX_ANALYSIS_RADIUS`.

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
        return np.minimum(edges, config.MAX_ANALYSIS_RADIUS)

    @property
    def shell_centers(self) -> np.ndarray:
        """Shell midpoints, shape (B,); `0.5 * (edge_low + edge_high)` per shell."""
        edges = self.shell_edges
        return 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------------
# SelectionConfig
# --------------------------------------------------------------------------


@dataclass
class SelectionConfig:
    """Placeholder selection knobs for the not-yet-built `mocks/` and `selection/` arms.

    Kept deliberately lean: only the fields that are already clearly needed
    (a mass floor matching the catalogue, an observer count and placement
    strategy for the MDPL2 mock-observer arm) rather than the old repo's full
    `OriginConfig` surface. Expect this to grow once `mocks/` and `selection/`
    are implemented.

    Attributes
    ----------
    mass_min : float
        Minimum halo mass, h^-1 M_sun. Defaults to 1e12, matching the
        catalogue floor (`mdpl2_rockstar_125_pid-1_mvir12.csv` is already cut
        at mvir >= 1e12).
    mass_max : float | None
        Maximum halo mass, h^-1 M_sun, or None for no upper cut.
    number_of_observers : int
        Number of mock observers to place in the box.
    observer_selection : str
        Observer placement strategy; one of "random" or "virgo".
    """

    #: Allowed values of `observer_selection`. Not a dataclass field (no type
    #: annotation), just a class-level constant the validator checks against.
    _ALLOWED_OBSERVER_SELECTIONS = frozenset({"random", "virgo"})

    mass_min: float = 1e12
    mass_max: float | None = None
    number_of_observers: int = 1000
    observer_selection: str = "random"

    def __post_init__(self) -> None:
        if self.mass_min <= 0.0:
            raise ValueError(f"SelectionConfig: mass_min must be > 0, got {self.mass_min}.")
        if self.mass_max is not None and self.mass_max <= self.mass_min:
            raise ValueError(
                f"SelectionConfig: mass_max ({self.mass_max}) must be greater "
                f"than mass_min ({self.mass_min})."
            )
        if self.number_of_observers < 1:
            raise ValueError(
                "SelectionConfig: number_of_observers must be >= 1, got "
                f"{self.number_of_observers}."
            )
        if self.observer_selection not in self._ALLOWED_OBSERVER_SELECTIONS:
            raise ValueError(
                f"SelectionConfig: observer_selection must be one of "
                f"{sorted(self._ALLOWED_OBSERVER_SELECTIONS)}, got "
                f"{self.observer_selection!r}."
            )


# --------------------------------------------------------------------------
# Settings — aggregator
# --------------------------------------------------------------------------


@dataclass
class Settings:
    """Aggregates one instance of every tunable settings group.

    Each field uses `default_factory` so that two `Settings` instances (e.g.
    two `default_settings()` calls) own independent sub-configs; mutating one
    instance's `shells` does not affect another's.

    Attributes
    ----------
    paths : PathsConfig
    cosmology : CosmologyConfig
    shells : ShellConfig
    selection : SelectionConfig
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    shells: ShellConfig = field(default_factory=ShellConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


def default_settings() -> Settings:
    """Return a fresh `Settings` with independent, default-valued sub-configs.

    Deliberately a factory rather than a shared module-level singleton: callers
    that need to override a field (e.g. a test pointing `paths.output_dir` at
    a tmp directory) get their own instance to mutate, without risk of
    clobbering another caller's settings.
    """
    return Settings()
