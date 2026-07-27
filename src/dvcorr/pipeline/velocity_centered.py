"""
velocity_centered.py
---------------------
Reusable pipeline stage functions for measuring the velocity-centered zeta_1
dipole (`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`) on
real MDPL2 halos: load-and-carve, candidate drawing, running the estimator,
normalizing the raw result (including a velocity-shuffle null), and building
the two-panel figure (the dipole alongside its monopole companion, CLAUDE.md
hard rule 6).

This is the single source of truth for that pipeline (CLAUDE.md's "one
library, two thin consumers" model). Two consumers call these stage functions
without reimplementing any of them:

- `scripts/plot_velocity_centered_dipole.py` -- a thin, headless driver that
  sets the Agg backend, builds a `RunConfig`, chains the stages below, and
  saves the PNG. `python -m scripts.plot_velocity_centered_dipole`.
- `notebooks/05_velocity_centered_dipole.ipynb` -- the exploratory twin; every
  code cell calls one of the stage functions below in sequence and adds plots
  / narrative, never reimplementing the pipeline.

This module's scope now also supports the velocity-frame comparison built on
top of it in `dvcorr.pipeline.velocity_frame_comparison`, which reuses
`RunConfig`, `load_and_carve`, `draw_candidates`, `global_number_density`,
`NormalizedDipole`, and `normalize_stacked_dipole` rather than duplicating
any of them (CLAUDE.md's "one library, two thin consumers" model, applied a
second time within the library itself: the comparison pipeline is additive
on top of this one, not a fork of it).

Matplotlib backend discipline
------------------------------
Importing this module does NOT select a matplotlib backend: `matplotlib.use`
is never called here, only in the thin script's own module body (before it
imports this module). A notebook that imports `make_figure` below keeps
whatever backend (e.g. an inline one) it already has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dvcorr import conventions
from dvcorr.config import PathsConfig, ShellConfig
from dvcorr.estimators.shell_dipole import (
    VelocityCenteredShellDipoleResult,
    center_standard_error,
    expected_shell_occupancy,
    velocity_centered_shell_dipole,
)

# Default radial binning for this run. min_radius defaults to radii_step
# (NOT 0.0): with candidates subsampled from the tracer array, every
# candidate is its own tracer at r = 0 -- per the documented coincident-
# tracer contract (velocity_centered_shell_dipole's Notes) that self-pair
# adds +1 to pair_count[0] and +u_alpha to monopole[0], a pure self-
# correlation term that would otherwise pollute exactly the hard-rule-6
# monopole diagnostic panel. Starting the innermost edge at radii_step
# excludes r = 0 (below-innermost-edge is excluded, same binning contract as
# `shell_dipole`). notebook 04's own innermost-bin turnover check found that
# bin to be one-halo/virial noise anyway, so nothing of value is lost.
_DEFAULT_RADII_STEP = 5.0
_DEFAULT_MAX_RADIUS = 60.0


def _default_shells() -> ShellConfig:
    """`RunConfig.shells`'s default_factory: min_radius = radii_step (see above)."""
    return ShellConfig(
        min_radius=_DEFAULT_RADII_STEP,
        max_radius=_DEFAULT_MAX_RADIUS,
        radii_step=_DEFAULT_RADII_STEP,
    )


@dataclass
class RunConfig:
    """Every tunable number for this run, in one place (CLAUDE.md hard rule 4).

    Radial binning is a composed `dvcorr.config.ShellConfig` (`shells`), not a
    trio of mirrored fields: `ShellConfig.__post_init__` already validates
    ordering, `radii_step > 0`, and
    `max_radius <= dvcorr.conventions.MAX_ANALYSIS_RADIUS`, and its
    `shell_edges` property already builds a strictly increasing, correctly
    clipped array -- reusing it here (rather than a second, drifting copy of
    the same logic) is the single source of truth for that validation.

    Attributes
    ----------
    sub_volume_radius : float
        Radius of the spherical sub-volume carved around the observer,
        h^-1 Mpc. Mirrors notebook 04's R_SUB.
    shells : ShellConfig
        Radial shell binning; see `_default_shells` for why the default
        `min_radius` is `radii_step`, not 0.
    n_candidate_centers : int
        Number of candidate velocity-object centers subsampled from the
        carved halos; `velocity_centered_shell_dipole`'s own core cut then
        decides how many of them survive as centers.
    seed : int
        Seed for the candidate-center subsample.
    shuffle_seed : int
        Seed for the velocity-shuffle null, kept distinct from `seed` so the
        null's randomness never silently reuses the center-selection stream.
    output_name : str
        Output figure filename, written under `PathsConfig().output_dir`.
    dpi : int
        Figure resolution.
    min_center_speed : float
        Minimum |v_alpha| (km/s) for a center to have a well-defined flow
        direction v_hat_alpha. Centers below it are dropped from BOTH
        frames' center sets at the ORCHESTRATION level
        (`dvcorr.pipeline.velocity_frame_comparison.select_shared_centers`),
        never inside an estimator, so both frames measure the identical
        center set and the only difference between them is the axis and the
        velocity scalar (see
        `dvcorr.estimators.velocity_frame_dipole.velocity_frame_shell_dipole`'s
        zero-speed ValueError, which this floor exists to prevent a caller
        from ever triggering). This is a TUNABLE data-quality knob, not a
        convention: the default (1.0 km/s) only removes pathologically slow
        halos (effectively zero-speed, within floating-point noise), not a
        physically meaningful cut; raise it to test sensitivity to poorly
        determined flow directions. It lives on `RunConfig` -- shared by both
        frames -- rather than on
        `dvcorr.pipeline.velocity_frame_comparison.ComparisonRunConfig`,
        because it filters the shared CENTER SET consumed by both estimators,
        not a comparison-only plotting or null-test knob.
    """

    sub_volume_radius: float = 300.0
    shells: ShellConfig = field(default_factory=_default_shells)
    n_candidate_centers: int = 4000
    seed: int = 42
    shuffle_seed: int = 43
    output_name: str = "velocity_centered_dipole.png"
    dpi: int = 150
    min_center_speed: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.sub_volume_radius <= 0.0
            or self.sub_volume_radius > conventions.MAX_ANALYSIS_RADIUS
        ):
            raise ValueError(
                f"RunConfig: sub_volume_radius ({self.sub_volume_radius}) must be "
                f"> 0 and <= dvcorr.conventions.MAX_ANALYSIS_RADIUS "
                f"({conventions.MAX_ANALYSIS_RADIUS})."
            )
        # self.shells' own __post_init__ already validated ordering, radii_step
        # > 0, and max_radius <= dvcorr.conventions.MAX_ANALYSIS_RADIUS -- not
        # re-checked here, so there is exactly one place that ceiling is
        # enforced.
        if self.min_center_speed < 0.0:
            raise ValueError(
                f"RunConfig: min_center_speed ({self.min_center_speed}) must be "
                ">= 0. 0.0 is allowed and means 'drop only exactly-zero-speed "
                "centers'."
            )


# Plot styling -- named here rather than inline (hard rule 4).
_COLOR_SIGNAL = "#2a78d6"
_COLOR_NULL = "#eb6834"
_COLOR_ZERO_LINE = "0.5"
_COLOR_MONOPOLE = "0.3"
_LABEL_COLOR = "#111111"
_ZERO_LINE_WIDTH = 0.8
_BAND_ALPHA = 0.2
_GRID_ALPHA = 0.3
_FIGSIZE = (8.0, 8.0)
_HEIGHT_RATIOS = (3, 1)


# ---------------------------------------------------------------------------
# Stage 1: load + carve
# ---------------------------------------------------------------------------


def load_and_carve(cfg: RunConfig, paths: PathsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load the MDPL2 catalog and carve a sphere around the observer.

    Loads only `dvcorr.conventions.POSITION_COLUMNS +
    dvcorr.conventions.VELOCITY_COLUMNS` (the ~4M-row catalog is not loaded
    fully otherwise, CLAUDE.md), then keeps halos within
    `cfg.sub_volume_radius` of `dvcorr.conventions.OBSERVER_POSITION`. Plain
    Euclidean distance IS the minimum image here: the sub-volume sits well
    inside the box, clear of the periodic faces (see geometry.py's PBC
    contract and notebooks/04_first_mdpl2_run.ipynb cell 7).

    Parameters
    ----------
    cfg : RunConfig
    paths : PathsConfig
        Supplies `mdpl2_catalog`'s path.

    Returns
    -------
    pos : ndarray, shape (N_carved, 3)
    vel : ndarray, shape (N_carved, 3)
        Carved halo positions and velocities, comoving h^-1 Mpc / km/s.

    Raises
    ------
    RuntimeError
        If the catalog loads zero rows, or zero halos survive the carve.
        Both are checked HERE, before any percentage is printed from them --
        a bare `n / 0` inside an f-string is a ZeroDivisionError with a
        misleading traceback, not a diagnosis.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    usecols = list(conventions.POSITION_COLUMNS) + list(conventions.VELOCITY_COLUMNS)

    print(f"loading {paths.mdpl2_catalog} (columns: {usecols}) ...")
    halos = pd.read_csv(paths.mdpl2_catalog, usecols=usecols)
    if len(halos) == 0:
        raise RuntimeError(f"{paths.mdpl2_catalog} loaded zero rows.")
    print(f"loaded {len(halos)} halos")

    pos_all = halos[list(conventions.POSITION_COLUMNS)].to_numpy(dtype=float)
    vel_all = halos[list(conventions.VELOCITY_COLUMNS)].to_numpy(dtype=float)

    d_obs = np.linalg.norm(pos_all - observer, axis=1)
    in_sub = d_obs <= cfg.sub_volume_radius
    pos = pos_all[in_sub]
    vel = vel_all[in_sub]
    n_carved = pos.shape[0]
    if n_carved == 0:
        raise RuntimeError(
            f"no halos survived the sub_volume_radius = {cfg.sub_volume_radius} "
            "carve; check the observer position and sub_volume_radius."
        )
    print(
        f"carved {n_carved} of {len(halos)} halos within "
        f"sub_volume_radius = {cfg.sub_volume_radius} h^-1 Mpc "
        f"({100.0 * n_carved / len(halos):.2f}%)"
    )
    return pos, vel


# ---------------------------------------------------------------------------
# Stage 2: candidate centers
# ---------------------------------------------------------------------------


def draw_candidates(
    cfg: RunConfig, pos: np.ndarray, vel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded subsample of the carved halos, as candidate velocity centers.

    `velocity_centered_shell_dipole`'s own core cut (`core_center_mask`)
    decides which of these candidates survive as centers; this stage only
    performs the subsample.

    Parameters
    ----------
    cfg : RunConfig
    pos, vel : ndarray, shape (N_carved, 3)
        Carved halo positions/velocities, as returned by `load_and_carve`.

    Returns
    -------
    s_candidates, v_candidates : ndarray, shape (N_candidates, 3)

    Raises
    ------
    RuntimeError
        If `cfg.n_candidate_centers <= 0` or `pos` is empty, either of which
        would otherwise surface later as a confusing zero-candidate estimator
        call.
    """
    n_candidates = min(cfg.n_candidate_centers, pos.shape[0])
    if n_candidates <= 0:
        raise RuntimeError(
            f"n_candidate_centers ({cfg.n_candidate_centers}) and the carved "
            f"population ({pos.shape[0]}) give zero candidates."
        )
    rng = np.random.default_rng(cfg.seed)
    candidate_idx = rng.choice(pos.shape[0], size=n_candidates, replace=False)
    return pos[candidate_idx], vel[candidate_idx]


# ---------------------------------------------------------------------------
# Stage 3: run the estimator
# ---------------------------------------------------------------------------


def run_estimator(
    cfg: RunConfig,
    s_candidates: np.ndarray,
    v_candidates: np.ndarray,
    s_tracers: np.ndarray,
    observer: np.ndarray,
) -> VelocityCenteredShellDipoleResult:
    """Run `velocity_centered_shell_dipole` and report the core-cut survival.

    Parameters
    ----------
    cfg : RunConfig
    s_candidates, v_candidates : ndarray, shape (N_candidates, 3)
        As returned by `draw_candidates`.
    s_tracers : ndarray, shape (N_t, 3)
        Density tracers -- ALL carved halos (`load_and_carve`'s `pos`).
    observer : ndarray, shape (3,)

    Returns
    -------
    VelocityCenteredShellDipoleResult

    Raises
    ------
    RuntimeError
        If zero candidates survive the core cut.
    """
    result = velocity_centered_shell_dipole(
        s_centers=s_candidates,
        v_centers=v_candidates,
        s_tracers=s_tracers,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )
    # result.n_candidates == s_candidates.shape[0], guaranteed > 0 by
    # draw_candidates' own guard, so this division is safe by construction.
    print(
        f"n_candidates = {result.n_candidates}, n_centers = {result.n_centers} "
        f"({100.0 * result.n_centers / result.n_candidates:.1f}% survive the core cut)"
    )
    if result.n_centers == 0:
        raise RuntimeError(
            "no candidate centers survived the core cut; widen "
            "sub_volume_radius or shrink the shell range."
        )
    return result


# ---------------------------------------------------------------------------
# Stage 4: normalize
# ---------------------------------------------------------------------------


def global_number_density(n_tracers: int, sub_volume_radius: float) -> float:
    """Global tracer number density n_bar over a spherical sub-volume.

    n_bar = n_tracers / ((4*pi/3) * sub_volume_radius**3) -- the Nusser
    (2017) eq. 24 normalization's n_bar, from the GLOBAL tracer count over
    the sub-volume (never estimated inside the estimator; see
    `dvcorr.estimators.shell_dipole.expected_shell_occupancy`).

    Parameters
    ----------
    n_tracers : int
        Total tracer count over the sub-volume (e.g. all carved halos).
    sub_volume_radius : float
        Radius of the spherical sub-volume, h^-1 Mpc.

    Returns
    -------
    float
        n_bar, tracers per (h^-1 Mpc)^3.
    """
    sub_volume = (4.0 / 3.0) * np.pi * sub_volume_radius**3  # 4pi/3: sphere volume, pure math
    return n_tracers / sub_volume


@dataclass(frozen=True)
class NormalizedDipole:
    """Normalised curves ready to plot: everything `make_figure` needs.

    Every array has shape (B,), aligned shell-for-shell with the
    `VelocityCenteredShellDipoleResult.shell_centers` it was built from.

    Attributes
    ----------
    zeta_hat : ndarray, shape (B,)
        Normalised velocity-centered dipole zeta_1(r), km/s.
    sem : ndarray, shape (B,)
        Standard error of `zeta_hat`, from `center_standard_error`, scaled by
        the same normalization.
    zeta_hat_shuffle : ndarray, shape (B,)
        The velocity-shuffle null, same normalization.
    sem_shuffle : ndarray, shape (B,)
        Standard error of `zeta_hat_shuffle`.
    monopole_norm : ndarray, shape (B,)
        Normalised ell=0 companion (CLAUDE.md hard rule 6): the residual-bulk
        / incomplete-shell diagnostic, km/s.
    """

    zeta_hat: np.ndarray
    sem: np.ndarray
    zeta_hat_shuffle: np.ndarray
    sem_shuffle: np.ndarray
    monopole_norm: np.ndarray


def shell_dipole_norm_scale(shell_edges: np.ndarray, n_bar: float) -> np.ndarray:
    """The per-shell dipole normalization scale, `3 / (sqrt(3/4pi) * nbar*V_b)`.

    ONE home for this factor (CLAUDE.md's DRY spirit, extracted specifically
    because two call sites now need it): `normalize_stacked_dipole` uses it
    for the STACKED curve, and
    `dvcorr.pipeline.velocity_frame_comparison.normalize_comparison` uses the
    identical factor for its per-center summary
    (`per_center_dipole * shell_dipole_norm_scale(...)`, deliberately without
    the `/ n_centers` stacking average). Before this was extracted, the two
    call sites carried the same three lines independently -- a genuine risk
    that the normalization convention could drift between the plotted stack
    and the per-center breakdown that is supposed to sum back to it.

        3 = 2*ell + 1, picks up <mu^2> = 1/3 over a full shell
        sqrt(3/4pi) = the Y_10 normalization constant, undone here so the
                      number is comparable (up to the (-1)^ell relation,
                      dvcorr.conventions.nusser_multipole_sign) to the
                      group-centered 3*Sum(u*mu)/N of
                      `dvcorr.estimators.shell_dipole.shell_dipole`.

    Parameters
    ----------
    shell_edges : ndarray, shape (B + 1,)
        Radial shell boundaries, h^-1 Mpc.
    n_bar : float
        Global tracer number density over the sub-volume, e.g. from
        `global_number_density`.

    Returns
    -------
    ndarray, shape (B,)
        The per-shell scale factor; multiplying a raw (per-center or
        stacked) dipole moment by this, after dividing by `n_centers` where
        a stacked average is wanted, gives the normalized zeta_hat_1.
    """
    nbar_v_b = expected_shell_occupancy(n_bar, shell_edges)
    y10_norm = np.sqrt(3.0 / (4.0 * np.pi))
    return 3.0 / (y10_norm * nbar_v_b)  # per-shell scale, (B,)


def normalize_stacked_dipole(
    shell_edges: np.ndarray,
    dipole: np.ndarray,
    monopole: np.ndarray,
    per_center_dipole: np.ndarray,
    null_dipole: np.ndarray,
    null_per_center_dipole: np.ndarray,
    n_centers: int,
    n_bar: float,
) -> NormalizedDipole:
    """The normalization arithmetic, factored out to ONE home.

    This contains exactly the arithmetic that used to live inline inside
    `normalize_result` (see that function, which now delegates here). It is
    pulled out to a stand-alone function because
    `dvcorr.pipeline.velocity_frame_comparison` needs the identical
    normalization for the velocity-frame result, and that frame's NULL is a
    different construction (a random-axis re-run, not a scalar permutation --
    see `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null` for
    why) -- so the one piece that legitimately differs between the two
    call sites, the null, is a parameter here rather than built inside this
    function.

        zeta_hat_1(r_b) = 3 * (dipole_b / n_centers) / (sqrt(3/4pi) * nbar*V_b)
          3            = 2*ell + 1, picks up <mu^2> = 1/3 over a full shell
          sqrt(3/4pi)  = the Y_10 normalization constant, undone here so the
                         number is comparable (up to the (-1)^ell relation,
                         dvcorr.conventions.nusser_multipole_sign) to the
                         group-centered 3*Sum(u*mu)/N of
                         `dvcorr.estimators.shell_dipole.shell_dipole`.
    The n_bar*V_b division happens HERE, at the call site, visibly and
    swappably -- never inside an estimator.

    Parameters
    ----------
    shell_edges : ndarray, shape (B + 1,)
        Radial shell boundaries, h^-1 Mpc, e.g. `result.shell_edges`.
    dipole : ndarray, shape (B,)
        Raw stacked dipole numerator, e.g. `result.dipole`.
    monopole : ndarray, shape (B,)
        Raw stacked monopole (ell=0 companion, hard rule 6), e.g.
        `result.monopole`.
    per_center_dipole : ndarray, shape (N_c, B)
        Per-center dipole breakdown, e.g. `result.per_center_dipole`, used
        for the across-center standard error via `center_standard_error`.
    null_dipole : ndarray, shape (B,)
        The raw stacked dipole of whatever NULL the caller has already
        built -- a shuffled/permuted recombination for the observer frame
        (`normalize_result`'s velocity-shuffle), or a random-axis re-run of
        the estimator for the velocity frame
        (`dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null`).
        This function does not know or care which; it only normalizes
        whatever it is handed. `zeta_hat_shuffle` on the returned
        `NormalizedDipole` therefore names "whatever null the caller built",
        not specifically a scalar permutation -- read it as the null curve,
        not as a promise about its construction.
    null_per_center_dipole : ndarray, shape (N_c, B)
        Per-center breakdown of the same null, for its own standard error.
    n_centers : int
        Number of surviving centers, e.g. `result.n_centers`. Shared by the
        signal and the null: both are stacks over the SAME center set.
    n_bar : float
        Global tracer number density over the sub-volume, e.g. from
        `global_number_density`.

    Returns
    -------
    NormalizedDipole
    """
    nbar_v_b = expected_shell_occupancy(n_bar, shell_edges)
    norm_scale = shell_dipole_norm_scale(shell_edges, n_bar)

    zeta_hat = (dipole / n_centers) * norm_scale
    sem = center_standard_error(per_center_dipole) * norm_scale

    zeta_hat_shuffle = (null_dipole / n_centers) * norm_scale
    sem_shuffle = center_standard_error(null_per_center_dipole) * norm_scale

    # ell=0 companion, same normalization convention minus the 3/Y10 factors
    # (hard rule 6: never plot the dipole without it).
    monopole_norm = (monopole / n_centers) / nbar_v_b

    return NormalizedDipole(
        zeta_hat=zeta_hat,
        sem=sem,
        zeta_hat_shuffle=zeta_hat_shuffle,
        sem_shuffle=sem_shuffle,
        monopole_norm=monopole_norm,
    )


def normalize_result(
    result: VelocityCenteredShellDipoleResult,
    n_bar: float,
    shuffle_seed: int,
) -> NormalizedDipole:
    """Turn a raw estimator result + n_bar into the plotted, normalized curves.

    Builds the velocity-shuffle null (a seeded permutation of
    `result.per_center_u`, recombined with `result.per_center_amplitude` --
    no second estimator pass needed) and delegates the actual normalization
    arithmetic to `normalize_stacked_dipole`, which is now the single home
    for it (see that function's docstring for the formula). This function's
    public signature, return type, and numerical output are UNCHANGED by
    that refactor -- it is a pure extraction, verified by construction (the
    arithmetic moved verbatim) and numerically in the task that introduced
    `normalize_stacked_dipole`.

    Parameters
    ----------
    result : VelocityCenteredShellDipoleResult
    n_bar : float
        Global tracer number density over the sub-volume, e.g. from
        `global_number_density`.
    shuffle_seed : int
        Seed for the velocity-shuffle null: a permutation of
        `result.per_center_u`, recombined with `result.per_center_amplitude`
        -- no second estimator pass.

    Returns
    -------
    NormalizedDipole
    """
    shuffle_rng = np.random.default_rng(shuffle_seed)
    perm = shuffle_rng.permutation(result.per_center_u.size)
    shuffled_per_center_dipole = (
        result.per_center_u[perm][:, None] * result.per_center_amplitude
    )
    null_dipole = shuffled_per_center_dipole.sum(axis=0)

    return normalize_stacked_dipole(
        shell_edges=result.shell_edges,
        dipole=result.dipole,
        monopole=result.monopole,
        per_center_dipole=result.per_center_dipole,
        null_dipole=null_dipole,
        null_per_center_dipole=shuffled_per_center_dipole,
        n_centers=result.n_centers,
        n_bar=n_bar,
    )


# ---------------------------------------------------------------------------
# Stage 5: figure
# ---------------------------------------------------------------------------


def make_figure(
    cfg: RunConfig,
    result: VelocityCenteredShellDipoleResult,
    normalized: NormalizedDipole,
) -> plt.Figure:
    """Build the two-panel figure. Does NOT save it -- the caller (a script's
    `main()` or a notebook cell) does, so this stays a pure builder callable
    from a notebook that wants to display the figure inline instead of / as
    well as writing a PNG.

    Two panels sharing the x-axis (CLAUDE.md hard rule 6 -- the dipole is
    never plotted alone): the normalized zeta_hat_1(r) with its SEM band and
    the shuffle null with its own band on top; the normalized monopole
    companion below.

    Parameters
    ----------
    cfg : RunConfig
    result : VelocityCenteredShellDipoleResult
    normalized : NormalizedDipole

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_dipole, ax_mono) = plt.subplots(
        2, 1, figsize=_FIGSIZE, sharex=True, gridspec_kw={"height_ratios": _HEIGHT_RATIOS}
    )

    r = result.shell_centers

    ax_dipole.fill_between(
        r, normalized.zeta_hat - normalized.sem, normalized.zeta_hat + normalized.sem,
        color=_COLOR_SIGNAL, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, normalized.zeta_hat, "o-", color=_COLOR_SIGNAL,
        label=fr"$\hat\zeta_1$  (N$_c$={result.n_centers})",
    )
    ax_dipole.fill_between(
        r, normalized.zeta_hat_shuffle - normalized.sem_shuffle,
        normalized.zeta_hat_shuffle + normalized.sem_shuffle,
        color=_COLOR_NULL, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(r, normalized.zeta_hat_shuffle, "s--", color=_COLOR_NULL, label="shuffle null")
    ax_dipole.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_dipole.set_ylabel(r"$\hat\zeta_1$  [km/s]", color=_LABEL_COLOR)
    ax_dipole.legend()
    ax_dipole.grid(alpha=_GRID_ALPHA)
    ax_dipole.spines["top"].set_visible(False)
    ax_dipole.spines["right"].set_visible(False)
    ax_dipole.set_title(
        f"velocity-centered dipole  "
        f"(R_sub={cfg.sub_volume_radius:.0f} h$^{{-1}}$Mpc, "
        f"r_max={cfg.shells.max_radius:.0f} h$^{{-1}}$Mpc, N_c={result.n_centers})"
    )

    ax_mono.plot(r, normalized.monopole_norm, "o-", color=_COLOR_MONOPOLE)
    ax_mono.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_mono.set_ylabel(r"$\hat\zeta_0$  [km/s]", color=_LABEL_COLOR)
    ax_mono.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
    ax_mono.grid(alpha=_GRID_ALPHA)
    ax_mono.spines["top"].set_visible(False)
    ax_mono.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
