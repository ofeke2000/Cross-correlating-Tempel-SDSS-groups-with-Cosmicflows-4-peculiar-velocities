"""
velocity_centered.py
---------------------
Reusable pipeline stage functions for measuring the velocity-centered zeta_1
dipole (`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`) on
real MDPL2 halos: load-and-carve, candidate drawing, running the estimator,
normalizing the raw result (including two nulls -- a velocity shuffle and a
matched-Gaussian draw, `normalize_result`), and building
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
`RunConfig`, `load_and_carve`, `draw_candidates`, `box_number_density`,
`NormalizedDipole`, and `normalize_stacked_dipole` rather than duplicating
any of them (CLAUDE.md's "one library, two thin consumers" model, applied a
second time within the library itself: the comparison pipeline is additive
on top of this one, not a fork of it), and the redshift-space comparison
built in `dvcorr.pipeline.redshift_space_comparison`, which additionally
reuses `SharedCenterSet` and `select_shared_centers` (moved here FROM
`velocity_frame_comparison.py` in that task, since both comparison pipelines
need them and this module is the shared base both import from -- leaving
them in `velocity_frame_comparison.py` would have made the redshift-space
pipeline depend on the velocity-frame one, which it has nothing else to do
with) and `_load_all_halos` (the single-CSV-read helper `load_and_carve`
itself is built on, factored out so the redshift-space pipeline's dual carve
-- a plain `sub_volume_radius` pass to derive `v_margin`, then a second,
buffered pass -- does not re-read the ~4M-row catalog from disk twice).

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
from dvcorr.config import (
    SPACING_LOG,
    CatalogConfig,
    PathsConfig,
    ShellConfig,
    volume_weighted_shell_radii,
)
from dvcorr.pipeline.catalog_conversion import IS_DISTINCT_COLUMN
from dvcorr.estimators.shell_dipole import (
    VelocityCenteredShellDipoleResult,
    center_standard_error,
    core_center_mask,
    expected_shell_occupancy,
    velocity_centered_shell_dipole,
)

# Default radial binning for this run: log spacing, min_radius = 1.0,
# max_radius = 64.0, n_bins = 12, plus the [0, 1) inner bin -- edges
# 0, 1, sqrt2, 2, 2sqrt2, 4, ... 64 (a ratio of exactly sqrt(2) between
# consecutive LOG edges: exact powers of two and their halves), so B = 13
# shells. Five of the twelve log bins sit below 5.66 h^-1 Mpc, which is the
# whole point of the change: the r ~ 5 regime, previously discarded by the
# old radii_step=5.0 linear default, is now resolved into several bins rather
# than folded into one.
#
# Why the [0, 1) inner bin (`include_zero_bin`), given that min_radius = 1.0
# is justified below as a floor: the floor says where the estimator is
# TRUSTWORTHY, not where it should be blind. A log ladder cannot reach zero,
# so without this bin the r < 1 region -- the one-halo regime -- is not
# merely noisy, it is unmeasured, and the innermost log bin has nothing to be
# read against. One linear bin covers it, and the monopole panel (hard rule
# 6) is where it earns its place: measured on the default catalog, that bin's
# realized occupancy is ~9x the uniform expectation, the top of the 1 + xi(r)
# rise. It is the only non-geometric shell in the array and its point on a
# figure is the only one whose horizontal extent does not scale with r --
# read it as a bracket over [0, 1), not as a measurement at r_eff = 0.75.
#
# The r = 0 self-pair. Candidates are subsampled from the tracer array
# (`draw_candidates`), so every center is its own tracer at exactly zero
# separation: exactly N_c self-pairs, all of them in [0, 1). Their weight
# depends entirely on the catalog, and on BOTH supported catalogs it is too
# large to admit:
#   full  (n_bar = 0.127 (h^-1 Mpc)^-3, the default) -- 4126 expected uniform
#         pairs at N_c = 7726, ~9x that realized, so N_c self-pairs would be
#         ~20% of the innermost bin: a 20% inflation of pair_count[0] and of
#         monopole[0].
#   mvir12 (n_bar ~ 4e-3) -- ~0.017 * N_c expected uniform pairs, so the
#         self-pairs would OUTNUMBER the real ones by more than an order of
#         magnitude and the innermost monopole would be essentially pure
#         self-correlation.
# The dipole is untouched either way (a directionless separation carries
# cos_theta = 0), which is precisely why this is worth a comment: the damage
# is confined to the monopole -- the diagnostic hard rule 6 has us judge the
# dipole BY -- and to the zeta_1/zeta_0 ratio, where it would pass unnoticed
# as a suppression rather than surfacing as a NaN. Until this binning they
# were excluded only INCIDENTALLY, by min_radius > 0 putting them below the
# innermost edge; with edges[0] = 0 that protection is gone, so the
# estimators now exclude them EXPLICITLY, on `r_mag > 0` in the in-range mask
# (all four r-binning sites; see `velocity_centered_shell_dipole`'s Notes).
# That is the stronger guarantee: it holds for any binning rather than only
# for binnings that happen to start above zero, and no config change can undo
# it.
#
# Why min_radius = 1.0. Statistically it is not binding on the default
# catalog: at n_bar = 0.127 (h^-1 Mpc)^-3 even [0, 1) holds ~4126 expected
# uniform pairs over the stack. It binds on `mvir12` (n_bar ~ 4e-3), where
# requiring >= 50 expected pairs gives r1 >= 0.94 -- which is where the 1.0
# came from and why it stays a sensible floor for the LOG ladder across both
# catalogs.
#
# Physically, what min_radius = 1.0 marks depends on the catalog, and the
# default changed underneath this number:
#   mvir12 is pid = -1 (distinct halos only), so every center carries a hard
#     exclusion hole of R_vir(host) -- 0.20 h^-1 Mpc at 1e12 Msun/h, 0.94 at
#     1e14, 2.0 at 1e15 -- and the massive centers, which dominate the
#     |u|-weighted stack, have the LARGEST holes. There, r < 1 is genuinely
#     an exclusion hole.
#   full (the default) INCLUDES subhalos, which sit inside their hosts'
#     virial radii, so there is no exclusion hole: the [0, 1) bin is
#     populated, and populated by host-subhalo pairs whose velocities are
#     virial rather than infall. Its steep monopole is the one-halo term, not
#     a cosmological signal.
# In neither case is MDPL2's force softening (5 h^-1 kpc) or mass resolution
# the limit; both sit ~100x below. The [0, 1) bin and the one or two log bins
# above it are the one-halo regime, deliberately SHOWN (alongside their
# monopole companion, hard rule 6) rather than hidden behind a single wide
# bin -- and on the default catalog they need reading as virial physics.
#
# Why max_radius = 64.0, not 60.0 (cost, paid deliberately): r_max 60 -> 64
# grows every `query_ball_point(s_alpha, edges[-1])` ball by (64/60)**3 ~=
# 1.21, and `core_margin` defaults to `cfg.shells.max_radius`
# (`select_shared_centers`, near line 469), tightening the core ball
# 240 -> 236 h^-1 Mpc so the core-cut survival rate falls from ~51.2% to
# 48.3% (-5.6%). That rate is geometric -- (236/300)**3 -- so it is the same
# for either catalog; measured on the default run it gives N_c = 7726 of
# 16000 candidates, the carve itself keeping 14418601 of 127388160 halos
# within R_sub = 300. Worth stating explicitly because it moves the N_c printed
# in every figure title. Runtime is independent of bin COUNT
# (`np.bincount` is O(pairs)), so 12 log bins cost the same as the old 11
# linear ones -- the cost above is entirely from r_max, not from n_bins.
_DEFAULT_SPACING = SPACING_LOG
_DEFAULT_MIN_RADIUS = 1.0
_DEFAULT_MAX_RADIUS = 64.0
_DEFAULT_N_BINS = 12
_DEFAULT_INCLUDE_ZERO_BIN = True


def _default_shells() -> ShellConfig:
    """`RunConfig.shells`'s default_factory: log, 1 to 64 h^-1 Mpc, 12 bins + [0, 1).

    Thirteen shells, not twelve: `include_zero_bin` prepends `[0, min_radius)`
    to the geometric ladder, so `shell_edges` starts at 0.0 exactly. See the
    module-level comment block above `_DEFAULT_SPACING` for why.

    `radii_step` is left at `ShellConfig`'s own class default (10.0) -- it is
    inert under `spacing=SPACING_LOG` (only read for `SPACING_LINEAR`,
    `ShellConfig`'s docstring), so there is nothing to set here.
    """
    return ShellConfig(
        min_radius=_DEFAULT_MIN_RADIUS,
        max_radius=_DEFAULT_MAX_RADIUS,
        spacing=_DEFAULT_SPACING,
        n_bins=_DEFAULT_N_BINS,
        include_zero_bin=_DEFAULT_INCLUDE_ZERO_BIN,
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
    catalog : CatalogConfig
        Which halo catalog this run reads and which halos it keeps from it.
        Composed for the same reason `shells` is: `CatalogConfig` already
        validates the catalog name and the mass bounds, and composing it means
        switching catalogs is one line at the call site with no pipeline
        signature change. Defaults to the full catalog, uncut -- see
        `dvcorr.config.catalog` for what that population contains and why the
        absence of a mass floor is a choice rather than an oversight.
    shells : ShellConfig
        Radial shell binning; see `_default_shells` (and its module-level
        comment block above `_DEFAULT_SPACING`) for why the default
        `min_radius` is `1.0` -- the exclusion-radius floor -- and why the
        `[0, min_radius)` bin is nonetheless measured and shown rather than
        dropped. `radii_step` (the old linear default's rationale) is inert
        under `SPACING_LOG` and not read here.
    n_candidate_centers : int
        Number of candidate velocity-object centers subsampled from the
        carved halos; `velocity_centered_shell_dipole`'s own core cut then
        decides how many of them survive as centers (~48%, geometrically).
        Cost is linear in the survivors -- one neighbor query each -- and
        measured at ~40 ms per center on the full catalog, where a query ball
        holds ~1.4e5 tracers. The standard error falls as 1/sqrt(N_centers)
        over this range, so this is a straight precision-for-runtime trade:
        16000 candidates give 7745 centers in ~310 s, against 1992 centers in
        ~87 s at the former default of 4000, for error bars smaller by ~1.97x.
        The ceiling is not this number but the carved population inside the
        core margin (~7e6 halos on the full catalog), and beyond it the limit
        is cosmic variance of the single sub-volume, which no number of
        centers inside it reduces.
    seed : int
        Seed for the candidate-center subsample.
    shuffle_seed : int
        Seed for the velocity-shuffle null, kept distinct from `seed` so the
        null's randomness never silently reuses the center-selection stream.
    gaussian_null_seed : int
        Seed for the matched-Gaussian null (`matched_gaussian_sample`,
        built into `normalize_result` alongside the shuffle null). Distinct
        from every other seed in the ladder -- `seed` = 42 (center draw),
        `shuffle_seed` = 43, `ComparisonRunConfig.axis_null_seed` = 44,
        `RedshiftSpaceRunConfig.redshift_shuffle_seed` = 45, this = 46,
        `ComparisonRunConfig.velocity_gaussian_null_seed` = 47,
        `RedshiftSpaceRunConfig.redshift_gaussian_null_seed` = 48 -- so no
        two nulls anywhere in the three pipelines share a random stream.
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
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    shells: ShellConfig = field(default_factory=_default_shells)
    n_candidate_centers: int = 16000
    seed: int = 42
    shuffle_seed: int = 43
    gaussian_null_seed: int = 46
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
_COLOR_GAUSSIAN_NULL = "#7a4fa3"
_COLOR_ZERO_LINE = "0.5"
_COLOR_MONOPOLE = "0.3"
_LABEL_COLOR = "#111111"
_ZERO_LINE_WIDTH = 0.8
_BAND_ALPHA = 0.2
_GRID_ALPHA = 0.3
_FIGSIZE = (8.0, 8.0)
_HEIGHT_RATIOS = (3, 1)

def _binning_description(shells: ShellConfig) -> str:
    """One-line summary of the shell binning, for a figure title.

    `SPACING_LOG` -> "r in [r_min, r_max] h^-1 Mpc, n_bins log bins", with
    " + [0, r_min) bin" appended under `ShellConfig.include_zero_bin` -- the
    bracketed range then reads from 0, so without that suffix the title would
    claim n_bins log bins spanning [0, r_max], which is not what was measured.
    `SPACING_LINEAR` -> "r in [r_min, r_max] h^-1 Mpc, step h^-1 Mpc" -- so a
    saved PNG states its own binning rather than only `r_max` (the titles'
    previous behavior), and a reader never has to cross-reference the
    `RunConfig` that produced the figure. Used by all four figure builders
    that carry a radial x-axis (`make_figure`,
    `dvcorr.pipeline.velocity_frame_comparison.make_comparison_figure`,
    `dvcorr.pipeline.redshift_space_comparison.make_redshift_comparison_figure`
    and `.make_single_center_figure`) -- one home, so the phrasing cannot
    drift between them.

    Parameters
    ----------
    shells : ShellConfig

    Returns
    -------
    str
    """
    edges = shells.shell_edges
    # :.4g, not :.2g -- :.2g renders 150.0 as "1.5e+02" (scientific notation),
    # which docs/architecture.md's own worked example does not reproduce
    # (item 3 review finding). :.4g stays in plain decimal form across the
    # range this project can reach (1 to conventions.MAX_ANALYSIS_RADIUS =
    # 500 h^-1 Mpc) while still trimming trailing zeros for a value like 20.
    r_range = f"r in [{edges[0]:.4g}, {edges[-1]:.4g}] h$^{{-1}}$Mpc"
    if shells.spacing == SPACING_LOG:
        zero_bin = (
            f" + [0, {shells.min_radius:.4g}) bin" if shells.include_zero_bin else ""
        )
        return f"{r_range}, {shells.n_bins} log bins{zero_bin}"
    return f"{r_range}, step={shells.radii_step:.4g} h$^{{-1}}$Mpc"


# ---------------------------------------------------------------------------
# Stage 1: load + carve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HaloArrays:
    """Every halo a run selected from its catalog, before any spatial carve.

    Returned by `_load_all_halos`. Arrays are float32 (see below) and share a
    single row order; row i of each describes the same halo.

    Why float32 here and float64 after the carve. The full catalog is ~127M
    halos, where each float64 (N, 3) array costs ~3 GB and each float32 one
    ~1.5 GB. Only ~11% of it survives the carve, so holding the whole catalog
    at reduced width and widening only the survivors (`load_and_carve`) is
    what keeps a full-catalog run inside memory. The narrowing is lossless
    relative to the source text -- see `dvcorr.pipeline.catalog_conversion`'s
    docstring -- but the estimators' documented contract is float64, so the
    widening is not optional either.

    Attributes
    ----------
    pos : ndarray, shape (N_total, 3), float32
        Positions, comoving h^-1 Mpc, in catalog row order.
    vel : ndarray, shape (N_total, 3), float32
        Peculiar velocities, km/s, same row order.
    mvir : ndarray, shape (N_total,), float32
        Virial masses, h^-1 M_sun. Carried so the selection funnel can be
        reported by mass (`dvcorr.pipeline.mass_diagnostics`) rather than only
        by count.
    is_distinct : ndarray, shape (N_total,), bool
        True where the halo is not a subhalo (`pid == -1`).
    n_total : int
        `pos.shape[0]`, after the catalog's own cuts.
    """

    pos: np.ndarray
    vel: np.ndarray
    mvir: np.ndarray
    is_distinct: np.ndarray
    n_total: int


def _load_all_halos(paths: PathsConfig, catalog: CatalogConfig) -> HaloArrays:
    """Read a halo catalog and apply its mass / subhalo cuts, uncarved.

    Factored out of `load_and_carve` (which calls this, then carves) so that
    `dvcorr.pipeline.redshift_space_comparison` can build its OWN dual carve
    -- a plain `sub_volume_radius` pass to derive the `v_margin` statistic,
    then a second, buffered pass at `sub_volume_radius + v_margin` -- from a
    SINGLE catalog read rather than one read per carve.

    This is the ONLY place a `CatalogConfig` becomes halo arrays, so it is the
    only place the catalog choice has to be resolved. It reads Parquet, not
    CSV: `dvcorr.pipeline.catalog_conversion` writes that file and its
    docstring explains why (the full catalog is 11.2 GB of text, ~10 minutes
    and ~12 GB of RAM to parse, versus seconds from the converted file).

    `mass_min` is pushed down as a Parquet row-group filter rather than
    applied after loading. On the full catalog -- which is stored in ascending
    mvir order -- that means a high floor skips most of the file unread
    instead of reading and discarding it. `mass_max` and `include_subhalos`
    are applied in memory afterwards, since neither can skip contiguous row
    groups the way a floor on the sort column can.

    Parameters
    ----------
    paths : PathsConfig
        Resolves `catalog.name` to a file via `halo_catalog`.
    catalog : CatalogConfig
        Which catalog, and which halos to keep from it.

    Returns
    -------
    HaloArrays

    Raises
    ------
    FileNotFoundError
        If the Parquet file has not been built yet, with the command that
        builds it -- this is a setup step, not a bug, and the message says so
        rather than surfacing as a bare missing-file traceback.
    RuntimeError
        If zero halos survive. Distinguishes an empty catalog from cuts that
        removed everything, since the fix differs.
    """
    parquet_path = paths.halo_catalog(catalog.name)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} does not exist. Build it once with:\n"
            f"    python -m scripts.convert_mdpl2_catalog {catalog.name}"
        )

    mass_col = conventions.HALO_COLUMNS["mass"]
    columns = [
        *conventions.POSITION_COLUMNS,
        *conventions.VELOCITY_COLUMNS,
        mass_col,
        IS_DISTINCT_COLUMN,
    ]
    filters = None if catalog.mass_min is None else [(mass_col, ">=", catalog.mass_min)]

    print(f"loading {parquet_path.name} [{catalog.describe_cuts()}] ...")
    halos = pd.read_parquet(parquet_path, columns=columns, filters=filters)
    n_read = len(halos)
    if n_read == 0:
        raise RuntimeError(
            f"{parquet_path} yielded zero rows"
            + ("." if catalog.mass_min is None else f" at mass_min = {catalog.mass_min:.3g}.")
        )

    keep = np.ones(n_read, dtype=bool)
    if catalog.mass_max is not None:
        keep &= halos[mass_col].to_numpy() <= catalog.mass_max
    if not catalog.include_subhalos:
        keep &= halos[IS_DISTINCT_COLUMN].to_numpy()

    pos = halos[list(conventions.POSITION_COLUMNS)].to_numpy()[keep]
    vel = halos[list(conventions.VELOCITY_COLUMNS)].to_numpy()[keep]
    mvir = halos[mass_col].to_numpy()[keep]
    is_distinct = halos[IS_DISTINCT_COLUMN].to_numpy()[keep]

    n_total = pos.shape[0]
    if n_total == 0:
        raise RuntimeError(
            f"every one of the {n_read} halos read from {parquet_path.name} was "
            f"removed by the catalog cuts ({catalog.describe_cuts()}); widen them."
        )
    print(f"loaded {n_total} halos ({n_read - n_total} removed by in-memory cuts)")

    return HaloArrays(
        pos=pos, vel=vel, mvir=mvir, is_distinct=is_distinct, n_total=n_total
    )


@dataclass(frozen=True)
class CarvedHalos:
    """The halos inside the analysis sub-volume, with the masses that selected them.

    Returned by `load_and_carve`. This is a dataclass rather than the bare
    `(pos, vel)` tuple it replaces because `mvir` and `is_distinct` now travel
    with the population: the mass-funnel diagnostic
    (`dvcorr.pipeline.mass_diagnostics`) reports which halos a run actually
    measured, and reconstructing that downstream would mean re-deriving the
    selection outside the one place that performs it.

    `pos` / `vel` are float64 (the estimators' contract); `mvir` and
    `is_distinct` stay at their loaded width, since nothing computes geometry
    from them.

    Attributes
    ----------
    pos : ndarray, shape (N_carved, 3), float64
    vel : ndarray, shape (N_carved, 3), float64
        Carved positions and velocities, comoving h^-1 Mpc / km/s.
    mvir : ndarray, shape (N_carved,)
        Carved virial masses, h^-1 M_sun, same row order.
    is_distinct : ndarray, shape (N_carved,), bool
        False for subhalos.
    n_carved : int
        `pos.shape[0]`.
    n_total : int
        Halos the catalog supplied before the carve, for the survival fraction.
    catalog_mvir : ndarray, shape (N_total,)
        Masses of the FULL pre-carve population. Kept so the mass funnel can
        show the carve against the catalog it was cut from -- a spatial carve
        should be mass-blind, and this is what lets that be checked rather
        than assumed. It is the one uncarved array retained; positions and
        velocities are released with the rest of `HaloArrays`.
    """

    pos: np.ndarray
    vel: np.ndarray
    mvir: np.ndarray
    is_distinct: np.ndarray
    n_carved: int
    n_total: int
    catalog_mvir: np.ndarray


def load_and_carve(cfg: RunConfig, paths: PathsConfig) -> CarvedHalos:
    """Load the selected halo catalog and carve a sphere around the observer.

    Loads via `_load_all_halos`, then keeps halos within
    `cfg.sub_volume_radius` of `dvcorr.conventions.OBSERVER_POSITION`. Plain
    Euclidean distance IS the minimum image here: the sub-volume sits well
    inside the box, clear of the periodic faces (see geometry.py's PBC
    contract and notebooks/04_first_mdpl2_run.ipynb cell 7).

    Parameters
    ----------
    cfg : RunConfig
        Supplies `sub_volume_radius` and, via `cfg.catalog`, which halos to
        read in the first place.
    paths : PathsConfig
        Resolves the catalog name to a file.

    Returns
    -------
    CarvedHalos

    Raises
    ------
    RuntimeError
        If the catalog loads zero rows, or zero halos survive the carve.
        Both are checked HERE, before any percentage is printed from them --
        a bare `n / 0` inside an f-string is a ZeroDivisionError with a
        misleading traceback, not a diagnosis.
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    halos = _load_all_halos(paths, cfg.catalog)

    d_obs = np.linalg.norm(halos.pos - observer, axis=1)
    in_sub = d_obs <= cfg.sub_volume_radius
    # float64 from here on: the carved subset is small enough to widen, and the
    # geometry primitives and estimators document a float64 contract. See
    # `HaloArrays` for why the uncarved catalog is not held this way.
    pos = halos.pos[in_sub].astype(float)
    vel = halos.vel[in_sub].astype(float)
    n_carved = pos.shape[0]
    if n_carved == 0:
        raise RuntimeError(
            f"no halos survived the sub_volume_radius = {cfg.sub_volume_radius} "
            "carve; check the observer position and sub_volume_radius."
        )
    print(
        f"carved {n_carved} of {halos.n_total} halos within "
        f"sub_volume_radius = {cfg.sub_volume_radius} h^-1 Mpc "
        f"({100.0 * n_carved / halos.n_total:.2f}%)"
    )
    return CarvedHalos(
        pos=pos,
        vel=vel,
        mvir=halos.mvir[in_sub],
        is_distinct=halos.is_distinct[in_sub],
        n_carved=n_carved,
        n_total=halos.n_total,
        catalog_mvir=halos.mvir,
    )


# ---------------------------------------------------------------------------
# Stage 2: candidate centers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateCenters:
    """The seeded subsample of carved halos offered as velocity centers.

    Returned by `draw_candidates`. Like `CarvedHalos`, a dataclass rather than
    a tuple so the drawn halos' masses travel with their positions.

    Attributes
    ----------
    s : ndarray, shape (N_candidates, 3), float64
    v : ndarray, shape (N_candidates, 3), float64
        Candidate positions and velocities, comoving h^-1 Mpc / km/s.
    mvir : ndarray, shape (N_candidates,)
        Candidate virial masses, h^-1 M_sun.
    is_distinct : ndarray, shape (N_candidates,), bool
        False for subhalos.
    draw_index : ndarray, shape (N_candidates,), int
        Rows of the `CarvedHalos` arrays these were drawn from. Retained so a
        caller can trace a candidate back to the carved population without
        re-running the seeded draw.
    """

    s: np.ndarray
    v: np.ndarray
    mvir: np.ndarray
    is_distinct: np.ndarray
    draw_index: np.ndarray


def draw_candidates_from_arrays(
    cfg: RunConfig,
    pos: np.ndarray,
    vel: np.ndarray,
    mvir: np.ndarray,
    is_distinct: np.ndarray,
) -> CandidateCenters:
    """Seeded subsample of a carved population, from bare arrays.

    The definition site of the draw. `draw_candidates` is the `CarvedHalos`
    convenience wrapper over it; `dvcorr.pipeline.redshift_space_comparison`
    calls this form directly, because its candidates come from a
    `BufferedCarve`'s core arrays rather than from a `CarvedHalos`. One
    implementation either way, so the two pipelines cannot drift into drawing
    differently.

    The draw is uniform over the carved population and therefore MASS-BLIND:
    on a catalog with no mass floor, the candidates inherit the catalog's mass
    distribution, which is dominated by its smallest halos. That is a property
    of the catalog, not of this function -- the mass funnel
    (`dvcorr.pipeline.mass_diagnostics`) is what makes it visible.

    THE SEED SELECTS HALOS, NOT FILE ROWS. `rng.choice` returns a fixed set of
    integers for a given seed; if those integers indexed the arrays as handed
    in, they would select whichever halos happened to occupy those ROWS, and
    the answer would depend on the order the catalog file stores them in. The
    two catalogs hold the same halos in different tie orders, so that made a
    run on one and a run on the equivalent cut of the other draw different
    samples of the same population -- measured: only 291 of 4000 candidates in
    common, and multipoles differing by ~1 sigma per shell.

    Sorting the population into a canonical, file-order-independent order
    before indexing removes that. Identical halos now give identical centers
    whichever file they were read from, so a comparison between the two
    catalogs shows the effect of the catalogs and nothing else. The order is
    lexicographic on position, computed on coordinates folded into
    [0, BOX_SIZE) -- folding is what makes the four halos the two files store
    at opposite box faces (`BOX_SIZE` in one, `0.0` in the other, the same
    object under PBC) sort to the same place. That folding is a canonicalization
    of IDENTITY, not a geometric minimum-image reduction; it never touches the
    coordinates handed to the estimators, and so does not intrude on the PBC
    contract in `docs/architecture.md`.

    Parameters
    ----------
    cfg : RunConfig
        Supplies `n_candidate_centers` and `seed`.
    pos, vel : ndarray, shape (N_carved, 3)
        Carved positions and velocities.
    mvir : ndarray, shape (N_carved,)
    is_distinct : ndarray, shape (N_carved,), bool
        Per-halo labels, row-aligned with `pos` / `vel`.

    Returns
    -------
    CandidateCenters
        `draw_index` holds rows of the INPUT arrays, not of the canonical
        ordering, so it remains directly usable against the caller's own
        arrays.

    Raises
    ------
    RuntimeError
        If `cfg.n_candidate_centers <= 0` or the carved population is empty,
        either of which would otherwise surface later as a confusing
        zero-candidate estimator call.
    """
    n_carved = pos.shape[0]
    n_candidates = min(cfg.n_candidate_centers, n_carved)
    if n_candidates <= 0:
        raise RuntimeError(
            f"n_candidate_centers ({cfg.n_candidate_centers}) and the carved "
            f"population ({n_carved}) give zero candidates."
        )
    folded = np.mod(pos, conventions.BOX_SIZE)
    canonical = np.lexsort((folded[:, 2], folded[:, 1], folded[:, 0]))
    rng = np.random.default_rng(cfg.seed)
    draw_index = canonical[rng.choice(n_carved, size=n_candidates, replace=False)]
    return CandidateCenters(
        s=pos[draw_index],
        v=vel[draw_index],
        mvir=np.asarray(mvir)[draw_index],
        is_distinct=np.asarray(is_distinct)[draw_index],
        draw_index=draw_index,
    )


def draw_candidates(cfg: RunConfig, carved: CarvedHalos) -> CandidateCenters:
    """Seeded subsample of the carved halos, as candidate velocity centers.

    `velocity_centered_shell_dipole`'s own core cut (`core_center_mask`)
    decides which of these candidates survive as centers; this stage only
    performs the subsample. Thin wrapper over
    `draw_candidates_from_arrays`, which documents the draw itself.

    Parameters
    ----------
    cfg : RunConfig
    carved : CarvedHalos
        The carved population, as returned by `load_and_carve`.

    Returns
    -------
    CandidateCenters
    """
    return draw_candidates_from_arrays(
        cfg, carved.pos, carved.vel, carved.mvir, carved.is_distinct
    )


# ---------------------------------------------------------------------------
# Shared center selection -- used by the comparison pipelines built on top of
# this one (dvcorr.pipeline.velocity_frame_comparison,
# dvcorr.pipeline.redshift_space_comparison), not by this module's own
# single-frame `run_estimator` path.
# ---------------------------------------------------------------------------
#
# MOVED HERE from `dvcorr.pipeline.velocity_frame_comparison` (where it was
# originally written for the observer-frame/velocity-frame comparison) rather
# than generalized in place, because the redshift-space comparison
# (`dvcorr.pipeline.redshift_space_comparison`) needs it too and this module
# is already the shared base BOTH comparison pipelines import `RunConfig`,
# `load_and_carve`, `draw_candidates`, and `box_number_density` from --
# leaving it in `velocity_frame_comparison.py` would have made the
# redshift-space pipeline depend on the velocity-frame one, which it has
# nothing else to do with. `velocity_frame_comparison.py` now imports both
# names from here unchanged; existing code that imports them FROM
# `velocity_frame_comparison` still works, since that module re-exports them.
#
# The only substantive change from the original: `core_margin` is now an
# explicit argument (previously hardcoded to `cfg.shells.max_radius` inside
# the function body), and the `cfg` type hint is loosened from
# `ComparisonRunConfig` to base `RunConfig` -- the function only ever touched
# `cfg.sub_volume_radius` and `cfg.min_center_speed`, both base-`RunConfig`
# fields, so the narrower comparison-subclass type was never required.


@dataclass(frozen=True)
class SharedCenterSet:
    """The single center set multiple pipelines are run on.

    Built by `select_shared_centers` in two cuts, funnel-style:
    `n_candidates` (input) -> core cut (`core_center_mask`) -> `n_core` ->
    speed floor (`cfg.min_center_speed`) -> `n_centers` (final). Selecting
    ONCE and handing the identical `s_centers` / `v_centers` arrays to every
    estimator run on them (see `dvcorr.pipeline.velocity_frame_comparison
    .run_both_frames` and `dvcorr.pipeline.redshift_space_comparison`) is
    what guarantees the only difference between runs built on the same
    `SharedCenterSet` is whatever each run deliberately varies -- not a
    difference in which centers were measured.

    Attributes
    ----------
    s_centers : ndarray, shape (N_c, 3)
        Surviving center positions, comoving h^-1 Mpc.
    v_centers : ndarray, shape (N_c, 3)
        Surviving center velocities, km/s, same row order as `s_centers`.
    n_candidates : int
        Number of candidates handed in, before either cut.
    n_core : int
        Number surviving `core_center_mask` alone (before the speed floor).
    n_dropped_slow : int
        `n_core - n_centers`: the number of otherwise-valid centers dropped
        by the speed floor alone. RETURNED as a field, not merely printed,
        so a caller (or a test) can assert on it directly.
    n_centers : int
        Final surviving count, `s_centers.shape[0]`.
    mvir_centers : ndarray, shape (N_c,) | None
        Surviving centers' virial masses, h^-1 M_sun, same row order as
        `s_centers` -- masked by the identical two cuts, so this is the mass
        distribution of the halos the estimator actually used. None when the
        caller supplied no masses (see `select_shared_centers`), which is what
        keeps the synthetic-array tests free of a catalog.
    is_distinct_centers : ndarray, shape (N_c,), bool | None
        Whether each surviving center is a distinct halo. None on the same
        condition as `mvir_centers`.
    """

    s_centers: np.ndarray
    v_centers: np.ndarray
    n_candidates: int
    n_core: int
    n_dropped_slow: int
    n_centers: int
    mvir_centers: np.ndarray | None = None
    is_distinct_centers: np.ndarray | None = None


def select_shared_centers(
    cfg: RunConfig,
    s_candidates: np.ndarray,
    v_candidates: np.ndarray,
    observer: np.ndarray,
    core_margin: float | None = None,
    *,
    mvir_candidates: np.ndarray | None = None,
    is_distinct_candidates: np.ndarray | None = None,
) -> SharedCenterSet:
    """Apply the core cut, then the speed floor, ONCE, for other stages to share.

    Two sequential cuts:

    1. `core_center_mask(s_candidates, cfg.sub_volume_radius, core_margin,
       observer=observer)` -- the IDENTICAL function every estimator applies
       internally (see e.g. `dvcorr.pipeline.velocity_frame_comparison
       .run_both_frames`'s docstring on why re-applying it there is
       idempotent). `core_margin` defaults to `cfg.shells.max_radius`,
       matching `velocity_centered_shell_dipole`'s own default
       (`core_margin=None` -> `shell_edges[-1]` = r_max), so a surviving
       center's full shell fits inside the carved sub-volume; a caller with a
       different margin requirement (e.g. `dvcorr.pipeline
       .redshift_space_comparison`, whose centers additionally need `r_max +
       v_margin` of clearance so their OWN redshift-space displacement stays
       inside the sub-volume) passes it explicitly rather than this function
       guessing.
    2. `speeds = |v_core|; keep = (speeds > 0.0) & (speeds >= cfg.min_center_speed)`
       -- the floor that gives every surviving center a well-defined flow
       direction (see
       `dvcorr.estimators.velocity_frame_dipole.velocity_frame_shell_dipole`'s
       zero-speed `ValueError`, which this floor exists to prevent a caller
       from ever triggering). The explicit `speeds > 0.0` conjunct matters
       specifically for `cfg.min_center_speed == 0.0`: `speeds >=
       cfg.min_center_speed` is trivially true for every speed (a norm can
       never be negative), so without it the documented "0.0 means drop only
       exactly-zero-speed centers" contract (`RunConfig.min_center_speed`'s
       docstring) would be false -- a floor of 0.0 would drop nothing at
       all, including the exact-zero centers the floor exists to guard
       against. For any `min_center_speed > 0.0` the extra conjunct is a
       no-op (a positive floor already implies `speeds > 0.0`).

    Parameters
    ----------
    cfg : RunConfig
        Only `cfg.sub_volume_radius`, `cfg.shells.max_radius` (the
        `core_margin` default), and `cfg.min_center_speed` are used -- base
        `RunConfig` fields, so any subclass (`ComparisonRunConfig`,
        `dvcorr.pipeline.redshift_space_comparison.RedshiftSpaceRunConfig`)
        works unchanged.
    s_candidates, v_candidates : ndarray, shape (N_cand, 3)
        Candidate positions/velocities, e.g. `CandidateCenters.s` / `.v` from
        `draw_candidates`.
    observer : ndarray, shape (3,)
    core_margin : float, optional
        Minimum clearance to the sub-volume boundary required of a
        surviving center (see `dvcorr.estimators.shell_dipole
        .core_center_mask`). `None` (the default) uses
        `cfg.shells.max_radius`, i.e. r_max.
    mvir_candidates : ndarray, shape (N_cand,), optional, keyword-only
    is_distinct_candidates : ndarray, shape (N_cand,), bool, optional, keyword-only
        Per-candidate labels to carry through both cuts, surfacing on the
        result as `mvir_centers` / `is_distinct_centers`. Optional because
        the cuts are purely geometric and kinematic: nothing here reads a
        mass, so a caller with no catalog (every synthetic-array test) passes
        neither and gets None back. Supplied by the real pipelines from
        `CandidateCenters`, which is what feeds
        `dvcorr.pipeline.mass_diagnostics`.

    Returns
    -------
    SharedCenterSet

    Raises
    ------
    RuntimeError
        If `s_candidates` is empty, or if zero centers survive both cuts.
        The empty-input case is checked BEFORE the core-cut survival
        percentage is printed, so a zero-candidate call raises this
        `RuntimeError` rather than a bare `n / 0` `ZeroDivisionError` with a
        misleading traceback (the same hazard `load_and_carve`'s docstring
        calls out and guards against).
    """
    n_candidates = s_candidates.shape[0]
    if n_candidates == 0:
        raise RuntimeError("select_shared_centers: s_candidates is empty.")

    if core_margin is None:
        core_margin = cfg.shells.max_radius

    core_mask = core_center_mask(
        s_candidates, cfg.sub_volume_radius, core_margin, observer=observer
    )
    s_core = s_candidates[core_mask]
    v_core = v_candidates[core_mask]
    n_core = s_core.shape[0]

    speeds = np.linalg.norm(v_core, axis=1)
    keep = (speeds > 0.0) & (speeds >= cfg.min_center_speed)
    s_centers = s_core[keep]
    v_centers = v_core[keep]
    n_centers = s_centers.shape[0]
    n_dropped_slow = n_core - n_centers

    # The optional per-candidate labels ride the IDENTICAL two masks, in the
    # same order, so `mvir_centers[i]` describes `s_centers[i]` by
    # construction rather than by a second, independently derived selection
    # that could drift from this one.
    def _survivors(labels: np.ndarray | None) -> np.ndarray | None:
        return None if labels is None else np.asarray(labels)[core_mask][keep]

    mvir_centers = _survivors(mvir_candidates)
    is_distinct_centers = _survivors(is_distinct_candidates)

    print(
        f"n_candidates = {n_candidates} -> core cut (core_margin = {core_margin}) "
        f"-> n_core = {n_core} ({100.0 * n_core / n_candidates:.1f}% survive) -> "
        f"speed floor (min_center_speed = {cfg.min_center_speed} km/s) -> "
        f"n_centers = {n_centers} ({n_dropped_slow} dropped as too slow)"
    )
    if n_centers == 0:
        raise RuntimeError(
            "select_shared_centers: zero centers survived the core cut and "
            "speed floor combined; widen sub_volume_radius, shrink the "
            "shell range, or lower min_center_speed."
        )

    return SharedCenterSet(
        s_centers=s_centers,
        v_centers=v_centers,
        n_candidates=n_candidates,
        n_core=n_core,
        n_dropped_slow=n_dropped_slow,
        n_centers=n_centers,
        mvir_centers=mvir_centers,
        is_distinct_centers=is_distinct_centers,
    )


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


def box_number_density(n_halos: int) -> float:
    """Mean halo number density n_bar over the WHOLE simulation box.

    n_bar = n_halos / BOX_SIZE**3 -- the Nusser (2017) eq. 24 normalization's
    n_bar, taken as the cosmic mean of the tracer population rather than as
    its density inside the analysis sub-volume (never estimated inside the
    estimator; see
    `dvcorr.estimators.shell_dipole.expected_shell_occupancy`).

    Why the box, not the sub-volume
    -----------------------------------
    This replaces the earlier `global_number_density(n_tracers,
    sub_volume_radius)`, which divided the CARVED count by the sub-volume's
    own volume. That made n_bar a property of one particular carve: the
    sphere of radius R_sub around the observer is a single realization of the
    density field, so its density is n_bar_cosmic * (1 + delta_sub) with
    delta_sub an unknown of order the rms fluctuation on the scale R_sub, and
    every normalized zeta_hat_1 inherited that unknown factor. Dividing by
    the box mean instead makes the denominator the true cosmic mean of the
    simulation -- exact, since the box IS the universe here -- so
    zeta_hat_1's amplitude no longer depends on where the observer sits or
    how large a sphere was carved around them.

    Two consequences worth knowing before comparing to older figures:

      - The change is a single multiplicative rescaling by
        (1 + delta_sub), FLAT in r: every shell moves by the same factor and
        no shape changes. On the default configuration (`full` catalog,
        R_sub = 300, observer at the box center) it is tiny -- the carve
        holds 14418601 halos, n_bar_sub = 0.12749, against
        n_bar_box = 0.12739, a 0.08% shift. It is small because a 300 h^-1
        Mpc sphere is already close to a fair sample of this box; it is NOT
        small by construction, and shrinking R_sub or moving the observer
        would grow it.
      - n_bar no longer depends on which population survived the CARVE, only
        on the population the catalog supplied (`CarvedHalos.n_total`,
        `BufferedCarve.n_total`). That deletes a whole class of denominator
        bug -- the buffered-carve inflation
        `dvcorr.pipeline.redshift_space_comparison` documents, where using
        the buffered count with the plain radius silently suppressed
        zeta_hat by ~32%: with a box-wide n_bar there is no radius to
        mismatch a count against.

    The count must still be the population that actually enters the shells:
    pass the catalog total AFTER `cfg.catalog`'s mass / subhalo cuts and
    BEFORE the spatial carve. Mixing populations here -- e.g. the raw file
    row count against mass-cut tracers -- rescales every curve by the ratio
    of the two abundances.

    Parameters
    ----------
    n_halos : int
        Number of halos of the tracer population in the whole box, i.e. the
        post-cut / pre-carve catalog count (`CarvedHalos.n_total`,
        `BufferedCarve.n_total`).

    Returns
    -------
    float
        n_bar, halos per (h^-1 Mpc)^3.
    """
    box_volume = conventions.BOX_SIZE**3
    return n_halos / box_volume


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
    zeta_hat_gaussian : ndarray, shape (B,)
        The matched-Gaussian null, same normalization: the same statistic with
        the per-center velocities REPLACED by a Gaussian draw matched to the
        sample's own mean and variance (`matched_gaussian_sample`). Read
        against `zeta_hat_shuffle`, which matches those first two moments
        exactly by construction -- the two curves therefore differ only in the
        higher moments of the velocity distribution, so a gap between them is
        the non-Gaussianity of that distribution and nothing else.

        Which construction sits here depends on the frame, exactly as for
        `zeta_hat_shuffle`: the observer frame draws a matched (N_c,) u
        (`normalize_result`, no estimator pass -- recombined against the
        fixed-axis amplitude), the velocity frame draws a matched (N_c, 3) v
        and RE-RUNS the estimator
        (`dvcorr.pipeline.velocity_frame_comparison.run_gaussian_velocity_null`),
        because there its axis and its weight both come from the drawn vector.
        NaN throughout if the caller passed no Gaussian null to
        `normalize_stacked_dipole` (see that function's Parameters).
    sem_gaussian : ndarray, shape (B,)
        Standard error of `zeta_hat_gaussian`.
    monopole_norm : ndarray, shape (B,)
        Normalised ell=0 companion (CLAUDE.md hard rule 6), km/s. Since the
        per-center weight is the SPEED |u_alpha|
        (`dvcorr.conventions.VELOCITY_AXIS_CONVENTION`), this sits near the
        mean radial speed <|u|>, NOT near zero, and on a clustered field it
        also inherits the 1 + xi_hh(r) occupancy shape. Read it as "what
        survives once that occupancy ratio is divided out" -- a residual
        TREND in r is the finite-distance / incomplete-shell diagnostic, not
        the absolute level.
    """

    zeta_hat: np.ndarray
    sem: np.ndarray
    zeta_hat_shuffle: np.ndarray
    sem_shuffle: np.ndarray
    zeta_hat_gaussian: np.ndarray
    sem_gaussian: np.ndarray
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
        Mean halo number density over the whole box, e.g. from
        `box_number_density`.

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


def matched_gaussian_sample(sample: np.ndarray, seed: int) -> np.ndarray:
    """Draw a Gaussian sample matched, column-wise, to `sample`'s mean and variance.

    ONE home for the matched-Gaussian null's distributional convention, shared
    by both frames: the observer frame matches the (N_c,) signed radial
    velocities u_alpha (`normalize_result`), the velocity frame matches the
    (N_c, 3) velocity vectors v_alpha
    (`dvcorr.pipeline.velocity_frame_comparison.run_gaussian_velocity_null`).
    Keeping the convention here means the two frames cannot drift apart on
    what "matched to the halos" means.

        mu    = sample.mean(axis=0)
        sigma = sample.std(axis=0, ddof=1)
        draw  ~ N(mu, sigma^2), independently per column

    WHY THE MEAN IS MATCHED AND NOT SET TO ZERO
    --------------------------------------------
    A zero-mean draw would be a "no signal at all" floor, but it would also
    delete the center sample's residual BULK MOTION -- and that bulk motion is
    exactly what the shuffle null retains (a permutation conserves Sigma_alpha
    u_alpha exactly, so E_pi[D_b^null] = N_c * <u> * <A_bar>_b, the
    finite-distance / incomplete-shell leakage floor that
    `dvcorr.estimators.shell_dipole.core_center_mask` documents as the ~+13
    km/s offset). Matching the mean here means the Gaussian and the shuffle
    null agree on their first TWO moments and differ ONLY in the higher ones,
    so the gap between the two curves isolates a single thing: the
    NON-GAUSSIANITY of the velocity distribution. A zero-mean draw would
    confound that with the bulk flow and the comparison would measure two
    effects at once.

    WHY ddof=1 AND WHY COLUMN-WISE
    -------------------------------
    `ddof=1` is the unbiased estimate of the population sigma being modeled,
    and matches `center_standard_error`'s convention so the two never disagree
    about what "the sample's spread" means. Column-wise (rather than a single
    pooled sigma) so the (N_c, 3) case reproduces the sample's own per-axis
    dispersion and mean -- i.e. its bulk-flow VECTOR -- instead of imposing an
    isotropy the halo sample does not have. The resulting 3-D null is
    therefore mildly anisotropic BY CONSTRUCTION, which is the point: it is
    what makes it a different question from
    `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null`'s
    strictly isotropic axis.

    Parameters
    ----------
    sample : ndarray, shape (N,) or (N, D)
        The empirical sample to match.
    seed : int
        Seed for `np.random.default_rng`. Callers pass a seed distinct from
        every other null's -- see `RunConfig.gaussian_null_seed` for the
        ladder.

    Returns
    -------
    ndarray
        Same shape as `sample`: (N,) in, (N,) out; (N, D) in, (N, D) out.
        All-NaN when N < 2, since a ddof=1 spread is undefined there. That
        mirrors `dvcorr.estimators.shell_dipole.center_standard_error`'s own
        N_c < 2 -> NaN convention rather than raising: a single-center run is
        a supported (if uninformative) configuration everywhere else in this
        pipeline, and it should degrade to a missing null curve, not to an
        exception thrown from inside `normalize_result`.

    Raises
    ------
    ValueError
        If `sample` is neither 1-D nor 2-D -- a shape misuse, not a
        degenerate sample.
    """
    values = np.asarray(sample, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError(
            f"matched_gaussian_sample: sample must be 1-D (N,) or 2-D (N, D), "
            f"got ndim={values.ndim} (shape {values.shape})."
        )
    if values.shape[0] < 2:
        return np.full(values.shape, np.nan)

    rng = np.random.default_rng(seed)
    mu = values.mean(axis=0)
    sigma = values.std(axis=0, ddof=1)
    return rng.normal(loc=mu, scale=sigma, size=values.shape)


def normalize_stacked_dipole(
    shell_edges: np.ndarray,
    dipole: np.ndarray,
    monopole: np.ndarray,
    per_center_dipole: np.ndarray,
    null_dipole: np.ndarray,
    null_per_center_dipole: np.ndarray,
    n_centers: int,
    n_bar: float,
    gaussian_null_dipole: np.ndarray | None = None,
    gaussian_null_per_center_dipole: np.ndarray | None = None,
) -> NormalizedDipole:
    """The normalization arithmetic, factored out to ONE home.

    This contains exactly the arithmetic that used to live inline inside
    `normalize_result` (see that function, which now delegates here). It is
    pulled out to a stand-alone function because
    `dvcorr.pipeline.velocity_frame_comparison` needs the identical
    normalization for the velocity-frame result, and that frame's NULLS are
    different constructions (a random-axis re-run, not a scalar permutation --
    see `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null` for
    why; and a re-run on drawn velocities rather than a recombination, see
    `.run_gaussian_velocity_null`) -- so the pieces that legitimately differ
    between the two call sites, BOTH nulls, are parameters here rather than
    built inside this function.

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
        Mean halo number density over the whole box, e.g. from
        `box_number_density`.
    gaussian_null_dipole : ndarray, shape (B,), optional
        Raw stacked dipole of the SECOND null, the matched-Gaussian one --
        again already built by the caller, and again differently per frame
        (`normalize_result` recombines a drawn u; `dvcorr.pipeline
        .velocity_frame_comparison.run_gaussian_velocity_null` re-runs the
        estimator on a drawn v). Both frames' production paths always supply
        it. It is optional ONLY so that a unit test exercising the
        normalization ARITHMETIC does not have to fabricate a second null it
        is not testing; omitting it fills `NormalizedDipole.zeta_hat_gaussian`
        and `.sem_gaussian` with NaN, which propagates to a visibly missing
        curve rather than to a wrong number.
    gaussian_null_per_center_dipole : ndarray, shape (N_c, B), optional
        Per-center breakdown of the same null, for its own standard error.
        Must be given together with `gaussian_null_dipole`.

    Returns
    -------
    NormalizedDipole

    Raises
    ------
    ValueError
        If exactly one of `gaussian_null_dipole` /
        `gaussian_null_per_center_dipole` is given: half a null would
        silently produce a curve with no error band, or a band with no curve.
    """
    if (gaussian_null_dipole is None) != (gaussian_null_per_center_dipole is None):
        raise ValueError(
            "normalize_stacked_dipole: gaussian_null_dipole and "
            "gaussian_null_per_center_dipole must be given together or not at "
            "all; got "
            f"gaussian_null_dipole={'None' if gaussian_null_dipole is None else 'array'}, "
            f"gaussian_null_per_center_dipole="
            f"{'None' if gaussian_null_per_center_dipole is None else 'array'}."
        )

    nbar_v_b = expected_shell_occupancy(n_bar, shell_edges)
    norm_scale = shell_dipole_norm_scale(shell_edges, n_bar)

    zeta_hat = (dipole / n_centers) * norm_scale
    sem = center_standard_error(per_center_dipole) * norm_scale

    zeta_hat_shuffle = (null_dipole / n_centers) * norm_scale
    sem_shuffle = center_standard_error(null_per_center_dipole) * norm_scale

    if gaussian_null_dipole is None:
        zeta_hat_gaussian = np.full_like(zeta_hat, np.nan)
        sem_gaussian = np.full_like(zeta_hat, np.nan)
    else:
        zeta_hat_gaussian = (gaussian_null_dipole / n_centers) * norm_scale
        sem_gaussian = center_standard_error(gaussian_null_per_center_dipole) * norm_scale

    # ell=0 companion, same normalization convention minus the 3/Y10 factors
    # (hard rule 6: never plot the dipole without it).
    monopole_norm = (monopole / n_centers) / nbar_v_b

    return NormalizedDipole(
        zeta_hat=zeta_hat,
        sem=sem,
        zeta_hat_shuffle=zeta_hat_shuffle,
        sem_shuffle=sem_shuffle,
        zeta_hat_gaussian=zeta_hat_gaussian,
        sem_gaussian=sem_gaussian,
        monopole_norm=monopole_norm,
    )


def normalize_result(
    result: VelocityCenteredShellDipoleResult,
    n_bar: float,
    shuffle_seed: int,
    gaussian_null_seed: int,
) -> NormalizedDipole:
    """Turn a raw estimator result + n_bar into the plotted, normalized curves.

    Builds BOTH of the observer frame's nulls -- the velocity shuffle and the
    matched-Gaussian draw -- and delegates the actual normalization arithmetic
    to `normalize_stacked_dipole`, which is the single home for it (see that
    function's docstring for the formula).

    The null, and why it undoes the axis flip first
    ---------------------------------------------------
    The estimator's axis is z_hat_alpha = sign(u_alpha) * n_hat_V,alpha
    (`dvcorr.conventions.VELOCITY_AXIS_CONVENTION`), so `per_center_amplitude`
    already carries that sign and the per-center weight is the positive-
    definite |u_alpha|. Permuting the weight against the untouched amplitudes
    would NOT be a null: a permutation of positive numbers leaves the
    alignment that produced the signal exactly where it was -- the same trap
    `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null`
    documents for the velocity frame, which is self-aligned for the same
    reason.

    So the flip is undone first, recovering the fixed-axis amplitude

        A_r,alpha,b = sign(u_alpha) * A_alpha,b      (axis pinned to +n_hat_V)

    and the SIGNED u_alpha is permuted against that. Because the axis is
    derived from the scalar's sign, permuting the signed scalar
    re-randomizes the axis and the weight TOGETHER, which is exactly what a
    null for this construction has to do; u_alpha averages to ~0 over an
    isotropic sample of lines of sight, so the recombined stack collapses.
    Still no second estimator pass.

    (Numerically this reproduces the pre-signed-axis null exactly --
    sign(u) * A = A(n_hat_V) -- so the plotted null curve is unchanged. The
    construction is spelled out because the OBVIOUS one-line version,
    permuting `per_center_speed`, is now silently wrong.)

    The second null: a matched Gaussian draw instead of a permutation
    ------------------------------------------------------------------
    Both nulls recombine a scalar against the SAME fixed-axis amplitude
    A_bar_alpha,b; they differ only in where the scalar comes from:

        shuffle  :  D_b = Sum_alpha u_{pi(alpha)}  * A_bar_alpha,b
        gaussian :  D_b = Sum_alpha u_gauss,alpha  * A_bar_alpha,b

    with u_gauss ~ N(<u>, sigma_u^2) from `matched_gaussian_sample` -- the
    sample's own mean and (ddof=1) variance, not a model fitted elsewhere.
    Because a permutation preserves the multiset {u_alpha} exactly, the
    shuffle null already has mean <u> and variance sigma_u^2, so the two nulls
    agree on the first two moments BY CONSTRUCTION and differ only in the
    higher ones. That is the whole point of carrying both: the gap between
    them is the non-Gaussianity of the halo radial-velocity distribution --
    the heavy tail contributed by satellites in clusters, and whatever
    `min_center_speed` and the core cut do to the shape -- and nothing else.
    The shuffle stays the primary null (it assumes no distribution at all, and
    gives an exact finite-sample test of the u <-> local-density pairing); the
    Gaussian is the parametric comparison, and is also the natural place to
    hang an error model later.

    Neither null costs an estimator pass: both are recombinations of arrays
    the result already carries. (The velocity frame gets no such discount for
    either of its nulls -- see
    `dvcorr.pipeline.velocity_frame_comparison.run_random_axis_null` and
    `.run_gaussian_velocity_null`.)

    Parameters
    ----------
    result : VelocityCenteredShellDipoleResult
    n_bar : float
        Mean halo number density over the whole box, e.g. from
        `box_number_density`.
    shuffle_seed : int
        Seed for the velocity-shuffle null (see above).
    gaussian_null_seed : int
        Seed for the matched-Gaussian null, distinct from `shuffle_seed` so
        the two nulls are independent draws rather than two views of one
        random stream. See `RunConfig.gaussian_null_seed` for the full ladder.

    Returns
    -------
    NormalizedDipole
    """
    shuffle_rng = np.random.default_rng(shuffle_seed)
    perm = shuffle_rng.permutation(result.per_center_u.size)

    # Undo the z_hat = sign(u) n_hat_V flip to recover the fixed-axis
    # amplitude, then permute the SIGNED u against it -- see the docstring.
    fixed_axis_amplitude = np.sign(result.per_center_u)[:, None] * result.per_center_amplitude
    shuffled_per_center_dipole = result.per_center_u[perm][:, None] * fixed_axis_amplitude
    null_dipole = shuffled_per_center_dipole.sum(axis=0)

    # Second null: the same recombination, against a matched Gaussian draw of
    # the signed u rather than a permutation of it.
    u_gaussian = matched_gaussian_sample(result.per_center_u, gaussian_null_seed)
    gaussian_per_center_dipole = u_gaussian[:, None] * fixed_axis_amplitude
    gaussian_null_dipole = gaussian_per_center_dipole.sum(axis=0)

    return normalize_stacked_dipole(
        shell_edges=result.shell_edges,
        dipole=result.dipole,
        monopole=result.monopole,
        per_center_dipole=result.per_center_dipole,
        null_dipole=null_dipole,
        null_per_center_dipole=shuffled_per_center_dipole,
        n_centers=result.n_centers,
        n_bar=n_bar,
        gaussian_null_dipole=gaussian_null_dipole,
        gaussian_null_per_center_dipole=gaussian_per_center_dipole,
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
    BOTH nulls, each with its own band, on top; the normalized monopole
    companion below. The two nulls (`normalize_result`: the u-shuffle, dashed,
    and the matched Gaussian, dotted) share their first two moments by
    construction, so the gap between the two dashed/dotted curves -- not
    either one's distance from zero -- is what carries information.

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

    # result.shell_centers (the plain midpoint) is deliberately left alone --
    # r_eff, the volume-weighted radius, is the plotting abscissa.
    r = volume_weighted_shell_radii(result.shell_edges)

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
    # Second null, matched to the same first two moments as the shuffle
    # (`normalize_result`), so the two dashed curves are read against EACH
    # OTHER: their separation is the non-Gaussianity of the u distribution.
    # All-NaN when the caller normalized without one -- matplotlib then draws
    # nothing, and the legend entry is the only trace, which is the intended
    # visible-absence behavior (`normalize_stacked_dipole`).
    ax_dipole.fill_between(
        r, normalized.zeta_hat_gaussian - normalized.sem_gaussian,
        normalized.zeta_hat_gaussian + normalized.sem_gaussian,
        color=_COLOR_GAUSSIAN_NULL, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, normalized.zeta_hat_gaussian, "^:", color=_COLOR_GAUSSIAN_NULL,
        label="gaussian null",
    )
    ax_dipole.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_dipole.set_ylabel(r"$\hat\zeta_1$  [km/s]", color=_LABEL_COLOR)
    ax_dipole.legend()
    ax_dipole.grid(alpha=_GRID_ALPHA)
    ax_dipole.spines["top"].set_visible(False)
    ax_dipole.spines["right"].set_visible(False)
    ax_dipole.set_title(
        f"velocity-centered dipole  "
        f"(R_sub={cfg.sub_volume_radius:.0f} h$^{{-1}}$Mpc, "
        f"{_binning_description(cfg.shells)}, N_c={result.n_centers})"
    )

    # No zero reference line here: with the |u_alpha| weight the monopole sits
    # near <|u|>, not near zero (NormalizedDipole.monopole_norm), so a y=0 line
    # would only compress the axis and imply a reference that no longer applies.
    ax_mono.plot(r, normalized.monopole_norm, "o-", color=_COLOR_MONOPOLE)
    ax_mono.set_ylabel(r"$\hat\zeta_0 \simeq \langle|u|\rangle$  [km/s]", color=_LABEL_COLOR)
    ax_mono.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
    ax_mono.grid(alpha=_GRID_ALPHA)
    ax_mono.spines["top"].set_visible(False)
    ax_mono.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
