"""
redshift_space_comparison.py
-------------------------------
Reusable pipeline stage functions for the real-space vs. redshift-space
comparison of the velocity-centered zeta_1 dipole
(`dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`), run on a
SHARED set of centers so the only difference between the two results is
which tracer positions -- real or redshift-space
(`dvcorr.redshift_space.to_redshift_space`) -- were used to build the
density field around them.

`velocity_centered_shell_dipole` ITSELF IS UNCHANGED. It already accepts
`s_centers`/`s_tracers` as plain arrays and has no notion of "real" or
"redshift" space; that agnosticism is exactly what makes this comparison
possible without touching the estimator, and this module is built to
preserve it, not work around it. Both runs below call the SAME function,
with different position arrays.

This is the single source of truth consumed by
`scripts/plot_redshift_space_comparison.py` and
`notebooks/07_redshift_space_comparison.ipynb` (CLAUDE.md's "one library, two
thin consumers" model), mirroring `dvcorr.pipeline.velocity_frame_comparison`'s
structure closely. It reuses `RunConfig` (subclassed below), `draw_candidates`,
`box_number_density`, `NormalizedDipole`, `normalize_result`,
`shell_dipole_norm_scale`, `SharedCenterSet`, and `select_shared_centers` from
`dvcorr.pipeline.velocity_centered` rather than duplicating any of them; it
does NOT reuse `load_and_carve` (see `load_and_carve_buffered` below for why
this pipeline needs a different, two-pass carve) but DOES reuse the
CSV-reading helper `_load_all_halos` that `load_and_carve` itself is built
on, so the two-pass carve costs one catalog read, not two.

Matplotlib backend discipline
------------------------------
Importing this module does NOT select a matplotlib backend: `matplotlib.use`
is never called here, only in the thin script's own module body (before it
imports this module) -- identical discipline to `dvcorr.pipeline
.velocity_centered` and `dvcorr.pipeline.velocity_frame_comparison`.

Volume construction -- two separate problems
-----------------------------------------------
Displacing every halo along its own line of sight (the redshift-space
transform, `dvcorr.redshift_space.to_redshift_space`) creates two DISTINCT
boundary problems, solved two different ways:

TRACERS. Tracers displace too, so the redshift-space tracer field near the
R_sub boundary is incomplete in BOTH directions at once: some real tracers
inside R_sub displace OUT past it (lost from the redshift-space density
field even though they are real members), and some real tracers just
OUTSIDE R_sub -- never loaded by a plain R_sub carve -- would have displaced
IN. This is exactly the mechanism `dvcorr.estimators.shell_dipole
.core_center_mask`'s docstring blames for the first MDPL2 run's ~+13 km/s
null offset, applied here to the TRACER field rather than the center
boundary. The fix: carve the tracer catalog at `R_sub + v_margin`
(`load_and_carve_buffered`, ~32% more rows for `v_margin` at its "max"
default -- negligible), displace every buffered tracer
(`dvcorr.redshift_space.to_redshift_space`), THEN restrict the result to
R_sub (`build_tracer_spaces`). Skipping the buffer and displacing only the
plain-R_sub carve would leave the boundary shell systematically incomplete
in the redshift-space run regardless of how generously centers are margined.

CENTERS. A uniform core margin `r_max + v_margin`, applied to REAL positions
-- never to the displaced position. `v_margin` is a configurable velocity
statistic (`RedshiftSpaceRunConfig.v_margin_statistic`: "max", the default,
or "percentile") defined on |v_r| -- the RADIAL velocity, never the 3-D
speed (`dvcorr.redshift_space.v_margin_from_statistic`'s docstring: a
percentile of |v| would under-cover the actual radial displacement, since
|v| >= |v_r| always). The statistic choice and the resulting value are
exposed on `RedshiftCenterSet` / logged, never recomputed inline at a second
call site.

    DO NOT select centers by whether their DISPLACED position clears the
    margin. That is velocity-conditioned selection: near the far boundary of
    R_sub it systematically drops outward-movers (whose displaced position
    falls outside the core radius) and keeps inward-movers, skewing the
    stacked <u> with position and coupling it to residual large-scale
    structure (Sigma Y_10) in a way that mimics a real dipole. This
    function never computes a displaced center position for the purpose of
    deciding whether to keep it -- see `select_redshift_shared_centers`'s
    docstring for the triangle-inequality argument for why this is not
    merely a stylistic avoidance but is actually UNNECESSARY: the
    `r_max + v_margin` real-position margin already GUARANTEES every
    surviving center's displaced position clears the plain `r_max` margin
    `velocity_centered_shell_dipole` re-applies internally, with no explicit
    displaced-position check anywhere. Do not "simplify" this by checking
    the displaced position directly -- that would reintroduce exactly the
    bias this construction avoids.

    Why the (separate) GLOBAL |v_r| cut, used only when `v_margin_statistic
    == "percentile"`, is legitimate where the displaced-position cut above
    is NOT: the global cut is SIGN-SYMMETRIC (it drops on |v_r|, so <u> among
    survivors is unbiased -- as many inward- as outward-movers are removed)
    and POSITION-INDEPENDENT (it does not depend on where in R_sub a center
    sits, so it cannot couple to Sigma Y_10 through position the way a
    boundary-localized cut does). The displaced-position cut fails BOTH
    tests: it depends on distance to the boundary (position-dependent) and it
    keeps inward- while dropping outward-movers near that boundary
    (sign-asymmetric). Do not conflate the two just because both involve a
    velocity-based cut on centers.

Cost of a like-for-like comparison. With the DEFAULT settings (R_sub = 300,
r_max = 64), the original single-frame run's core margin is `r_max = 64`,
giving a core radius of 236 and a candidate volume fraction (236/300)**3 ~=
49% (measured on the real MDPL2 run: n_candidates = 4000, n_centers = 1933,
48.3% survive; carve 464764 of 4093751 halos within R_sub = 300). With
`v_margin` at its "max" default (empirically ~30 h^-1 Mpc on this catalog),
the redshift-space-comparable margin is `r_max + v_margin ~= 94`, giving a
core radius of ~206 and a candidate volume fraction (206/300)**3 ~= 32% --
roughly a third fewer centers than the single-frame run. This DEGRADES
the real-space run relative to the currently published single-frame result
(`dvcorr.pipeline.velocity_centered`'s own run, margined at `r_max` alone).
That is the correct price of a like-for-like comparison, paid deliberately,
not a regression to "fix" by loosening the redshift-space margin back down.

n_bar normalization -- read this before touching the `n_bar` argument
------------------------------------------------------------------------------------------
`n_bar` is the mean halo density of the WHOLE BOX,
`dvcorr.pipeline.velocity_centered.box_number_density(carve.n_total)` --
the catalog's post-cut halo count over BOX_SIZE**3. It is supplied by the
CALLER (`normalize_redshift_comparison`'s `n_bar` parameter), exactly as
`dvcorr.pipeline.velocity_frame_comparison.normalize_comparison` takes it,
so the normalization stays a visible call-site choice rather than something
a stage function derives on its own.

That definition is deliberately blind to the carve, which is what makes it
safe HERE in particular. This pipeline's tracer array
(`build_tracer_spaces`'s `s_tracers_real`/`s_tracers_redshift`) is built
from a BUFFERED carve (`load_and_carve_buffered`) and then restricted back
to R_sub, so under the OLD sub-volume normalization (carved count / carve
volume) the count and the radius could disagree: feeding the buffered count
against the plain radius inflated n_bar by ~(R_sub + v_margin)**3 /
R_sub**3 (~32% at v_margin's "max" default) and suppressed the normalized
zeta_hat by the same ~32% in BOTH runs, flat in r, with no shape change to
reveal it. A box-wide n_bar has no radius to mismatch a count against, so
that failure mode is gone by construction -- the only thing that can still
be wrong is passing a count from a DIFFERENT tracer population (a different
catalog or mass cut) than the one in the shells.

DECISION (unchanged): both runs use the SAME n_bar. The tracer population is
identical between runs -- only its placement differs -- so a PER-RUN n_bar
would fold the small real-vs-redshift boundary-flux difference into the
amplitude ratio, which is the headline quantity this whole comparison exists
to measure. Under the box-wide definition this is now automatic (both runs
read the same catalog, so both would compute the same number anyway), but
the single shared argument keeps it explicit. Both realized counts
(`n_real_inside_r_sub`, `n_redshift_inside_r_sub`) are still logged and
returned on `RedshiftSpaceComparison` so the flux stays visible -- they are
a DIAGNOSTIC now, no longer an input to the normalization: with the buffer
capturing inflow and outflow both, the two counts are EXPECTED to agree to
well under a percent (see `build_tracer_spaces`'s docstring); a larger gap
means the buffer radius (`v_margin`) is misconfigured, not that RSD moved a
meaningfully large fraction of the tracer mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from dvcorr import conventions
from dvcorr.config import PathsConfig, volume_weighted_shell_radii
from dvcorr.estimators.shell_dipole import VelocityCenteredShellDipoleResult, velocity_centered_shell_dipole
from dvcorr.redshift_space import (
    FLIP_GUARD_FLOOR,
    radial_velocity,
    to_redshift_space,
    v_margin_from_statistic,
)

# Stage functions and the styling constants reused verbatim from the base
# pipeline -- see this module's docstring for the full list. _COLOR_SIGNAL is
# imported as `_COLOR_REAL` so the real-space curve in this comparison figure
# is the visually identical blue used for zeta_hat in the single-frame
# figure, mirroring `velocity_frame_comparison.py`'s `_COLOR_OBS` pattern.
from dvcorr.pipeline.velocity_centered import (
    _BAND_ALPHA,
    _COLOR_ZERO_LINE,
    _FIGSIZE,
    _GRID_ALPHA,
    _HEIGHT_RATIOS,
    _LABEL_COLOR,
    _ZERO_LINE_WIDTH,
    NormalizedDipole,
    RunConfig,
    SharedCenterSet,
    _binning_description,
    _load_all_halos,
    normalize_result,
    select_shared_centers,
    shell_dipole_norm_scale,
)
from dvcorr.pipeline.velocity_centered import _COLOR_SIGNAL as _COLOR_REAL

_COLOR_REAL_NULL = "#a9c9ee"       # lighter tint of _COLOR_REAL: the real-space run's u-shuffle null
_COLOR_REDSHIFT = "#1b9e77"        # distinct hue for the redshift-space run's curves
_COLOR_REDSHIFT_NULL = "#a8ddc9"   # lighter tint of _COLOR_REDSHIFT: the redshift run's u-shuffle null
# Each run's SECOND null, the matched-Gaussian one: a darker, desaturated
# companion to that run's null tint, so it groups with its own run first and
# reads against that run's u-shuffle null second.
_COLOR_REAL_GAUSSIAN = "#5b7fa6"       # real space, matched-Gaussian null
_COLOR_REDSHIFT_GAUSSIAN = "#4f8f77"   # redshift space, matched-Gaussian null


@dataclass
class RedshiftSpaceRunConfig(RunConfig):
    """`RunConfig`, plus the knobs the real-space vs. redshift-space comparison needs.

    Everything shared with the base single-frame run -- `sub_volume_radius`,
    `shells`, `n_candidate_centers`, `seed`, `shuffle_seed`, `min_center_speed`
    -- stays on the inherited `RunConfig`; this subclass adds only the knobs
    with no meaning outside this comparison.

    Attributes
    ----------
    v_margin_statistic : {"max", "percentile"}
        Which statistic of |v_r| (radial velocity magnitude) sets `v_margin`
        -- see `dvcorr.redshift_space.v_margin_from_statistic` and this
        module's docstring, "Volume construction". "max" (the default) is a
        TRUE bound: every object's radial displacement is covered, so no
        further velocity cut is ever needed. "percentile" is cheaper (a
        smaller buffer/margin) but is no longer a true bound, so
        `select_redshift_shared_centers` additionally drops centers whose
        |v_r| exceeds it globally (see that function and the module
        docstring's legitimate-vs-illegitimate-cut argument).
    v_margin_percentile : float
        Percentile in [0, 100], used only when `v_margin_statistic ==
        "percentile"`. Default 99.9.
    flip_guard_floor : float
        Passed straight through to `dvcorr.redshift_space.to_redshift_space`
        for every displacement this pipeline performs (tracers and centers
        alike). Defaults to `dvcorr.redshift_space.FLIP_GUARD_FLOOR`.
    redshift_shuffle_seed : int
        Seed for the redshift-space run's own velocity-shuffle null
        (`dvcorr.pipeline.velocity_centered.normalize_result`), kept DISTINCT
        from `shuffle_seed` (the real-space run's null) so the two nulls
        never silently share a permutation stream.
    redshift_gaussian_null_seed : int
        Seed for the redshift-space run's own matched-Gaussian null, the
        second null `dvcorr.pipeline.velocity_centered.normalize_result`
        builds. Stands to the inherited `gaussian_null_seed` (the real-space
        run's) exactly as `redshift_shuffle_seed` stands to `shuffle_seed`:
        four nulls reach the figure here, two per run, and no two of them
        share a stream. See
        `dvcorr.pipeline.velocity_centered.RunConfig.gaussian_null_seed` for
        the full ladder.
    comparison_output_name : str
        Output filename for `make_redshift_comparison_figure`'s PNG, written
        under `PathsConfig().output_dir`.
    single_center_output_name : str
        Output filename for `make_single_center_figure`'s PNG.
    example_center_index : int
        Index into the shared center set (`RedshiftCenterSet`, row-aligned
        across both runs) plotted by `make_single_center_figure`. A plain
        `int`, not re-derived at the call site, so the SAME example center is
        used everywhere it is referenced.
    """

    v_margin_statistic: str = "max"
    v_margin_percentile: float = 99.9
    flip_guard_floor: float = FLIP_GUARD_FLOOR
    redshift_shuffle_seed: int = 45
    redshift_gaussian_null_seed: int = 48
    comparison_output_name: str = "redshift_space_comparison.png"
    single_center_output_name: str = "redshift_space_single_center.png"
    example_center_index: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.v_margin_statistic not in ("max", "percentile"):
            raise ValueError(
                "RedshiftSpaceRunConfig: v_margin_statistic must be 'max' or "
                f"'percentile', got {self.v_margin_statistic!r}."
            )
        if self.flip_guard_floor <= 0.0:
            raise ValueError(
                f"RedshiftSpaceRunConfig: flip_guard_floor ({self.flip_guard_floor}) "
                "must be > 0."
            )
        if self.example_center_index < 0:
            raise ValueError(
                "RedshiftSpaceRunConfig: example_center_index "
                f"({self.example_center_index}) must be >= 0."
            )


# ---------------------------------------------------------------------------
# Stage 1: buffered load (two carves, one catalog read)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferedCarve:
    """The plain-R_sub carve's `v_margin`, plus the buffered tracer carve it sets.

    Attributes
    ----------
    pos_buffer : ndarray, shape (N_buffer, 3)
    vel_buffer : ndarray, shape (N_buffer, 3)
        Every halo within `buffered_radius` of the observer, comoving
        h^-1 Mpc / km/s. Row order is the STABLE buffer-row id used
        throughout this module's churn diagnostics (`build_tracer_spaces`).
    pos_core : ndarray, shape (N_core, 3)
    vel_core : ndarray, shape (N_core, 3)
        Every halo within the PLAIN `sub_volume_radius` (a strict subset of
        `pos_buffer`/`vel_buffer`) -- the population `v_margin` was derived
        from, and the correct population to draw CANDIDATE centers from
        (`dvcorr.pipeline.velocity_centered.draw_candidates`): a candidate
        center must itself be a real member of the analysis sub-volume, not
        merely of the wider tracer buffer. Exposed here so callers never
        need to re-derive this subset by re-filtering `pos_buffer` (the
        boolean mask CLAUDE.md's "scripts/notebooks are orchestration only"
        rule would otherwise push into a script or notebook cell).
    n_core : int
        `pos_core.shape[0]`.
    n_buffer : int
        `pos_buffer.shape[0]`.
    n_total : int
        Halos the catalog supplied over the WHOLE box, before either carve --
        the same quantity `dvcorr.pipeline.velocity_centered.CarvedHalos
        .n_total` holds, and the count `box_number_density` must be given to
        build this pipeline's `n_bar` (`normalize_redshift_comparison`).
        Carried as its own field rather than left to `catalog_mvir.size` so
        the normalization's input is a named number, not a shape read off an
        array kept for a different purpose.
    v_margin_kms : float
        The `v_margin` statistic actually used, km/s.
    v_margin_mpc : float
        `v_margin_kms / 100` (H0 = 100 h km/s/Mpc -> h^-1 Mpc; same
        definitional literal as `dvcorr.redshift_space.to_redshift_space`
        and `dvcorr.config.cosmology.CosmologyConfig.h`).
    buffered_radius : float
        `sub_volume_radius + v_margin_mpc`, h^-1 Mpc.
    mvir_core : ndarray, shape (N_core,)
    is_distinct_core : ndarray, shape (N_core,), bool
        Virial masses and distinct-halo flags of the PLAIN carve, row-aligned
        with `pos_core` / `vel_core`. Carried for the same reason
        `dvcorr.pipeline.velocity_centered.CarvedHalos` carries them: the mass
        funnel reports the population a run measured, and the candidate
        centers are drawn from the core carve, not the buffer.
    catalog_mvir : ndarray, shape (N_total,)
        Masses of the full pre-carve population, for the funnel's first stage.
    num_of_subhalos_core : ndarray, shape (N_core,), int32 | None
        Daughter counts of the PLAIN carve, row-aligned with `pos_core` /
        `vel_core`. Carried for the core carve only, and not for the buffer,
        for the same reason `mvir_core` is: candidate centers are drawn from
        the core, while the buffer exists solely to supply a complete
        redshift-space TRACER field, and a tracer's halo class is never asked
        about. None unless `load_and_carve_buffered` was asked for it.
    """

    pos_buffer: np.ndarray
    vel_buffer: np.ndarray
    pos_core: np.ndarray
    vel_core: np.ndarray
    n_core: int
    n_buffer: int
    n_total: int
    mvir_core: np.ndarray
    is_distinct_core: np.ndarray
    catalog_mvir: np.ndarray
    v_margin_kms: float
    v_margin_mpc: float
    buffered_radius: float
    num_of_subhalos_core: np.ndarray | None = None


def load_and_carve_buffered(
    cfg: RedshiftSpaceRunConfig, paths: PathsConfig, *, with_num_of_subhalos: bool = False
) -> BufferedCarve:
    """Load the MDPL2 catalog once; carve twice (plain, then buffered).

    Pass 1: a plain `cfg.sub_volume_radius` carve, used ONLY to compute
    `v_margin` from that population's |v_r| (`dvcorr.redshift_space
    .radial_velocity`, `dvcorr.redshift_space.v_margin_from_statistic`).
    Pass 2: a buffered carve at `sub_volume_radius + v_margin`, the actual
    tracer population `build_tracer_spaces` displaces and restricts. Both
    passes filter the SAME in-memory arrays from a single
    `dvcorr.pipeline.velocity_centered._load_all_halos` call, so this two-pass
    design costs one catalog read, not two. Which catalog, and which halos
    from it, comes from `cfg.catalog` -- resolved once inside that helper.

    Parameters
    ----------
    cfg : RedshiftSpaceRunConfig
    paths : PathsConfig
    with_num_of_subhalos : bool, keyword-only
        Passed through to `_load_all_halos`; surfaces on
        `BufferedCarve.num_of_subhalos_core`. Only
        `dvcorr.pipeline.halo_class_comparison` sets it.

    Returns
    -------
    BufferedCarve

    Raises
    ------
    RuntimeError
        If the catalog loads zero rows (via `_load_all_halos`), or zero
        halos survive the plain `sub_volume_radius` carve (checked before
        printing any percentage, mirroring `load_and_carve`'s guard).
    """
    observer = np.asarray(conventions.OBSERVER_POSITION, dtype=float)
    halos = _load_all_halos(paths, cfg.catalog, with_num_of_subhalos=with_num_of_subhalos)
    d_obs = np.linalg.norm(halos.pos - observer, axis=1)

    in_core = d_obs <= cfg.sub_volume_radius
    # float64 from here on, as in `load_and_carve` -- see `HaloArrays`.
    pos_core = halos.pos[in_core].astype(float)
    vel_core = halos.vel[in_core].astype(float)
    n_core = pos_core.shape[0]
    if n_core == 0:
        raise RuntimeError(
            f"no halos survived the plain sub_volume_radius = "
            f"{cfg.sub_volume_radius} carve; cannot derive v_margin."
        )

    v_r_core = radial_velocity(pos_core, vel_core, observer=observer)
    v_margin_kms = v_margin_from_statistic(
        v_r_core, cfg.v_margin_statistic, cfg.v_margin_percentile
    )
    v_margin_mpc = v_margin_kms / 100.0  # H0 = 100 h km/s/Mpc -> h^-1 Mpc

    buffered_radius = cfg.sub_volume_radius + v_margin_mpc
    in_buffer = d_obs <= buffered_radius
    pos_buffer = halos.pos[in_buffer].astype(float)
    vel_buffer = halos.vel[in_buffer].astype(float)
    n_buffer = pos_buffer.shape[0]

    print(
        f"v_margin ({cfg.v_margin_statistic}) = {v_margin_kms:.1f} km/s = "
        f"{v_margin_mpc:.2f} h^-1 Mpc; buffered carve: {n_buffer} halos within "
        f"{buffered_radius:.2f} h^-1 Mpc, {100.0 * (n_buffer / n_core - 1.0):.1f}% "
        f"more than the plain {cfg.sub_volume_radius:.0f} h^-1 Mpc carve ({n_core} halos)"
    )

    return BufferedCarve(
        pos_buffer=pos_buffer,
        vel_buffer=vel_buffer,
        pos_core=pos_core,
        vel_core=vel_core,
        n_core=n_core,
        n_buffer=n_buffer,
        n_total=halos.n_total,
        v_margin_kms=v_margin_kms,
        v_margin_mpc=v_margin_mpc,
        buffered_radius=buffered_radius,
        mvir_core=halos.mvir[in_core],
        is_distinct_core=halos.is_distinct[in_core],
        catalog_mvir=halos.mvir,
        num_of_subhalos_core=(
            None if halos.num_of_subhalos is None else halos.num_of_subhalos[in_core]
        ),
    )


# ---------------------------------------------------------------------------
# Stage 2: tracer fields in both spaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TracerSpaces:
    """Real- and redshift-space tracer positions restricted to R_sub.

    Both are built from the SAME buffered carve (`BufferedCarve`), so every
    row of `s_tracers_real` / `s_tracers_redshift` corresponds to a STABLE
    buffer-row id (`real_ids` / `redshift_ids`) usable for set algebra
    (`membership_diagnostics`) across the two, positionally-different,
    subsets.

    Attributes
    ----------
    s_tracers_real : ndarray, shape (N_real, 3)
        Buffered real positions restricted to `|r - observer| <= R_sub`.
    real_ids : ndarray, shape (N_real,), int
        Stable buffer-row id of each `s_tracers_real` row (an index into
        `BufferedCarve.pos_buffer`/`vel_buffer`).
    s_tracers_redshift : ndarray, shape (N_z, 3)
        Buffered positions, displaced
        (`dvcorr.redshift_space.to_redshift_space`), restricted to
        `|s - observer| <= R_sub`.
    redshift_ids : ndarray, shape (N_z,), int
        Stable buffer-row id of each `s_tracers_redshift` row.
    n_flip_dropped : int
        Buffered tracers dropped by the through-observer flip guard, before
        the R_sub restriction.
    n_real_inside : int
        `s_tracers_real.shape[0]` -- the count `normalize_redshift_comparison`
        uses for the boundary-flux diagnostic (see this module's docstring).
    n_redshift_inside : int
        `s_tracers_redshift.shape[0]` -- logged alongside `n_real_inside` as
        the boundary-flux diagnostic; NOT used for n_bar.
    """

    s_tracers_real: np.ndarray
    real_ids: np.ndarray
    s_tracers_redshift: np.ndarray
    redshift_ids: np.ndarray
    n_flip_dropped: int
    n_real_inside: int
    n_redshift_inside: int


def build_tracer_spaces(
    cfg: RedshiftSpaceRunConfig,
    buffer: BufferedCarve,
    observer: np.ndarray,
) -> TracerSpaces:
    """Displace the buffered tracer field, then restrict both spaces to R_sub.

    Order matters (see the module docstring's "TRACERS" section): carve at
    `R_sub + v_margin` (already done, `buffer`), displace the FULL buffer,
    THEN restrict to R_sub -- restricting first would throw away exactly the
    tracers whose displacement was the point of buffering for.

    Parameters
    ----------
    cfg : RedshiftSpaceRunConfig
    buffer : BufferedCarve
        From `load_and_carve_buffered`.
    observer : ndarray, shape (3,)

    Returns
    -------
    TracerSpaces
    """
    buffer_ids = np.arange(buffer.pos_buffer.shape[0])

    d_real = np.linalg.norm(buffer.pos_buffer - observer, axis=1)
    in_real = d_real <= cfg.sub_volume_radius
    s_tracers_real = buffer.pos_buffer[in_real]
    real_ids = buffer_ids[in_real]

    transform = to_redshift_space(
        buffer.pos_buffer, buffer.vel_buffer,
        observer=observer, flip_guard_floor=cfg.flip_guard_floor,
    )
    kept_ids = buffer_ids[transform.kept_mask]
    d_redshift = np.linalg.norm(transform.s_redshift - observer, axis=1)
    in_redshift = d_redshift <= cfg.sub_volume_radius
    s_tracers_redshift = transform.s_redshift[in_redshift]
    redshift_ids = kept_ids[in_redshift]

    n_real_inside = s_tracers_real.shape[0]
    n_redshift_inside = s_tracers_redshift.shape[0]
    gap_pct = (
        100.0 * abs(n_redshift_inside - n_real_inside) / n_real_inside
        if n_real_inside > 0 else float("nan")
    )
    print(
        f"tracer flux check: real-inside-R_sub = {n_real_inside}, "
        f"redshift-inside-R_sub = {n_redshift_inside} (gap = {gap_pct:.3f}%; "
        "expected well under 1% -- the buffer captures inflow and outflow "
        "both, so a much larger gap means the buffer is misconfigured, not "
        "that RSD moved a meaningfully large fraction of the tracer mass); "
        f"{transform.n_dropped} of {buffer.n_buffer} buffered tracers dropped "
        "by the through-observer flip guard before the R_sub restriction"
    )

    return TracerSpaces(
        s_tracers_real=s_tracers_real,
        real_ids=real_ids,
        s_tracers_redshift=s_tracers_redshift,
        redshift_ids=redshift_ids,
        n_flip_dropped=transform.n_dropped,
        n_real_inside=n_real_inside,
        n_redshift_inside=n_redshift_inside,
    )


# ---------------------------------------------------------------------------
# Stage 3: the shared center set, in both spaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedshiftCenterSet:
    """The single center set both the real-space and redshift-space runs share.

    `s_centers_real` and `s_centers_redshift` are ROW-ALIGNED: row alpha is
    the SAME physical halo in both, at its real and displaced position
    respectively. `v_centers` is a SINGLE array shared by both runs -- not an
    oversight, but the concrete statement of the module docstring's n_hat
    invariance: `velocity_centered_shell_dipole` recomputes n_hat_V,alpha
    (and hence u_alpha, z_hat_alpha) from whichever `s_centers` it is given,
    and that recomputation is provably identical whether `s_centers` is real
    or displaced (see `dvcorr.redshift_space`'s module docstring). There is
    therefore exactly one `v_centers` to pass to either run.

    Attributes
    ----------
    s_centers_real : ndarray, shape (N_c, 3)
    s_centers_redshift : ndarray, shape (N_c, 3)
    v_centers : ndarray, shape (N_c, 3)
    base : SharedCenterSet
        The underlying core-cut-plus-speed-floor selection (on REAL
        positions, `core_margin = r_max + v_margin`), before the optional
        global |v_r| percentile cut and the through-observer guard.
    n_dropped_v_r_percentile : int
        Centers dropped by the global |v_r| cut. Always 0 when
        `cfg.v_margin_statistic == "max"` (no object exceeds a true max by
        definition).
    n_dropped_flip : int
        Centers dropped by the through-observer flip guard on their OWN
        displacement. Expected to be 0 or very small in practice -- see
        `select_redshift_shared_centers`'s docstring for the
        triangle-inequality argument for why the margin construction makes
        this guard nearly always a no-op, kept only as a genuine safety net.
    v_margin_kms : float
    v_margin_mpc : float
    n_centers : int
        Final surviving count, shared by both runs.
    """

    s_centers_real: np.ndarray
    s_centers_redshift: np.ndarray
    v_centers: np.ndarray
    base: SharedCenterSet
    n_dropped_v_r_percentile: int
    n_dropped_flip: int
    v_margin_kms: float
    v_margin_mpc: float
    n_centers: int
    mvir_centers: np.ndarray | None = None
    is_distinct_centers: np.ndarray | None = None


def select_redshift_shared_centers(
    cfg: RedshiftSpaceRunConfig,
    s_candidates: np.ndarray,
    v_candidates: np.ndarray,
    observer: np.ndarray,
    v_margin_kms: float,
    v_margin_mpc: float,
    *,
    mvir_candidates: np.ndarray | None = None,
    is_distinct_candidates: np.ndarray | None = None,
) -> RedshiftCenterSet:
    """Select centers usable, identically, by both the real- and redshift-space runs.

    Three sequential cuts, funnel-style:

    1. `select_shared_centers(cfg, s_candidates, v_candidates, observer,
       core_margin=cfg.shells.max_radius + v_margin_mpc)` -- core cut plus
       speed floor, on REAL positions, with the WIDENED margin `r_max +
       v_margin` (never the plain `r_max` the single-frame/velocity-frame
       pipelines use, and never applied to a displaced position -- see the
       module docstring's "CENTERS" section for why the widened REAL margin
       is required and the displaced-position alternative is not just
       avoided but unnecessary, by the triangle-inequality argument below).
    2. If `cfg.v_margin_statistic == "percentile"`: drop centers with
       `|v_r| > v_margin_kms`, GLOBALLY (position-independent) -- see the
       module docstring's legitimate-vs-illegitimate-cut argument. A no-op
       when the statistic is "max" (nothing exceeds a true max).
    3. The through-observer flip guard
       (`dvcorr.redshift_space.to_redshift_space`) on the survivors' OWN
       displacement, so BOTH runs see exactly the survivors of this guard --
       never just the redshift-space run silently missing a few centers the
       real-space run still has.

    Why step 3 is (almost always) a no-op, by construction
    -----------------------------------------------------------
    Every surviving center has REAL distance from the observer
    `R_real <= R_sub - (r_max + v_margin_mpc)` (step 1) and, after step 2,
    `|v_r| <= v_margin_kms` (trivially true for the "max" statistic; enforced
    explicitly for "percentile"). Its displaced distance is therefore bounded

        R_z = R_real + v_r / 100 <= R_real + v_margin_mpc
            <= R_sub - (r_max + v_margin_mpc) + v_margin_mpc = R_sub - r_max,

    i.e. EVERY surviving center's displaced position automatically clears the
    plain `r_max`-margin core cut `velocity_centered_shell_dipole` re-applies
    internally by default -- WITHOUT this function ever computing or
    inspecting a displaced position to decide who survives. This is the
    concrete reason `select_redshift_shared_centers` does not need (and must
    not add) an explicit "does the displaced position still clear the
    margin" check: doing so would be redundant when the bound holds, and
    exactly the illegitimate velocity-conditioned selection the module
    docstring warns against if anyone ever loosens the margin below this
    bound "to get more centers." Step 3's flip guard is therefore a genuine
    safety net for the rare near-observer, fast-inward object (where R_real
    itself is small enough that the bound above, while still valid, does not
    prevent `s_mag` from crossing `FLIP_GUARD_FLOOR`), not the mechanism
    keeping the two runs' center sets aligned in the ordinary case.

    Parameters
    ----------
    cfg : RedshiftSpaceRunConfig
    s_candidates, v_candidates : ndarray, shape (N_cand, 3)
        Candidate positions/velocities, e.g. from
        `dvcorr.pipeline.velocity_centered.draw_candidates`.
    observer : ndarray, shape (3,)
    v_margin_kms, v_margin_mpc : float
        From `load_and_carve_buffered`'s `BufferedCarve` -- computed ONCE
        there and passed through, never recomputed here (this module's
        docstring: "never recomputed inline at a second call site").

    Returns
    -------
    RedshiftCenterSet

    Raises
    ------
    RuntimeError
        Propagated from `select_shared_centers` if zero centers survive
        steps 1, or raised here if zero survive steps 2-3 combined.
    """
    core_margin = cfg.shells.max_radius + v_margin_mpc
    base = select_shared_centers(
        cfg,
        s_candidates,
        v_candidates,
        observer,
        core_margin=core_margin,
        mvir_candidates=mvir_candidates,
        is_distinct_candidates=is_distinct_candidates,
    )

    s_real = base.s_centers
    v_centers = base.v_centers
    # Labels ride every cut below alongside the arrays they describe, exactly
    # as they rode the two inside `select_shared_centers`. `base` keeps the
    # pre-cut set for the funnel's earlier stage; these are the FINAL centers'.
    mvir_centers = base.mvir_centers
    is_distinct_centers = base.is_distinct_centers

    n_dropped_v_r_percentile = 0
    if cfg.v_margin_statistic == "percentile":
        v_r = radial_velocity(s_real, v_centers, observer=observer)
        keep_v_r = np.abs(v_r) <= v_margin_kms
        n_dropped_v_r_percentile = int(np.sum(~keep_v_r))
        s_real = s_real[keep_v_r]
        v_centers = v_centers[keep_v_r]
        if mvir_centers is not None:
            mvir_centers = mvir_centers[keep_v_r]
        if is_distinct_centers is not None:
            is_distinct_centers = is_distinct_centers[keep_v_r]

    transform = to_redshift_space(
        s_real, v_centers, observer=observer, flip_guard_floor=cfg.flip_guard_floor
    )
    s_centers_real = s_real[transform.kept_mask]
    v_centers = v_centers[transform.kept_mask]
    s_centers_redshift = transform.s_redshift
    n_dropped_flip = transform.n_dropped
    n_centers = s_centers_real.shape[0]
    if mvir_centers is not None:
        mvir_centers = mvir_centers[transform.kept_mask]
    if is_distinct_centers is not None:
        is_distinct_centers = is_distinct_centers[transform.kept_mask]

    print(
        f"redshift-space center selection: base n_centers = {base.n_centers} "
        f"(core_margin = r_max + v_margin = {core_margin:.2f} h^-1 Mpc) -> "
        f"global |v_r| <= {v_margin_kms:.1f} km/s cut drops "
        f"{n_dropped_v_r_percentile} -> through-observer guard drops "
        f"{n_dropped_flip} -> n_centers = {n_centers}"
    )
    if n_centers == 0:
        raise RuntimeError(
            "select_redshift_shared_centers: zero centers survived the "
            "global |v_r| cut and through-observer guard combined."
        )

    return RedshiftCenterSet(
        s_centers_real=s_centers_real,
        s_centers_redshift=s_centers_redshift,
        v_centers=v_centers,
        base=base,
        n_dropped_v_r_percentile=n_dropped_v_r_percentile,
        n_dropped_flip=n_dropped_flip,
        v_margin_kms=v_margin_kms,
        v_margin_mpc=v_margin_mpc,
        n_centers=n_centers,
        mvir_centers=mvir_centers,
        is_distinct_centers=is_distinct_centers,
    )


# ---------------------------------------------------------------------------
# Stage 4: run the (unchanged) estimator, twice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedshiftSpaceFrameResults:
    """Both runs' raw results, plus the centers and tracers they share.

    Attributes
    ----------
    real_result : VelocityCenteredShellDipoleResult
        `velocity_centered_shell_dipole` on real positions, both centers and
        tracers.
    redshift_result : VelocityCenteredShellDipoleResult
        `velocity_centered_shell_dipole` on redshift-space positions, both
        centers and tracers. Same `v_centers`, same `shell_edges`, same
        `sub_volume_radius`, same `observer` as `real_result` -- the ONLY
        difference in either call's arguments is the position arrays.
    centers : RedshiftCenterSet
    tracers : TracerSpaces
    """

    real_result: VelocityCenteredShellDipoleResult
    redshift_result: VelocityCenteredShellDipoleResult
    centers: RedshiftCenterSet
    tracers: TracerSpaces


def run_both_spaces(
    cfg: RedshiftSpaceRunConfig,
    centers: RedshiftCenterSet,
    tracers: TracerSpaces,
    observer: np.ndarray,
) -> RedshiftSpaceFrameResults:
    """Run `velocity_centered_shell_dipole` on both spaces, unchanged.

    Mirrors `dvcorr.pipeline.velocity_frame_comparison.run_both_frames`'s
    row-alignment check: both calls must report `n_centers ==
    centers.n_centers`, which `select_redshift_shared_centers`'s
    triangle-inequality argument guarantees for the redshift-space call (see
    that function's docstring) and which trivially holds for the real-space
    call (its own margin is only WIDER than what `core_center_mask`
    re-checks internally).

    Parameters
    ----------
    cfg : RedshiftSpaceRunConfig
    centers : RedshiftCenterSet
        From `select_redshift_shared_centers`.
    tracers : TracerSpaces
        From `build_tracer_spaces`.
    observer : ndarray, shape (3,)

    Returns
    -------
    RedshiftSpaceFrameResults

    Raises
    ------
    RuntimeError
        If either estimator's `n_centers` does not equal `centers.n_centers`.
    """
    real_result = velocity_centered_shell_dipole(
        s_centers=centers.s_centers_real,
        v_centers=centers.v_centers,
        s_tracers=tracers.s_tracers_real,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )
    redshift_result = velocity_centered_shell_dipole(
        s_centers=centers.s_centers_redshift,
        v_centers=centers.v_centers,
        s_tracers=tracers.s_tracers_redshift,
        shell_edges=cfg.shells.shell_edges,
        sub_volume_radius=cfg.sub_volume_radius,
        observer=observer,
    )

    if real_result.n_centers != centers.n_centers or redshift_result.n_centers != centers.n_centers:
        raise RuntimeError(
            "run_both_spaces: row-alignment invariant violated -- "
            f"centers.n_centers = {centers.n_centers}, "
            f"real_result.n_centers = {real_result.n_centers}, "
            f"redshift_result.n_centers = {redshift_result.n_centers}. See "
            "select_redshift_shared_centers's triangle-inequality argument "
            "for the invariant this violates (e.g. cfg.sub_volume_radius or "
            "cfg.shells changed between selection and this call)."
        )

    return RedshiftSpaceFrameResults(
        real_result=real_result, redshift_result=redshift_result,
        centers=centers, tracers=tracers,
    )


# ---------------------------------------------------------------------------
# Stage 5: normalize (shared, box-wide n_bar)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedshiftSpaceComparison:
    """Normalized curves for both spaces, plus the shared-n_bar bookkeeping.

    Attributes
    ----------
    real : NormalizedDipole
        Real-space normalized dipole/monopole/nulls, from `normalize_result`
        with `cfg.shuffle_seed` and `cfg.gaussian_null_seed`.
    redshift : NormalizedDipole
        Redshift-space normalized dipole/monopole/nulls, from
        `normalize_result` with `cfg.redshift_shuffle_seed` and
        `cfg.redshift_gaussian_null_seed`.
    shell_centers : ndarray, shape (B,)
        Shared by both spaces (identical `shell_edges`).
    n_bar : float
        The SHARED, box-wide n_bar both curves were normalized with (see
        this module's docstring, "n_bar normalization").
    n_real_inside_r_sub : int
    n_redshift_inside_r_sub : int
        The boundary-flux diagnostic counts (`TracerSpaces.n_real_inside` /
        `.n_redshift_inside`), carried through for convenience.
    """

    real: NormalizedDipole
    redshift: NormalizedDipole
    shell_centers: np.ndarray
    n_bar: float
    n_real_inside_r_sub: int
    n_redshift_inside_r_sub: int


def normalize_redshift_comparison(
    cfg: RedshiftSpaceRunConfig,
    results: RedshiftSpaceFrameResults,
    n_bar: float,
) -> RedshiftSpaceComparison:
    """Normalize both runs with the SAME, box-wide n_bar.

    Both runs use the ONE `n_bar` handed in (the decision documented at the
    top of this module) -- a per-run n_bar would fold the boundary-flux
    difference into the headline amplitude ratio. The caller builds it with
    `dvcorr.pipeline.velocity_centered.box_number_density(carve.n_total)`,
    the catalog's halo count over the whole box; see this module's n_bar
    section for why that definition, unlike the carved-count-over-carve-volume
    one it replaced, cannot be silently mismatched against this pipeline's
    buffered carve.

    Parameters
    ----------
    cfg : RedshiftSpaceRunConfig
    results : RedshiftSpaceFrameResults
    n_bar : float
        Mean halo number density over the whole box, e.g. from
        `dvcorr.pipeline.velocity_centered.box_number_density`.

    Returns
    -------
    RedshiftSpaceComparison
    """
    real = normalize_result(
        results.real_result,
        n_bar,
        cfg.shuffle_seed,
        cfg.gaussian_null_seed,
        cfg.n_null_realizations,
    )
    redshift = normalize_result(
        results.redshift_result,
        n_bar,
        cfg.redshift_shuffle_seed,
        cfg.redshift_gaussian_null_seed,
        cfg.n_null_realizations,
    )

    return RedshiftSpaceComparison(
        real=real,
        redshift=redshift,
        shell_centers=results.real_result.shell_centers,
        n_bar=n_bar,
        n_real_inside_r_sub=results.tracers.n_real_inside,
        n_redshift_inside_r_sub=results.tracers.n_redshift_inside,
    )


# ---------------------------------------------------------------------------
# Membership diagnostics -- standalone, NOT part of the estimator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MembershipDiagnostics:
    """Per-shell tracer-membership comparison between the two tracer fields.

    Built by `membership_diagnostics`, which runs its OWN `cKDTree` queries
    against `s_tracers_real`/`s_tracers_redshift` -- deliberately NOT part of
    `velocity_centered_shell_dipole`, so the estimator stays agnostic to
    which space it was handed and never needs to store an N_c x N_members
    index set purely to support this diagnostic.

    Attributes
    ----------
    shell_edges : ndarray, shape (B + 1,)
    shell_centers : ndarray, shape (B,)
    net_change : ndarray, shape (B,)
        `Sigma_alpha (N_redshift,alpha,b - N_real,alpha,b)` -- aggregated net
        membership change per shell. A shell with unchanged `net_change` can
        still have fully swapped membership; see `churn_*` below.
    per_center_net_change : ndarray, shape (N_c, B)
        The same quantity, per center.
    churn_only_real : ndarray, shape (B,)
        Aggregated count of (center, tracer) pairs present in the real-space
        shell but NOT the redshift-space shell for that same center,
        compared by STABLE buffer-row id (see `membership_diagnostics`'s
        docstring for why positional indices are unsafe here).
    churn_only_redshift : ndarray, shape (B,)
        The mirror image: present in redshift-space, not real-space.
    churn_intersection : ndarray, shape (B,)
        Count of (center, tracer) id pairs present in BOTH -- so
        `churn_only_real + churn_intersection` recovers the real-space
        occupancy and `churn_only_redshift + churn_intersection` the
        redshift-space one.
    per_center_churn_only_real : ndarray, shape (N_c, B)
    per_center_churn_only_redshift : ndarray, shape (N_c, B)
    per_center_churn_intersection : ndarray, shape (N_c, B)
        Per-center breakdowns of the three churn counts above.
    n_centers : int
    """

    shell_edges: np.ndarray
    shell_centers: np.ndarray
    net_change: np.ndarray
    per_center_net_change: np.ndarray
    churn_only_real: np.ndarray
    churn_only_redshift: np.ndarray
    churn_intersection: np.ndarray
    per_center_churn_only_real: np.ndarray
    per_center_churn_only_redshift: np.ndarray
    per_center_churn_intersection: np.ndarray
    n_centers: int


def _ids_by_shell(
    tree: cKDTree | None,
    tracer_positions: np.ndarray,
    tracer_ids: np.ndarray,
    center: np.ndarray,
    shell_edges: np.ndarray,
) -> list[np.ndarray]:
    """Stable tracer ids per shell around ONE center, from ONE tree query.

    A single `query_ball_point` at the outermost edge, then binned by radius
    exactly as `dvcorr.estimators.shell_dipole.velocity_centered_shell_dipole`
    bins its own tracers (left-closed/right-open, outer edge folded into the
    last shell, the r = 0 self-pair excluded on `r_mag > 0`) -- so this
    diagnostic's shell membership matches the estimator's own binning
    contract precisely.

    Returns SORTED ARRAYS, not Python sets. The set-based version this
    replaces built its result with a per-element `.add(int(...))` loop, which
    costs a Python-level iteration per tracer-shell membership. That is
    tolerable at the ~4e3 neighbors a ball holds in the mvir >= 1e12 catalog
    and not at the ~1.4e5 it holds in the full one, where the same loop runs
    ~5e8 times per pipeline invocation. Sorted arrays let `membership_diagnostics`
    use `np.intersect1d` / `np.setdiff1d` with `assume_unique=True` instead,
    which is vectorized and returns the identical cardinalities.

    Ids within a shell are unique by construction -- a tracer occupies exactly
    one shell, and `tracer_ids` is a stable per-row id -- which is what makes
    `assume_unique=True` safe downstream.

    Returns
    -------
    list of ndarray, length B
        `result[b]` holds the STABLE `tracer_ids` values in shell b, sorted
        ascending. Empty arrays for empty shells, so every entry is indexable
        without a None check.
    """
    n_bins = shell_edges.size - 1
    empty = [np.empty(0, dtype=np.int64) for _ in range(n_bins)]
    if tree is None or tracer_positions.shape[0] == 0:
        return empty

    idx = np.asarray(tree.query_ball_point(center, shell_edges[-1]), dtype=int)
    if idx.size == 0:
        return empty

    r_mag = np.linalg.norm(tracer_positions[idx] - center, axis=1)
    # r_mag > 0 drops the center's own self-pair, matching the estimator term
    # for term -- without it this diagnostic would report a shell membership
    # the estimator does not count, and the "net change" of a bin whose only
    # occupant is the self-pair would be pure artefact. A no-op unless
    # shell_edges[0] == 0 (`dvcorr.config.ShellConfig.include_zero_bin`).
    in_range = (r_mag > 0.0) & (r_mag >= shell_edges[0]) & (r_mag <= shell_edges[-1])
    bin_index = np.digitize(r_mag, shell_edges) - 1
    bin_index = np.where(bin_index == n_bins, n_bins - 1, bin_index)

    ids_in = np.asarray(tracer_ids)[idx[in_range]].astype(np.int64, copy=False)
    bins_in = bin_index[in_range]

    # Group by shell with one sort rather than n_bins boolean scans: sort the
    # (shell, id) pairs by shell, then split at the shell boundaries. Sorting
    # by id within each group as well makes every returned array sorted, which
    # is what the set operations downstream rely on.
    order = np.lexsort((ids_in, bins_in))
    bins_sorted = bins_in[order]
    ids_sorted = ids_in[order]
    boundaries = np.searchsorted(bins_sorted, np.arange(n_bins + 1))
    return [ids_sorted[boundaries[b] : boundaries[b + 1]] for b in range(n_bins)]


def membership_diagnostics(
    s_centers: np.ndarray,
    s_tracers_real: np.ndarray,
    real_ids: np.ndarray,
    s_tracers_redshift: np.ndarray,
    redshift_ids: np.ndarray,
    shell_edges: np.ndarray,
) -> MembershipDiagnostics:
    """Compare shell membership between the real- and redshift-space tracer fields.

    Runs its own `cKDTree` on each tracer array -- see `MembershipDiagnostics`'s
    docstring for why this is deliberately standalone rather than folded into
    `velocity_centered_shell_dipole`.

    Always queried around `s_centers`, the REAL center positions, for BOTH
    tracer fields: membership is a question about which tracers fall near a
    given PHYSICAL center, not about how the center's own apparent position
    moved -- using the center's redshift-space position for the
    redshift-side query would conflate "the tracer field moved" with "the
    center moved" and muddy exactly the comparison this function exists to
    make.

    CHURN MUST USE STABLE IDS. `s_tracers_real` and `s_tracers_redshift` are
    two DIFFERENT positional subsets of the same underlying buffer
    (`dvcorr.pipeline.redshift_space_comparison.TracerSpaces`); row i of one
    and row i of the other are, in general, different halos.
    `cKDTree.query_ball_point` returns indices POSITIONAL in whichever array
    it queried, so set algebra on those raw indices would silently compare
    the wrong objects. `_ids_by_shell` immediately maps every positional
    index through `real_ids`/`redshift_ids` before any set operation, so
    every comparison below is on STABLE ids throughout.

    Parameters
    ----------
    s_centers : ndarray, shape (N_c, 3)
        REAL center positions (e.g. `RedshiftCenterSet.s_centers_real`).
    s_tracers_real : ndarray, shape (N_real, 3)
    real_ids : ndarray, shape (N_real,), int
        E.g. `TracerSpaces.s_tracers_real` / `.real_ids`.
    s_tracers_redshift : ndarray, shape (N_z, 3)
    redshift_ids : ndarray, shape (N_z,), int
        E.g. `TracerSpaces.s_tracers_redshift` / `.redshift_ids`.
    shell_edges : ndarray, shape (B + 1,)
        Radial shell boundaries, h^-1 Mpc, matching
        `velocity_centered_shell_dipole`'s binning contract.

    Returns
    -------
    MembershipDiagnostics
    """
    edges = np.asarray(shell_edges, dtype=float)
    n_bins = edges.size - 1
    s_centers = np.atleast_2d(np.asarray(s_centers, dtype=float))
    n_centers = s_centers.shape[0]

    per_center_net_change = np.zeros((n_centers, n_bins), dtype=float)
    per_center_churn_only_real = np.zeros((n_centers, n_bins), dtype=float)
    per_center_churn_only_redshift = np.zeros((n_centers, n_bins), dtype=float)
    per_center_churn_intersection = np.zeros((n_centers, n_bins), dtype=float)

    tree_real = cKDTree(s_tracers_real) if s_tracers_real.shape[0] > 0 else None
    tree_redshift = cKDTree(s_tracers_redshift) if s_tracers_redshift.shape[0] > 0 else None

    for a in range(n_centers):
        center = s_centers[a]
        ids_real_by_shell = _ids_by_shell(tree_real, s_tracers_real, real_ids, center, edges)
        ids_redshift_by_shell = _ids_by_shell(
            tree_redshift, s_tracers_redshift, redshift_ids, center, edges
        )
        for b in range(n_bins):
            ids_real = ids_real_by_shell[b]
            ids_redshift = ids_redshift_by_shell[b]

            # Both arrays are sorted and duplicate-free by `_ids_by_shell`'s
            # contract, so `assume_unique=True` is a statement of that fact,
            # not an optimistic shortcut. Only the intersection is computed:
            # the two one-sided churns follow from it by subtraction, which
            # avoids two more set passes over the same arrays and cannot
            # disagree with it the way three independent calls could.
            n_real = ids_real.size
            n_redshift = ids_redshift.size
            n_shared = np.intersect1d(ids_real, ids_redshift, assume_unique=True).size

            per_center_net_change[a, b] = n_redshift - n_real
            per_center_churn_only_real[a, b] = n_real - n_shared
            per_center_churn_only_redshift[a, b] = n_redshift - n_shared
            per_center_churn_intersection[a, b] = n_shared

    shell_centers = 0.5 * (edges[:-1] + edges[1:])

    return MembershipDiagnostics(
        shell_edges=edges,
        shell_centers=shell_centers,
        net_change=per_center_net_change.sum(axis=0),
        per_center_net_change=per_center_net_change,
        churn_only_real=per_center_churn_only_real.sum(axis=0),
        churn_only_redshift=per_center_churn_only_redshift.sum(axis=0),
        churn_intersection=per_center_churn_intersection.sum(axis=0),
        per_center_churn_only_real=per_center_churn_only_real,
        per_center_churn_only_redshift=per_center_churn_only_redshift,
        per_center_churn_intersection=per_center_churn_intersection,
        n_centers=n_centers,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_redshift_comparison_figure(
    cfg: RedshiftSpaceRunConfig,
    results: RedshiftSpaceFrameResults,
    comparison: RedshiftSpaceComparison,
) -> plt.Figure:
    """Center-averaged comparison: real-space vs. redshift-space zeta_hat_1 + monopole.

    Two panels sharing the x-axis (separation r), height ratios (3, 1) --
    identical layout convention to `dvcorr.pipeline.velocity_centered
    .make_figure` and `dvcorr.pipeline.velocity_frame_comparison
    .make_comparison_figure` (CLAUDE.md hard rule 6: the dipole never travels
    without its monopole). Unlike the observer-vs-velocity-frame comparison,
    a SINGLE shared y-axis suffices for the monopole panel here: both curves
    are the SAME kind of quantity (Sigma |u_alpha| N_alpha,b, normalized the
    same way) on the SAME set of u_alpha values (n_hat invariance), differing
    only through shell occupancy N_alpha,b -- there is no projection-factor
    scale gap to hide behind a twin axis.

    TOP panel: `zeta_hat` for both spaces, each with its own across-center
    SEM band and its own TWO nulls -- the velocity shuffle dashed in a
    lighter tint, the matched-Gaussian null dotted in a darker one, grouped
    by space rather than by null type. `real`'s pair uses `cfg.shuffle_seed`
    / `cfg.gaussian_null_seed`, `redshift`'s uses
    `cfg.redshift_shuffle_seed` / `cfg.redshift_gaussian_null_seed` (four
    distinct streams, `normalize_redshift_comparison`). Within a space the
    two nulls are matched on their first two moments by construction
    (`dvcorr.pipeline.velocity_centered.matched_gaussian_sample`), so their
    separation isolates the non-Gaussianity of the u distribution -- and
    since u_alpha is IDENTICAL between the two spaces (n_hat invariance, see
    the bottom panel), any difference between the two spaces' Gaussian-vs-
    shuffle gaps is occupancy, not velocities. The SEM-band independence caveat is the
    same as every other stacked estimate in this project
    (`dvcorr.estimators.shell_dipole.center_standard_error`'s docstring):
    centers sharing tracers or large-scale velocity coherence are not
    independent samples, so the bands UNDERSTATE the true uncertainty for
    BOTH curves.

    BOTTOM panel: both monopoles, single shared y-axis (see above). Neither
    sits at zero (the `|u_alpha|` weight has no near/far cancellation), and
    the panel is the finite-distance / incomplete-shell trust diagnostic:
    since `u_alpha` is identical between spaces, any TREND difference between
    the two monopole curves is coming from shell occupancy alone -- exactly
    what displacing the tracer field is expected to perturb, and exactly why
    this panel, not just the dipole, is the thing to look at first when
    judging whether the redshift-space run's dipole is trustworthy.

    Parameters
    ----------
    cfg : RedshiftSpaceRunConfig
    results : RedshiftSpaceFrameResults
    comparison : RedshiftSpaceComparison

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_dipole, ax_mono) = plt.subplots(
        2, 1, figsize=_FIGSIZE, sharex=True, gridspec_kw={"height_ratios": _HEIGHT_RATIOS}
    )

    # comparison.shell_centers (the plain midpoint) is deliberately left
    # alone -- r_eff, the volume-weighted radius, is the plotting abscissa.
    r = volume_weighted_shell_radii(results.real_result.shell_edges)

    ax_dipole.fill_between(
        r, comparison.real.zeta_hat - comparison.real.sem, comparison.real.zeta_hat + comparison.real.sem,
        color=_COLOR_REAL, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.real.zeta_hat, "o-", color=_COLOR_REAL,
        label=fr"$\hat\zeta_1^{{real}}$  (N$_c$={results.real_result.n_centers})",
    )
    ax_dipole.plot(
        r, comparison.real.zeta_hat_shuffle, "o--", color=_COLOR_REAL_NULL,
        label="real null (u-shuffle)",
    )
    ax_dipole.plot(
        r, comparison.real.zeta_hat_gaussian, "o:", color=_COLOR_REAL_GAUSSIAN,
        label="real null (gaussian)",
    )

    ax_dipole.fill_between(
        r, comparison.redshift.zeta_hat - comparison.redshift.sem,
        comparison.redshift.zeta_hat + comparison.redshift.sem,
        color=_COLOR_REDSHIFT, alpha=_BAND_ALPHA,
    )
    ax_dipole.plot(
        r, comparison.redshift.zeta_hat, "s-", color=_COLOR_REDSHIFT,
        label=fr"$\hat\zeta_1^{{z}}$  (N$_c$={results.redshift_result.n_centers})",
    )
    ax_dipole.plot(
        r, comparison.redshift.zeta_hat_shuffle, "s--", color=_COLOR_REDSHIFT_NULL,
        label="redshift null (u-shuffle)",
    )
    ax_dipole.plot(
        r, comparison.redshift.zeta_hat_gaussian, "s:", color=_COLOR_REDSHIFT_GAUSSIAN,
        label="redshift null (gaussian)",
    )

    ax_dipole.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_dipole.set_ylabel(r"$\hat\zeta_1$  [km/s]", color=_LABEL_COLOR)
    ax_dipole.legend(fontsize="small")
    ax_dipole.grid(alpha=_GRID_ALPHA)
    ax_dipole.spines["top"].set_visible(False)
    ax_dipole.spines["right"].set_visible(False)
    ax_dipole.set_title(
        "real space vs. redshift space  "
        f"(R_sub={cfg.sub_volume_radius:.0f} h$^{{-1}}$Mpc, "
        f"{_binning_description(cfg.shells)}, "
        f"N$_c$={results.centers.n_centers}, "
        f"v_margin={results.centers.v_margin_kms:.0f} km/s)\n"
        "SEM bands treat centers as independent and UNDERSTATE the true "
        "uncertainty (center_standard_error)",
        fontsize=9,
    )

    ax_mono.plot(r, comparison.real.monopole_norm, "o-", color=_COLOR_REAL, label="real")
    ax_mono.plot(r, comparison.redshift.monopole_norm, "s-", color=_COLOR_REDSHIFT, label="redshift")
    ax_mono.set_ylabel(r"$\hat\zeta_0 \simeq \langle|u|\rangle$  [km/s]", color=_LABEL_COLOR)
    ax_mono.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
    ax_mono.legend(fontsize="small")
    ax_mono.grid(alpha=_GRID_ALPHA)
    ax_mono.spines["top"].set_visible(False)
    ax_mono.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def make_single_center_figure(
    cfg: RedshiftSpaceRunConfig,
    results: RedshiftSpaceFrameResults,
    n_bar: float,
) -> plt.Figure:
    """One example center's own dipole + monopole, real vs. redshift, no band.

    Same (3, 1) two-panel layout (hard rule 6). `cfg.example_center_index`
    (a config parameter, not hardcoded) indexes the SHARED, row-aligned
    center set (`RedshiftCenterSet`), so `results.real_result
    .per_center_dipole[a]` and `results.redshift_result.per_center_dipole[a]`
    describe the SAME physical halo in the two spaces.

    No error band: a single center has no across-center population to form
    a standard error from -- this figure is a shape/sanity read-out, not a
    stacked estimate.

    Dipole: `per_center_dipole[a] * shell_dipole_norm_scale(shell_edges,
    n_bar)`, the SAME per-shell normalization
    `dvcorr.pipeline.velocity_frame_comparison.normalize_comparison` uses for
    its per-center summaries, WITHOUT the `/n_centers` stacking average
    (there is only one center here, so no stack to average over).

    Monopole: `per_center_speed[a] * per_center_count[a]`, deliberately the
    RAW speed-weighted occupancy for this one center -- no n_bar*V_b
    division. For a single center the question is its own realized
    shell-by-shell shape (how many tracers landed in each shell, times its
    own speed), not a comparison to the expected occupancy; dividing by an
    ensemble-level expectation would not sharpen that reading.

    Parameters
    ----------
    cfg : RedshiftSpaceRunConfig
    results : RedshiftSpaceFrameResults
    n_bar : float
        The SAME shared, box-wide n_bar used for the center-averaged
        figure (e.g. `RedshiftSpaceComparison.n_bar`) -- kept identical so
        the two figures' dipole y-axes are on the same normalization.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If `cfg.example_center_index` is out of range for the shared center
        set's `n_centers`.
    """
    a = cfg.example_center_index
    n_centers = results.real_result.n_centers
    if not (0 <= a < n_centers):
        raise ValueError(
            f"example_center_index ({a}) out of range for n_centers = {n_centers}."
        )

    # results.real_result.shell_centers (the plain midpoint) is deliberately
    # left alone -- r_eff, the volume-weighted radius, is the plotting
    # abscissa; already reading .shell_edges on the next line regardless.
    r = volume_weighted_shell_radii(results.real_result.shell_edges)
    norm_scale = shell_dipole_norm_scale(results.real_result.shell_edges, n_bar)

    real_dipole = results.real_result.per_center_dipole[a] * norm_scale
    redshift_dipole = results.redshift_result.per_center_dipole[a] * norm_scale

    real_mono = results.real_result.per_center_speed[a] * results.real_result.per_center_count[a]
    redshift_mono = (
        results.redshift_result.per_center_speed[a] * results.redshift_result.per_center_count[a]
    )

    fig, (ax_dipole, ax_mono) = plt.subplots(
        2, 1, figsize=_FIGSIZE, sharex=True, gridspec_kw={"height_ratios": _HEIGHT_RATIOS}
    )

    ax_dipole.plot(r, real_dipole, "o-", color=_COLOR_REAL, label="real")
    ax_dipole.plot(r, redshift_dipole, "s-", color=_COLOR_REDSHIFT, label="redshift")
    ax_dipole.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_dipole.set_ylabel(r"$\hat\zeta_1$ (single center)  [km/s]", color=_LABEL_COLOR)
    ax_dipole.legend(fontsize="small")
    ax_dipole.grid(alpha=_GRID_ALPHA)
    ax_dipole.spines["top"].set_visible(False)
    ax_dipole.spines["right"].set_visible(False)
    ax_dipole.set_title(
        f"single-center example (index {a} of {n_centers}, "
        f"{_binning_description(cfg.shells)})",
        fontsize=9,
    )

    ax_mono.plot(r, real_mono, "o-", color=_COLOR_REAL, label="real")
    ax_mono.plot(r, redshift_mono, "s-", color=_COLOR_REDSHIFT, label="redshift")
    ax_mono.set_ylabel(r"$|u_\alpha|\, N_{\alpha,b}$  [km/s]", color=_LABEL_COLOR)
    ax_mono.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
    ax_mono.legend(fontsize="small")
    ax_mono.grid(alpha=_GRID_ALPHA)
    ax_mono.spines["top"].set_visible(False)
    ax_mono.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
