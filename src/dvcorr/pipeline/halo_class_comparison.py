"""
halo_class_comparison.py
------------------------
Both existing comparisons -- observer frame vs. velocity frame
(`dvcorr.pipeline.velocity_frame_comparison`) and real space vs. redshift space
(`dvcorr.pipeline.redshift_space_comparison`) -- run TWICE over, once with the
centers restricted to SUBHALOS and once with them restricted to PARENT halos,
on the same catalog, the same tracers, and the same shells.

Neither estimator, and neither comparison pipeline, is modified or
reimplemented here. This module changes exactly one thing -- WHICH halos are
offered as candidate centers -- and then calls
`velocity_frame_comparison.run_both_frames` and
`redshift_space_comparison.run_both_spaces` unchanged. Everything downstream
of the center set (normalization, nulls, per-center bookkeeping, the two
existing figure builders) is imported, not copied.

The two center classes
----------------------
    subhalo : `pid != -1`, i.e. `not is_distinct`. A satellite: its peculiar
              velocity is dominated by its ORBIT inside a host halo, not by
              the large-scale flow the dipole is trying to measure.
    parent  : `pid == -1` AND at least `min_num_of_subhalos` other halos name
              it as their parent. A top-level host that actually hosts
              something -- the high-mass, group-scale end of the population.

The two classes are DISJOINT by construction, and neither is the whole
catalog: an isolated field halo (distinct, no daughters) is in neither, and is
the population both runs implicitly exclude. That disjointness is the point --
a halo is a satellite, a host, or neither, never two of those, so a difference
between the two curves cannot be a shared-membership artifact.

Reading the expected result: this is a bias/velocity-contamination test, not a
null test. Both classes trace the same underlying velocity field, so a
DIFFERENCE between their dipoles is real physics plus real contamination, and
the two are worth separating. Subhalo centers carry an orbital velocity
component that is uncorrelated with the surrounding density field on the
scales the shells cover, which DILUTES the dipole -- entering noise in the
axis while the tracer geometry is unchanged, exactly the direction
`dvcorr.pipeline.velocity_frame_comparison.random_axis_null_dipoles` randomizes on
purpose. Parent centers should therefore show the LARGER amplitude of the two,
and a subhalo run that does not sit below its parent counterpart is the
surprising outcome, not the expected one.

Why `num_of_subhalos` and not just `is_distinct`
------------------------------------------------
"Is this a subhalo" is free: it is `pid == -1`, a fact about the halo's own
catalog row, and `dvcorr.pipeline.catalog_conversion` has always written it.
"Does this halo HOST subhalos" is not: no property of a halo's own row answers
it, because the answer lives in every OTHER row. It has to be read off the
daughters -- the set of `pid` values appearing anywhere in the catalog IS the
set of halos with at least one subhalo. That is why
`NUM_OF_SUBHALOS_COLUMN` exists, why building it costs the converter a second
pass over the CSV, and why a catalog that was pre-cut to distinct halos
(`CATALOG_MVIR12`) cannot answer the question at all: its subhalos were
removed before it was written, so the evidence is simply not in the file.
`require_classifiable_catalog` refuses such a catalog by name rather than
letting the `NUM_OF_SUBHALOS_UNKNOWN` sentinel be read as a count of zero.

One carve, two classes, shared tracers
---------------------------------------
Both classes are drawn from ONE catalog load (`load_and_carve_buffered`, whose
core arrays `carved_from_buffer` re-presents as the plain carve the frame
comparison wants) and measured against the IDENTICAL tracer field: every
carved halo, subhalos and parents and isolated halos alike. Only the CENTERS
differ. Restricting the tracers as well would change two things at once and
make the resulting difference uninterpretable -- the density field is the
density field, and which halos are used to probe it is a separate question
from what it is made of.

Candidates are drawn AFTER the class filter, not before
--------------------------------------------------------
`select_class_population` filters the carved population down to the class,
and `draw_candidates_from_arrays` then draws `cfg.n_candidate_centers` from
THAT. Drawing first and filtering second would leave each run with only its
class's share of the draw -- roughly 12% for subhalos and a few percent for
parents -- so the parent run would be starved of centers and its wider error
band would be an artifact of the draw rather than a statement about parents.
Filtering first gives both runs the same candidate count and therefore
comparable noise floors, which is the whole basis on which the two curves are
being read against each other.

Matplotlib backend discipline
------------------------------
Importing this module does NOT select a matplotlib backend: `matplotlib.use`
is never called here, only in the thin script's own module body -- identical
discipline to the three pipelines it builds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from dvcorr.config import (
    CATALOG_MVIR12,
    SPACING_LINEAR,
    ShellConfig,
    volume_weighted_shell_radii,
)
from dvcorr.estimators.shell_dipole import (
    center_standard_error,
    expected_shell_occupancy,
    per_center_shell_counts,
)
from dvcorr.pipeline.catalog_conversion import NUM_OF_SUBHALOS_UNKNOWN
from dvcorr.pipeline.redshift_space_comparison import (
    _COLOR_REDSHIFT,
    BufferedCarve,
    RedshiftCenterSet,
    RedshiftSpaceComparison,
    RedshiftSpaceFrameResults,
    RedshiftSpaceRunConfig,
    TracerSpaces,
    normalize_redshift_comparison,
    run_both_spaces,
    select_redshift_shared_centers,
)
from dvcorr.pipeline.redshift_space_comparison import _COLOR_REAL as _COLOR_REAL_SPACE
from dvcorr.pipeline.velocity_centered import (
    _BAND_ALPHA,
    _COLOR_ZERO_LINE,
    _FIGSIZE,
    _GRID_ALPHA,
    _HEIGHT_RATIOS,
    _LABEL_COLOR,
    _ZERO_LINE_WIDTH,
    CarvedHalos,
    SharedCenterSet,
    _binning_description,
    draw_candidates_from_arrays,
    select_shared_centers,
)
from dvcorr.pipeline.velocity_frame_comparison import (
    _COLOR_OBS,
    _COLOR_VEL,
    ComparisonRunConfig,
    FrameComparison,
    FrameRunResults,
    normalize_comparison,
    run_both_frames,
)

#: Centers restricted to satellites: `pid != -1`, i.e. `not is_distinct`.
CENTER_CLASS_SUBHALO: str = "subhalo"

#: Centers restricted to top-level hosts that host something: `pid == -1` AND
#: named as the parent of at least `HaloClassRunConfig.min_num_of_subhalos`
#: other halos.
CENTER_CLASS_PARENT: str = "parent"

#: Every value `HaloClassRunConfig.center_classes` may contain. Named constants
#: rather than bare strings so a typo'd "subhalos" raises in the constructor
#: instead of silently selecting an empty population -- the same guard
#: `dvcorr.config.catalog.VALID_CATALOGS` provides.
VALID_CENTER_CLASSES: frozenset[str] = frozenset({CENTER_CLASS_SUBHALO, CENTER_CLASS_PARENT})

#: Marker per class, for the contrast figures. Class is carried by MARKER and
#: fill, never by linestyle: dashed and dotted already mean "null curve"
#: everywhere else in this project (see `velocity_frame_comparison
#: .make_comparison_figure`), and reusing them for a signal curve here would
#: make the two figure families contradict each other.
_CLASS_MARKER: dict[str, str] = {CENTER_CLASS_PARENT: "o", CENTER_CLASS_SUBHALO: "^"}

#: Marker fill per class: parents solid, subhalos hollow. `None` means "use the
#: line color", matplotlib's default.
_CLASS_FACECOLOR: dict[str, str | None] = {
    CENTER_CLASS_PARENT: None,
    CENTER_CLASS_SUBHALO: "none",
}

_ERRORBAR_CAPSIZE: float = 2.0   # SEM cap width on the contrast figures, points
_CONTRAST_LINEWIDTH: float = 1.4  # signal lines on the contrast figures

#: Default binning for the OCCUPANCY histogram only (`ClassShellOccupancy`) --
#: ten 0.1 h^-1 Mpc shells from 0 to 1. Deliberately unrelated to
#: `RunConfig.shells`, which starts at the estimator's r_min and runs to r_max:
#: this ladder sits entirely INSIDE the innermost dipole shell, where the
#: one-halo term lives and where the two center classes are expected to differ
#: most (a satellite is by definition inside a host's own occupied volume).
#: Linear, not log, so every bar covers the same dr and the bar heights are
#: directly comparable as counts; the r**3 growth of the shell volume is then
#: the only geometric trend in the figure.
_OCCUPANCY_MIN_RADIUS: float = 0.0
_OCCUPANCY_MAX_RADIUS: float = 1.0
_OCCUPANCY_RADII_STEP: float = 0.1

_OCCUPANCY_FIGSIZE: tuple[float, float] = (8.0, 5.0)  # single-panel, so not _FIGSIZE
_OCCUPANCY_BAR_WIDTH_FRACTION: float = 0.8  # of a bin width, shared by all classes
_OCCUPANCY_BAR_EDGE_WIDTH: float = 1.2      # bar outline, points
_OCCUPANCY_EXPECTED_LINEWIDTH: float = 1.4  # the n_bar * V_b reference curve

#: Bar color per class on the occupancy histogram. Class is carried by COLOR
#: here, which the contrast figures deliberately refuse to do -- but only
#: because this figure has no series axis at all (no frames, no spaces, no
#: nulls: one quantity, two populations), so there is nothing for color to
#: collide with. The two hues are chosen clear of `_COLOR_OBS`, `_COLOR_VEL`,
#: `_COLOR_REAL_SPACE` and `_COLOR_REDSHIFT` so a reader never mistakes a bar
#: for a frame or a space. Fill still follows `_CLASS_FACECOLOR`'s convention
#: -- parents solid, subhalos hollow -- so the two channels agree.
_CLASS_BAR_COLOR: dict[str, str] = {
    CENTER_CLASS_PARENT: "#3f5d7d",
    CENTER_CLASS_SUBHALO: "#b5651d",
}


def _default_occupancy_shells() -> ShellConfig:
    """The `[0, 0.1), [0.1, 0.2), ... [0.9, 1.0]` occupancy ladder.

    A `ShellConfig` rather than a bare edge array so the histogram's binning is
    a config object like every other binning in the project, adjustable on a
    `HaloClassRunConfig` instance without touching module constants. `spacing`
    is pinned to `SPACING_LINEAR` explicitly: `min_radius = 0` is legal only
    for linear spacing (`log_shell_edges` is undefined at zero), so this
    factory must not inherit a log default if one is ever introduced.
    """
    return ShellConfig(
        min_radius=_OCCUPANCY_MIN_RADIUS,
        max_radius=_OCCUPANCY_MAX_RADIUS,
        radii_step=_OCCUPANCY_RADII_STEP,
        spacing=SPACING_LINEAR,
    )


@dataclass
class HaloClassRunConfig(ComparisonRunConfig, RedshiftSpaceRunConfig):
    """One config driving BOTH comparisons, so their shared knobs cannot drift.

    Inherits from `ComparisonRunConfig` AND `RedshiftSpaceRunConfig` -- both of
    which extend `dvcorr.pipeline.velocity_centered.RunConfig`, so the MRO
    resolves to a single base and the shared fields (`sub_volume_radius`,
    `shells`, `catalog`, `n_candidate_centers`, `seed`, `shuffle_seed`,
    `gaussian_null_seed`, `min_center_speed`) exist exactly once. That single
    inheritance is the POINT rather than a convenience: this module runs the
    frame comparison and the redshift-space comparison on the same halos, and
    two separate config objects would let one of them silently acquire a
    different shell binning or a different candidate seed, at which point the
    four curves on the two figures would no longer describe the same centers.

    Every comparison-specific field of both parents comes along unchanged
    (`axis_null_seed`, `velocity_gaussian_null_seed`, `v_margin_statistic`,
    `redshift_shuffle_seed`, ...), so each pipeline still gets exactly the
    knobs it documents.

    ONE COLLIDING FIELD, and it is not used here
    --------------------------------------------
    Both parents define `comparison_output_name`, meaning different files.
    Dataclass fields are collected in REVERSE MRO order and later definitions
    win, so `ComparisonRunConfig`'s value ("velocity_frame_comparison.png")
    silently shadows `RedshiftSpaceRunConfig`'s ("redshift_space_comparison.png")
    -- the inherited field is therefore ambiguous by construction and nothing
    in this module reads it. `frame_output_name` and `redshift_output_name`
    below are its unambiguous replacements, and a consumer writing per-class
    figures must use those. The inherited field is left in place rather than
    overridden: overriding it would pick a winner and make the ambiguity
    invisible, whereas leaving it visibly unused keeps the collision findable.
    (The two CONTRAST filenames have no such problem; neither parent defines
    them.)

    Attributes
    ----------
    center_classes : tuple of str
        Which classes to run, in plot order; each must be in
        `VALID_CENTER_CLASSES`. Default is both, parents first so the filled
        marker leads the legend.
    min_num_of_subhalos : int
        How many daughters a distinct halo needs before it counts as a
        `CENTER_CLASS_PARENT`. 1 (the default) is the plain reading of
        "is a parent"; raising it selects progressively richer groups, which
        is the knob to reach for if the parent population turns out to be
        dominated by hosts of a single small satellite. A config attribute
        rather than an inline literal (CLAUDE.md hard rule 4).
    frame_output_name : str
        Filename stem for the PER-CLASS frame-comparison figure
        (`velocity_frame_comparison.make_comparison_figure`), prefixed with the
        class by `class_output_name`. Replaces the ambiguous inherited
        `comparison_output_name` -- see the note above.
    redshift_output_name : str
        The same, for the per-class redshift-space figure
        (`redshift_space_comparison.make_redshift_comparison_figure`).
    frame_contrast_output_name : str
        Output filename for `make_frame_class_contrast_figure`'s PNG, written
        under `dvcorr.config.PathsConfig().output_dir`.
    redshift_contrast_output_name : str
        Output filename for `make_redshift_class_contrast_figure`'s PNG.
    occupancy_shells : ShellConfig
        Binning for the occupancy histogram ALONE (`class_shell_occupancy`,
        `make_class_occupancy_figure`) -- ten 0.1 h^-1 Mpc shells from 0 to 1
        by default. Separate from the inherited `shells` on purpose, and not
        derived from it: this ladder probes the sub-h^-1-Mpc region entirely
        inside the innermost dipole shell, and tying the two together would
        mean the histogram silently re-binned whenever the estimator's shells
        were re-tuned. See `_default_occupancy_shells`.
    occupancy_output_name : str
        Output filename for `make_class_occupancy_figure`'s PNG.
    """

    center_classes: tuple[str, ...] = (CENTER_CLASS_PARENT, CENTER_CLASS_SUBHALO)
    min_num_of_subhalos: int = 1
    frame_output_name: str = "halo_class_frame_comparison.png"
    redshift_output_name: str = "halo_class_redshift_comparison.png"
    frame_contrast_output_name: str = "halo_class_frame_contrast.png"
    redshift_contrast_output_name: str = "halo_class_redshift_contrast.png"
    occupancy_shells: ShellConfig = field(default_factory=_default_occupancy_shells)
    occupancy_output_name: str = "halo_class_shell_occupancy.png"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.center_classes:
            raise ValueError("HaloClassRunConfig: center_classes must not be empty.")
        unknown = [c for c in self.center_classes if c not in VALID_CENTER_CLASSES]
        if unknown:
            raise ValueError(
                f"HaloClassRunConfig: unknown center class(es) {unknown}; expected "
                f"values from {sorted(VALID_CENTER_CLASSES)}."
            )
        if len(set(self.center_classes)) != len(self.center_classes):
            raise ValueError(
                f"HaloClassRunConfig: center_classes {self.center_classes} repeats a "
                "class; each would run twice and plot on top of itself."
            )
        if self.min_num_of_subhalos < 1:
            raise ValueError(
                f"HaloClassRunConfig: min_num_of_subhalos ({self.min_num_of_subhalos}) "
                "must be >= 1 -- a parent with zero daughters is not a parent."
            )


def class_output_name(center_class: str, output_name: str) -> str:
    """Prefix a per-class figure filename with its class.

    `"subhalo", "halo_class_frame_comparison.png"` ->
    `"subhalo_halo_class_frame_comparison.png"`. One definition site so the
    script and the notebook cannot disagree about where a class's figure went,
    and so the two classes' runs cannot silently overwrite each other's PNG --
    which is what a bare `cfg.frame_output_name` would do, the second run
    quietly replacing the first.

    Parameters
    ----------
    center_class : str
    output_name : str
        E.g. `HaloClassRunConfig.frame_output_name`.

    Returns
    -------
    str
    """
    return f"{center_class}_{output_name}"


# ---------------------------------------------------------------------------
# Stage 1: classify halos
# ---------------------------------------------------------------------------


def require_classifiable_catalog(cfg: HaloClassRunConfig) -> None:
    """Refuse a catalog that cannot say which of its halos host subhalos.

    `CATALOG_MVIR12` was cut to `pid == -1` before it was written, so it
    contains no subhalos: it can supply no `CENTER_CLASS_SUBHALO` centers at
    all, and it cannot identify its parents either, because the daughters that
    would have named them are not in the file. `catalog_conversion` records
    that honestly by writing `NUM_OF_SUBHALOS_UNKNOWN` rather than 0 (see its
    docstring), and this check turns that into a refusal at the top of a run
    rather than an empty-population failure several stages down.

    Called for its side effect; `select_class_population` re-checks the
    sentinel on the actual array, so a hand-built config that bypasses this
    still cannot silently read the sentinel as a count.

    Parameters
    ----------
    cfg : HaloClassRunConfig

    Raises
    ------
    RuntimeError
        If `cfg.catalog.name` is `CATALOG_MVIR12`.
    """
    if cfg.catalog.name == CATALOG_MVIR12:
        raise RuntimeError(
            f"halo_class_comparison cannot run on the {CATALOG_MVIR12!r} catalog: it "
            "was cut to distinct halos before it was written, so it contains no "
            "subhalos to use as centers and no daughters from which to identify its "
            "parents (its num_of_subhalos column is the NUM_OF_SUBHALOS_UNKNOWN "
            "sentinel throughout). Use the full catalog, which retains both."
        )


def center_class_mask(
    is_distinct: np.ndarray,
    num_of_subhalos: np.ndarray | None,
    center_class: str,
    *,
    min_num_of_subhalos: int = 1,
) -> np.ndarray:
    """Boolean mask selecting one center class from a halo population.

    The definition site of both classes, and the only place either is spelled
    out:

        subhalo : ~is_distinct
        parent  :  is_distinct & (num_of_subhalos >= min_num_of_subhalos)

    Note that `num_of_subhalos` is consulted for the PARENT class only. The
    subhalo class is a statement about a halo's own `pid` and needs no
    knowledge of anyone else's -- which is exactly the asymmetry this module's
    docstring describes, made concrete. The column is still REQUIRED for both,
    because its sentinel is what proves the catalog is able to answer the
    parent question at all, and running only the subhalo half against a
    catalog that cannot answer the other half would produce a comparison with
    one side missing.

    Sub-subhalos: a subhalo that itself hosts a sub-subhalo has
    `num_of_subhalos >= 1` but `is_distinct == False`, so it lands in the
    subhalo class and NOT in the parent class. The `is_distinct` conjunct is
    what makes the two classes disjoint, and dropping it would put such halos
    in both.

    Parameters
    ----------
    is_distinct : ndarray, shape (N,), bool
        True where the halo is not a subhalo (`pid == -1`).
    num_of_subhalos : ndarray, shape (N,), int | None
        Daughter counts, row-aligned with `is_distinct`.
    center_class : str
        One of `VALID_CENTER_CLASSES`.
    min_num_of_subhalos : int, keyword-only
        Daughter-count floor for `CENTER_CLASS_PARENT`; ignored for
        `CENTER_CLASS_SUBHALO`.

    Returns
    -------
    ndarray, shape (N,), bool

    Raises
    ------
    ValueError
        If `center_class` is not a known class.
    RuntimeError
        If `num_of_subhalos` is None (the catalog was loaded without it -- pass
        `with_num_of_subhalos=True`), if it is not row-aligned with
        `is_distinct`, or if it carries the `NUM_OF_SUBHALOS_UNKNOWN` sentinel
        anywhere (the catalog cannot answer the parent question).
    """
    if center_class not in VALID_CENTER_CLASSES:
        raise ValueError(
            f"center_class_mask: unknown center_class {center_class!r}; expected one "
            f"of {sorted(VALID_CENTER_CLASSES)}."
        )

    is_distinct = np.asarray(is_distinct, dtype=bool)
    if num_of_subhalos is None:
        raise RuntimeError(
            "center_class_mask: num_of_subhalos is None -- the catalog was loaded "
            "without it. Pass with_num_of_subhalos=True to load_and_carve / "
            "load_and_carve_buffered."
        )

    num_of_subhalos = np.asarray(num_of_subhalos)
    if num_of_subhalos.shape != is_distinct.shape:
        raise RuntimeError(
            "center_class_mask: num_of_subhalos "
            f"{num_of_subhalos.shape} is not row-aligned with is_distinct "
            f"{is_distinct.shape}; the two must describe the same halos in the "
            "same order."
        )
    if bool(np.any(num_of_subhalos == NUM_OF_SUBHALOS_UNKNOWN)):
        raise RuntimeError(
            f"center_class_mask: num_of_subhalos holds the NUM_OF_SUBHALOS_UNKNOWN "
            f"sentinel ({NUM_OF_SUBHALOS_UNKNOWN}), meaning the source catalog "
            "contained no subhalos and so could not say which halos host any. "
            "Reading the sentinel as a count would classify every halo as "
            "'not a parent', which is not the same statement. Use the full catalog."
        )

    if center_class == CENTER_CLASS_SUBHALO:
        return ~is_distinct
    return is_distinct & (num_of_subhalos >= min_num_of_subhalos)


@dataclass(frozen=True)
class ClassPopulation:
    """The carved halos of ONE center class, ready to draw candidates from.

    Built by `select_class_population`. Holds only what
    `draw_candidates_from_arrays` needs plus the counts that make the
    class-selection funnel reportable -- `num_of_subhalos` itself is
    deliberately NOT carried past this point: once the population has been
    filtered every member is in the class by construction, so keeping the
    column would only invite a second, independently derived classification
    downstream that could drift from this one.

    Attributes
    ----------
    center_class : str
    pos : ndarray, shape (N_class, 3), float64
    vel : ndarray, shape (N_class, 3), float64
    mvir : ndarray, shape (N_class,)
    is_distinct : ndarray, shape (N_class,), bool
        Uniform within a class (all False for subhalos, all True for parents),
        carried anyway because `select_shared_centers` accepts it as a
        pass-through label and dropping it here would make the center set's
        `is_distinct_centers` None for no reason.
    n_class : int
        `pos.shape[0]` -- how many carved halos are in this class.
    n_source : int
        How many carved halos were offered, before the class filter, for the
        funnel's fraction.
    """

    center_class: str
    pos: np.ndarray
    vel: np.ndarray
    mvir: np.ndarray
    is_distinct: np.ndarray
    n_class: int
    n_source: int

    def describe(self) -> str:
        """One-line human-readable summary, for run logs and plot titles."""
        return (
            f"{self.center_class}: {self.n_class:,} of {self.n_source:,} carved halos "
            f"({100.0 * self.n_class / max(self.n_source, 1):.2f}%)"
        )


def select_class_population(
    cfg: HaloClassRunConfig,
    center_class: str,
    pos: np.ndarray,
    vel: np.ndarray,
    mvir: np.ndarray,
    is_distinct: np.ndarray,
    num_of_subhalos: np.ndarray | None,
) -> ClassPopulation:
    """Filter a carved population down to one center class.

    Takes bare arrays rather than a `CarvedHalos` for the same reason
    `draw_candidates_from_arrays` does: the redshift-space path's population
    arrives as a `BufferedCarve`'s core arrays, not as a `CarvedHalos`, and one
    implementation serving both is what stops the two paths drifting into
    classifying differently.

    Parameters
    ----------
    cfg : HaloClassRunConfig
        Supplies `min_num_of_subhalos`.
    center_class : str
        One of `VALID_CENTER_CLASSES`.
    pos, vel : ndarray, shape (N_carved, 3)
    mvir : ndarray, shape (N_carved,)
    is_distinct : ndarray, shape (N_carved,), bool
    num_of_subhalos : ndarray, shape (N_carved,), int | None
        All row-aligned with `pos`.

    Returns
    -------
    ClassPopulation

    Raises
    ------
    RuntimeError
        Propagated from `center_class_mask`, or raised here if the class is
        empty -- a zero-halo population would otherwise surface as a confusing
        zero-candidate failure inside `draw_candidates_from_arrays`.
    """
    mask = center_class_mask(
        is_distinct,
        num_of_subhalos,
        center_class,
        min_num_of_subhalos=cfg.min_num_of_subhalos,
    )
    population = ClassPopulation(
        center_class=center_class,
        pos=pos[mask],
        vel=vel[mask],
        mvir=np.asarray(mvir)[mask],
        is_distinct=np.asarray(is_distinct, dtype=bool)[mask],
        n_class=int(mask.sum()),
        n_source=int(mask.size),
    )
    print(f"class filter -> {population.describe()}")
    if population.n_class == 0:
        raise RuntimeError(
            f"select_class_population: no carved halo is in class {center_class!r} "
            f"(min_num_of_subhalos = {cfg.min_num_of_subhalos}); widen "
            "sub_volume_radius, lower min_num_of_subhalos, or check the catalog cuts."
        )
    return population


def carved_from_buffer(buffer: BufferedCarve) -> CarvedHalos:
    """Re-present a `BufferedCarve`'s core arrays as the plain `CarvedHalos`.

    `BufferedCarve.pos_core` / `.vel_core` / `.mvir_core` / `.is_distinct_core`
    / `.num_of_subhalos_core` ARE the plain `cfg.sub_volume_radius` carve --
    `load_and_carve_buffered` computes exactly that as its first pass, in order
    to derive `v_margin` from it, before carving the wider tracer buffer. So
    the population `dvcorr.pipeline.velocity_frame_comparison` wants is already
    in hand, and calling `load_and_carve` afterwards would re-read the same
    11 GB catalog to rebuild an array this object already holds.

    This is a REPACKAGING, not a second selection: no cut is applied, no row is
    dropped or reordered, and the two carves are the same code path
    (`d_obs <= cfg.sub_volume_radius`) against the same loaded arrays. The
    saving is one full catalog load per run, which on the full catalog is a
    couple of minutes and ~16 GB of transient memory.

    `catalog_mvir` and `n_total` come straight across, so the mass funnel and
    `box_number_density` see exactly what they would have seen from
    `load_and_carve`.

    Parameters
    ----------
    buffer : BufferedCarve
        From `load_and_carve_buffered`, ideally with
        `with_num_of_subhalos=True`.

    Returns
    -------
    CarvedHalos
    """
    return CarvedHalos(
        pos=buffer.pos_core,
        vel=buffer.vel_core,
        mvir=buffer.mvir_core,
        is_distinct=buffer.is_distinct_core,
        n_carved=buffer.n_core,
        n_total=buffer.n_total,
        catalog_mvir=buffer.catalog_mvir,
        num_of_subhalos=buffer.num_of_subhalos_core,
    )


# ---------------------------------------------------------------------------
# Stage 2: run each comparison, once per class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassFrameRun:
    """One center class's complete observer-frame vs. velocity-frame run.

    Attributes
    ----------
    center_class : str
    population : ClassPopulation
        The carved halos this class contributed, before the candidate draw.
    centers : SharedCenterSet
        The centers BOTH frames were run on -- shared within the class, not
        across classes (the two classes are disjoint populations, so there is
        no cross-class center set to share and none is implied).
    results : FrameRunResults
    comparison : FrameComparison
    """

    center_class: str
    population: ClassPopulation
    centers: SharedCenterSet
    results: FrameRunResults
    comparison: FrameComparison


def run_frame_comparison_for_class(
    cfg: HaloClassRunConfig,
    center_class: str,
    carved: CarvedHalos,
    observer: np.ndarray,
    n_bar: float,
) -> ClassFrameRun:
    """Run `velocity_frame_comparison` end to end for ONE center class.

    Four stages, all of them imported: class filter
    (`select_class_population`), candidate draw
    (`draw_candidates_from_arrays`), center selection
    (`select_shared_centers`, `core_margin` left at its default r_max exactly
    as `velocity_frame_comparison` does), then
    `run_both_frames` + `normalize_comparison` untouched.

    TRACERS ARE THE FULL CARVE, not the class population: `carved.pos` is
    passed to `run_both_frames`, so the density field both frames measure is
    every halo inside the sub-volume regardless of class. See this module's
    docstring -- restricting the tracers too would vary two things at once.

    Parameters
    ----------
    cfg : HaloClassRunConfig
    center_class : str
        One of `VALID_CENTER_CLASSES`.
    carved : CarvedHalos
        The full carved population, loaded with `with_num_of_subhalos=True`
        (or repackaged by `carved_from_buffer`).
    observer : ndarray, shape (3,)
    n_bar : float
        Mean halo number density over the whole box, shared by every class --
        `dvcorr.pipeline.velocity_centered.box_number_density(carved.n_total)`.
        Taken as an argument rather than derived here so both classes are
        visibly normalized by the same number; a per-class n_bar would fold
        the class's own abundance into the amplitude, which is precisely the
        quantity being compared.

    Returns
    -------
    ClassFrameRun
    """
    print(f"\n--- frame comparison, centers = {center_class} ---")
    population = select_class_population(
        cfg, center_class, carved.pos, carved.vel, carved.mvir,
        carved.is_distinct, carved.num_of_subhalos,
    )
    candidates = draw_candidates_from_arrays(
        cfg, population.pos, population.vel, population.mvir, population.is_distinct
    )
    centers = select_shared_centers(
        cfg,
        candidates.s,
        candidates.v,
        observer,
        mvir_candidates=candidates.mvir,
        is_distinct_candidates=candidates.is_distinct,
    )
    results = run_both_frames(cfg, centers, carved.pos, observer)
    comparison = normalize_comparison(cfg, results, n_bar)

    return ClassFrameRun(
        center_class=center_class,
        population=population,
        centers=centers,
        results=results,
        comparison=comparison,
    )


@dataclass(frozen=True)
class ClassRedshiftRun:
    """One center class's complete real-space vs. redshift-space run.

    Attributes
    ----------
    center_class : str
    population : ClassPopulation
    centers : RedshiftCenterSet
        Row-aligned real and displaced center positions for this class.
    results : RedshiftSpaceFrameResults
    comparison : RedshiftSpaceComparison
    """

    center_class: str
    population: ClassPopulation
    centers: RedshiftCenterSet
    results: RedshiftSpaceFrameResults
    comparison: RedshiftSpaceComparison


def run_redshift_comparison_for_class(
    cfg: HaloClassRunConfig,
    center_class: str,
    buffer: BufferedCarve,
    tracers: TracerSpaces,
    observer: np.ndarray,
    n_bar: float,
) -> ClassRedshiftRun:
    """Run `redshift_space_comparison` end to end for ONE center class.

    Same shape as `run_frame_comparison_for_class`, with that pipeline's own
    center-selection stage in place of the plain one: candidates come from the
    CORE carve (`buffer.pos_core`, never the wider tracer buffer -- a candidate
    center must be a real member of the analysis sub-volume), and
    `select_redshift_shared_centers` applies the widened `r_max + v_margin`
    real-position margin, the optional global |v_r| cut, and the
    through-observer flip guard, all unchanged.

    `tracers` is shared across classes and built ONCE, by `build_tracer_spaces`
    from the full buffer: the redshift-space displacement of the tracer field
    has nothing to do with which halos are being used as centers, so rebuilding
    it per class would repeat several minutes of work to get an identical
    answer.

    Parameters
    ----------
    cfg : HaloClassRunConfig
    center_class : str
    buffer : BufferedCarve
        From `load_and_carve_buffered(..., with_num_of_subhalos=True)`.
    tracers : TracerSpaces
        From `build_tracer_spaces`, shared by every class.
    observer : ndarray, shape (3,)
    n_bar : float
        `box_number_density(buffer.n_total)` -- the same box-wide, class-blind
        normalization both spaces and both classes use.

    Returns
    -------
    ClassRedshiftRun
    """
    print(f"\n--- redshift-space comparison, centers = {center_class} ---")
    population = select_class_population(
        cfg, center_class, buffer.pos_core, buffer.vel_core, buffer.mvir_core,
        buffer.is_distinct_core, buffer.num_of_subhalos_core,
    )
    candidates = draw_candidates_from_arrays(
        cfg, population.pos, population.vel, population.mvir, population.is_distinct
    )
    centers = select_redshift_shared_centers(
        cfg,
        candidates.s,
        candidates.v,
        observer,
        buffer.v_margin_kms,
        buffer.v_margin_mpc,
        mvir_candidates=candidates.mvir,
        is_distinct_candidates=candidates.is_distinct,
    )
    results = run_both_spaces(cfg, centers, tracers, observer)
    comparison = normalize_redshift_comparison(cfg, results, n_bar)

    return ClassRedshiftRun(
        center_class=center_class,
        population=population,
        centers=centers,
        results=results,
        comparison=comparison,
    )


# ---------------------------------------------------------------------------
# Stage 3: small-scale shell occupancy, per class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassShellOccupancy:
    """Mean tracer occupancy per shell, averaged ACROSS one class's centers.

    A pure-geometry companion to the two comparisons -- no velocities enter it
    at all -- measured on `HaloClassRunConfig.occupancy_shells` rather than on
    the estimator's own shells: the sub-h^-1-Mpc region inside the innermost
    dipole shell, which is where a subhalo center and a parent center have the
    most obviously different neighborhoods and which no dipole figure shows.

    Attributes
    ----------
    center_class : str
    shell_edges : ndarray, shape (B + 1,)
        The occupancy binning used, h^-1 Mpc.
    mean_count : ndarray, shape (B,)
        <N_alpha,b> over this class's centers -- the histogram's bar heights.
    sem_count : ndarray, shape (B,)
        Across-center standard error of `mean_count`
        (`center_standard_error`), and carrying that function's caveat
        unchanged: centers are not independent samples, so this UNDERSTATES
        the true uncertainty.
    expected_count : ndarray, shape (B,)
        n_bar * V_b, the occupancy the same shells would hold in a uniform
        field at the cosmic mean density (`expected_shell_occupancy`). Not a
        normalization here, just the reference the bars are read against:
        `mean_count / expected_count` is 1 + xi_hh(r) averaged over the shell,
        so on these scales it is the one-halo term made visible.
    n_centers : int
        Number of centers averaged over.
    """

    center_class: str
    shell_edges: np.ndarray
    mean_count: np.ndarray
    sem_count: np.ndarray
    expected_count: np.ndarray
    n_centers: int


def class_shell_occupancy(
    cfg: HaloClassRunConfig,
    center_class: str,
    s_centers: np.ndarray,
    tracer_pos: np.ndarray,
    n_bar: float,
) -> ClassShellOccupancy:
    """Average tracer occupancy per fine shell over one class's centers.

    Thin: `per_center_shell_counts` does the counting, `center_standard_error`
    the error bar, `expected_shell_occupancy` the reference. This function
    exists to bind them to `cfg.occupancy_shells` and to one center class, so
    the script and the notebook cannot bin the same histogram differently.

    TRACERS ARE THE FULL CARVE, exactly as in
    `run_frame_comparison_for_class`: the neighbors being counted are every
    halo in the sub-volume, not the class's own halos. Restricting them would
    answer a different question ("how many parents surround a parent") and
    would no longer describe the density field the estimators actually see.

    NO CORE CUT is applied. The centers passed in have already been through
    `select_shared_centers`' r_max clearance, which is far larger than
    `cfg.occupancy_shells.max_radius`; every occupancy shell therefore sits
    well inside the carve, with no boundary deficit to correct for.

    Parameters
    ----------
    cfg : HaloClassRunConfig
        Only `cfg.occupancy_shells` is read.
    center_class : str
        Recorded on the result; not used to filter anything, since
        `s_centers` is already this class's center set.
    s_centers : ndarray, shape (N_c, 3)
        The centers to average over -- `SharedCenterSet.s_centers` from the
        frame run, or `RedshiftCenterSet.s_centers_real` from the redshift
        run.
    tracer_pos : ndarray, shape (N_t, 3)
        Tracer positions, comoving h^-1 Mpc.
    n_bar : float
        Mean halo number density over the whole box, shared by every class --
        the same number `run_frame_comparison_for_class` normalizes with.

    Returns
    -------
    ClassShellOccupancy
    """
    edges = cfg.occupancy_shells.shell_edges
    counts = per_center_shell_counts(s_centers, tracer_pos, edges)

    return ClassShellOccupancy(
        center_class=center_class,
        shell_edges=edges,
        mean_count=counts.mean(axis=0),
        sem_count=center_standard_error(counts),
        expected_count=expected_shell_occupancy(n_bar, edges),
        n_centers=counts.shape[0],
    )


def class_shell_occupancy_for_runs(
    cfg: HaloClassRunConfig,
    runs: dict[str, ClassFrameRun],
    tracer_pos: np.ndarray,
    n_bar: float,
) -> dict[str, ClassShellOccupancy]:
    """`class_shell_occupancy` over a whole dict of frame runs, in plot order.

    Reuses the centers each class's frame run already selected rather than
    re-drawing them, so the histogram describes exactly the halos the frame
    contrast figure's curves were measured on.

    Parameters
    ----------
    cfg : HaloClassRunConfig
    runs : dict of str -> ClassFrameRun
        Keyed by center class, in plot order. `ClassRedshiftRun` is NOT
        accepted here: its `RedshiftCenterSet` carries `s_centers_real` and
        `s_centers_redshift` rather than a single `s_centers`, so which of the
        two to count around is a choice the caller must make explicitly by
        calling `class_shell_occupancy` itself.
    tracer_pos : ndarray, shape (N_t, 3)
    n_bar : float

    Returns
    -------
    dict of str -> ClassShellOccupancy
    """
    return {
        center_class: class_shell_occupancy(
            cfg, center_class, run.centers.s_centers, tracer_pos, n_bar
        )
        for center_class, run in runs.items()
    }


# ---------------------------------------------------------------------------
# Figures -- the CONTRAST between classes
# ---------------------------------------------------------------------------
#
# The per-class figures are NOT rebuilt here: `velocity_frame_comparison
# .make_comparison_figure` and `redshift_space_comparison
# .make_redshift_comparison_figure` already draw one class's full result,
# nulls and all, and a consumer calls them once per class. What is genuinely
# new -- and therefore the only thing defined below -- is the figure that puts
# the two classes side by side with the nulls suppressed, which no existing
# builder can produce.


def _plot_class_curves(
    ax: plt.Axes,
    r: np.ndarray,
    runs: dict[str, object],
    curve: str,
    color: str,
    series_label: str,
    *,
    with_sem: bool = True,
) -> None:
    """Draw one series (a frame, or a space) for every class on one axis.

    Factored out because the two contrast figures draw the identical four
    curves with only the attribute name and color changing; two copies of this
    loop is exactly the duplication CLAUDE.md's graduation rule exists to
    prevent.

    Class is carried by marker and fill (`_CLASS_MARKER`, `_CLASS_FACECOLOR`),
    series by color, so a reader separates "which frame" from "which halos"
    without consulting the legend twice. SEM is drawn as an errorbar rather
    than the `fill_between` band the single-class figures use: four
    overlapping translucent bands in two colors are unreadable, whereas four
    errorbar sets stay distinguishable.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    r : ndarray, shape (B,)
        Volume-weighted shell radii, the plotting abscissa.
    runs : dict of str -> ClassFrameRun | ClassRedshiftRun
        Keyed by center class, in plot order.
    curve : str
        Attribute name on `run.comparison` holding this series'
        `NormalizedDipole` -- "obs"/"vel", or "real"/"redshift".
    color : str
    series_label : str
        Human-readable series name for the legend, e.g. "obs".
    with_sem : bool, keyword-only
        Draw the across-center standard error. False for the monopole panel,
        where `NormalizedDipole` carries no monopole SEM.
    """
    for center_class, run in runs.items():
        normalized = getattr(run.comparison, curve)
        marker = _CLASS_MARKER[center_class]
        facecolor = _CLASS_FACECOLOR[center_class]
        label = f"{series_label} · {center_class} (N$_c$={run.centers.n_centers})"
        if with_sem:
            ax.errorbar(
                r, normalized.zeta_hat, yerr=normalized.sem,
                fmt=f"{marker}-", color=color, markerfacecolor=facecolor,
                lw=_CONTRAST_LINEWIDTH, capsize=_ERRORBAR_CAPSIZE, alpha=1.0 - _BAND_ALPHA,
                label=label,
            )
        else:
            ax.plot(
                r, normalized.monopole_norm, f"{marker}-", color=color,
                markerfacecolor=facecolor, lw=_CONTRAST_LINEWIDTH, label=label,
            )


def make_frame_class_contrast_figure(
    cfg: HaloClassRunConfig, runs: dict[str, ClassFrameRun]
) -> plt.Figure:
    """Subhalo-centered vs. parent-centered, in both velocity frames.

    Two panels sharing the x-axis, height ratios (3, 1) -- the layout
    convention every figure in this project follows, and CLAUDE.md hard rule 6:
    the dipole is never plotted without its monopole.

    TOP panel: four zeta_hat curves, {obs, vel} x {parent, subhalo}, each with
    its across-center SEM errorbar. Color is the FRAME (blue observer, purple
    velocity -- the identical hues `velocity_frame_comparison
    .make_comparison_figure` uses, so the two figures read as one family) and
    marker is the CLASS (filled circle parent, hollow triangle subhalo).

    NO NULL CURVES. Each class's own four nulls are on its own
    `make_comparison_figure`, which a consumer calls alongside this one;
    reproducing eight of them here would leave twelve curves on one axis and
    obscure the single comparison this figure exists to make. The nulls are
    what say whether a curve is distinguishable from zero; THIS figure asks
    the different question of whether the two classes are distinguishable from
    each other.

    What to look for, and what it would mean. A subhalo curve sitting BELOW
    its parent counterpart in amplitude is the expected result, and the
    mechanism is dilution rather than cancellation: a satellite's velocity is
    dominated by its orbit inside its host, which is uncorrelated with the
    density field on shell scales, so it enters the stack as a near-random
    axis carrying a real speed -- the same thing
    `velocity_frame_comparison.random_axis_null_dipoles` does deliberately, and it
    drives the statistic toward zero without changing the tracer geometry. The
    gap between the classes is therefore a rough measure of how much orbital
    velocity is contaminating a mixed-population run.

    Two readings that are NOT available from this figure. First, the two
    classes have different MASS distributions -- parents are group-scale hosts,
    subhalos are typically far lighter -- so part of any gap is halo bias, not
    orbital contamination, and this figure cannot separate the two (a
    mass-matched run is what would). Second, the SEM caveat applies to every
    curve here exactly as elsewhere (`dvcorr.estimators.shell_dipole
    .center_standard_error`): centers are not independent samples, so all four
    error bars UNDERSTATE the true uncertainty, and a class gap of order one
    error bar is not evidence of anything.

    BOTTOM panel (monopole): the obs-frame monopole on the LEFT y-axis and the
    velocity-frame monopole on the RIGHT, twinned, for the same reason
    `make_comparison_figure` twins them -- the two frames' monopoles sit near
    <|u|> and <|v|> respectively, a projection factor apart, and one shared
    axis would compress the smaller and hide the r-dependent trend that is the
    finite-distance diagnostic. Both classes are drawn on both axes, so the
    panel also shows directly whether the two classes' shell occupancy differs,
    which is the first thing to check before believing any dipole gap: parents
    live in denser environments and will show the higher monopole regardless of
    velocities.

    Builder only: never saves, never calls `plt.show`.

    Parameters
    ----------
    cfg : HaloClassRunConfig
    runs : dict of str -> ClassFrameRun
        Keyed by center class, in plot order. Every value must have been run
        on the same `shell_edges` -- guaranteed when they share one `cfg`.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If `runs` is empty.
    """
    if not runs:
        raise ValueError("make_frame_class_contrast_figure: runs is empty.")

    any_run = next(iter(runs.values()))
    r = volume_weighted_shell_radii(any_run.results.obs_result.shell_edges)

    fig, (ax_dipole, ax_mono_obs) = plt.subplots(
        2, 1, figsize=_FIGSIZE, sharex=True, gridspec_kw={"height_ratios": _HEIGHT_RATIOS}
    )

    _plot_class_curves(ax_dipole, r, runs, "obs", _COLOR_OBS, "obs")
    _plot_class_curves(ax_dipole, r, runs, "vel", _COLOR_VEL, "vel")

    ax_dipole.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_dipole.set_ylabel(r"$\hat\zeta_1$  [km/s]", color=_LABEL_COLOR)
    ax_dipole.legend(fontsize="small")
    ax_dipole.grid(alpha=_GRID_ALPHA)
    ax_dipole.spines["top"].set_visible(False)
    ax_dipole.spines["right"].set_visible(False)
    ax_dipole.set_title(
        "center halo class: subhalo vs. parent, both frames  "
        f"(R_sub={cfg.sub_volume_radius:.0f} h$^{{-1}}$Mpc, "
        f"{_binning_description(cfg.shells)}, "
        f"parent = distinct with $\\geq${cfg.min_num_of_subhalos} subhalo)\n"
        "nulls are on the per-class figures; SEM bars UNDERSTATE the true "
        "uncertainty (center_standard_error)",
        fontsize=9,
    )

    ax_mono_vel = ax_mono_obs.twinx()
    _plot_class_curves(ax_mono_obs, r, runs, "obs", _COLOR_OBS, "obs", with_sem=False)
    _plot_class_curves(ax_mono_vel, r, runs, "vel", _COLOR_VEL, "vel", with_sem=False)

    ax_mono_obs.set_ylabel(
        r"$\hat\zeta_0^{obs} \simeq \langle|u|\rangle$  [km/s]", color=_COLOR_OBS
    )
    ax_mono_obs.tick_params(axis="y", labelcolor=_COLOR_OBS)
    ax_mono_obs.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
    ax_mono_obs.grid(alpha=_GRID_ALPHA)
    ax_mono_obs.spines["top"].set_visible(False)

    ax_mono_vel.set_ylabel(r"$\hat\zeta_0^{vel}$  [km/s]", color=_COLOR_VEL)
    ax_mono_vel.tick_params(axis="y", labelcolor=_COLOR_VEL)
    ax_mono_vel.spines["top"].set_visible(False)

    fig.tight_layout()
    return fig


def make_redshift_class_contrast_figure(
    cfg: HaloClassRunConfig, runs: dict[str, ClassRedshiftRun]
) -> plt.Figure:
    """Subhalo-centered vs. parent-centered, in real and redshift space.

    The redshift-space twin of `make_frame_class_contrast_figure`, and read the
    same way: color is the SPACE (blue real, green redshift -- the hues
    `redshift_space_comparison.make_redshift_comparison_figure` uses), marker
    is the CLASS, nulls are omitted and live on the per-class figures.

    TOP panel: four zeta_hat curves, {real, redshift} x {parent, subhalo}, with
    SEM errorbars. The quantity of interest is a RATIO OF RATIOS -- how much
    redshift-space distortion costs each class -- rather than any single curve:
    if the redshift/real gap is wider for subhalos than for parents, the extra
    loss is the satellites' own orbital velocity smearing them along the line
    of sight, which is the finger-of-God effect reaching the estimator through
    the center positions.

    BOTTOM panel (monopole): a SINGLE shared y-axis, unlike the frame contrast
    figure's twinned pair. Both spaces weight by the same `|u_alpha|` on the
    same centers (n_hat invariance -- the displaced center's line of sight is
    the same line, so u is unchanged), so the two monopoles are the same kind
    of quantity and differ only through shell occupancy; there is no
    projection-factor scale gap for a twin axis to rescue. Any trend
    difference here is occupancy alone, which is exactly the redshift-space
    trust diagnostic.

    Builder only: never saves, never calls `plt.show`.

    Parameters
    ----------
    cfg : HaloClassRunConfig
    runs : dict of str -> ClassRedshiftRun
        Keyed by center class, in plot order.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If `runs` is empty.
    """
    if not runs:
        raise ValueError("make_redshift_class_contrast_figure: runs is empty.")

    any_run = next(iter(runs.values()))
    r = volume_weighted_shell_radii(any_run.results.real_result.shell_edges)

    fig, (ax_dipole, ax_mono) = plt.subplots(
        2, 1, figsize=_FIGSIZE, sharex=True, gridspec_kw={"height_ratios": _HEIGHT_RATIOS}
    )

    _plot_class_curves(ax_dipole, r, runs, "real", _COLOR_REAL_SPACE, "real")
    _plot_class_curves(ax_dipole, r, runs, "redshift", _COLOR_REDSHIFT, "redshift")

    ax_dipole.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_dipole.set_ylabel(r"$\hat\zeta_1$  [km/s]", color=_LABEL_COLOR)
    ax_dipole.legend(fontsize="small")
    ax_dipole.grid(alpha=_GRID_ALPHA)
    ax_dipole.spines["top"].set_visible(False)
    ax_dipole.spines["right"].set_visible(False)
    ax_dipole.set_title(
        "center halo class: subhalo vs. parent, real vs. redshift space  "
        f"(R_sub={cfg.sub_volume_radius:.0f} h$^{{-1}}$Mpc, "
        f"{_binning_description(cfg.shells)}, "
        f"v_margin={any_run.centers.v_margin_kms:.0f} km/s)\n"
        "nulls are on the per-class figures; SEM bars UNDERSTATE the true "
        "uncertainty (center_standard_error)",
        fontsize=9,
    )

    _plot_class_curves(ax_mono, r, runs, "real", _COLOR_REAL_SPACE, "real", with_sem=False)
    _plot_class_curves(
        ax_mono, r, runs, "redshift", _COLOR_REDSHIFT, "redshift", with_sem=False
    )

    ax_mono.set_ylabel(r"$\hat\zeta_0 \simeq \langle|u|\rangle$  [km/s]", color=_LABEL_COLOR)
    ax_mono.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
    ax_mono.legend(fontsize="small")
    ax_mono.grid(alpha=_GRID_ALPHA)
    ax_mono.spines["top"].set_visible(False)
    ax_mono.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def make_class_occupancy_figure(
    cfg: HaloClassRunConfig,
    occupancies: dict[str, ClassShellOccupancy],
    *,
    log_y: bool = False,
) -> plt.Figure:
    """Mean tracer count per fine shell, one bar group per shell, per class.

    A histogram, not a curve: `cfg.occupancy_shells` gives ten 0.1 h^-1 Mpc
    shells from 0 to 1, each class's bars drawn side by side within a shell so
    the classes are compared at fixed r rather than across it. Bar height is
    <N_alpha,b> across that class's centers, with the across-center SEM as the
    error bar (understated, per `center_standard_error`).

    The dashed reference line is `n_bar * V_b`, the occupancy of the same shell
    in a uniform field at the cosmic mean density. It is what makes the bars
    mean something: a bar sitting three decades above it says the neighborhood
    is ~1000x overdense on that scale, which on sub-h^-1-Mpc shells is the
    one-halo term and nothing else.

    WHY THIS ISN'T A DIPOLE FIGURE, and why hard rule 6 is not in play: there
    is no dipole here to pair a monopole with. This is the pure counting
    geometry, velocities never entering, and it is the diagnostic BEHIND the
    monopole panels on the two contrast figures rather than a competitor to
    them -- those show occupancy on the estimator's shells, folded through the
    speed weight; this shows it raw, on scales none of those shells reach.

    What to read. Subhalo centers should carry the far higher count at small r,
    by construction: a satellite sits inside a host halo, surrounded by that
    host's other satellites, whereas a parent center at these radii sees only
    its own. The size of that gap is a direct measure of how much one-halo
    structure a subhalo-centered run is sitting in, and therefore of how much
    of the dipole dilution the contrast figures show is a small-scale
    environmental effect rather than a purely kinematic one.

    Builder only: never saves, never calls `plt.show`.

    Parameters
    ----------
    cfg : HaloClassRunConfig
    occupancies : dict of str -> ClassShellOccupancy
        Keyed by center class, in plot order. Every value must share one
        `shell_edges` -- guaranteed when they come from one `cfg`.
    log_y : bool, keyword-only, default False
        Log-scale the count axis. LINEAR by default, as everywhere else in
        this project, and here it is not merely a house rule: the r**3 growth
        of the shell volume is largely cancelled by the falling correlation
        function, so the measured counts span barely a decade over the whole
        ladder (0.26 to 4.1 on the full MDPL2 carve) and the bars are directly
        comparable by eye. Only the `n_bar * V_b` reference dives out of
        range, and it stays legible even so.

        Set True to read the BAR-TO-REFERENCE ratio -- 1 + xi_hh(r) -- off the
        figure instead of off `expected_count`, which a linear axis compresses
        away. The x-axis stays linear either way: the binning is linear, and a
        log abscissa would misrepresent equal-width bars.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If `occupancies` is empty, or if two classes were binned differently.
    """
    if not occupancies:
        raise ValueError("make_class_occupancy_figure: occupancies is empty.")

    any_occupancy = next(iter(occupancies.values()))
    edges = np.asarray(any_occupancy.shell_edges, dtype=float)
    for occupancy in occupancies.values():
        if not np.array_equal(np.asarray(occupancy.shell_edges, dtype=float), edges):
            raise ValueError(
                "make_class_occupancy_figure: classes were binned differently "
                f"({occupancy.center_class} disagrees with "
                f"{any_occupancy.center_class}); they must share one cfg."
            )

    left = edges[:-1]
    widths = np.diff(edges)
    n_classes = len(occupancies)
    bar_width = _OCCUPANCY_BAR_WIDTH_FRACTION * widths / n_classes

    fig, ax = plt.subplots(figsize=_OCCUPANCY_FIGSIZE)

    for position, (center_class, occupancy) in enumerate(occupancies.items()):
        color = _CLASS_BAR_COLOR[center_class]
        facecolor = _CLASS_FACECOLOR[center_class]
        # Bars of the `position`-th class sit `position` slots into the
        # centered group: the group spans `_OCCUPANCY_BAR_WIDTH_FRACTION` of
        # the bin, leaving the rest as the gap between neighboring shells.
        offset = (
            0.5 * widths * (1.0 - _OCCUPANCY_BAR_WIDTH_FRACTION)
            + (position + 0.5) * bar_width
        )
        ax.bar(
            left + offset,
            occupancy.mean_count,
            width=bar_width,
            yerr=occupancy.sem_count,
            color=color if facecolor is None else facecolor,
            edgecolor=color,
            linewidth=_OCCUPANCY_BAR_EDGE_WIDTH,
            ecolor=color,
            capsize=_ERRORBAR_CAPSIZE,
            label=f"{center_class} (N$_c$={occupancy.n_centers})",
        )

    ax.plot(
        0.5 * (edges[:-1] + edges[1:]),
        any_occupancy.expected_count,
        "--",
        color=_COLOR_ZERO_LINE,
        lw=_OCCUPANCY_EXPECTED_LINEWIDTH,
        label=r"uniform field  $\bar{n}\,V_b$",
    )

    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks(edges)
    ax.set_xlabel(r"separation $r$  [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"mean tracers per shell  $\langle N_{\alpha,b}\rangle$")
    ax.legend(fontsize="small")
    ax.grid(alpha=_GRID_ALPHA, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "small-scale shell occupancy by center class  "
        f"(R_sub={cfg.sub_volume_radius:.0f} h$^{{-1}}$Mpc, "
        f"{_binning_description(cfg.occupancy_shells)}, "
        f"parent = distinct with $\\geq${cfg.min_num_of_subhalos} subhalo)\n"
        "tracers are the FULL carve; self-pairs excluded; SEM bars UNDERSTATE "
        "the true uncertainty",
        fontsize=9,
    )

    fig.tight_layout()
    return fig
