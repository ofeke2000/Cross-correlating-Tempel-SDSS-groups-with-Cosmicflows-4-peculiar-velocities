"""
robustness.py
-------------
Is a feature in zeta_hat_1(r) REAL, or is it one draw of a noisy statistic?

This module answers that question for the velocity-centered shell dipole, and
it exists because of a specific near-miss. On the constrained Gadget-4 box
(`notebooks/DM_simulation.ipynb`), the innermost `[0, min_radius)` bin read
301.9 km/s on one center set and 80.4 km/s on a 58% subset of the SAME
centers, measured with the same estimator on the same tracer field. Read as a
curve, the second looked like a physical collapse of the correlation inside the
virial radius. It was not: at 33x more centers that bin sits ABOVE the inward
continuation of its neighbours. The difference was sampling scatter in a bin
whose per-center contributions are heavy-tailed enough that a few thousand
centers do not pin down their own mean.

Nothing here is specific to that bin or that box. The five tests below are the
general recipe for deciding whether any shell's value is a measurement.

THE FIVE TESTS
--------------
1. PRECISION LADDER (`precision_ladder`) -- does the scatter fall as
   1/sqrt(N_c)? Draw random center subsets of increasing size and watch both
   the across-draw spread and the internal standard error. If they agree and
   both track 1/sqrt(N_c), the error bar is honest and you can read off the
   N_c a target precision needs. If the across-draw spread sits ABOVE the
   internal SEM, the centers are not independent samples (they share tracers;
   see `dvcorr.estimators.shell_dipole.center_standard_error`'s Notes) and the
   SEM is understating the true uncertainty.

2. BOOTSTRAP INTERVAL (`bootstrap_confidence`) -- resample the centers with
   replacement and take a percentile interval. The SEM is a Gaussian
   statement about a sum whose terms are emphatically not Gaussian; the
   bootstrap makes no such assumption, and on the inner shells the two can
   disagree. Report the interval, never the marker alone.

3. TRIM LADDER (`trim_ladder`) -- drop the k largest-|contribution| centers
   and restack. A stack that is a genuine population mean barely moves; a
   stack that is three centers wearing a trenchcoat collapses. The value is
   expected to move (the distribution IS heavy-tailed); what must survive is
   the SIGN and the order of magnitude.

4. SEED REPEATS (`seed_repeats`) -- redraw the center set from a different
   random stream and re-run the estimator. This is the only test here that
   costs an estimator pass, and the only one that probes the center DRAW
   rather than the centers already drawn. The across-seed spread is the
   honest reproducibility of the number.

5. MECHANISM SPLIT (`property_split`) -- split the centers on the per-center
   variable the proposed explanation names, and check the trend. This is the
   test that distinguishes "real effect" from "real fluctuation", because a
   mechanism makes a MONOTONE prediction while a fluctuation makes none. For
   the near-miss above, the claim was that virialized environments cancel the
   dipole; splitting on shell occupancy showed crowded centers giving the
   LARGEST POSITIVE contributions, which falsifies the mechanism regardless of
   what any single stack reads.

   Measure the environment at a DIFFERENT scale than the shell under test
   (`RobustnessRunConfig.crowding_shell`), or the test grades its own
   homework: a center with no tracers in a shell contributes exactly zero to
   that shell by construction, so the emptiest group reads 0.0 whatever the
   physics.

WHY THESE ARE CHEAP
-------------------
Tests 1, 2, 3 and 5 are all RE-SUMS of arrays the estimator already returned --
`per_center_dipole`, `per_center_count`, `per_center_speed`. Nothing re-queries
the KDTree, exactly as every null in `dvcorr.pipeline.velocity_centered` and
`dvcorr.pipeline.velocity_frame_comparison` is a recombination rather than a
second pass. `PerCenterStack` is the bundle those arrays travel in, and
`merge_shells` coarsens that binning when a resolved ladder has to be compared
to a wide-binned run, and `restack` is the single place a center subset is
turned back into a normalized curve -- one home for the normalization convention, so a subset curve and the
published curve cannot drift apart. `restack(stack)` with no subset reproduces
`normalize_stacked_dipole`'s `zeta_hat`, `sem` and `monopole_norm` exactly.

Only test 4 runs the estimator, once per seed.

MONOPOLE
--------
`restack` returns the normalized monopole beside every dipole it builds
(CLAUDE.md hard rule 6). A subset's monopole is the diagnostic that says
whether that subset's shells are still complete: a trim or a split that also
moves the monopole has changed the GEOMETRY of the sample, and its dipole is
not comparable to the parent's.

FRAME
-----
Written against the observer-frame `VelocityCenteredShellDipoleResult`. The
observer-free `dvcorr.estimators.velocity_frame_dipole
.VelocityFrameShellDipoleResult` carries the same four field names
(`per_center_dipole`, `per_center_count`, `per_center_speed`, `shell_centers`),
so `per_center_stack` accepts either -- `|u_alpha|` becomes `|v_alpha|` and
every test reads unchanged.

Backend discipline: this module imports `matplotlib.pyplot` but never selects
a backend. The CONSUMER (a script, a notebook, a test) does that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Sequence
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from dvcorr.estimators.shell_dipole import (
    center_standard_error,
    expected_shell_occupancy,
)
from dvcorr.pipeline.velocity_centered import (
    DipoleFigures,
    RunConfig,
    draw_candidates_from_arrays,
    run_estimator,
    select_shared_centers,
    shell_dipole_norm_scale,
    step_band,
)

# Percent, not fraction -- np.percentile's units, so the conversion happens
# once here rather than at three call sites.
_PERCENT = 100.0
_MIN_BOOTSTRAP_RESAMPLES = 2
_MIN_SUBSET_DRAWS = 2
_MIN_SEEDS = 2


class PerCenterResult(Protocol):
    """The four arrays every test in this module reads off an estimator result.

    Both `dvcorr.estimators.shell_dipole.VelocityCenteredShellDipoleResult`
    and `dvcorr.estimators.velocity_frame_dipole.VelocityFrameShellDipoleResult`
    satisfy this structurally, which is what makes the module frame-agnostic
    (see the module docstring's FRAME section). Declared as a Protocol rather
    than a union so a future third frame needs no edit here.
    """

    shell_edges: np.ndarray
    shell_centers: np.ndarray
    per_center_dipole: np.ndarray
    per_center_count: np.ndarray
    per_center_speed: np.ndarray


@dataclass
class RobustnessRunConfig(RunConfig):
    """`RunConfig` plus the knobs the five tests need (CLAUDE.md hard rule 4).

    Subclasses `RunConfig` for the same reason
    `dvcorr.pipeline.velocity_frame_comparison.ComparisonRunConfig` does: the
    tests run on a center set drawn and cut by the inherited machinery, so
    `sub_volume_radius`, `shells`, `n_candidate_centers`, `seed` and
    `min_center_speed` all mean exactly what they mean everywhere else, and a
    robustness run is configured by editing one object rather than by keeping
    two in sync.

    Attributes
    ----------
    subset_fractions : tuple[float, ...]
        Test 1. Fractions of the full center set to draw, each strictly in
        (0, 1]. The ladder is a ladder because the diagnostic is the SLOPE:
        one point tells you a number, several tell you whether that number is
        converging. Must span at least a decade in N_c for the 1/sqrt(N_c)
        reference to be readable.
    n_subset_draws : int
        Test 1. Independent subsets per rung. The across-draw standard
        deviation is itself an estimate, so this is the precision of the
        precision measurement; >= 2 is required for a standard deviation to
        exist at all, and the default trades runtime for a stable rung.
    subset_seed : int
        Test 1. Extends the package seed ladder, whose existing rungs are
        listed in `RunConfig.gaussian_null_seed`. No two random streams
        anywhere in `dvcorr.pipeline` share a seed.
    n_bootstrap : int
        Test 2. Resamples. The percentile interval's own granularity is
        1 / n_bootstrap, so a 95% interval needs enough resamples that its
        tails are resolved by more than a handful of draws.
    bootstrap_seed : int
        Test 2. See `subset_seed`.
    confidence : float
        Test 2. Central interval mass, in (0, 1). 0.95 is the default because
        it is what "the error bar" is normally read as meaning; the interval
        is reported as explicit low/high edges, never as a symmetric +/-,
        because on the inner shells it is not symmetric.
    trim_counts : tuple[int, ...]
        Test 3. How many top-|contribution| centers to drop at each rung.
        Includes 0 (the untrimmed stack) so the ladder carries its own
        baseline and the caller never has to splice one in. Spans several
        decades because the question is at what depth the answer stops moving.
    trim_rank_shell : int
        Test 3. Which shell's contribution the centers are RANKED by. The
        ranking must come from one shell: a center that dominates the inner
        bin is not the one that dominates the outer bins, and re-ranking per
        shell would trim a different sample in every column, which is not a
        subset of the center set and has no interpretation. 0 is the innermost
        shell, the one the trim test is usually asked about.
    crowding_shell : int | None
        Test 5. Which shell's occupancy is the ENVIRONMENT variable the
        centers are split on. None means `trim_rank_shell`, which is the
        right default when that shell is well populated.

        Set it to a wider shell when the shell under test is not. Splitting
        on the occupancy of the very shell whose dipole is being measured is
        partly circular -- a center with zero tracers there contributes
        exactly zero to that shell's dipole by construction, so the lowest
        group reads 0.0 whatever the physics -- and the circularity stops
        being harmless once most centers fall in it. Resolving down to the
        force softening does exactly that: the innermost shells hold O(0)
        tracers per center, and the whole split collapses into its trivial
        group.

        Pointing this at a shell around the mean interparticle spacing keeps
        the test non-circular AND informative: it measures environment at the
        smallest scale where a count is a density rather than a coincidence,
        then asks whether that environment orders the contribution to the
        shell under test -- which is the mechanism question, stated without
        the estimator grading its own homework.
    crowding_edges : tuple[float, ...]
        Test 5. Bin edges, in NUMBER OF TRACERS, for splitting centers by
        their occupancy in `trim_rank_shell`. Occupancy is the environment
        proxy that needs no extra input: it is `per_center_count`, which the
        estimator already returned, and inside the innermost shell it is a
        direct count of how virialized the center's neighbourhood is. The
        last edge is +inf so the top group is open and no center falls out of
        the split.
    speed_percentiles : tuple[float, ...]
        Test 5. Percentile edges for the companion split on the per-center
        weight |u_alpha| (or |v_alpha|). The weight enters the stack
        multiplicatively, so this split shows directly how much of the answer
        rides on the fastest few percent of centers -- the systematic a
        global velocity cut would introduce.
    figure_max_radius : float | None
        Plot-only. Upper x-limit for the figure's RADIAL panels, h^-1 Mpc.
        None shows the full shell range. Every test still runs, and every
        table still reports, on all shells -- this trims the view, never the
        measurement.

        It exists because log binning plus cartesian axes (the house style,
        pinned by `tests/test_plot_wiring.py`) puts a decade of resolved
        inner shells into the leftmost few percent of a full-range axis. When
        the question IS the inner shells, that is a figure that cannot answer
        it.
    n_seed_repeats : int
        Test 4. How many independent center DRAWS to re-run the estimator on.
        Costs one estimator pass each; every other test here is free.
        `seed_repeats` derives the seeds from `RunConfig.seed` by spawning,
        the same construction `null_realization_seeds` uses, so a repeat set
        is reproducible from the run's own seed.
    """

    subset_fractions: tuple[float, ...] = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
    n_subset_draws: int = 64
    subset_seed: int = 49
    n_bootstrap: int = 2000
    bootstrap_seed: int = 50
    confidence: float = 0.95
    trim_counts: tuple[int, ...] = (0, 1, 10, 100, 1000)
    trim_rank_shell: int = 0
    crowding_shell: int | None = None
    figure_max_radius: float | None = None
    crowding_edges: tuple[float, ...] = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, np.inf)
    speed_percentiles: tuple[float, ...] = (0.0, 50.0, 90.0, 99.0, 99.9, 100.0)
    n_seed_repeats: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.subset_fractions or any(
            not 0.0 < f <= 1.0 for f in self.subset_fractions
        ):
            raise ValueError(
                "RobustnessRunConfig: every subset_fraction must be in (0, 1]; "
                f"got {self.subset_fractions}."
            )
        if self.n_subset_draws < _MIN_SUBSET_DRAWS:
            raise ValueError(
                f"RobustnessRunConfig: n_subset_draws ({self.n_subset_draws}) must "
                f"be >= {_MIN_SUBSET_DRAWS}; each rung is reported as an "
                "across-draw mean and standard deviation."
            )
        if self.n_bootstrap < _MIN_BOOTSTRAP_RESAMPLES:
            raise ValueError(
                f"RobustnessRunConfig: n_bootstrap ({self.n_bootstrap}) must be "
                f">= {_MIN_BOOTSTRAP_RESAMPLES}."
            )
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(
                f"RobustnessRunConfig: confidence ({self.confidence}) must be in "
                "(0, 1) -- it is an interval mass, not a percentage."
            )
        if any(k < 0 for k in self.trim_counts):
            raise ValueError(
                f"RobustnessRunConfig: trim_counts must be >= 0; got {self.trim_counts}."
            )
        if self.trim_rank_shell < 0:
            raise ValueError(
                f"RobustnessRunConfig: trim_rank_shell ({self.trim_rank_shell}) "
                "must be >= 0."
            )
        if self.crowding_shell is not None and self.crowding_shell < 0:
            raise ValueError(
                f"RobustnessRunConfig: crowding_shell ({self.crowding_shell}) "
                "must be >= 0, or None to follow trim_rank_shell."
            )
        if self.figure_max_radius is not None and self.figure_max_radius <= 0.0:
            raise ValueError(
                f"RobustnessRunConfig: figure_max_radius "
                f"({self.figure_max_radius}) must be > 0, or None for the full "
                "shell range."
            )
        if list(self.crowding_edges) != sorted(self.crowding_edges):
            raise ValueError(
                f"RobustnessRunConfig: crowding_edges must be increasing; got "
                f"{self.crowding_edges}."
            )
        if self.n_seed_repeats < _MIN_SEEDS:
            raise ValueError(
                f"RobustnessRunConfig: n_seed_repeats ({self.n_seed_repeats}) must "
                f"be >= {_MIN_SEEDS}; the across-seed spread needs two samples."
            )


# ---------------------------------------------------------------------------
# The shared re-sum primitive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerCenterStack:
    """Everything needed to rebuild a normalized curve from a center subset.

    The bundle the four free tests travel on. Built once by
    `per_center_stack` from an estimator result and the run's n_bar, then
    passed to `restack` with whichever rows a test wants -- so the
    normalization convention lives in exactly one place and a subset's curve
    is guaranteed to be the same arithmetic as the published one.

    Attributes
    ----------
    shell_edges : ndarray, shape (B + 1,)
        Radial bin boundaries, h^-1 Mpc.
    shell_centers : ndarray, shape (B,)
        Bin midpoints, h^-1 Mpc.
    per_center_dipole : ndarray, shape (N_c, B)
        The estimator's raw per-center l=1 numerators, |u_alpha| * A_alpha,b.
    per_center_count : ndarray, shape (N_c, B)
        Realized tracer occupancy per center per shell. Pure geometry -- it
        is what `property_split`'s crowding variable is read from, and what
        the monopole is built from.
    per_center_speed : ndarray, shape (N_c,)
        The per-center weight actually used, |u_alpha| (observer frame) or
        |v_alpha| (velocity frame). Positive-definite.
    norm_scale : ndarray, shape (B,)
        `shell_dipole_norm_scale(shell_edges, n_bar)` -- cached because every
        test multiplies by it and recomputing it per test is one more place
        the convention could drift.
    expected_occupancy : ndarray, shape (B,)
        `expected_shell_occupancy(n_bar, shell_edges)`, the monopole's
        denominator, cached for the same reason.
    n_bar : float
        The number density both of the above were built from, retained so a
        stack is self-describing.
    n_centers : int
        N_c, the number of rows.
    """

    shell_edges: np.ndarray
    shell_centers: np.ndarray
    per_center_dipole: np.ndarray
    per_center_count: np.ndarray
    per_center_speed: np.ndarray
    norm_scale: np.ndarray
    expected_occupancy: np.ndarray
    n_bar: float
    n_centers: int


def per_center_stack(result: PerCenterResult, n_bar: float) -> PerCenterStack:
    """Bundle an estimator result and its n_bar into a re-summable stack.

    Parameters
    ----------
    result : PerCenterResult
        Any estimator result carrying the per-center arrays -- the observer
        frame's `VelocityCenteredShellDipoleResult` or the velocity frame's
        `VelocityFrameShellDipoleResult`.
    n_bar : float
        Mean tracer number density over the WHOLE box, the same value the
        published curve was normalized with (`box_number_density` or
        `dvcorr.pipeline.dm_snapshot.snapshot_number_density`). Passing the
        sub-volume density instead would rescale every curve here by a
        constant and every RATIO reported by the tests by nothing -- which is
        exactly why it is taken as an explicit argument rather than derived.

    Returns
    -------
    PerCenterStack

    Raises
    ------
    ValueError
        If the result holds no centers -- there is nothing to resample.
    """
    per_center_dipole = np.asarray(result.per_center_dipole, dtype=float)
    if per_center_dipole.shape[0] == 0:
        raise ValueError(
            "per_center_stack: the result holds zero centers; every test in "
            "this module resamples centers and none is defined on an empty set."
        )
    shell_edges = np.asarray(result.shell_edges, dtype=float)
    return PerCenterStack(
        shell_edges=shell_edges,
        shell_centers=np.asarray(result.shell_centers, dtype=float),
        per_center_dipole=per_center_dipole,
        per_center_count=np.asarray(result.per_center_count, dtype=float),
        per_center_speed=np.asarray(result.per_center_speed, dtype=float),
        norm_scale=shell_dipole_norm_scale(shell_edges, n_bar),
        expected_occupancy=expected_shell_occupancy(n_bar, shell_edges),
        n_bar=float(n_bar),
        n_centers=int(per_center_dipole.shape[0]),
    )


@dataclass(frozen=True)
class RestackedCurve:
    """One normalized curve rebuilt from a subset of centers.

    Attributes
    ----------
    zeta_hat : ndarray, shape (B,)
        The subset's normalized dipole, km/s.
    sem : ndarray, shape (B,)
        Across-center standard error WITHIN the subset. NaN when the subset
        holds fewer than two centers.
    monopole_norm : ndarray, shape (B,)
        The subset's normalized l=0 companion, km/s (CLAUDE.md hard rule 6).
        Read it beside `zeta_hat`: a subset whose monopole differs from the
        parent's has a different shell GEOMETRY, and its dipole is not a
        like-for-like comparison.
    n_centers : int
        How many centers went into this curve. The divisor is this number,
        not the parent's -- a trimmed or split stack is the mean over the
        centers it actually kept.
    """

    zeta_hat: np.ndarray
    sem: np.ndarray
    monopole_norm: np.ndarray
    n_centers: int


def restack(stack: PerCenterStack, rows: np.ndarray | None = None) -> RestackedCurve:
    """Rebuild the normalized dipole and monopole from a subset of centers.

    The single home for "turn these centers into a curve". With `rows = None`
    this reproduces `dvcorr.pipeline.velocity_centered.normalize_stacked_dipole`'s
    `zeta_hat`, `sem` and `monopole_norm` exactly, which is what licenses
    reading any subset curve on the same axes as the published one.

    Parameters
    ----------
    stack : PerCenterStack
    rows : ndarray, shape (M,), int | None
        Row indices into the stack's per-center arrays. May repeat (the
        bootstrap resamples with replacement) and need not be sorted. None
        means all centers.

    Returns
    -------
    RestackedCurve

    Raises
    ------
    ValueError
        If `rows` is empty -- a curve over no centers is 0/0, and returning
        NaN silently would let an over-aggressive trim look like a measurement.
    """
    if rows is None:
        dipole_rows = stack.per_center_dipole
        count_rows = stack.per_center_count
        speed_rows = stack.per_center_speed
    else:
        rows = np.asarray(rows, dtype=int)
        if rows.size == 0:
            raise ValueError(
                "restack: `rows` is empty. A curve over zero centers is "
                "undefined; loosen the trim or widen the split bin."
            )
        dipole_rows = stack.per_center_dipole[rows]
        count_rows = stack.per_center_count[rows]
        speed_rows = stack.per_center_speed[rows]

    n_rows = dipole_rows.shape[0]
    zeta_hat = (dipole_rows.sum(axis=0) / n_rows) * stack.norm_scale
    if n_rows < 2:
        sem = np.full(stack.shell_centers.shape[0], np.nan)
    else:
        sem = center_standard_error(dipole_rows) * stack.norm_scale
    monopole = (speed_rows[:, None] * count_rows).sum(axis=0)
    monopole_norm = (monopole / n_rows) / stack.expected_occupancy
    return RestackedCurve(
        zeta_hat=zeta_hat,
        sem=sem,
        monopole_norm=monopole_norm,
        n_centers=int(n_rows),
    )


def merge_shells(stack: PerCenterStack, boundaries: Sequence[int]) -> PerCenterStack:
    """Coarsen a stack's radial binning by merging adjacent shells.

    The bridge between a finely-resolved run and a coarse one. A resolved
    ladder cannot be compared to a single wide bin by eye, or by averaging the
    fine bins' `zeta_hat`: each shell's normalization divides by its OWN
    `nbar * V_b`, so a plain average is a volume-mis-weighted quantity that
    equals nothing. Merging at the level of the RAW per-center numerators, and
    normalizing once against the merged shell's own volume, is the exact
    answer -- the number the coarse run would have produced from the same
    centers.

    Parameters
    ----------
    stack : PerCenterStack
    boundaries : Sequence[int]
        Indices into `stack.shell_edges`, strictly increasing, at least two.
        Merged bin `j` spans original shells `boundaries[j] ...
        boundaries[j+1] - 1`, i.e. the radial range
        `[shell_edges[boundaries[j]], shell_edges[boundaries[j+1]])`.
        `[0, B]` collapses everything into one bin; `range(B + 1)` is the
        identity.

    Returns
    -------
    PerCenterStack
        A new stack over `len(boundaries) - 1` shells, same centers and same
        `n_bar`. Every test in this module accepts it unchanged, so a coarse
        view gets the same bootstrap, trim and split treatment as the fine one.

    Raises
    ------
    ValueError
        If `boundaries` is shorter than two, not strictly increasing, or
        outside `[0, B]`. Shells cannot be dropped or reordered by this
        function -- the merged bins must tile a contiguous radial range, or
        the volume the result is normalized by is not the volume the tracers
        came from.
    """
    edges_index = np.asarray(boundaries, dtype=int)
    n_shells = stack.shell_centers.shape[0]
    if edges_index.size < 2:
        raise ValueError(
            f"merge_shells: `boundaries` needs at least two entries, got "
            f"{edges_index.size}."
        )
    if np.any(np.diff(edges_index) <= 0):
        raise ValueError(
            f"merge_shells: `boundaries` must be strictly increasing; got "
            f"{edges_index.tolist()}."
        )
    if edges_index[0] < 0 or edges_index[-1] > n_shells:
        raise ValueError(
            f"merge_shells: `boundaries` must lie in [0, {n_shells}]; got "
            f"{edges_index.tolist()}."
        )

    lo, hi = edges_index[:-1], edges_index[1:]
    # np.add.reduceat would be the terse form, but it silently returns the
    # single element (not a sum) wherever lo[j] == lo[j+1]; the strictly-
    # increasing guard above rules that out, and the explicit loop keeps the
    # column blocks obvious.
    dipole = np.column_stack(
        [stack.per_center_dipole[:, a:b].sum(axis=1) for a, b in zip(lo, hi)]
    )
    count = np.column_stack(
        [stack.per_center_count[:, a:b].sum(axis=1) for a, b in zip(lo, hi)]
    )
    shell_edges = stack.shell_edges[edges_index]
    return PerCenterStack(
        shell_edges=shell_edges,
        shell_centers=0.5 * (shell_edges[:-1] + shell_edges[1:]),
        per_center_dipole=dipole,
        per_center_count=count,
        per_center_speed=stack.per_center_speed,
        norm_scale=shell_dipole_norm_scale(shell_edges, stack.n_bar),
        expected_occupancy=expected_shell_occupancy(stack.n_bar, shell_edges),
        n_bar=stack.n_bar,
        n_centers=stack.n_centers,
    )


# ---------------------------------------------------------------------------
# Test 1 -- precision ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionLadder:
    """Test 1: how the answer's scatter falls with the number of centers.

    Attributes
    ----------
    n_centers : ndarray, shape (F,)
        Subset size at each rung, ascending.
    zeta_hat_mean : ndarray, shape (F, B)
        Mean over `n_subset_draws` independent subsets, per rung per shell.
        Unbiased at every rung by construction -- a subset mean is a mean --
        so this column should be FLAT in N_c. A trend in it means the subsets
        are not exchangeable with the parent (a selection leaked in), which is
        a different and more serious problem than noise.
    zeta_hat_spread : ndarray, shape (F, B)
        Standard deviation (ddof=1) ACROSS those draws -- the empirical
        sampling scatter of a stack that size. This is the number to compare
        against the published error bar, but compare it at the SMALL rungs.
        Drawing without replacement carries the finite-population correction
        `sqrt((N_full - N) / (N_full - 1))`, so this column necessarily sits a
        factor ~sqrt(1 - N/N_full) BELOW `reference_spread` at large
        fractions -- a subset that is half the parent shares half its centers
        with it and cannot scatter freely. That deficit is arithmetic, not
        evidence that the error bar is conservative; at a few percent of the
        parent the correction is negligible and the comparison is clean.
        `seed_repeats` is the version with no such correction at all.
    sem_mean : ndarray, shape (F, B)
        Mean of the subsets' own INTERNAL `center_standard_error`. Agreement
        with `zeta_hat_spread` is the statement that the SEM is honest;
        `sem_mean` sitting BELOW `zeta_hat_spread` means the centers share
        tracers and the quoted error bar is too small.
    reference_spread : ndarray, shape (F, B)
        The 1/sqrt(N_c) law drawn through the largest rung, as
        `sem(N_full) * sqrt(N_full / N)`. Anchored on the full set's INTERNAL
        SEM rather than on any rung's empirical spread, because the full
        rung's spread is structurally zero (see `precision_ladder`) and the
        SEM is the quantity a reader would otherwise have extrapolated from.
        Deviation of `zeta_hat_spread` from this line is the diagnostic.
    n_draws : int
        Draws per rung, echoed for the plot legend.
    """

    n_centers: np.ndarray
    zeta_hat_mean: np.ndarray
    zeta_hat_spread: np.ndarray
    sem_mean: np.ndarray
    reference_spread: np.ndarray
    n_draws: int


def precision_ladder(cfg: RobustnessRunConfig, stack: PerCenterStack) -> PrecisionLadder:
    """Draw center subsets of increasing size; report scatter vs. N_c.

    Subsets are drawn WITHOUT replacement (they stand in for smaller runs,
    and a smaller run would not measure the same center twice), and every rung
    is drawn from the same `cfg.subset_seed` stream so the ladder is
    reproducible.

    The full rung (`fraction = 1.0`) is a degenerate case on purpose: without
    replacement it can only return the whole set, so its across-draw spread is
    exactly zero. That zero is the ladder's own consistency marker -- it is
    where `sem_mean` alone carries the uncertainty -- and it is left in rather
    than dropped so the rung's `zeta_hat_mean` is the published value the
    other rungs are being compared against.

    Parameters
    ----------
    cfg : RobustnessRunConfig
        Reads `subset_fractions`, `n_subset_draws`, `subset_seed`.
    stack : PerCenterStack

    Returns
    -------
    PrecisionLadder
        Rungs sorted ascending in subset size, duplicates removed (two
        fractions can round to the same integer on a small center set).
    """
    rng = np.random.default_rng(cfg.subset_seed)
    n_full = stack.n_centers
    sizes = sorted({max(1, int(round(f * n_full))) for f in cfg.subset_fractions})

    n_shells = stack.shell_centers.shape[0]
    zeta_mean = np.empty((len(sizes), n_shells))
    zeta_spread = np.empty((len(sizes), n_shells))
    sem_mean = np.empty((len(sizes), n_shells))

    for i, size in enumerate(sizes):
        draws = np.empty((cfg.n_subset_draws, n_shells))
        sems = np.empty((cfg.n_subset_draws, n_shells))
        for d in range(cfg.n_subset_draws):
            rows = rng.choice(n_full, size=size, replace=False)
            curve = restack(stack, rows)
            draws[d] = curve.zeta_hat
            sems[d] = curve.sem
        zeta_mean[i] = draws.mean(axis=0)
        zeta_spread[i] = draws.std(axis=0, ddof=1)
        sem_mean[i] = np.nanmean(sems, axis=0)

    # The 1/sqrt(N) law anchored on the FULL set's internal SEM rather than on
    # a rung's empirical spread: the full rung's spread is structurally zero
    # (see the docstring), and the internal SEM is the quantity a reader would
    # otherwise have used to extrapolate.
    full_sem = restack(stack).sem
    n_centers = np.asarray(sizes, dtype=float)
    reference = full_sem[None, :] * np.sqrt(n_full / n_centers[:, None])

    return PrecisionLadder(
        n_centers=np.asarray(sizes, dtype=int),
        zeta_hat_mean=zeta_mean,
        zeta_hat_spread=zeta_spread,
        sem_mean=sem_mean,
        reference_spread=reference,
        n_draws=cfg.n_subset_draws,
    )


# ---------------------------------------------------------------------------
# Test 2 -- bootstrap interval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapInterval:
    """Test 2: a distribution-free interval on every shell.

    Attributes
    ----------
    zeta_hat : ndarray, shape (B,)
        The point estimate over all centers -- the published curve, repeated
        here so the interval never has to be matched up by hand.
    low, high : ndarray, shape (B,)
        Percentile interval edges, km/s, at `confidence` central mass. NOT
        symmetric about `zeta_hat` on the inner shells, which is the reason
        this test exists.
    bootstrap_std : ndarray, shape (B,)
        Standard deviation of the resampled stacks. Comparable to `sem`;
        a large gap between them is non-Gaussianity of the per-center
        contributions, the same diagnostic the trim ladder measures directly.
    sem : ndarray, shape (B,)
        The Gaussian error bar, carried alongside for exactly that comparison.
    confidence : float
        Echo of the interval mass used.
    n_resamples : int
        Echo of the resample count.
    """

    zeta_hat: np.ndarray
    low: np.ndarray
    high: np.ndarray
    bootstrap_std: np.ndarray
    sem: np.ndarray
    confidence: float
    n_resamples: int


def bootstrap_confidence(
    cfg: RobustnessRunConfig, stack: PerCenterStack
) -> BootstrapInterval:
    """Resample centers WITH replacement; report a percentile interval per shell.

    Resampling the centers (not the tracers) is the right unit here: a center
    is what the stack averages over, and its whole shell -- every tracer in it
    -- travels with it, so the resample preserves the within-shell correlation
    that makes the per-center contributions heavy-tailed in the first place.

    Parameters
    ----------
    cfg : RobustnessRunConfig
        Reads `n_bootstrap`, `bootstrap_seed`, `confidence`.
    stack : PerCenterStack

    Returns
    -------
    BootstrapInterval
    """
    rng = np.random.default_rng(cfg.bootstrap_seed)
    n_shells = stack.shell_centers.shape[0]
    resampled = np.empty((cfg.n_bootstrap, n_shells))
    for b in range(cfg.n_bootstrap):
        rows = rng.integers(0, stack.n_centers, size=stack.n_centers)
        resampled[b] = restack(stack, rows).zeta_hat

    tail = (1.0 - cfg.confidence) / 2.0
    low, high = np.percentile(
        resampled, [_PERCENT * tail, _PERCENT * (1.0 - tail)], axis=0
    )
    full = restack(stack)
    return BootstrapInterval(
        zeta_hat=full.zeta_hat,
        low=low,
        high=high,
        bootstrap_std=resampled.std(axis=0, ddof=1),
        sem=full.sem,
        confidence=cfg.confidence,
        n_resamples=cfg.n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Test 3 -- trim ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrimLadder:
    """Test 3: how much of the answer rides on its largest contributors.

    Attributes
    ----------
    trim_counts : ndarray, shape (K,)
        Centers dropped at each rung, ascending; the first is 0.
    n_centers : ndarray, shape (K,)
        Centers remaining at each rung.
    zeta_hat : ndarray, shape (K, B)
        The trimmed stacks. Row 0 is the untrimmed curve.
    monopole_norm : ndarray, shape (K, B)
        Their monopoles (hard rule 6). A trim that moves the monopole has
        changed which SHELLS are represented, not just which velocities --
        the trimmed dipole is then not measuring the same geometry.
    rank_shell : int
        Which shell's |contribution| set the ranking, echoed because the
        answer depends on it.
    mean_contribution, median_contribution : float
        Mean and median normalized per-center contribution in `rank_shell`,
        km/s. Their RATIO is the compact statement of how heavy the tail is:
        near 1 is a well-behaved average, orders of magnitude apart means the
        stack is a sum of rare large terms and the trim ladder is the only
        honest error statement.

        On a SPARSE shell the median is exactly 0.0 and the ratio is
        undefined. That is not a degenerate case to guard around, it is the
        measurement: more than half the centers have no tracer in the shell
        at all, so the statistic is carried entirely by a minority.
        `n_contributing` is the number to report instead, and a caller
        forming the ratio must check the median first.
    n_contributing : int
        Centers with a NON-ZERO contribution in `rank_shell`. Equal to
        `n_centers` on a well-populated shell; far below it once the shell is
        narrow enough that most centers are empty -- which is the regime any
        resolved r -> 0 ladder enters, and the context every other number in
        this dataclass has to be read in.
    """

    trim_counts: np.ndarray
    n_centers: np.ndarray
    zeta_hat: np.ndarray
    monopole_norm: np.ndarray
    rank_shell: int
    mean_contribution: float
    median_contribution: float
    n_contributing: int


def trim_ladder(cfg: RobustnessRunConfig, stack: PerCenterStack) -> TrimLadder:
    """Drop the k largest-|contribution| centers and restack, for each k.

    Ranking is by |contribution| rather than by contribution, so the trim
    removes the terms that dominate the VARIANCE (both signs) rather than
    shaving one tail and manufacturing a drift toward the other. Trimming a
    single tail is a biased estimator dressed as a robustness check; this is
    the symmetric version.

    Parameters
    ----------
    cfg : RobustnessRunConfig
        Reads `trim_counts`, `trim_rank_shell`.
    stack : PerCenterStack

    Returns
    -------
    TrimLadder
        Rungs with `k >= n_centers` are dropped rather than returned empty --
        there is no such thing as a stack over no centers.

    Raises
    ------
    IndexError
        If `cfg.trim_rank_shell` is not a valid shell index. Raised rather
        than clamped: a silently-clamped rank shell would rank by the wrong
        column and every number below would be quietly about a different bin.
    """
    n_shells = stack.shell_centers.shape[0]
    if not 0 <= cfg.trim_rank_shell < n_shells:
        raise IndexError(
            f"trim_ladder: trim_rank_shell ({cfg.trim_rank_shell}) is outside "
            f"the {n_shells} shells this stack carries."
        )
    contribution = stack.per_center_dipole[:, cfg.trim_rank_shell] * (
        stack.norm_scale[cfg.trim_rank_shell]
    )
    order = np.argsort(-np.abs(contribution))  # largest |contribution| first

    counts = sorted({int(k) for k in cfg.trim_counts if k < stack.n_centers})
    zeta = np.empty((len(counts), n_shells))
    monopole = np.empty((len(counts), n_shells))
    n_kept = np.empty(len(counts), dtype=int)
    for i, k in enumerate(counts):
        curve = restack(stack, order[k:])
        zeta[i] = curve.zeta_hat
        monopole[i] = curve.monopole_norm
        n_kept[i] = curve.n_centers

    return TrimLadder(
        trim_counts=np.asarray(counts, dtype=int),
        n_centers=n_kept,
        zeta_hat=zeta,
        monopole_norm=monopole,
        rank_shell=cfg.trim_rank_shell,
        mean_contribution=float(contribution.mean()),
        median_contribution=float(np.median(contribution)),
        n_contributing=int(np.count_nonzero(contribution)),
    )


# ---------------------------------------------------------------------------
# Test 4 -- seed repeats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRepeats:
    """Test 4: the same run, on independently drawn center sets.

    Attributes
    ----------
    seeds : ndarray, shape (S,)
        The center-draw seeds used, derived from `RunConfig.seed`.
    n_centers : ndarray, shape (S,)
        Surviving centers per seed. These differ slightly run to run (the
        core cut and speed floor act on a different draw), which is expected;
        a LARGE spread means the cuts are marginal for this geometry.
    zeta_hat : ndarray, shape (S, B)
        One curve per seed.
    monopole_norm : ndarray, shape (S, B)
        Their monopoles (hard rule 6).
    mean, spread : ndarray, shape (B,)
        Across-seed mean and standard deviation (ddof=1) of `zeta_hat`. The
        spread is the reproducibility of the number under a redraw, and is
        the one error statement here that includes the center SELECTION, not
        just the centers selected.
    """

    seeds: np.ndarray
    n_centers: np.ndarray
    zeta_hat: np.ndarray
    monopole_norm: np.ndarray
    mean: np.ndarray
    spread: np.ndarray


def seed_repeats(
    cfg: RobustnessRunConfig,
    s_pool: np.ndarray,
    v_pool: np.ndarray,
    mvir_pool: np.ndarray,
    is_distinct_pool: np.ndarray,
    s_tracers: np.ndarray,
    observer: np.ndarray,
    n_bar: float,
) -> SeedRepeats:
    """Re-draw the center set from `n_seed_repeats` independent streams and re-run.

    The ONLY test in this module that costs estimator passes -- one per seed.
    It is also the only one that can catch a bad center DRAW, because every
    other test conditions on the centers already in hand.

    Seeds are `SeedSequence(cfg.seed).spawn(...)`-derived rather than
    `cfg.seed + i`, for the reason
    `dvcorr.pipeline.velocity_centered.null_realization_seeds` gives: adjacent
    integer seeds are not guaranteed to give independent streams, and the
    whole point of the test is independence.

    Parameters
    ----------
    cfg : RobustnessRunConfig
        Reads `n_seed_repeats`, `seed`, and everything the draw/cut/estimator
        stages read.
    s_pool, v_pool : ndarray, shape (N_pool, 3)
        The carved population candidates are drawn FROM -- `CarvedHalos.pos` /
        `.vel`, or the particle path's `carved.pos` / `.vel`.
    mvir_pool : ndarray, shape (N_pool,)
    is_distinct_pool : ndarray, shape (N_pool,), bool
        Required by `draw_candidates_from_arrays`' signature. On the particle
        path they are the synthetic uniform-mass / all-distinct arrays that
        notebook `DM_simulation` builds, and nothing downstream reads them.
    s_tracers : ndarray, shape (N_t, 3)
        The density tracer field, held FIXED across seeds -- only the centers
        are redrawn, so the test isolates center-draw variance from anything
        else.
    observer : ndarray, shape (3,)
    n_bar : float
        As for `per_center_stack`.

    Returns
    -------
    SeedRepeats
    """
    seeds = np.asarray(
        [
            int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(cfg.seed).spawn(cfg.n_seed_repeats)
        ],
        dtype=np.int64,
    )

    curves: list[np.ndarray] = []
    monopoles: list[np.ndarray] = []
    counts: list[int] = []
    for s in seeds:
        seed_cfg = replace(cfg, seed=int(s))
        candidates = draw_candidates_from_arrays(
            seed_cfg, s_pool, v_pool, mvir_pool, is_distinct_pool
        )
        centers = select_shared_centers(
            seed_cfg, candidates.s, candidates.v, observer
        )
        result = run_estimator(
            seed_cfg, centers.s_centers, centers.v_centers, s_tracers, observer
        )
        curve = restack(per_center_stack(result, n_bar))
        curves.append(curve.zeta_hat)
        monopoles.append(curve.monopole_norm)
        counts.append(curve.n_centers)

    zeta_hat = np.vstack(curves)
    return SeedRepeats(
        seeds=seeds,
        n_centers=np.asarray(counts, dtype=int),
        zeta_hat=zeta_hat,
        monopole_norm=np.vstack(monopoles),
        mean=zeta_hat.mean(axis=0),
        spread=zeta_hat.std(axis=0, ddof=1),
    )


# ---------------------------------------------------------------------------
# Test 5 -- mechanism split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertySplit:
    """Test 5: the stack broken down by a per-center property.

    Attributes
    ----------
    label : str
        What the centers were split on, for the plot and the printed table.
    group_edges : ndarray, shape (G + 1,)
        The bin edges in the property's own units.
    group_labels : tuple[str, ...]
        Human-readable `[lo, hi)` strings, one per occupied group.
    n_centers : ndarray, shape (G,)
        Centers per group. Groups that came out empty are dropped before this
        array is built, so every entry is >= 1 and `group_labels` matches.
    mean_contribution : ndarray, shape (G, B)
        The group's MEAN normalized per-center contribution, km/s -- i.e.
        `restack(group).zeta_hat`. This is the number a mechanism predicts a
        trend in, because it is per center and therefore comparable between
        groups of very different size.
    share_of_stack : ndarray, shape (G, B)
        The fraction of the FULL stack's dipole this group supplies. Sums to
        1 across groups in every shell (up to floating point). Read beside
        `mean_contribution`: a group can have the largest mean and still
        supply almost nothing if it holds twenty centers, and the pair of
        columns is what separates "this is where the signal lives" from "this
        is where the outliers live".
    monopole_norm : ndarray, shape (G, B)
        Per-group monopole (hard rule 6) -- here it does double duty as the
        split's own sanity check, since a split on occupancy MUST order the
        monopole monotonically and a split on speed must not.
    """

    label: str
    group_edges: np.ndarray
    group_labels: tuple[str, ...]
    n_centers: np.ndarray
    mean_contribution: np.ndarray
    share_of_stack: np.ndarray
    monopole_norm: np.ndarray


def _format_group_label(low: float, high: float) -> str:
    """`[lo, hi)` with an infinite upper edge printed as `inf`."""
    high_text = "inf" if not np.isfinite(high) else f"{high:g}"
    return f"[{low:g}, {high_text})"


def property_split(
    stack: PerCenterStack,
    values: np.ndarray,
    edges: np.ndarray,
    label: str,
) -> PropertySplit:
    """Split the centers on a per-center property and restack each group.

    The mechanism test. A proposed physical explanation for a feature always
    names a per-center variable -- "virialized environments", "the fastest
    centers", "the ones near the boundary" -- and this turns that name into a
    falsifiable prediction: the groups must ORDER in the direction the
    mechanism claims. A feature that is a fluctuation shows no ordering, and
    a mechanism that predicts the wrong direction is refuted by the ordering
    it does show, regardless of what the total stack reads.

    Parameters
    ----------
    stack : PerCenterStack
    values : ndarray, shape (N_c,)
        The per-center property. Must be row-aligned with the stack's
        per-center arrays -- e.g. `stack.per_center_count[:, 0]` for inner-
        shell crowding, or `stack.per_center_speed` for the weight.
    edges : ndarray, shape (G + 1,)
        Increasing bin edges in the property's units; the last may be +inf.
        Values below the first edge or above the last are DROPPED, and
        because that is a silent loss of centers the caller is expected to
        bracket the data (the config defaults do).
    label : str
        Name of the property, carried into the figure and the table.

    Returns
    -------
    PropertySplit

    Raises
    ------
    ValueError
        If `values` is not row-aligned with the stack, or if `edges` is not
        increasing.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != (stack.n_centers,):
        raise ValueError(
            f"property_split: `values` has shape {values.shape}, expected "
            f"({stack.n_centers},) -- it must be one number per center, "
            "row-aligned with the stack."
        )
    edges = np.asarray(edges, dtype=float)
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"property_split: `edges` must be increasing; got {edges}.")

    full_dipole_sum = stack.per_center_dipole.sum(axis=0)
    labels: list[str] = []
    counts: list[int] = []
    means: list[np.ndarray] = []
    shares: list[np.ndarray] = []
    monopoles: list[np.ndarray] = []
    for low, high in zip(edges[:-1], edges[1:]):
        rows = np.flatnonzero((values >= low) & (values < high))
        if rows.size == 0:
            continue
        curve = restack(stack, rows)
        labels.append(_format_group_label(low, high))
        counts.append(curve.n_centers)
        means.append(curve.zeta_hat)
        # Share of the RAW stacked numerator, so the group shares sum to 1
        # exactly; the norm_scale is a per-shell constant and cancels.
        with np.errstate(invalid="ignore", divide="ignore"):
            shares.append(stack.per_center_dipole[rows].sum(axis=0) / full_dipole_sum)
        monopoles.append(curve.monopole_norm)

    return PropertySplit(
        label=label,
        group_edges=edges,
        group_labels=tuple(labels),
        n_centers=np.asarray(counts, dtype=int),
        mean_contribution=np.vstack(means),
        share_of_stack=np.vstack(shares),
        monopole_norm=np.vstack(monopoles),
    )


def crowding_shell_index(cfg: RobustnessRunConfig) -> int:
    """Which shell's occupancy `crowding_split` reads: `crowding_shell`, else
    `trim_rank_shell`.

    One resolution site, because `cfg.crowding_shell or cfg.trim_rank_shell`
    is a live bug: shell 0 is falsy, so an explicit `crowding_shell = 0` would
    silently fall through to `trim_rank_shell`.
    """
    return cfg.trim_rank_shell if cfg.crowding_shell is None else cfg.crowding_shell


def crowding_split(cfg: RobustnessRunConfig, stack: PerCenterStack) -> PropertySplit:
    """`property_split` on shell occupancy -- the environment proxy.

    Splits on `per_center_count[:, crowding_shell_index(cfg)]`: how many
    tracers the center has in that shell. A count in a shell of fixed volume
    IS the local overdensity of the center's own neighbourhood, so the top
    group is the virialized-environment sample any "small scales are dominated
    by virial motion" claim is really about.

    Which shell to read it in is `RobustnessRunConfig.crowding_shell`'s
    docstring -- the short version being that it defaults to the shell under
    test and must NOT stay there once that shell is sparse enough for most
    centers to hold nothing in it.

    The contribution the groups are compared on is still the one in
    `trim_rank_shell` (`make_robustness_figure` panel 5): the split says where
    a center LIVES, the contribution says what it DOES to the shell in
    question, and keeping those two indices separable is what makes the test
    non-circular.
    """
    shell = crowding_shell_index(cfg)
    return property_split(
        stack,
        stack.per_center_count[:, shell],
        np.asarray(cfg.crowding_edges, dtype=float),
        label=(
            f"tracers in [{stack.shell_edges[shell]:.3g}, "
            f"{stack.shell_edges[shell + 1]:.3g}) $h^{{-1}}$ Mpc"
        ),
    )


def speed_split(cfg: RobustnessRunConfig, stack: PerCenterStack) -> PropertySplit:
    """`property_split` on the per-center weight |u| (or |v|), by percentile.

    The companion to `crowding_split`, and the one that prices a velocity
    cut: the weight multiplies the whole contribution, so the top percentile
    group's `share_of_stack` is, to first order, exactly what a cut at that
    percentile would remove from the answer.

    Percentile edges are converted to value edges on THIS sample, so the
    group sizes are fixed by construction (a 50-90 group holds 40% of the
    centers) and only the contributions vary between groups.
    """
    edges = np.percentile(stack.per_center_speed, np.asarray(cfg.speed_percentiles))
    # A percentile edge can repeat when the sample is degenerate; nudge the
    # top edge open so the fastest center is never dropped by the half-open
    # bin (property_split's bins are [lo, hi)).
    edges = np.asarray(edges, dtype=float)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return property_split(stack, stack.per_center_speed, edges, label="|u| percentile")


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

# Styling, named rather than inline (hard rule 4). The signal colour matches
# `velocity_centered._COLOR_SIGNAL` so a robustness panel reads as the same
# curve as the pipeline figure it is auditing.
_COLOR_SIGNAL = "#2a78d6"
_COLOR_REFERENCE = "#eb6834"
_COLOR_SEM = "#7a4fa3"
_COLOR_MONOPOLE = "0.3"
_COLOR_ZERO_LINE = "0.5"
_ZERO_LINE_WIDTH = 0.8
_BAND_ALPHA = 0.25
_GRID_ALPHA = 0.3
#: The five non-monopole tests, one 3x2 sheet with the sixth cell unused
#: (CLAUDE.md hard rule 6 -- the monopole companion is a SEPARATE figure,
#: below, sized on its own).
_FIGSIZE = (13.0, 13.5)
_MONOPOLE_FIGSIZE = (8.0, 3.0)
_TRIM_CMAP = "viridis"
_SPLIT_CMAP = "plasma"
_COLOR_SPAN = 0.85  # keep colormap ends off pure white/yellow
_ANNOTATION_HEADROOM = 1.18  # y-limit slack for the split panel's bar labels


def _shell_axis(
    ax: plt.Axes,
    max_radius: float | None = None,
    xlabel: str = r"$r$  [$h^{-1}$ Mpc]",
) -> None:
    """Cartesian radial axis, zero line, grid -- the house style.

    Log BINNING never implies log AXES in this package; that coupling is
    pinned out by `tests/test_plot_wiring.py` for the pipeline figures and
    followed here for the same reason. `max_radius`
    (`RobustnessRunConfig.figure_max_radius`) trims the VIEW only; the curves
    plotted on it still carry every shell, so autoscaled y-limits are set by
    the whole range and a zoomed panel stays on the same y-scale as an
    untrimmed one.

    The left edge is always pinned to r = 0 (CLAUDE.md hard rule 10 -- every
    figure starts at r = 0), independent of `max_radius`; passing
    `right=None` when `max_radius` is None leaves the autoscaled upper bound
    untouched.
    """
    ax.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=_GRID_ALPHA)
    ax.set_xlim(left=0.0, right=max_radius)


def make_robustness_figure(
    cfg: RobustnessRunConfig,
    stack: PerCenterStack,
    ladder: PrecisionLadder,
    interval: BootstrapInterval,
    trim: TrimLadder,
    repeats: SeedRepeats,
    split: PropertySplit,
) -> DipoleFigures:
    """The five tests on one sheet, plus the monopole companion as its own figure.

    Returned as a `DipoleFigures` (CLAUDE.md hard rule 6): `.dipole` is the
    5-panel sheet below (all five tests overlay or diagnose the DIPOLE, so
    they stay one figure together -- the rule requires only that the
    monopole not be merged into it), `.monopole` is panel 6, alone.

    Panels, reading across then down (`.dipole`):

    1. Precision ladder -- across-draw spread and mean internal SEM in the
       ranked shell against N_c, with the 1/sqrt(N_c) reference. LOG-LOG:
       this axis is not radial, and a power law is what it exists to show.
       The full rung's structurally-zero spread is omitted from that one
       series (see the comment at the call site); it is in `ladder`.
    2. Bootstrap interval -- the full curve with its percentile band and, for
       contrast, the Gaussian SEM band.
    3. Trim ladder -- one curve per trim depth.
    4. Seed repeats -- one curve per redrawn center set, with the across-seed
       band.
    5. Mechanism split -- per-group mean contribution in the ranked shell,
       with each group's share of the stack annotated.

    `.monopole`:

    6. Monopole (hard rule 6) -- the geometry companion of panel 2.

    Panels 2-4 and 6 are STEPS (`ax.stairs`), held flat across each bin's
    full width from `stack.shell_edges[0]` (0.0 under the default binning)
    -- CLAUDE.md hard rule 10 -- rather than a point at the shell midpoint;
    `anchor_dipole_at_origin` is therefore not needed here either. Panels 1
    and 5 are not radial (N_c and group index respectively) and are
    unaffected.

    Parameters
    ----------
    cfg : RobustnessRunConfig
    stack : PerCenterStack
    ladder, interval, trim, repeats, split
        The five test results, all built from `stack`.

    Returns
    -------
    DipoleFigures
    """
    edges = stack.shell_edges
    b = cfg.trim_rank_shell
    shell_text = (
        f"[{stack.shell_edges[b]:g}, {stack.shell_edges[b + 1]:g}) "
        r"$h^{-1}$ Mpc"
    )

    fig, axes = plt.subplots(3, 2, figsize=_FIGSIZE)
    ax_ladder, ax_boot, ax_trim, ax_seed, ax_split, ax_unused = axes.ravel()
    fig.delaxes(ax_unused)

    # --- 1: precision ladder -------------------------------------------------
    # The full rung's across-draw spread is EXACTLY zero by construction
    # (`precision_ladder`: without replacement there is one way to draw the
    # whole set). It belongs in the table and not on a log axis, where a zero
    # is not a small number but an unplottable one -- left in, it drags the
    # y-limits down by ten orders of magnitude and flattens every other curve
    # to a line. Dropped from THIS SERIES ONLY; the rung's mean and internal
    # SEM are still drawn, and `PrecisionLadder` still carries the zero.
    # Identified STRUCTURALLY (the rung that is the whole set) rather than by
    # testing the spread against zero: np.std over identical values returns a
    # denormal ~1e-12, not 0.0, which passes a `> 0` guard and lands twelve
    # decades below the rest of the panel.
    drawn = ladder.n_centers < stack.n_centers
    ax_ladder.plot(
        ladder.n_centers[drawn],
        ladder.zeta_hat_spread[drawn, b],
        "o-",
        color=_COLOR_SIGNAL,
        label=f"across-draw spread ({ladder.n_draws} draws)",
    )
    ax_ladder.plot(
        ladder.n_centers,
        ladder.sem_mean[:, b],
        "s--",
        color=_COLOR_SEM,
        label="mean internal SEM",
    )
    ax_ladder.plot(
        ladder.n_centers,
        ladder.reference_spread[:, b],
        ":",
        color=_COLOR_REFERENCE,
        label=r"$1/\sqrt{N_c}$ reference",
    )
    ax_ladder.set_xscale("log")
    ax_ladder.set_yscale("log")
    ax_ladder.set_xlabel(r"$N_c$")
    ax_ladder.set_ylabel(r"uncertainty on $\hat\zeta_1$  [km/s]")
    ax_ladder.set_title(f"1 - precision ladder, shell {shell_text}")
    ax_ladder.grid(alpha=_GRID_ALPHA, which="both")
    ax_ladder.legend(fontsize="small")

    # --- 2: bootstrap interval ----------------------------------------------
    # Panels 2, 3, 4 and the monopole panel below are all STEPS (`ax.stairs`),
    # held flat across each bin's full width from `edges[0]` (0.0 under the
    # default binning) -- CLAUDE.md hard rule 10 -- so the curve already
    # starts at the exact origin without a separate anchor.
    step_band(
        ax_boot, edges, interval.zeta_hat - interval.sem, interval.zeta_hat + interval.sem,
        color=_COLOR_SEM, alpha=_BAND_ALPHA, label="Gaussian SEM",
    )
    step_band(
        ax_boot, edges, interval.low, interval.high,
        color=_COLOR_SIGNAL, alpha=_BAND_ALPHA,
        label=f"{interval.confidence:.0%} bootstrap ({interval.n_resamples} resamples)",
    )
    ax_boot.stairs(interval.zeta_hat, edges, color=_COLOR_SIGNAL, label=r"$\hat\zeta_1$")
    _shell_axis(ax_boot, cfg.figure_max_radius)
    ax_boot.set_ylabel(r"$\hat\zeta_1(r)$  [km/s]")
    ax_boot.set_title(f"2 - bootstrap interval, $N_c$ = {stack.n_centers}")
    ax_boot.legend(fontsize="small")

    # --- 3: trim ladder ------------------------------------------------------
    trim_colors = plt.get_cmap(_TRIM_CMAP)(
        np.linspace(0.0, _COLOR_SPAN, trim.trim_counts.size)
    )
    for i, k in enumerate(trim.trim_counts):
        ax_trim.stairs(
            trim.zeta_hat[i], edges, color=trim_colors[i],
            label=f"drop top {k}" if k else "untrimmed",
        )
    _shell_axis(ax_trim, cfg.figure_max_radius)
    ax_trim.set_ylabel(r"$\hat\zeta_1(r)$  [km/s]")
    # The heavy-tail headline, in whichever form the shell admits. A median of
    # exactly zero is not a failure to guard around on a sparse inner shell --
    # it says most centers hold no tracer there at all -- so the panel reports
    # the contributing fraction instead of an infinite ratio.
    if trim.median_contribution != 0.0:
        tail_text = f"mean/median = {trim.mean_contribution / trim.median_contribution:.0f}"
    else:
        contributing = trim.n_contributing / trim.n_centers[0]
        tail_text = f"median = 0; {contributing:.1%} of centers contribute"
    ax_trim.set_title(f"3 - trim ladder (ranked on {shell_text}); {tail_text}")
    ax_trim.legend(fontsize="small")

    # --- 4: seed repeats -----------------------------------------------------
    step_band(
        ax_seed, edges, repeats.mean - repeats.spread, repeats.mean + repeats.spread,
        color=_COLOR_SIGNAL, alpha=_BAND_ALPHA, label="across-seed spread",
    )
    for i in range(repeats.seeds.size):
        ax_seed.stairs(
            repeats.zeta_hat[i], edges, lw=1.0, alpha=0.8,
            label=f"$N_c$ = {repeats.n_centers[i]}",
        )
    ax_seed.stairs(repeats.mean, edges, color=_COLOR_SIGNAL, label="across-seed mean")
    _shell_axis(ax_seed, cfg.figure_max_radius)
    ax_seed.set_ylabel(r"$\hat\zeta_1(r)$  [km/s]")
    ax_seed.set_title(f"4 - {repeats.seeds.size} independent center draws")
    ax_seed.legend(fontsize="x-small", ncol=2)

    # --- 5: mechanism split --------------------------------------------------
    n_groups = split.n_centers.size
    split_colors = plt.get_cmap(_SPLIT_CMAP)(np.linspace(0.0, _COLOR_SPAN, n_groups))
    positions = np.arange(n_groups)
    ax_split.bar(
        positions, split.mean_contribution[:, b], color=split_colors, edgecolor="none"
    )
    # Labels go OUTSIDE the bar in whichever direction it points. A group
    # contributing negatively is the informative case here (virialized centers
    # cancelling the shell under test), so its label must not be buried under
    # the zero line where a fixed upward offset would put it.
    values = split.mean_contribution[:, b]
    for x, (share, n) in enumerate(zip(split.share_of_stack[:, b], split.n_centers)):
        above = values[x] >= 0.0
        ax_split.annotate(
            f"{share:.0%}\nN={n}",
            (x, values[x]),
            textcoords="offset points",
            xytext=(0, 4 if above else -4),
            ha="center",
            va="bottom" if above else "top",
            fontsize="x-small",
        )
    ax_split.axhline(0.0, color=_COLOR_ZERO_LINE, lw=_ZERO_LINE_WIDTH)
    ax_split.set_xticks(positions)
    ax_split.set_xticklabels(split.group_labels, rotation=30, ha="right", fontsize="small")
    # Headroom for the share/N annotations, on BOTH sides -- the split is
    # signed and the largest bar is not necessarily the positive one. Zero is
    # always included so the sign of every bar is readable.
    span = np.abs(values).max()
    ax_split.set_ylim(
        min(0.0, values.min()) - (_ANNOTATION_HEADROOM - 1.0) * span,
        max(0.0, values.max()) + (_ANNOTATION_HEADROOM - 1.0) * span,
    )
    ax_split.set_xlabel(split.label)
    ax_split.set_ylabel(r"mean per-center $\hat\zeta_1$  [km/s]")
    ax_split.set_title(f"5 - mechanism split in {shell_text}")
    ax_split.grid(alpha=_GRID_ALPHA, axis="y")

    fig.tight_layout()

    # --- 6: the monopole companion (hard rule 6), its own figure ------------
    full = restack(stack)
    fig_mono, ax_mono = plt.subplots(figsize=_MONOPOLE_FIGSIZE)
    ax_mono.stairs(full.monopole_norm, edges, color=_COLOR_MONOPOLE)
    _shell_axis(ax_mono, cfg.figure_max_radius)
    ax_mono.set_ylabel(r"normalized monopole  [km/s]")
    ax_mono.set_title("6 - monopole companion (hard rule 6)")
    fig_mono.tight_layout()

    return DipoleFigures(dipole=fig, monopole=fig_mono)
