"""
mass_diagnostics.py
-------------------
The mass funnel: which halos a run actually measured, by mass, at every stage
of its selection.

Cross-cutting, used by all three pipelines. A run narrows its population four
times -- catalog -> spatial carve -> seeded candidate draw -> core cut and
speed floor -- and reports each narrowing only as a COUNT. A count says how
many halos survived; it does not say WHICH, and on a catalog spanning six
decades in mass those are different questions with different answers.

What this exists to make visible
--------------------------------
The full MDPL2 catalog reaches down to 2 * `conventions.PARTICLE_MASS`, and
about half of it lies below 33 particles, where Rockstar peculiar velocities
are not merely noisy but biased. The candidate draw
(`dvcorr.pipeline.velocity_centered.draw_candidates_from_arrays`) is uniform
over the carved population and therefore mass-blind, so on an uncut catalog
the velocity CENTERS -- the objects whose motion the dipole is measured from
-- inherit that mass distribution. Whether that is happening, and how far it
goes, is a question about the selection, and this module answers it directly
rather than leaving it to be inferred from the catalog's own histogram.

The two comparisons the figure is built around:

- **carved against catalog.** The carve is spatial and should therefore be
  mass-blind. If the two distributions differ in shape, something has coupled
  position to mass and the carve is not doing what it claims.
- **centers against carved, split by distinct/subhalo.** This is the one that
  carries a physical consequence. Subhalo velocities are orbital rather than
  large-scale flow, so an over-representation of subhalos among centers
  dilutes the dipole and inflates the monopole at small separations.

Reading the figure
------------------
Bins are uniform in log10(mvir) and the x-axis is LINEAR in log10(mvir) --
cartesian axes even under logarithmic binning, matching this project's plotting
convention. The top axis is the same axis relabelled in particle number
(`mvir / conventions.PARTICLE_MASS`, exact by quantization -- see that
constant), not a second scale for a second measure; the markers on it are
`conventions.RESOLVED_PARTICLE_COUNT` and
`conventions.CONVERGED_PARTICLE_COUNT`.

Stage colors are a single hue, light to dark, because the stages are ORDERED
and nested rather than independent categories: each population is a subset of
the one before it. Lightness carries that ordering, and monotone lightness is
also what keeps the four curves separable under color-vision deficiency
(verified: minimum adjacent OKLab dE ~ 15 under simulated deuteranopia and
protanopia, against a target of 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from dvcorr import conventions

# Plot styling -- named here rather than inline (hard rule 4).
#: Sequential single-hue ramp, light -> dark, one step per funnel stage.
_STAGE_COLORS: tuple[str, ...] = ("#bcd7f5", "#7fb2e8", "#2a78d6", "#12406f")
_COLOR_SUBHALO = "#eb6834"
_COLOR_MARKER_LINE = "0.45"
_LABEL_COLOR = "#111111"
_FIGSIZE = (9.0, 6.0)
_GRID_ALPHA = 0.3
_STAGE_LINE_WIDTH = 1.8
_SPLIT_LINE_WIDTH = 1.3
_MARKER_LINE_WIDTH = 0.9
_FILL_ALPHA = 0.12

#: Histogram bins across the catalog's full log10 mass range. Enough to
#: resolve the particle-count quantization at the low-mass end without turning
#: the high-mass tail into single-halo spikes.
_DEFAULT_MASS_BINS: int = 60

#: Stage labels, in funnel order. The order is load-bearing: it pairs with
#: `_STAGE_COLORS` and encodes the nesting each stage's population obeys.
_STAGE_LABELS: tuple[str, ...] = ("catalog", "carved", "candidates", "centers")


@dataclass(frozen=True)
class MassFunnel:
    """Virial masses of the halos surviving each stage of a run's selection.

    Every array holds `mvir` in h^-1 M_sun for one stage, and each stage is a
    subset of the one before it. Built by `mass_funnel`.

    Attributes
    ----------
    catalog : ndarray, shape (N_total,)
        Every halo the catalog supplied, after its own mass / subhalo cuts and
        before the spatial carve.
    carved : ndarray, shape (N_carved,)
        Halos inside the analysis sub-volume. These are the density TRACERS.
    candidates : ndarray, shape (N_candidates,)
        The seeded subsample offered as velocity centers.
    centers : ndarray, shape (N_c,)
        Candidates surviving the core cut and speed floor -- the halos whose
        velocities the dipole is actually measured from.
    is_distinct_centers : ndarray, shape (N_c,), bool
        Whether each surviving center is a distinct halo rather than a subhalo.
    catalog_name : str
        Which catalog this run read.
    catalog_cuts : str
        `CatalogConfig.describe_cuts()`, so a figure or printout carries the
        selection that produced it rather than relying on it being remembered.
    """

    catalog: np.ndarray
    carved: np.ndarray
    candidates: np.ndarray
    centers: np.ndarray
    is_distinct_centers: np.ndarray
    catalog_name: str
    catalog_cuts: str

    @property
    def stages(self) -> tuple[np.ndarray, ...]:
        """The four stage arrays in funnel order, aligned with `_STAGE_LABELS`."""
        return (self.catalog, self.carved, self.candidates, self.centers)


def particle_counts(mvir: np.ndarray) -> np.ndarray:
    """Convert virial masses to particle counts.

    Exact rather than approximate: Rockstar's `mvir` is quantized in multiples
    of `conventions.PARTICLE_MASS`, so this recovers the integer particle count
    the halo was built from (up to float representation).

    Parameters
    ----------
    mvir : ndarray, shape (N,)
        Virial masses, h^-1 M_sun.

    Returns
    -------
    ndarray, shape (N,)
        Particle counts.
    """
    return np.asarray(mvir, dtype=float) / conventions.PARTICLE_MASS


def mass_funnel(
    catalog_mvir: np.ndarray,
    carved_mvir: np.ndarray,
    candidate_mvir: np.ndarray,
    center_mvir: np.ndarray | None,
    is_distinct_centers: np.ndarray | None,
    catalog_name: str,
    catalog_cuts: str,
) -> MassFunnel:
    """Assemble a `MassFunnel` from the arrays each pipeline stage already carries.

    Takes bare arrays rather than the stage dataclasses so that all three
    pipelines can call it: the velocity-centered and velocity-frame pipelines
    hold a `CarvedHalos`, the redshift-space one a `BufferedCarve`, and their
    center sets are different types too. The arrays themselves are the common
    currency.

    Parameters
    ----------
    catalog_mvir : ndarray, shape (N_total,)
        E.g. `CarvedHalos.catalog_mvir` or `BufferedCarve.catalog_mvir`.
    carved_mvir : ndarray, shape (N_carved,)
        E.g. `CarvedHalos.mvir` or `BufferedCarve.mvir_core`.
    candidate_mvir : ndarray, shape (N_candidates,)
        `CandidateCenters.mvir`.
    center_mvir : ndarray, shape (N_c,) | None
        `SharedCenterSet.mvir_centers` / `RedshiftCenterSet.mvir_centers`.
    is_distinct_centers : ndarray, shape (N_c,), bool | None
        The matching distinct-halo flags.
    catalog_name, catalog_cuts : str
        `CatalogConfig.name` and `CatalogConfig.describe_cuts()`.

    Returns
    -------
    MassFunnel

    Raises
    ------
    ValueError
        If `center_mvir` is None, or its length does not match
        `is_distinct_centers`. A funnel whose final stage is missing is not a
        funnel -- the centers are the whole point of the diagnostic -- and a
        length mismatch means the two arrays came from different cuts, which
        would mislabel every subhalo in the figure.
    """
    if center_mvir is None or is_distinct_centers is None:
        raise ValueError(
            "mass_funnel: center_mvir and is_distinct_centers are both "
            "required. Pass mvir_candidates / is_distinct_candidates to "
            "select_shared_centers so they are carried through the cuts."
        )
    center_mvir = np.asarray(center_mvir)
    is_distinct_centers = np.asarray(is_distinct_centers)
    if center_mvir.shape != is_distinct_centers.shape:
        raise ValueError(
            f"mass_funnel: center_mvir {center_mvir.shape} and "
            f"is_distinct_centers {is_distinct_centers.shape} must match; a "
            "mismatch means they survived different cuts."
        )

    return MassFunnel(
        catalog=np.asarray(catalog_mvir),
        carved=np.asarray(carved_mvir),
        candidates=np.asarray(candidate_mvir),
        centers=center_mvir,
        is_distinct_centers=is_distinct_centers,
        catalog_name=catalog_name,
        catalog_cuts=catalog_cuts,
    )


def print_mass_funnel(funnel: MassFunnel) -> None:
    """Print the funnel as a table: count, median mass, and resolution fractions.

    The companion to the figure, for runs that are not plotting. The column
    that usually matters most is `< N_res`, the fraction of a stage below
    `conventions.RESOLVED_PARTICLE_COUNT` particles: on an uncut catalog it is
    large at every stage, and it being equally large at the `centers` stage is
    the concrete statement that the dipole is being measured from unresolved
    halos.

    Parameters
    ----------
    funnel : MassFunnel
    """
    n_res = conventions.RESOLVED_PARTICLE_COUNT
    n_conv = conventions.CONVERGED_PARTICLE_COUNT
    print(f"\nmass funnel [{funnel.catalog_cuts}]")
    print(
        f"  {'stage':11s} {'N':>13s} {'median M':>11s} {'median N_p':>11s} "
        f"{f'< {n_res}p':>8s} {f'< {n_conv}p':>8s} {'subhalo':>8s}"
    )
    for label, mvir in zip(_STAGE_LABELS, funnel.stages):
        if mvir.size == 0:
            print(f"  {label:11s} {0:>13,}")
            continue
        n_particles = particle_counts(mvir)
        median = float(np.median(mvir))
        frac_res = float(np.mean(n_particles < n_res))
        frac_conv = float(np.mean(n_particles < n_conv))
        # The subhalo fraction is only known for the centers: it is the only
        # stage whose distinct-halo flags are carried through the cuts.
        if label == _STAGE_LABELS[-1]:
            subhalo = f"{100.0 * float(np.mean(~funnel.is_distinct_centers)):7.2f}%"
        else:
            subhalo = f"{'-':>8s}"
        print(
            f"  {label:11s} {mvir.size:>13,} {median:11.3e} "
            f"{np.median(n_particles):11.1f} {100.0 * frac_res:7.2f}% "
            f"{100.0 * frac_conv:7.2f}% {subhalo}"
        )


def make_mass_histogram_figure(
    funnel: MassFunnel, bins: int = _DEFAULT_MASS_BINS
) -> plt.Figure:
    """Plot log10(mvir) distributions for every funnel stage on one axis.

    See the module docstring for what the figure is for and how its axes are
    built. Counts are on a log y-axis: the stages span ~1e8 to ~1e3 objects, and
    a linear count axis would render every stage but the catalog as a flat line.

    Parameters
    ----------
    funnel : MassFunnel
    bins : int
        Number of bins, uniform in log10(mvir), spanning the catalog's range.

    Returns
    -------
    plt.Figure
        Two-panel figure: stage distributions above, the centers'
        distinct/subhalo split below.
    """
    log_mvir = np.log10(funnel.catalog)
    edges = np.linspace(log_mvir.min(), log_mvir.max(), bins + 1)

    fig, (ax_stages, ax_split) = plt.subplots(
        2, 1, figsize=_FIGSIZE, sharex=True, height_ratios=(2, 1)
    )

    for label, mvir, color in zip(_STAGE_LABELS, funnel.stages, _STAGE_COLORS):
        if mvir.size == 0:
            continue
        counts, _ = np.histogram(np.log10(mvir), bins=edges)
        ax_stages.stairs(
            counts, edges, color=color, linewidth=_STAGE_LINE_WIDTH,
            label=f"{label} (N = {mvir.size:,})",
        )
        ax_stages.stairs(counts, edges, color=color, fill=True, alpha=_FILL_ALPHA)

    ax_stages.set_yscale("log")
    ax_stages.set_ylabel("halos per bin", color=_LABEL_COLOR)
    ax_stages.legend(frameon=False, fontsize="small")
    ax_stages.grid(alpha=_GRID_ALPHA)
    ax_stages.set_title(
        f"mass funnel -- {funnel.catalog_cuts}", color=_LABEL_COLOR
    )

    # Lower panel: the split that carries a physical consequence.
    centers = funnel.centers
    distinct = centers[funnel.is_distinct_centers]
    subhalos = centers[~funnel.is_distinct_centers]
    for subset, color, label in (
        (distinct, _STAGE_COLORS[-1], f"distinct (N = {distinct.size:,})"),
        (subhalos, _COLOR_SUBHALO, f"subhalo (N = {subhalos.size:,})"),
    ):
        if subset.size == 0:
            continue
        counts, _ = np.histogram(np.log10(subset), bins=edges)
        ax_split.stairs(
            counts, edges, color=color, linewidth=_SPLIT_LINE_WIDTH, label=label
        )
    ax_split.set_yscale("log")
    ax_split.set_ylabel("centers per bin", color=_LABEL_COLOR)
    ax_split.set_xlabel(
        r"$\log_{10}\,(m_{\rm vir}\ /\ h^{-1}M_\odot)$", color=_LABEL_COLOR
    )
    ax_split.legend(frameon=False, fontsize="small")
    ax_split.grid(alpha=_GRID_ALPHA)

    # Resolution markers on both panels, drawn in mass coordinates so they sit
    # on the same axis as the data.
    for n_particles, style in (
        (conventions.RESOLVED_PARTICLE_COUNT, ":"),
        (conventions.CONVERGED_PARTICLE_COUNT, "--"),
    ):
        x = np.log10(n_particles * conventions.PARTICLE_MASS)
        for ax in (ax_stages, ax_split):
            ax.axvline(
                x, color=_COLOR_MARKER_LINE, linestyle=style,
                linewidth=_MARKER_LINE_WIDTH,
            )
        ax_stages.annotate(
            f"{n_particles}p", xy=(x, 1.0), xycoords=("data", "axes fraction"),
            xytext=(2, -10), textcoords="offset points",
            color=_COLOR_MARKER_LINE, fontsize="small",
        )

    # Top axis: the SAME axis relabelled in particle number. In log space the
    # conversion is a constant shift, so this is a unit relabelling of one
    # scale -- not a second scale for a second measure.
    shift = np.log10(conventions.PARTICLE_MASS)
    secondary = ax_stages.secondary_xaxis(
        "top", functions=(lambda x: x - shift, lambda x: x + shift)
    )
    secondary.set_xlabel(r"$\log_{10}\,N_{\rm particles}$", color=_LABEL_COLOR)

    fig.tight_layout()
    return fig
