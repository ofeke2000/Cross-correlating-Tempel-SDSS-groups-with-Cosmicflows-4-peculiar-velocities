"""
galaxy_bias.py
--------------
The large-scale linear bias `b` of the halo samples this project measures
`zeta_1` on, in REAL space, at the snapshot redshift, from configuration-space
correlation functions.

Why this module exists
======================
Every `zeta_1` amplitude this project reports is proportional to the bias of
the tracer population whose density field the estimator counts. The linear
prediction

    zeta_1(r) = -(b * a * H * f / (6 * pi**2)) * Int dk k P_lin(k) j_1(k r)

(`zeta_one_linear_prediction`) cannot be overlaid on a measurement until `b`
is a number rather than a hope, and the two catalogs
(`dvcorr.config.catalog.CATALOG_FULL`, `CATALOG_MVIR12`) do not share one.

Definitions -- which `b`, exactly
=================================
The large-scale, linear, REAL-space bias, from three correlation functions
measured on the same box:

    b_cross(r) = xi_hm(r) / xi_mm(r)                 <- primary
    b_auto(r)  = sqrt(xi_hh(r) / xi_mm(r))           <- consistency check
    r_cc(r)    = xi_hm(r) / sqrt(xi_hh(r) xi_mm(r))  <- stochasticity check

`b_cross` is primary because it is LINEAR in `b`: a 5% error in the ratio is a
5% error in `b`, not 2.5%, and it is far less sensitive to halo shot noise than
the auto route. `r_cc` must sit at 1.00 within a few percent
(`R_CC_UNITY_TOLERANCE`) over the fit range; if it does not, "the bias" is not
a single number and no constant fit is meaningful.

Everything is measured at `SNAPSHOT_REDSHIFT` in real space -- raw MDPL2 box
coordinates, never the redshift-space displaced coordinates of
`dvcorr.redshift_space` -- on the production sample as
`dvcorr.pipeline.velocity_centered` selects it: whatever `CatalogConfig` says,
subhalos included by default, no extra cut the estimator does not also apply.

Configuration space, not P(k)
-----------------------------
Deliberate. Poisson shot noise contributes to the halo auto-correlation only at
ZERO separation, so `xi_hh(r > 0)` is unbiased in the mean and needs none of
the `1/n_bar` subtraction `P(k)` would force. Shot noise still inflates the
VARIANCE, which is why `BiasRunConfig.max_tracers` is large.

Subsampling is not a cut
------------------------
`CATALOG_FULL` holds ~127M halos. Pair-counting cost scales as
`N**2 / BOX_SIZE**3 * r_max**3`, so the full population to 120 h^-1 Mpc is
~12 hours where 8M halos is ~3 minutes. A UNIFORM RANDOM subsample is drawn
instead (`load_tracer_positions`). This is not a selection: a
position-independent random dilution leaves `xi` unchanged in the mean and only
adds shot noise, which -- see above -- lives at r = 0. The mass distribution,
the subhalo fraction and the clustering of the subsample are those of the
parent population, which is exactly what "the exact production sample" requires.

The matter field
================
`xi_mm` and `xi_hm` need the dark-matter particles, and MDPL2's are NOT
distributed as a downsampled table -- see `MATTER_FIELD_AVAILABILITY`. Until a
particle set is in hand, `linear_matter_xi` supplies a THEORY `xi_mm` from CAMB
at the MDPL2 cosmology. Read `BiasEstimates.matter_source` before quoting any
number that came through it: a theory `xi_mm` inherits every sigma8 and
snapshot-redshift error, and cannot produce `r_cc` at all (there is no measured
`xi_hm` to put in the numerator). It is a placeholder, not the answer.

Independent cross-check
=======================
`effective_tinker_bias` averages the Tinker (2010) `b(M)` over the sample's own
mass function -- no correlation function, no matter field, no shared failure
mode with the route above. Subhalos take their HOST's mass
(`resolve_host_masses`): a satellite's large-scale bias is its host's. The two
routes disagreeing by more than `TINKER_CROSSCHECK_TOLERANCE` means one of them
is wrong, and it is usually a mass-definition mismatch -- hence
`HALO_MASS_DEFINITION`, pinned to Rockstar's `mvir`.

Layering
========
Imports `dvcorr.conventions`, `dvcorr.config`; does NOT import the other
`dvcorr.pipeline` modules' run machinery except `velocity_centered`'s loader
config types, and nothing under `dvcorr` imports this module back. Corrfunc,
camb, mcfit and colossus are imported lazily inside the functions that need
them, so importing this module (and running the tests that do not exercise
those paths) costs nothing and does not require them installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dvcorr import conventions
from dvcorr.config import CatalogConfig, CosmologyConfig, PathsConfig
from dvcorr.config.catalog import CATALOG_FULL

# ---------------------------------------------------------------------------
# Frozen-ish analysis constants (CLAUDE.md hard rule 4)
# ---------------------------------------------------------------------------

#: Redshift of MDPL2 snapshot 125 / snapdir 130. The whole module is a
#: single-snapshot measurement, so this is metadata rather than a knob.
SNAPSHOT_REDSHIFT: float = 0.0

#: Radial range and binning of every correlation function here, h^-1 Mpc.
#: Logarithmic, from well inside the one-halo regime out past the fit range,
#: so the flatness of b(r) can be read against scales where it is NOT expected
#: to be flat. The ceiling stays under `conventions.MAX_ANALYSIS_RADIUS`.
XI_MIN_RADIUS: float = 1.0
XI_MAX_RADIUS: float = 120.0
XI_N_BINS: int = 24

#: The range RATIOS are fitted over, h^-1 Mpc -- the `zeta_1` amplitude ratio
#: and the matter-field-free bias ratio. Large enough that linear,
#: deterministic bias is expected to hold; small enough that a 1 h^-1 Gpc box
#: still has many independent modes. A ratio may be fitted this low because
#: `xi_mm` and the box's sample variance cancel in it exactly.
FIT_MIN_RADIUS: float = 20.0
FIT_MAX_RADIUS: float = 60.0

#: The range an ABSOLUTE bias is fitted over, h^-1 Mpc. Starts higher than
#: `FIT_MIN_RADIUS` because nothing cancels here: the denominator is the LINEAR
#: `xi_mm` (see `theory_matter_xi`), and below ~30 h^-1 Mpc the true `xi_mm`
#: exceeds it by enough to drag `b_auto` measurably low. Fitting a ratio and
#: fitting an absolute amplitude are different jobs with different floors, so
#: they get different constants rather than one shared compromise.
ABSOLUTE_BIAS_MIN_RADIUS: float = 30.0

#: How close to 1 the cross-correlation coefficient must sit over the fit
#: range for a single-number bias to be meaningful ("a few percent").
R_CC_UNITY_TOLERANCE: float = 0.03

#: How closely the mass-function route must reproduce the clustering route
#: before the disagreement counts as a bug in one of them rather than scatter.
TINKER_CROSSCHECK_TOLERANCE: float = 0.10

#: Tinker et al. (2010) halo bias, and the mass definition it is evaluated at.
#: Rockstar's `mvir` is the Bryan & Norman (1998) virial overdensity, which
#: colossus spells "vir"; feeding it a `200m` mass instead is the single most
#: likely cause of a 10%-level disagreement with the clustering route.
TINKER_BIAS_MODEL: str = "tinker10"
HALO_MASS_DEFINITION: str = "vir"

#: Seed for the uniform random tracer subsample. Not a null -- it does not
#: belong to the null-seed ladder documented in docs/architecture.md
#: ("Every null gets its own random stream", seeds 42-48) -- but it is
#: allocated as the next free integer anyway so the two can never collide.
SUBSAMPLE_SEED: int = 49

#: Default cap on the number of halos entering a pair count. See the module
#: docstring, "Subsampling is not a cut", for why capping is legitimate and
#: what it costs.
DEFAULT_MAX_TRACERS: int = 8_000_000

#: `matter_source` tags carried on `BiasEstimates`, so a plot or a printed
#: number always says which matter field it came from.
MATTER_SOURCE_PARTICLES: str = "particles"
MATTER_SOURCE_CAMB_LINEAR: str = "camb-linear (PLACEHOLDER)"
MATTER_SOURCE_CAMB_HALOFIT: str = "camb-halofit (PLACEHOLDER)"

#: What was actually checked, on 2026-08-09, when looking for the MDPL2
#: particle subsample this module's primary estimator needs. Recorded here
#: rather than in a commit message because the next person to reach for
#: `xi_mm` will reach for it from this file.
MATTER_FIELD_AVAILABILITY: str = (
    "MDPL2 has NO downsampled particle table. The CosmoSim TAP tableset "
    "(https://www.cosmosim.org/tap/tables) lists exactly 13 mdpl2.* tables -- "
    "availhalos, fof, fof1-5, galacticus, linklength, redshifts, rockstar, "
    "sag, sage -- and no particle table; the equivalents that DO exist for the "
    "older runs (bolshoi_particles, mdr1_particles) have no mdpl2 counterpart. "
    "The only MDPL2 particle product is the full GADGET snapshot "
    "(snapdir_130 = z = 0): 1270 files, ~1.1 TB, login-gated (HTTP 403 "
    "unauthenticated), and spatially decomposed, so a partial download is a "
    "contiguous sub-region rather than a fair subsample and would need "
    "Landy-Szalay against its own footprint. Skies & Universes lists particle "
    "sets for MDPL/SMD/BigMD but not for MDPL2, and its download host "
    "(astronomy.nmsu.edu/aklypin/SUsimulations) no longer serves them."
)

#: CAMB sampling for the theory `P_lin(k)`, h/Mpc. `k_max` is well past the
#: scales the fit range needs but keeps the Hankel transform's high-k tail
#: honest; `n_k` is the log-spaced sample count mcfit transforms.
CAMB_K_MIN: float = 1e-5
CAMB_K_MAX: float = 50.0
CAMB_N_K: int = 2048

#: Cap on how many times `resolve_host_masses` follows a `pid` pointer.
#: MEASURED on `CATALOG_FULL`: `pid` is the IMMEDIATE (least massive) parent, so
#: the chain is genuinely deep -- 15,429,923 halos climb one hop, and the
#: remainder falls off by a factor ~5.7 per level (2,309,109 / 380,529 / 65,134
#: / 11,523 / 1,994 / 309 / 36 / 3), reaching zero at the NINTH hop. Eight was
#: not enough and mistook a real hierarchy for a cycle. Cycles are caught by
#: the strict-decrease check in `resolve_host_masses`, not by this number, which
#: is only a backstop.
MAX_HOST_CHAIN_DEPTH: int = 32

#: Rockstar's "no parent" sentinel; a halo with this `pid` IS a host.
DISTINCT_PARENT_ID: int = -1

#: Filename suffix of the cached host-mass column (CLAUDE.md hard rule 7:
#: check column -> skip if present -> compute -> save). Row-aligned with the
#: catalog Parquet, which `dvcorr.pipeline.catalog_conversion` writes in the
#: source CSV's row order.
HOST_MASS_CACHE_SUFFIX: str = "_host_mvir.parquet"
HOST_MASS_COLUMN: str = "host_mvir"

#: CSV chunk size for the host-mass pass. The full catalog is 11.2 GB / 127M
#: rows, so the pass is streamed rather than loaded.
HOST_MASS_CHUNK_ROWS: int = 5_000_000


def default_xi_bins() -> np.ndarray:
    """Logarithmic radial bin edges for every correlation function here, (XI_N_BINS + 1,).

    Returns
    -------
    ndarray, shape (XI_N_BINS + 1,)
        Strictly increasing edges from `XI_MIN_RADIUS` to `XI_MAX_RADIUS`,
        h^-1 Mpc.
    """
    return np.geomspace(XI_MIN_RADIUS, XI_MAX_RADIUS, XI_N_BINS + 1)


@dataclass
class BiasRunConfig:
    """One bias measurement: which halos, how many, how many threads.

    Attributes
    ----------
    catalog : CatalogConfig
        Which halos. Defaults to `CatalogConfig()` -- the production sample:
        `CATALOG_FULL`, no mass cut, subhalos included.
    max_tracers : int
        Cap on the number of halos entering a pair count; `DEFAULT_MAX_TRACERS`
        by default. A sample smaller than this is used whole and no draw is
        made. See the module docstring, "Subsampling is not a cut".
    seed : int
        Seed for the subsample draw (`SUBSAMPLE_SEED`).
    n_threads : int
        OpenMP threads handed to Corrfunc; defaults to the machine's CPU count.
    bins : ndarray, shape (B + 1,)
        Radial bin edges, h^-1 Mpc (`default_xi_bins`).
    fit_min_radius, fit_max_radius : float
        The constant-bias fit range, h^-1 Mpc (`FIT_MIN_RADIUS`,
        `FIT_MAX_RADIUS`). Fields rather than bare constants precisely because
        the acceptance criterion is flatness: if `b(r)` is still rising at
        `FIT_MIN_RADIUS`, the correct response is to move this range up and
        report that it moved.
    """

    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    max_tracers: int = DEFAULT_MAX_TRACERS
    seed: int = SUBSAMPLE_SEED
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 1)
    bins: np.ndarray = field(default_factory=default_xi_bins)
    fit_min_radius: float = FIT_MIN_RADIUS
    fit_max_radius: float = FIT_MAX_RADIUS

    def __post_init__(self) -> None:
        if self.max_tracers < 2:
            raise ValueError(
                f"BiasRunConfig: max_tracers must be at least 2, got {self.max_tracers}."
            )
        if self.n_threads < 1:
            raise ValueError(
                f"BiasRunConfig: n_threads must be >= 1, got {self.n_threads}."
            )
        edges = np.asarray(self.bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
            raise ValueError("BiasRunConfig: bins must be 1-D and strictly increasing.")
        if edges[-1] > conventions.MAX_ANALYSIS_RADIUS:
            raise ValueError(
                f"BiasRunConfig: outermost bin edge ({edges[-1]}) exceeds "
                f"conventions.MAX_ANALYSIS_RADIUS ({conventions.MAX_ANALYSIS_RADIUS}); "
                "periodic minimum-image geometry is not well defined beyond it "
                "(CLAUDE.md hard rule 3)."
            )
        self.bins = edges
        if not self.fit_max_radius > self.fit_min_radius:
            raise ValueError(
                f"BiasRunConfig: fit_max_radius ({self.fit_max_radius}) must exceed "
                f"fit_min_radius ({self.fit_min_radius})."
            )


# ---------------------------------------------------------------------------
# Tracer loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TracerSample:
    """A population of halo positions ready for pair counting.

    Attributes
    ----------
    pos : ndarray, shape (N, 3)
        Box coordinates, comoving h^-1 Mpc, float64 (Corrfunc's double
        precision path). Half-open [0, BOX_SIZE) -- guaranteed by
        `dvcorr.pipeline.catalog_conversion`'s fold, not re-imposed here.
    n_total : int
        Size of the post-cut population BEFORE the subsample draw. This, not
        `n_used`, is the number density of the tracer population.
    label : str
        Human-readable description of the cuts, for plot titles and logs.
    subsampled : bool
        Whether a draw was made (`n_used < n_total`).
    """

    pos: np.ndarray
    n_total: int
    label: str
    subsampled: bool

    @property
    def n_used(self) -> int:
        """Number of halos actually pair-counted."""
        return int(self.pos.shape[0])

    @property
    def number_density(self) -> float:
        """Number density of the POST-CUT population, halos per (h^-1 Mpc)^3.

        Built from `n_total`, i.e. before the subsample draw: the draw changes
        how many objects are counted, never how many the universe holds. The
        density that enters an `xi` normalization is `n_used / BOX_SIZE**3`
        and is computed inside the estimators from the array actually passed.
        """
        return self.n_total / conventions.BOX_SIZE**3


def load_tracer_positions(
    cfg: BiasRunConfig, paths: PathsConfig | None = None
) -> TracerSample:
    """Read halo positions for `cfg.catalog`, apply its cuts, subsample.

    Reads only the columns the cuts require: with no mass bounds and subhalos
    included (the production default) that is `x, y, z` alone, so the full
    catalog costs ~1.5 GB rather than the ~7 GB a full load would.

    Parameters
    ----------
    cfg : BiasRunConfig
    paths : PathsConfig, optional
        Defaults to `PathsConfig()`.

    Returns
    -------
    TracerSample
        `pos` shape (N, 3) with N = min(post-cut count, `cfg.max_tracers`).

    Notes
    -----
    The draw is `numpy.random.default_rng(cfg.seed).choice(..., replace=False)`
    over ROW INDICES of the post-cut array. Unlike
    `dvcorr.pipeline.velocity_centered.draw_candidates_from_arrays`, no
    canonical position sort is applied first: that sort exists so the SAME
    halos are drawn from either catalog file, which matters when two runs are
    being compared halo-for-halo. Here the two catalogs are deliberately
    different populations and the draw is a variance-reduction convenience,
    not an identity -- `xi` is invariant under which fair subsample was taken.
    """
    import pyarrow.parquet as pq

    paths = paths or PathsConfig()
    catalog = cfg.catalog
    parquet_path = paths.halo_catalog(catalog.name)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"load_tracer_positions: {parquet_path} does not exist. Run "
            "`python -m scripts.convert_mdpl2_catalog` first (CSV -> Parquet)."
        )

    needs_mass = catalog.mass_min is not None or catalog.mass_max is not None
    columns = ["x", "y", "z"]
    if needs_mass:
        columns.append("mass")
    if not catalog.include_subhalos:
        columns.append("is_distinct")
    columns = [conventions.HALO_COLUMNS.get(c, c) for c in columns]

    table = pq.read_table(parquet_path, columns=columns)
    pos = np.column_stack(
        [table[c].to_numpy().astype(np.float64, copy=False) for c in conventions.POSITION_COLUMNS]
    )

    keep = np.ones(pos.shape[0], dtype=bool)
    if needs_mass:
        mvir = table[conventions.HALO_COLUMNS["mass"]].to_numpy()
        if catalog.mass_min is not None:
            keep &= mvir >= catalog.mass_min
        if catalog.mass_max is not None:
            keep &= mvir <= catalog.mass_max
    if not catalog.include_subhalos:
        keep &= table["is_distinct"].to_numpy()
    if not keep.all():
        pos = pos[keep]

    n_total = int(pos.shape[0])
    subsampled = n_total > cfg.max_tracers
    if subsampled:
        rng = np.random.default_rng(cfg.seed)
        pick = rng.choice(n_total, size=cfg.max_tracers, replace=False)
        pick.sort()  # locality: Corrfunc grids the array anyway, but this keeps the read sequential
        pos = pos[pick]

    return TracerSample(
        pos=np.ascontiguousarray(pos),
        n_total=n_total,
        label=catalog.describe_cuts(),
        subsampled=subsampled,
    )


def load_velocity_sample(
    cfg: BiasRunConfig, paths: PathsConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Positions AND velocities for `measure_velocity_density_correlation`.

    Separate from `load_tracer_positions` rather than a flag on it: the pair
    counters never want velocities and the full catalog's are ~3 GB, so the
    common path should not be able to load them by accident. Uses the same
    catalog, cuts, cap and seed, so the two samples are drawn from the same
    population -- though not the same halos, since this reads different
    columns and applies the draw independently.

    Parameters
    ----------
    cfg : BiasRunConfig
    paths : PathsConfig, optional

    Returns
    -------
    pos : ndarray, shape (N, 3)
        Box coordinates, comoving h^-1 Mpc.
    vel : ndarray, shape (N, 3)
        Peculiar velocities, km/s.
    """
    import pyarrow.parquet as pq

    paths = paths or PathsConfig()
    columns = list(conventions.POSITION_COLUMNS) + list(conventions.VELOCITY_COLUMNS)
    table = pq.read_table(paths.halo_catalog(cfg.catalog.name), columns=columns)
    n_rows = table.num_rows

    if n_rows > cfg.max_tracers:
        rng = np.random.default_rng(cfg.seed)
        pick = np.sort(rng.choice(n_rows, size=cfg.max_tracers, replace=False))
    else:
        pick = slice(None)

    def _stack(names: tuple[str, ...]) -> np.ndarray:
        return np.column_stack([table[c].to_numpy()[pick].astype(np.float64) for c in names])

    return _stack(conventions.POSITION_COLUMNS), _stack(conventions.VELOCITY_COLUMNS)


# ---------------------------------------------------------------------------
# Correlation functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationFunction:
    """One measured `xi(r)` in a periodic box.

    Attributes
    ----------
    edges : ndarray, shape (B + 1,)
        Radial bin edges, h^-1 Mpc.
    r_eff : ndarray, shape (B,)
        Pair-count-weighted mean separation in each bin, h^-1 Mpc -- the
        abscissa the ordinate was actually measured at. Deliberately NOT the
        bin midpoint and not
        `dvcorr.config.shells.volume_weighted_shell_radii`: those are the right
        abscissa for a statistic whose denominator is a uniform-field shell
        volume, which is exactly what `xi` divides out. Here the numerator is a
        clustered pair count and the weight is `r**2 (1 + xi(r)) dr`, so the
        estimator's own `ravg` is the honest answer.
    xi : ndarray, shape (B,)
        The correlation function in each bin.
    npairs : ndarray, shape (B,)
        Raw pair count in each bin.
    n1, n2 : int
        Object counts of the two samples (`n1 == n2` for an auto-correlation).
    label : str
        Human-readable name, e.g. "hh (full)".
    """

    edges: np.ndarray
    r_eff: np.ndarray
    xi: np.ndarray
    npairs: np.ndarray
    n1: int
    n2: int
    label: str

    @property
    def is_auto(self) -> bool:
        """Whether this is an auto-correlation."""
        return self.n1 == self.n2


def _shell_volumes(edges: np.ndarray) -> np.ndarray:
    """Volume of each spherical shell, shape (B,), (h^-1 Mpc)^3."""
    # 4/3 pi (r2^3 - r1^3): the volume of a spherical shell, part of the
    # geometry rather than a tunable number.
    return (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def periodic_auto_xi(
    sample: TracerSample, cfg: BiasRunConfig, label: str
) -> CorrelationFunction:
    """`xi(r)` of one sample with itself, periodic-box natural estimator.

    Uses Corrfunc's `theory.xi`, which counts `DD` with the minimum-image
    convention over `conventions.BOX_SIZE` and divides by the analytic `RR` of
    a uniform Poisson field at the SAME number density:

        xi(r) = DD(r) / RR(r) - 1,   RR(r) = N * (N - 1) / 2 * V_shell / V_box

    No random catalog and no Landy-Szalay: in a periodic box `RR` is known
    exactly, and LS exists to cancel geometry that a periodic box does not
    have. Shot noise enters only at zero separation (module docstring), so no
    `1/n_bar` term is subtracted.

    Parameters
    ----------
    sample : TracerSample
    cfg : BiasRunConfig
        Supplies `bins` and `n_threads`.
    label : str
        Name carried onto the result.

    Returns
    -------
    CorrelationFunction
        `n1 == n2 == sample.n_used`.
    """
    from Corrfunc.theory.xi import xi as corrfunc_xi

    pos = sample.pos
    result = corrfunc_xi(
        conventions.BOX_SIZE,
        cfg.n_threads,
        cfg.bins,
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        output_ravg=True,
    )
    return CorrelationFunction(
        edges=np.asarray(cfg.bins, dtype=float),
        r_eff=np.asarray(result["ravg"], dtype=float),
        xi=np.asarray(result["xi"], dtype=float),
        npairs=np.asarray(result["npairs"], dtype=np.int64),
        n1=sample.n_used,
        n2=sample.n_used,
        label=label,
    )


def periodic_cross_xi(
    sample_a: TracerSample, sample_b: TracerSample, cfg: BiasRunConfig, label: str
) -> CorrelationFunction:
    """`xi(r)` between two different samples in the same periodic box.

    The cross counterpart of `periodic_auto_xi`. Corrfunc has no periodic-box
    cross-`xi` convenience wrapper, so `DD(autocorr=0, ...)` supplies the pair
    count and the uniform-field expectation is formed here:

        xi(r) = DD(r) / (N_a * N_b * V_shell / V_box) - 1

    -- ordered pairs (each a-b pair counted once), hence no factor of 1/2,
    which is the one place a cross-correlation differs from the auto case and
    the one place a factor of 2 can silently halve the answer.

    Parameters
    ----------
    sample_a, sample_b : TracerSample
        Must occupy the same box. Order is irrelevant to the result.
    cfg : BiasRunConfig
    label : str

    Returns
    -------
    CorrelationFunction
    """
    from Corrfunc.theory.DD import DD

    a, b = sample_a.pos, sample_b.pos
    result = DD(
        0,  # autocorr = 0: two distinct samples
        cfg.n_threads,
        cfg.bins,
        a[:, 0],
        a[:, 1],
        a[:, 2],
        X2=b[:, 0],
        Y2=b[:, 1],
        Z2=b[:, 2],
        periodic=True,
        boxsize=conventions.BOX_SIZE,
        output_ravg=True,
    )
    edges = np.asarray(cfg.bins, dtype=float)
    npairs = np.asarray(result["npairs"], dtype=np.int64)
    n_a, n_b = sample_a.n_used, sample_b.n_used
    expected = n_a * n_b * _shell_volumes(edges) / conventions.BOX_SIZE**3
    return CorrelationFunction(
        edges=edges,
        r_eff=np.asarray(result["ravg"], dtype=float),
        xi=npairs / expected - 1.0,
        npairs=npairs,
        n1=n_a,
        n2=n_b,
        label=label,
    )


# ---------------------------------------------------------------------------
# Bias estimates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiasEstimates:
    """`b_cross(r)`, `b_auto(r)`, `r_cc(r)` on a shared radial grid.

    Attributes
    ----------
    r : ndarray, shape (B,)
        Pair-weighted separations, h^-1 Mpc, from the halo auto-correlation.
    b_cross : ndarray, shape (B,)
        `xi_hm / xi_mm` -- the primary estimate. All-NaN when no measured
        `xi_hm` was supplied (the theory-`xi_mm` placeholder case).
    b_auto : ndarray, shape (B,)
        `sqrt(xi_hh / xi_mm)`, NaN wherever the ratio is negative (noise
        driving `xi` through zero at large r; never silently clipped to 0).
    r_cc : ndarray, shape (B,)
        `xi_hm / sqrt(xi_hh xi_mm)`. All-NaN in the placeholder case, for the
        same reason as `b_cross`: there is no measured `xi_hm`.
    xi_hh, xi_mm, xi_hm : ndarray, shape (B,)
        The three inputs, on the same grid; `xi_hm` all-NaN when absent.
    matter_source : str
        One of `MATTER_SOURCE_PARTICLES`, `MATTER_SOURCE_CAMB_LINEAR`,
        `MATTER_SOURCE_CAMB_HALOFIT`. Anything but the first means the number
        is a placeholder -- see the module docstring.
    label : str
        The halo sample this describes.
    """

    r: np.ndarray
    b_cross: np.ndarray
    b_auto: np.ndarray
    r_cc: np.ndarray
    xi_hh: np.ndarray
    xi_mm: np.ndarray
    xi_hm: np.ndarray
    matter_source: str
    label: str

    @property
    def is_placeholder(self) -> bool:
        """Whether the matter field is theory rather than measured particles."""
        return self.matter_source != MATTER_SOURCE_PARTICLES


def _safe_sqrt_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """`sqrt(numerator / denominator)`, NaN where the ratio is not positive.

    A negative `xi` in a noisy outer bin is data, not an error; taking its
    square root is what would be the error. NaN propagates visibly through the
    plot and the fit instead of a clipped zero that looks like a measurement.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(numerator, dtype=float) / np.asarray(denominator, dtype=float)
        return np.where(ratio > 0.0, np.sqrt(np.abs(ratio)), np.nan)


def bias_estimates(
    xi_hh: CorrelationFunction,
    xi_mm: np.ndarray,
    matter_source: str,
    xi_hm: np.ndarray | None = None,
    label: str = "",
) -> BiasEstimates:
    """Form `b_cross`, `b_auto` and `r_cc` from correlation functions on one grid.

    Parameters
    ----------
    xi_hh : CorrelationFunction
        The halo auto-correlation; supplies the radial grid.
    xi_mm : ndarray, shape (B,)
        Matter auto-correlation on `xi_hh.r_eff`, measured or theory.
    matter_source : str
        Provenance tag; see `BiasEstimates.matter_source`.
    xi_hm : ndarray, shape (B,), optional
        Halo-matter cross-correlation on the same grid. None (the default)
        leaves `b_cross` and `r_cc` all-NaN -- the placeholder case.
    label : str

    Returns
    -------
    BiasEstimates
    """
    xi_hh_values = np.asarray(xi_hh.xi, dtype=float)
    xi_mm_values = np.asarray(xi_mm, dtype=float)
    if xi_mm_values.shape != xi_hh_values.shape:
        raise ValueError(
            f"bias_estimates: xi_mm has shape {xi_mm_values.shape}, expected "
            f"{xi_hh_values.shape} to match xi_hh."
        )

    if xi_hm is None:
        nan_like = np.full_like(xi_hh_values, np.nan)
        b_cross, r_cc, xi_hm_values = nan_like, nan_like.copy(), nan_like.copy()
    else:
        xi_hm_values = np.asarray(xi_hm, dtype=float)
        if xi_hm_values.shape != xi_hh_values.shape:
            raise ValueError(
                f"bias_estimates: xi_hm has shape {xi_hm_values.shape}, expected "
                f"{xi_hh_values.shape}."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            b_cross = xi_hm_values / xi_mm_values
            r_cc = xi_hm_values / np.sqrt(
                np.where(xi_hh_values * xi_mm_values > 0.0, xi_hh_values * xi_mm_values, np.nan)
            )

    return BiasEstimates(
        r=np.asarray(xi_hh.r_eff, dtype=float),
        b_cross=b_cross,
        b_auto=_safe_sqrt_ratio(xi_hh_values, xi_mm_values),
        r_cc=r_cc,
        xi_hh=xi_hh_values,
        xi_mm=xi_mm_values,
        xi_hm=xi_hm_values,
        matter_source=matter_source,
        label=label or xi_hh.label,
    )


@dataclass(frozen=True)
class ConstantFit:
    """A constant fitted to `y(r)` over a radial window, with its flatness.

    Attributes
    ----------
    value : float
        Unweighted mean over the bins inside the window. Unweighted rather
        than inverse-variance weighted because neighbouring bins of a
        correlation function are strongly correlated: a weighted mean would
        claim a precision the covariance does not support.
    scatter : float
        Standard deviation across those bins (ddof = 1). This is the FLATNESS
        of the curve, not the error on the mean -- it is the number the
        acceptance criterion is read off, and dividing it by sqrt(n_bins) to
        make it look like an error bar would be wrong for the same reason the
        mean is unweighted.
    n_bins : int
        How many finite bins entered.
    r_min, r_max : float
        The window, h^-1 Mpc.
    trend : float
        Least-squares slope of `y` against `log(r)` over the window, per dex.
        The quantitative form of "is it still rising at r_min": a value
        consistent with zero is what flatness means.
    """

    value: float
    scatter: float
    n_bins: int
    r_min: float
    r_max: float
    trend: float

    @property
    def fractional_scatter(self) -> float:
        """`scatter / |value|` -- the flatness as a fraction."""
        return float(self.scatter / abs(self.value)) if self.value else np.nan

    def summary(self) -> str:
        """One-line human-readable summary, for logs and plot annotations."""
        return (
            f"{self.value:.4f} +/- {self.scatter:.4f} (scatter across "
            f"{self.n_bins} bins, {self.r_min:.0f}-{self.r_max:.0f} h^-1 Mpc; "
            f"{100.0 * self.fractional_scatter:.1f}%, trend "
            f"{self.trend:+.4f}/dex)"
        )


def fit_constant(
    r: np.ndarray, y: np.ndarray, r_min: float, r_max: float
) -> ConstantFit:
    """Fit a constant to `y(r)` over `[r_min, r_max]` and measure its flatness.

    Parameters
    ----------
    r : ndarray, shape (B,)
        Separations, h^-1 Mpc.
    y : ndarray, shape (B,)
        The quantity to fit; NaNs are dropped.
    r_min, r_max : float
        Inclusive window, h^-1 Mpc.

    Returns
    -------
    ConstantFit

    Raises
    ------
    ValueError
        If fewer than two finite bins fall inside the window -- a "constant"
        fitted to one point has no scatter to report and no flatness to check,
        which is the whole acceptance criterion.
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = (r >= r_min) & (r <= r_max) & np.isfinite(y) & np.isfinite(r)
    n = int(inside.sum())
    if n < 2:
        raise ValueError(
            f"fit_constant: only {n} finite bin(s) inside [{r_min}, {r_max}] "
            "h^-1 Mpc; a constant needs at least two so its scatter -- the "
            "flatness acceptance criterion -- is defined."
        )
    r_in, y_in = r[inside], y[inside]
    slope = float(np.polyfit(np.log10(r_in), y_in, 1)[0])
    return ConstantFit(
        value=float(np.mean(y_in)),
        scatter=float(np.std(y_in, ddof=1)),
        n_bins=n,
        r_min=float(r_min),
        r_max=float(r_max),
        trend=slope,
    )


def r_cc_is_acceptable(fit: ConstantFit, tolerance: float = R_CC_UNITY_TOLERANCE) -> bool:
    """Whether `r_cc` sits at 1 within `tolerance` over the fit window.

    Both the LEVEL and the SCATTER must clear the tolerance: an `r_cc` that
    averages 1.00 while swinging by 10% bin to bin is not a deterministic bias
    either. Returns False on an all-NaN fit (the placeholder case), because
    "not measured" is not "passed".
    """
    if not np.isfinite(fit.value) or not np.isfinite(fit.scatter):
        return False
    return abs(fit.value - 1.0) <= tolerance and fit.scatter <= tolerance


# ---------------------------------------------------------------------------
# Independent cross-check: Tinker (2010) bias over the sample's mass function
# ---------------------------------------------------------------------------


def _cached_host_mass_path(paths: PathsConfig, catalog_name: str) -> Path:
    """Path of the cached host-mass column for `catalog_name`."""
    catalog_path = paths.halo_catalog(catalog_name, parquet=True)
    return catalog_path.with_name(catalog_path.stem + HOST_MASS_CACHE_SUFFIX)


def resolve_host_masses(
    paths: PathsConfig | None = None,
    catalog_name: str = CATALOG_FULL,
    *,
    chunk_rows: int = HOST_MASS_CHUNK_ROWS,
    verbose: bool = True,
) -> np.ndarray:
    """Host `mvir` for every halo in a catalog, row-aligned with its Parquet.

    A satellite's large-scale bias is its HOST's, so the Tinker cross-check
    must evaluate `b(M)` at the host mass for subhalos and at the halo's own
    mass for hosts. Neither `is_distinct` nor `num_of_subhalos` can supply it:
    the answer lives in a DIFFERENT row, reached through `pid`, and `pid` and
    `rockstarid` are the two columns `dvcorr.pipeline.catalog_conversion`
    consumes and does not write out. The mapping is therefore rebuilt from the
    source CSV.

    Rockstar nests subhaloes: a sub-subhalo's `pid` names another subhalo, not
    the top-level host. The pointer is followed until it reaches a halo with
    `pid == DISTINCT_PARENT_ID`, at most `MAX_HOST_CHAIN_DEPTH` times.

    Cache-and-skip (CLAUDE.md hard rule 7): the result is written next to the
    catalog as `<catalog>_host_mvir.parquet` and re-read on later calls. The
    build is a full streamed pass over the 11.2 GB CSV and takes ~10-15
    minutes for `CATALOG_FULL`; the cached read takes under a second.

    Parameters
    ----------
    paths : PathsConfig, optional
    catalog_name : str
        One of `dvcorr.config.catalog.VALID_CATALOGS`.
    chunk_rows : int
        CSV rows per streamed chunk.
    verbose : bool
        Print progress during the (long) build.

    Returns
    -------
    ndarray, shape (N,), float32
        Host `mvir`, h^-1 M_sun, in the catalog Parquet's row order. Equal to
        the halo's own `mvir` for distinct halos.

    Notes
    -----
    Row alignment rests on `catalog_conversion` writing Parquet in the CSV's
    row order; the row count is asserted against the Parquet's, which catches
    the only way that could silently stop being true.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = paths or PathsConfig()
    cache_path = _cached_host_mass_path(paths, catalog_name)
    parquet_path = paths.halo_catalog(catalog_name, parquet=True)
    n_rows = pq.ParquetFile(parquet_path).metadata.num_rows

    if cache_path.exists():
        cached = pq.read_table(cache_path)[HOST_MASS_COLUMN].to_numpy()
        if cached.size != n_rows:
            raise ValueError(
                f"resolve_host_masses: cached {cache_path} has {cached.size} rows "
                f"but {parquet_path} has {n_rows}; delete the cache and rebuild."
            )
        return cached.astype(np.float32, copy=False)

    csv_path = paths.halo_catalog(catalog_name, parquet=False)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"resolve_host_masses: {csv_path} is needed to rebuild the host-mass "
            "column (pid / rockstarid are not carried into the Parquet) and does "
            "not exist."
        )

    id_col = conventions.HALO_COLUMNS["id"]
    parent_col = conventions.HALO_COLUMNS["parent_id"]
    mass_col = conventions.HALO_COLUMNS["mass"]

    ids_chunks: list[np.ndarray] = []
    pids_chunks: list[np.ndarray] = []
    mass_chunks: list[np.ndarray] = []
    reader = pd.read_csv(
        csv_path,
        usecols=[id_col, parent_col, mass_col],
        dtype={id_col: "int64", parent_col: "int64", mass_col: "float32"},
        chunksize=chunk_rows,
    )
    for n_chunk, chunk in enumerate(reader, start=1):
        ids_chunks.append(chunk[id_col].to_numpy())
        pids_chunks.append(chunk[parent_col].to_numpy())
        mass_chunks.append(chunk[mass_col].to_numpy())
        if verbose:
            print(f"  host-mass pass: {n_chunk * chunk_rows:,} rows read", flush=True)

    ids = np.concatenate(ids_chunks)
    pids = np.concatenate(pids_chunks)
    mvir = np.concatenate(mass_chunks)
    del ids_chunks, pids_chunks, mass_chunks
    if ids.size != n_rows:
        raise ValueError(
            f"resolve_host_masses: CSV has {ids.size} rows but Parquet has "
            f"{n_rows}; the two are not row-aligned."
        )

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    def _row_of(target_ids: np.ndarray) -> np.ndarray:
        """Row index of each id, or -1 where the id is absent."""
        slot = np.searchsorted(sorted_ids, target_ids)
        slot_clipped = np.clip(slot, 0, sorted_ids.size - 1)
        found = sorted_ids[slot_clipped] == target_ids
        return np.where(found, order[slot_clipped], -1)

    # Follow pid -> row -> that row's pid, one level per iteration. The chase
    # state is `current_parent`, the parent id of the halo `host_row` currently
    # points at -- NOT the original row's `pid`, which never changes and would
    # make every iteration re-resolve the same first hop.
    host_row = np.arange(n_rows, dtype=np.int64)
    current_parent = pids.copy()
    previous_active = n_rows + 1
    for depth in range(MAX_HOST_CHAIN_DEPTH):
        active = np.flatnonzero(current_parent != DISTINCT_PARENT_ID)
        if active.size == 0:
            break
        if active.size >= previous_active:
            # Every hop must strictly shrink the unresolved set. It not doing so
            # is what a cycle in the pid graph looks like; depth alone cannot
            # distinguish that from a legitimately deep hierarchy.
            raise ValueError(
                f"resolve_host_masses: hop {depth + 1} left {active.size:,} halos "
                f"unresolved, no fewer than the {previous_active:,} before it -- "
                "the pid graph has a cycle."
            )
        previous_active = active.size
        step = _row_of(current_parent[active])
        found = step >= 0
        climbed = active[found]
        host_row[climbed] = step[found]
        current_parent[climbed] = pids[step[found]]
        # A `pid` naming a halo the catalog does not contain cannot be climbed;
        # the halo keeps whatever host it has reached so far rather than
        # spinning here forever.
        current_parent[active[~found]] = DISTINCT_PARENT_ID
        if verbose:
            remaining = int((current_parent != DISTINCT_PARENT_ID).sum())
            print(
                f"  host chain depth {depth + 1}: climbed {climbed.size:,}, "
                f"{remaining:,} still nested",
                flush=True,
            )
    still_nested = int((current_parent != DISTINCT_PARENT_ID).sum())
    if still_nested:
        raise ValueError(
            f"resolve_host_masses: {still_nested:,} halos are still nested after "
            f"MAX_HOST_CHAIN_DEPTH = {MAX_HOST_CHAIN_DEPTH} hops. Rockstar does not "
            "nest that deep, so this is a cycle in the pid graph, not a deep "
            "hierarchy -- raising rather than silently assigning a wrong host mass."
        )

    host_mvir = mvir[host_row].astype(np.float32, copy=False)
    pq.write_table(
        pa.table({HOST_MASS_COLUMN: pa.array(host_mvir, type=pa.float32())}), cache_path
    )
    if verbose:
        print(f"  wrote {cache_path}", flush=True)
    return host_mvir


def tinker_bias_of_mass(
    masses: np.ndarray,
    cosmology: CosmologyConfig | None = None,
    z: float = SNAPSHOT_REDSHIFT,
) -> np.ndarray:
    """Tinker et al. (2010) large-scale bias `b(M)`, shape (N,).

    Evaluated at `HALO_MASS_DEFINITION` -- Rockstar's `mvir` is the
    Bryan & Norman (1998) virial overdensity, colossus's `"vir"`. Passing a
    `200m` mass here instead is the classic ~10% cross-check failure.

    Parameters
    ----------
    masses : ndarray, shape (N,)
        Halo masses, h^-1 M_sun. For subhalos this must already be the HOST
        mass (`resolve_host_masses`).
    cosmology : CosmologyConfig, optional
    z : float

    Returns
    -------
    ndarray, shape (N,)
        `b(M_i)`.

    Notes
    -----
    Tinker (2010) is calibrated on halos far above MDPL2's 2-particle floor
    (`conventions.PARTICLE_MASS`). Applied to `CATALOG_FULL` uncut it is an
    EXTRAPOLATION over roughly the lower half of the sample, which is a real
    limitation of this cross-check and not of the clustering measurement it is
    checking. It is fully inside the calibrated range for `CATALOG_MVIR12`.
    """
    from colossus.cosmology import cosmology as colossus_cosmology
    from colossus.lss import bias as colossus_bias

    cosmology = cosmology or CosmologyConfig()
    colossus_cosmology.setCosmology("mdpl2", **cosmology.to_colossus_dict())
    return np.asarray(
        colossus_bias.haloBias(
            np.asarray(masses, dtype=float),
            z=z,
            mdef=HALO_MASS_DEFINITION,
            model=TINKER_BIAS_MODEL,
        ),
        dtype=float,
    )


def effective_tinker_bias(
    masses: np.ndarray,
    cosmology: CosmologyConfig | None = None,
    z: float = SNAPSHOT_REDSHIFT,
) -> float:
    """`b_eff = mean_i b_Tinker(M_i)` over a sample, a single float.

    The NUMBER-weighted mean, matching what a number-weighted correlation
    function measures: every halo contributes one object to `xi_hh`, so every
    halo contributes one `b(M_i)` here. A mass-weighted mean would be the
    right answer to a different question.

    Evaluated over the sample's DISTINCT masses, weighted by their
    multiplicities, rather than over all N masses one by one. This is exact,
    not an approximation: `dvcorr.conventions.PARTICLE_MASS` quantizes
    Rockstar's `mvir` into an integer particle count, so a 127M-halo catalog
    holds only ~10^5-10^6 distinct mass values and `b(M)` need only be
    evaluated once per value. Calling colossus 127M times instead would cost
    minutes and several GB for the identical number.

    Parameters
    ----------
    masses : ndarray, shape (N,)
        Host masses (see `tinker_bias_of_mass`), h^-1 M_sun.
    cosmology : CosmologyConfig, optional
    z : float

    Returns
    -------
    float
    """
    distinct, counts = np.unique(np.asarray(masses), return_counts=True)
    bias = tinker_bias_of_mass(distinct, cosmology, z)
    return float(np.average(bias, weights=counts))


# ---------------------------------------------------------------------------
# Theory: P_lin(k), the placeholder xi_mm, and the zeta_1 prediction
# ---------------------------------------------------------------------------


def linear_power_spectrum(
    cosmology: CosmologyConfig | None = None,
    z: float = SNAPSHOT_REDSHIFT,
    *,
    nonlinear: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """`P_m(k)` at the MDPL2 cosmology from CAMB, renormalized to its sigma8.

    CAMB is parameterized by `A_s`, not `sigma8`, so `A_s` is rescaled by
    `(sigma8_target / sigma8_camb)**2` -- exact, since `P` is linear in `A_s`
    and `sigma8**2` is linear in `P`. This is what makes the theory curve carry
    MDPL2's sigma8 = `CosmologyConfig.sigma8` rather than CAMB's default
    normalization, and it is the single largest lever on any bias derived from
    a theory `xi_mm`.

    Parameters
    ----------
    cosmology : CosmologyConfig, optional
    z : float
    nonlinear : bool
        Apply halofit. The linear spectrum is what the `zeta_1` prediction
        wants (`zeta_one_linear_prediction`); halofit is the better stand-in
        for a MEASURED `xi_mm` on 20-60 h^-1 Mpc, where nonlinear corrections
        are small but not zero.

    Returns
    -------
    k : ndarray, shape (CAMB_N_K,)
        Wavenumbers, h/Mpc, log-spaced from `CAMB_K_MIN` to `CAMB_K_MAX`.
    pk : ndarray, shape (CAMB_N_K,)
        `P_m(k)`, (h^-1 Mpc)^3.

    Notes
    -----
    `P` is the matter power spectrum here and nowhere else does this project
    use that symbol -- Legendre polynomials are `L_ell`
    (`dvcorr.conventions`, Notation).
    """
    import camb

    cosmology = cosmology or CosmologyConfig()
    h = cosmology.h
    pars = camb.set_params(
        H0=cosmology.H0,
        ombh2=cosmology.Ob0 * h**2,
        omch2=(cosmology.Om0 - cosmology.Ob0) * h**2,
        ns=cosmology.ns,
        # Derived rather than assumed flat: CosmologyConfig carries Ode0
        # independently, so a future non-flat entry stays self-consistent
        # instead of silently being forced to omk = 0.
        omk=1.0 - cosmology.Om0 - cosmology.Ode0,
    )
    pars.set_matter_power(redshifts=[z], kmax=CAMB_K_MAX)
    pars.NonLinear = camb.model.NonLinear_none

    sigma8_camb = camb.get_results(pars).get_sigma8_0()
    # P is linear in A_s and sigma8**2 is linear in P, so this hits the target
    # exactly rather than iteratively.
    pars.InitPower.As *= (cosmology.sigma8 / sigma8_camb) ** 2
    if nonlinear:
        pars.NonLinear = camb.model.NonLinear_both

    results = camb.get_results(pars)
    k, _, pk = results.get_matter_power_spectrum(
        minkh=CAMB_K_MIN, maxkh=CAMB_K_MAX, npoints=CAMB_N_K
    )
    return np.asarray(k, dtype=float), np.asarray(pk[0], dtype=float)


def _log_interp(r_target: np.ndarray, r: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cubic-spline interpolate `y(r)` onto `r_target` in log r, shape (len(r_target),)."""
    from scipy.interpolate import CubicSpline

    order = np.argsort(r)
    return CubicSpline(np.log(r[order]), y[order])(np.log(np.asarray(r_target, dtype=float)))


def theory_matter_xi(
    r: np.ndarray,
    cosmology: CosmologyConfig | None = None,
    z: float = SNAPSHOT_REDSHIFT,
    *,
    nonlinear: bool = False,
) -> tuple[np.ndarray, str]:
    """A PLACEHOLDER `xi_mm(r)` from CAMB, with its provenance tag.

    Read the module docstring before using this. It stands in for the measured
    particle `xi_mm` that MDPL2 does not distribute
    (`MATTER_FIELD_AVAILABILITY`); it inherits every sigma8 and
    snapshot-redshift error, and it cannot produce `r_cc` or `b_cross` because
    those need a measured `xi_hm`. Do not put a number derived from it in a
    final plot without the tag it returns.

    Parameters
    ----------
    r : ndarray, shape (B,)
        Separations, h^-1 Mpc.
    cosmology : CosmologyConfig, optional
    z : float
    nonlinear : bool
        halofit, or LINEAR (the default). Linear is the default because the
        `b` this denominator produces exists to feed a LINEAR-theory overlay
        (`zeta_one_linear_prediction`), and a `b` divided out of halofit and
        then multiplied back into linear theory is not the same number: on
        20-60 h^-1 Mpc halofit's `xi_mm` sits above linear, so a halofit
        denominator biases `b` LOW and the overlay then sits low by the same
        factor. Use halofit only when the target really is a measured `xi_mm`.
        `ABSOLUTE_BIAS_MIN_RADIUS` is the matching fit floor.

    Returns
    -------
    xi_mm : ndarray, shape (B,)
    matter_source : str
        `MATTER_SOURCE_CAMB_LINEAR` or `MATTER_SOURCE_CAMB_HALOFIT`.
    """
    import mcfit

    k, pk = linear_power_spectrum(cosmology, z, nonlinear=nonlinear)
    # P2xi(l=0): xi_0(r) = Int dk k**2 / (2 pi**2) P(k) j_0(k r) -- mcfit
    # carries the (2 pi**2) and the k**2 measure itself.
    r_grid, xi_grid = mcfit.P2xi(k, l=0, lowring=True)(pk, extrap=True)
    tag = MATTER_SOURCE_CAMB_HALOFIT if nonlinear else MATTER_SOURCE_CAMB_LINEAR
    return _log_interp(r, r_grid, np.real(xi_grid)), tag


def zeta_one_linear_prediction(
    r: np.ndarray,
    b: float,
    cosmology: CosmologyConfig | None = None,
    z: float = SNAPSHOT_REDSHIFT,
) -> np.ndarray:
    """The linear velocity-centered dipole prediction, km/s, shape (B,).

        zeta_1(r) = -(b a H f / (6 pi**2)) Int dk k P_lin(k) j_1(k r)

    Sign, and why it is not applied blindly
    ---------------------------------------
    The integral is positive, so the expression as written is NEGATIVE. That is
    the GROUP-CENTERED multipole `xi_Tu,1`, which `dvcorr.conventions` fixes
    negative for coherent infall (hard rule 2). This project's `zeta_1` is the
    VELOCITY-centered one, related by
    `conventions.nusser_multipole_sign(1) = -1`, so the returned array is
    POSITIVE for infall and directly comparable to
    `dvcorr.pipeline.velocity_centered.NormalizedDipole.zeta_hat`. The factor
    is applied here, explicitly and once, rather than absorbed into a sign
    somewhere -- see CLAUDE.md's Notation section.

    Units: `Int dk k P j_1` carries (h Mpc^-1)^2 x (h^-1 Mpc)^3 = h^-1 Mpc, and
    `a H` is evaluated in km/s per h^-1 Mpc -- `H(z) / h`, which at z = 0 is
    exactly 100 -- so the product is km/s with no stray factor of h. This
    conversion is the most dangerous line in the module: getting it upside down
    scales every prediction by h**2 with no change of shape, so it cannot be
    caught by looking at the plot, only by checking against a directly measured
    `<delta_h v.rhat>` (which is what `tests/test_galaxy_bias.py`'s
    `test_hubble_in_inverse_h_units_is_exactly_100_at_z0` now pins).

    Parameters
    ----------
    r : ndarray, shape (B,)
        Separations, h^-1 Mpc.
    b : float
        Linear bias of the tracer population whose density field is counted.
    cosmology : CosmologyConfig, optional
    z : float

    Returns
    -------
    ndarray, shape (B,)
        `zeta_1(r)`, km/s, in this project's velocity-centered sign convention.
    """
    import mcfit

    cosmology = cosmology or CosmologyConfig()
    k, pk = linear_power_spectrum(cosmology, z, nonlinear=False)

    # mcfit's P2xi(l=1) returns  i * Int dk k**2/(2 pi**2) Q(k) j_1(k r).
    # Feeding it Q = P / k and taking 2 pi**2 * Im(...) recovers
    # Int dk k P(k) j_1(k r) exactly (verified against direct quadrature).
    r_grid, transform = mcfit.P2xi(k, l=1, lowring=True)(pk / k, extrap=True)
    integral = _log_interp(r, r_grid, 2.0 * np.pi**2 * np.imag(transform))

    a = 1.0 / (1.0 + z)
    hubble_z = cosmology.H0 * np.sqrt(
        cosmology.Om0 * (1.0 + z) ** 3 + cosmology.Ode0
    )  # km/s/Mpc, flat LCDM
    # km/s per h^-1 Mpc. DIVIDE by h, do not multiply: one h^-1 Mpc is 1/h Mpc,
    # so H [km/s/Mpc] * (1/h) [Mpc per h^-1 Mpc] = H/h. At z = 0 this is exactly
    # 100, which is the whole point of h^-1 Mpc units and the one-line check
    # that this conversion is the right way up. Multiplying instead costs a
    # factor h**2 ~ 0.46 -- the amplitude comes out low by 2.2x, flat in r, and
    # looks exactly like a normalisation problem in the estimator.
    hubble_h_units = hubble_z / cosmology.h
    growth_rate = (
        cosmology.Om0 * (1.0 + z) ** 3 / (cosmology.Om0 * (1.0 + z) ** 3 + cosmology.Ode0)
    ) ** cosmology.growth_index

    xi_tu_one = -(b * a * hubble_h_units * growth_rate / (6.0 * np.pi**2)) * integral
    return conventions.nusser_multipole_sign(1) * xi_tu_one


#: Sub-samples per shell used to bin-average the theory curve
#: (`shell_averaged_zeta_one`). The integrand is smooth over a shell, so this
#: is generous rather than marginal.
SHELL_AVERAGE_SUBSAMPLES: int = 128

#: Smallest radius the theory curves are evaluated at, h^-1 Mpc. The Hankel
#: transform of `P(k)` sampled over [`CAMB_K_MIN`, `CAMB_K_MAX`] returns an `r`
#: grid reaching down to ~1/`CAMB_K_MAX`; below that, interpolating it is
#: extrapolation, and a shell whose inner edge is 0 (`ShellConfig.
#: include_zero_bin`) would otherwise ask for exactly that and get a
#: cubic-spline excursion of arbitrary size. Linear theory is meaningless there
#: anyway, so the quadrature floor is honest rather than merely defensive.
THEORY_MIN_RADIUS: float = 1.0 / CAMB_K_MAX


def shell_averaged_zeta_one(
    shell_edges: np.ndarray,
    b: float,
    xi_hh: CorrelationFunction | None = None,
    cosmology: CosmologyConfig | None = None,
    z: float = SNAPSHOT_REDSHIFT,
) -> np.ndarray:
    """`zeta_one_linear_prediction` averaged over each shell the way it is measured, (B,).

    The estimator does not evaluate `zeta_1` at a point: it sums over every
    tracer that falls in a shell, so what it reports is the PAIR-WEIGHTED mean
    of `zeta_1` over that shell. Comparing a point-evaluated theory curve
    against it mismatches ordinate and abscissa, and does so worst in the
    innermost shells where the curve is steepest and the shells are widest in
    absolute terms.

    The weight is `r**2 (1 + xi_hh(r)) dr`: `r**2 dr` is the uniform-field
    shell measure, and `(1 + xi_hh)` is the clustering modulation that turns it
    into the number of pairs actually counted. Supplying the MEASURED `xi_hh`
    is what makes this "bin-averaged with your measured pair counts" rather
    than with an assumed uniform field -- a per-shell total pair count cannot
    do the job on its own, since the weighting needed here is the distribution
    of pairs WITHIN each shell.

    Parameters
    ----------
    shell_edges : ndarray, shape (B + 1,)
        The estimator's own shell edges, h^-1 Mpc.
    b : float
        Linear bias of the tracer population.
    xi_hh : CorrelationFunction, optional
        The measured halo auto-correlation, log-interpolated onto the
        sub-shell grid and clipped to its own radial range. None falls back to
        the uniform-field weight `r**2 dr` alone, which is the right thing only
        where `xi_hh << 1`.
    cosmology : CosmologyConfig, optional
    z : float

    Returns
    -------
    ndarray, shape (B,)
        Shell-averaged `zeta_1`, km/s, in this project's velocity-centered
        sign convention. NaN for any shell lying entirely below
        `THEORY_MIN_RADIUS` -- not measured-and-zero, and not a spline
        excursion dressed up as a prediction.
    """
    edges = np.asarray(shell_edges, dtype=float)
    out = np.empty(edges.size - 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi <= THEORY_MIN_RADIUS:
            out[i] = np.nan
            continue
        # A shell reaching r = 0 has no pairs at the origin, and the theory
        # grid does not reach there either; start the quadrature at the floor
        # rather than at a radius where the interpolation is extrapolation.
        sub = np.linspace(max(lo, THEORY_MIN_RADIUS), hi, SHELL_AVERAGE_SUBSAMPLES)
        weight = sub**2
        if xi_hh is not None:
            r_ref = np.asarray(xi_hh.r_eff, dtype=float)
            clipped = np.clip(sub, r_ref.min(), r_ref.max())
            weight = weight * (1.0 + _log_interp(clipped, r_ref, np.asarray(xi_hh.xi, dtype=float)))
        values = zeta_one_linear_prediction(sub, b, cosmology, z)
        out[i] = np.trapezoid(values * weight, sub) / np.trapezoid(weight, sub)
    return out


# ---------------------------------------------------------------------------
# The direct velocity-density correlation -- an estimator-free reference
# ---------------------------------------------------------------------------

#: Halos per chunk in `measure_velocity_density_correlation`'s neighbour query.
#: Small enough that the returned neighbour lists stay in cache, large enough
#: that the per-call overhead is amortised.
VELOCITY_CORRELATION_CHUNK: int = 2000


@dataclass(frozen=True)
class VelocityDensityCorrelation:
    """`C(r) = <delta_h(x + r) v(x).rhat>` measured straight from halo pairs.

    The reference the `zeta_1` pipeline can be checked against WITHOUT going
    through the pipeline and without trusting a theory curve: it shares no
    code with `dvcorr.estimators`, uses the full 3-D velocity instead of its
    line-of-sight projection, and needs no observer. Since
    `dvcorr.pipeline.velocity_centered`'s normalized `zeta_hat` equals
    `C(r) / 3` -- the `<mu**2> = 1/3` of a uniform shell, the same 1/3 that
    turns the `1/(2 pi**2)` of linear theory into the `1/(6 pi**2)` of
    `zeta_one_linear_prediction` -- the comparison `3 * zeta_hat` against `C`
    is a direct, absolute test of the estimator's normalization.

    Attributes
    ----------
    edges : ndarray, shape (B + 1,)
        Radial bin edges, h^-1 Mpc.
    r_eff : ndarray, shape (B,)
        Volume-weighted shell radii, h^-1 Mpc.
    mean_v_radial : ndarray, shape (B,)
        `<v_1 . rhat_12>` over ordered pairs, km/s, with `rhat` pointing from
        the halo whose velocity is used to its neighbour. POSITIVE for infall:
        halo 1 moves towards halo 2.
    xi_hh : ndarray, shape (B,)
        The halo auto-correlation of the same sample, from the same pair
        counts -- needed because `mean_v_radial` is pair-weighted and `C` is
        not.
    C : ndarray, shape (B,)
        `mean_v_radial * (1 + xi_hh)`, km/s: the pair weighting undone.
    npairs : ndarray, shape (B,)
        Ordered pair count per bin.
    n_halos : int
    """

    edges: np.ndarray
    r_eff: np.ndarray
    mean_v_radial: np.ndarray
    xi_hh: np.ndarray
    C: np.ndarray
    npairs: np.ndarray
    n_halos: int


def measure_velocity_density_correlation(
    pos: np.ndarray, vel: np.ndarray, edges: np.ndarray
) -> VelocityDensityCorrelation:
    """Measure `C(r) = <delta_h(x + r) v(x).rhat>` from halo pairs directly.

    Periodic throughout (CLAUDE.md hard rule 3): the neighbour search is a
    `scipy.spatial.cKDTree` built with `boxsize=conventions.BOX_SIZE`, and the
    separation vectors it returns neighbours for are reduced to the minimum
    image explicitly -- the tree matches across the box faces but hands back
    raw coordinates, so a plain difference would give a ~BOX_SIZE separation
    for exactly the pairs periodicity was supposed to fix.

    No carving and no observer: this works on the whole box, which is why it
    is a check on the carved, observer-centered pipeline rather than a second
    copy of it.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Box coordinates, comoving h^-1 Mpc, in [0, BOX_SIZE).
    vel : ndarray, shape (N, 3)
        Peculiar velocities, km/s, same row order as `pos`. Must be finite:
        a missing velocity is missing data and the caller drops it
        (CLAUDE.md hard rule 5).
    edges : ndarray, shape (B + 1,)
        Radial bin edges, h^-1 Mpc, strictly increasing, outermost
        <= `conventions.MAX_ANALYSIS_RADIUS`.

    Returns
    -------
    VelocityDensityCorrelation

    Notes
    -----
    Cost is the pair count, `N**2 / BOX_SIZE**3 * (4 pi / 3) r_max**3`, in
    Python-level chunks -- ~40 s for 300k halos to 70 h^-1 Mpc. It is a
    diagnostic, not a production estimator, so it is written for clarity
    rather than for the tens of millions of halos `periodic_auto_xi` handles.
    """
    from scipy.spatial import cKDTree

    pos = np.ascontiguousarray(np.asarray(pos, dtype=float))
    vel = np.ascontiguousarray(np.asarray(vel, dtype=float))
    edges = np.asarray(edges, dtype=float)
    if pos.shape != vel.shape or pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(
            f"measure_velocity_density_correlation: pos {pos.shape} and vel "
            f"{vel.shape} must both be (N, 3) and match."
        )
    if not np.all(np.isfinite(vel)):
        raise ValueError(
            "measure_velocity_density_correlation: vel contains a non-finite "
            "value; a missing peculiar velocity is missing data and must be "
            "dropped by the caller, never entered as 0 (CLAUDE.md hard rule 5)."
        )
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("measure_velocity_density_correlation: edges must be increasing.")
    if edges[-1] > conventions.MAX_ANALYSIS_RADIUS:
        raise ValueError(
            f"measure_velocity_density_correlation: outermost edge {edges[-1]} "
            f"exceeds conventions.MAX_ANALYSIS_RADIUS "
            f"({conventions.MAX_ANALYSIS_RADIUS})."
        )

    n_halos = pos.shape[0]
    n_bins = edges.size - 1
    tree = cKDTree(pos, boxsize=conventions.BOX_SIZE)
    sum_v_radial = np.zeros(n_bins)
    npairs = np.zeros(n_bins)

    for start in range(0, n_halos, VELOCITY_CORRELATION_CHUNK):
        stop = min(start + VELOCITY_CORRELATION_CHUNK, n_halos)
        neighbours = tree.query_ball_point(pos[start:stop], r=edges[-1], workers=-1)
        for offset, raw in enumerate(neighbours):
            a = start + offset
            idx = np.asarray(raw, dtype=np.int64)
            idx = idx[idx != a]  # the self-pair has no direction
            if idx.size == 0:
                continue
            separation = pos[idx] - pos[a]
            separation -= conventions.BOX_SIZE * np.round(separation / conventions.BOX_SIZE)
            r_mag = np.sqrt((separation * separation).sum(axis=1))
            inside = (r_mag >= edges[0]) & (r_mag < edges[-1]) & (r_mag > 0.0)
            if not inside.any():
                continue
            r_mag = r_mag[inside]
            v_radial = (separation[inside] @ vel[a]) / r_mag
            bin_index = np.digitize(r_mag, edges) - 1
            sum_v_radial += np.bincount(bin_index, weights=v_radial, minlength=n_bins)[:n_bins]
            npairs += np.bincount(bin_index, minlength=n_bins)[:n_bins]

    mean_v_radial = sum_v_radial / npairs
    number_density = n_halos / conventions.BOX_SIZE**3
    xi_hh = npairs / (n_halos * number_density * _shell_volumes(edges)) - 1.0
    return VelocityDensityCorrelation(
        edges=edges,
        r_eff=(3.0 / 4.0) * (edges[1:] ** 4 - edges[:-1] ** 4) / (edges[1:] ** 3 - edges[:-1] ** 3),
        mean_v_radial=mean_v_radial,
        xi_hh=xi_hh,
        C=mean_v_radial * (1.0 + xi_hh),
        npairs=npairs.astype(np.int64),
        n_halos=n_halos,
    )


# ---------------------------------------------------------------------------
# The free consistency test: does the zeta_1 amplitude ratio track b?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmplitudeRatioTest:
    """`zeta_1^A(r) / zeta_1^B(r)` against the independently measured `b_A / b_B`.

    Two samples measured with the SAME estimator in the SAME box differ in
    `zeta_1` amplitude only through their tracer bias, because the velocity
    field they are correlated against is the same unbiased field. So the ratio
    must be flat in `r` and must equal the bias ratio. It is simultaneously the
    bias cross-check and a genuine pipeline validation: if the amplitude ratio
    does not track the bias ratio, the fault is in the estimator, not the
    cosmology.

    Attributes
    ----------
    r : ndarray, shape (B,)
        Shell radii of the two `zeta_1` runs, h^-1 Mpc. Both runs must share
        this grid.
    ratio : ndarray, shape (B,)
        `zeta_1^A / zeta_1^B` per shell.
    ratio_error : ndarray, shape (B,)
        Propagated from the two runs' per-shell standard errors, treating them
        as independent -- they are measured on overlapping halos and the same
        density field, so this is an UNDER-estimate at large r where the two
        samples share most of their pairs. Quoted as a guide to which shells
        carry information, not as a p-value.
    ratio_fit : ConstantFit
        The constant fitted to `ratio` over the window.
    bias_ratio : float
        `b_A / b_B` from the independent (clustering or mass-function) route.
    label_a, label_b : str
    """

    r: np.ndarray
    ratio: np.ndarray
    ratio_error: np.ndarray
    ratio_fit: ConstantFit
    bias_ratio: float
    label_a: str
    label_b: str

    @property
    def discrepancy(self) -> float:
        """`ratio_fit.value / bias_ratio - 1`: fractional disagreement."""
        return float(self.ratio_fit.value / self.bias_ratio - 1.0)

    def passes(self, tolerance: float = TINKER_CROSSCHECK_TOLERANCE) -> bool:
        """Whether the amplitude ratio tracks the bias ratio within `tolerance`.

        Requires BOTH agreement in level and flatness in `r`: a ratio that
        happens to average the right number while sloping across the window has
        not tracked anything.
        """
        if not np.isfinite(self.discrepancy):
            return False
        return (
            abs(self.discrepancy) <= tolerance
            and self.ratio_fit.fractional_scatter <= tolerance
        )


def zeta_amplitude_ratio(
    r: np.ndarray,
    zeta_a: np.ndarray,
    sem_a: np.ndarray,
    zeta_b: np.ndarray,
    sem_b: np.ndarray,
    bias_ratio: float,
    label_a: str,
    label_b: str,
    r_min: float = FIT_MIN_RADIUS,
    r_max: float = FIT_MAX_RADIUS,
) -> AmplitudeRatioTest:
    """Build the `zeta_1` amplitude-ratio test; see `AmplitudeRatioTest`.

    Parameters
    ----------
    r : ndarray, shape (B,)
        Shared shell radii, h^-1 Mpc.
    zeta_a, zeta_b : ndarray, shape (B,)
        Normalized `zeta_1` of the two samples, km/s
        (`dvcorr.pipeline.velocity_centered.NormalizedDipole.zeta_hat`).
    sem_a, sem_b : ndarray, shape (B,)
        Their per-shell standard errors, km/s.
    bias_ratio : float
        `b_A / b_B`, measured independently.
    label_a, label_b : str
    r_min, r_max : float
        Fit window, h^-1 Mpc.

    Returns
    -------
    AmplitudeRatioTest
    """
    r = np.asarray(r, dtype=float)
    zeta_a = np.asarray(zeta_a, dtype=float)
    zeta_b = np.asarray(zeta_b, dtype=float)
    shapes = {a.shape for a in (r, zeta_a, zeta_b, np.asarray(sem_a), np.asarray(sem_b))}
    if len(shapes) != 1:
        raise ValueError(
            f"zeta_amplitude_ratio: r, zeta and sem arrays must share one shape, got {shapes}."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = zeta_a / zeta_b
        rel = np.sqrt(
            (np.asarray(sem_a, dtype=float) / zeta_a) ** 2
            + (np.asarray(sem_b, dtype=float) / zeta_b) ** 2
        )
    return AmplitudeRatioTest(
        r=r,
        ratio=ratio,
        ratio_error=np.abs(ratio) * rel,
        ratio_fit=fit_constant(r, ratio, r_min, r_max),
        bias_ratio=float(bias_ratio),
        label_a=label_a,
        label_b=label_b,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_correlation_figure(
    curves: list[CorrelationFunction],
    cfg: BiasRunConfig,
    theory: tuple[np.ndarray, np.ndarray, str] | None = None,
):
    """`r**2 xi(r)` for every measured correlation function, one panel.

    `r**2 xi` rather than `xi` so four decades of dynamic range fit on one
    linear ordinate and the 20-60 h^-1 Mpc window -- where every number in
    this module is fitted -- is legible rather than crushed against zero.
    Cartesian axes with a log abscissa, matching the project's plotting habit.

    Parameters
    ----------
    curves : list of CorrelationFunction
    cfg : BiasRunConfig
        Supplies the shaded fit window.
    theory : (r, xi, label), optional
        A theory curve to overlay, e.g. the CAMB placeholder `xi_mm`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for curve in curves:
        ax.plot(curve.r_eff, curve.r_eff**2 * curve.xi, marker="o", ms=3, label=curve.label)
    if theory is not None:
        r_theory, xi_theory, theory_label = theory
        ax.plot(r_theory, r_theory**2 * xi_theory, ls="--", color="k", lw=1.2, label=theory_label)
    ax.axvspan(cfg.fit_min_radius, cfg.fit_max_radius, color="0.85", zorder=0, label="fit range")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$  [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$r^{2}\,\xi(r)$  [$(h^{-1}\,\mathrm{Mpc})^{2}$]")
    ax.set_title("Real-space correlation functions, MDPL2 snapshot 125")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_bias_figure(estimates: BiasEstimates, cfg: BiasRunConfig, fits: dict[str, ConstantFit]):
    """The flatness plot: `b(r)` and `r_cc(r)` against `r`, two stacked panels.

    "A number without the flatness plot is not a measurement" -- so the fitted
    constant is drawn ON the curve it was fitted to, over the window it was
    fitted over, and nowhere else.

    Parameters
    ----------
    estimates : BiasEstimates
    cfg : BiasRunConfig
    fits : dict of str -> ConstantFit
        Keyed by "b_cross", "b_auto", "r_cc"; missing keys are simply not
        annotated (the placeholder case has no `b_cross` or `r_cc`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, (ax_b, ax_r) = plt.subplots(
        2, 1, figsize=(8.0, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    for key, values, color in (
        ("b_cross", estimates.b_cross, "C0"),
        ("b_auto", estimates.b_auto, "C1"),
    ):
        if not np.any(np.isfinite(values)):
            continue
        label = {"b_cross": r"$b_{\rm cross}=\xi_{hm}/\xi_{mm}$",
                 "b_auto": r"$b_{\rm auto}=\sqrt{\xi_{hh}/\xi_{mm}}$"}[key]
        ax_b.plot(estimates.r, values, marker="o", ms=4, color=color, label=label)
        fit = fits.get(key)
        if fit is not None:
            ax_b.hlines(
                fit.value, fit.r_min, fit.r_max, color=color, ls="--", lw=2,
                label=f"{key} = {fit.value:.3f} ± {fit.scatter:.3f}",
            )

    ax_b.axvspan(cfg.fit_min_radius, cfg.fit_max_radius, color="0.85", zorder=0)
    ax_b.set_xscale("log")
    ax_b.set_ylabel(r"$b(r)$")
    ax_b.set_title(f"Linear bias — {estimates.label}\nmatter field: {estimates.matter_source}")
    ax_b.legend(fontsize=8)

    if np.any(np.isfinite(estimates.r_cc)):
        ax_r.plot(estimates.r, estimates.r_cc, marker="o", ms=4, color="C2")
        fit = fits.get("r_cc")
        if fit is not None:
            ax_r.hlines(fit.value, fit.r_min, fit.r_max, color="C2", ls="--", lw=2)
    else:
        ax_r.text(
            0.5, 0.5,
            "$r_{cc}$ needs a measured $\\xi_{hm}$\n(no MDPL2 particle subsample — see\n"
            "galaxy_bias.MATTER_FIELD_AVAILABILITY)",
            ha="center", va="center", transform=ax_r.transAxes, fontsize=9, color="0.35",
        )
    ax_r.axhline(1.0, color="0.5", lw=0.8)
    ax_r.axhspan(1.0 - R_CC_UNITY_TOLERANCE, 1.0 + R_CC_UNITY_TOLERANCE, color="C2", alpha=0.12)
    ax_r.axvspan(cfg.fit_min_radius, cfg.fit_max_radius, color="0.85", zorder=0)
    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"$r$  [$h^{-1}$ Mpc]")
    ax_r.set_ylabel(r"$r_{cc}(r)$")

    fig.tight_layout()
    return fig


def make_amplitude_ratio_figure(test: AmplitudeRatioTest, cfg: BiasRunConfig):
    """The free consistency test as a figure: the `zeta_1` ratio against `b_A/b_B`.

    Parameters
    ----------
    test : AmplitudeRatioTest
    cfg : BiasRunConfig

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(
        test.r, test.ratio, yerr=test.ratio_error, marker="o", ms=4, lw=1.2, capsize=2,
        label=rf"$\zeta_1^{{\rm {test.label_a}}} / \zeta_1^{{\rm {test.label_b}}}$",
    )
    ax.axhline(
        test.bias_ratio, color="C3", ls="-", lw=2,
        label=rf"$b_{{\rm {test.label_a}}}/b_{{\rm {test.label_b}}} = {test.bias_ratio:.3f}$",
    )
    ax.hlines(
        test.ratio_fit.value, test.ratio_fit.r_min, test.ratio_fit.r_max,
        color="C0", ls="--", lw=2,
        label=f"fit = {test.ratio_fit.value:.3f} ± {test.ratio_fit.scatter:.3f}",
    )
    ax.axvspan(cfg.fit_min_radius, cfg.fit_max_radius, color="0.85", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$  [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$\zeta_1$ amplitude ratio")
    ax.set_title(
        f"Free consistency test: does the ζ₁ ratio track the bias ratio?\n"
        f"discrepancy {100.0 * test.discrepancy:+.1f}%"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
