"""
dm_snapshot.py
--------------
Load the Gadget-4 dark-matter particle snapshot as a tracer field the existing
velocity-centered pipelines can consume unchanged.

What this module is for
=======================
`dvcorr.pipeline.velocity_frame_comparison` (notebook 06) and
`dvcorr.pipeline.redshift_space_comparison` (notebook 07) are written against
*arrays* -- positions, velocities, an observer, an `n_bar` -- and take every
one of those as an explicit argument. The ONLY place either pipeline reaches
for a halo catalog, and the only place either reads
`dvcorr.conventions.OBSERVER_POSITION`, is its loader:
`dvcorr.pipeline.velocity_centered.load_and_carve` and
`dvcorr.pipeline.redshift_space_comparison.load_and_carve_buffered`.

This module supplies the two replacement loaders. Everything downstream --
`draw_candidates_from_arrays`, `select_shared_centers`, `run_both_frames`,
`build_tracer_spaces`, `select_redshift_shared_centers`, `run_both_spaces`,
both `normalize_*` stages and every figure builder -- is called exactly as
notebooks 06 and 07 call it, with no signature change and no forked copy of a
single estimator. `notebooks/DM_simulation.ipynb` is the thin consumer.

The box is NOT MDPL2, and that is the whole point of this file
==============================================================
`dvcorr.conventions` is frozen around MDPL2: `BOX_SIZE` = 1000, hence
`MAX_ANALYSIS_RADIUS` = 500 and `OBSERVER_POSITION` = (500, 500, 500). This
snapshot is a **300 h^-1 Mpc** box, so all three are wrong for it, and CLAUDE.md
hard rule 1 forbids editing them to suit one dataset.

The resolution is that none of the three has to be edited, because none of them
is load-bearing on this path:

  - `OBSERVER_POSITION` -- every stage downstream of the loaders takes
    `observer` as an argument. `snapshot_observer` supplies this box's own
    center, and the notebook threads it exactly as 06/07 thread
    `conventions.OBSERVER_POSITION`. Nothing is redefined; a different value is
    passed to a parameter that already existed.
  - `BOX_SIZE` -- reached only by `box_number_density` (which this module
    replaces with `snapshot_number_density`) and by
    `draw_candidates_from_arrays`' canonical-ordering fold, where
    `np.mod(pos, 1000)` on coordinates already inside [0, 300) is the identity
    and the ordering is unaffected.
  - `MAX_ANALYSIS_RADIUS` -- the estimators check shell edges and
    `sub_volume_radius` against 500, which every radius in a 300 h^-1 Mpc box
    trivially passes. That check is therefore *inert*, not wrong, and this box
    needs a TIGHTER one: `SnapshotBox.max_analysis_radius` = 150.
    `check_run_geometry` enforces it at load time, before any expensive work.
    A tighter guard added on top of the frozen one is not a weakening of it.

What "PBC everywhere" means here (hard rule 3)
==============================================
This box is periodic and the minimum-image convention applies to it, exactly as
it does to MDPL2. The pipelines discharge that rule the same way for both: they
**carve a sphere** of `sub_volume_radius` around the observer and hand the
estimators a population that sits clear of the periodic faces, so plain
Euclidean geometry *is* the minimum image inside the carve. The carve is what
makes that true, and it is only true while the largest radius the run touches
stays under `box.max_analysis_radius`. For the redshift-space pipeline the
largest radius is the BUFFERED one (`sub_volume_radius + v_margin`), not
`sub_volume_radius` -- `check_run_geometry` takes the radius it must actually
protect as an argument, so the caller cannot check the wrong one.

Particles are not halos
=======================
Three differences that change how the output is read, none of them fixable in
code, all of them stated here rather than discovered from a plot:

1. **b = 1 by construction.** The tracer field is the matter field itself, so
   there is no halo bias between the density and the velocity. This is the
   reason to run the box at all: every MDPL2 measurement is of biased tracers,
   and this one is not.

2. **Centers are mass-weighted, and carry virial motion.** A center drawn from
   this field is a random *particle*, so the center population samples the
   matter field by mass rather than by object. Its velocity is its halo's bulk
   flow PLUS its own virial velocity, and the virial part is uncorrelated with
   the surrounding large-scale density -- it tilts the axis away from the flow
   and dilutes the dipole, while simultaneously raising the |v| weight of
   centers in dense regions. Both effects run the same way in every shell, so
   they rescale the amplitude rather than change its shape, and the monopole
   panel reports the mean speed they act through. Do NOT read a lower
   ζ̂₁ against notebook 06's halo-centered number as a statement about the
   density field.

3. **`mvir` and `is_distinct` do not exist.** Every particle carries the same
   mass (`SnapshotBox.particle_mass`), and "is this a subhalo" is not a
   property a particle has. `CarvedParticles` therefore does not fabricate
   either column; the two optional per-candidate arguments the shared center
   selectors accept are simply left at `None`, and the mass-funnel diagnostics
   (`dvcorr.pipeline.mass_diagnostics`) do not apply to this pipeline.
   `load_and_carve_particles_buffered` must fill `BufferedCarve`'s two required
   mass fields to satisfy that dataclass; it fills them with the uniform
   particle mass and an all-True flag, and nothing on the 07 path reads either.

Subsampling
===========
The snapshot holds 16 777 216 particles in 300^3 (h^-1 Mpc)^3, i.e.
n_bar = 0.62 (h^-1 Mpc)^-3 -- ~4.9x the MDPL2 full catalog's 0.127. Estimator
cost is dominated by the neighbor count in a ball of `shells.max_radius`, so
that density is a ~5x runtime multiplier for shot noise that was already
negligible.

`subsample_factor` keeps a seeded random 1-in-N of the particles. This is
UNBIASED for the density field -- independent Bernoulli thinning changes no
correlation function, it only adds Poisson noise -- so the measured ζ̂₁ is
unchanged in expectation, at a 1/N runtime.

The draw is a random MASK, never a stride. Gadget writes particles in
Peano-Hilbert domain order, so `pos[::N]` would be a spatially clustered
subset, not a fair sample, and `ParticleIDs` is the Lagrangian grid index, so
striding on that would sample a sub-LATTICE -- the one subsample that aliases
against the initial conditions. Neither is a fair thinning; only the random
mask is.

Units
=====
Checked against the file's own `Parameters` group rather than assumed, so a
snapshot written in different code units fails loudly instead of silently
producing positions scaled by ~1e3 (`_check_unit_system`):

  - `UnitLength_in_cm`   = 3.085678e24  -> 1 h^-1 Mpc  (positions)
  - `UnitVelocity_in_cm_per_s` = 1e5    -> 1 km/s      (velocities)
  - `UnitMass_in_g`      = 1.989e43     -> 1e10 h^-1 M_sun (`MassTable`)

which is exactly the project's unit system (CLAUDE.md, Units), so no rescaling
of positions or masses is needed. Velocities DO carry one factor: Gadget-4
stores `v_snapshot = v_peculiar / sqrt(a)` (the `a_scaling = 0.5` attribute on
the dataset), so `peculiar_velocity_scale` applies sqrt(a). This snapshot is at
a = 1 to within double-precision round-off, where that factor is exactly 1 --
it is applied anyway, so the loader stays correct if a z > 0 snapshot is ever
pointed at it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from dvcorr.pipeline.redshift_space_comparison import BufferedCarve, RedshiftSpaceRunConfig
from dvcorr.pipeline.velocity_centered import RunConfig
from dvcorr.redshift_space import radial_velocity, v_margin_from_statistic

# ---------------------------------------------------------------------------
# The snapshot's own layout and unit system (hard rule 4: named, not inline)
# ---------------------------------------------------------------------------

#: HDF5 group holding the dark-matter particles. Gadget's type 1 is the
#: collisionless dark-matter species; this snapshot has no other populated
#: type (`NumPart_Total` = [0, 16777216]).
PARTICLE_TYPE_GROUP: str = "PartType1"

#: Gadget's index for that species inside the header's `MassTable`.
PARTICLE_TYPE_INDEX: int = 1

_HEADER_GROUP: str = "Header"
_PARAMETERS_GROUP: str = "Parameters"
_COORDINATES_DATASET: str = "Coordinates"
_VELOCITIES_DATASET: str = "Velocities"

#: Gadget's `UnitMass_in_g` = 1.989e43 g = 1e10 M_sun, so a `MassTable` entry
#: is a mass in 1e10 h^-1 M_sun. This converts it to the project's h^-1 M_sun.
GADGET_MASS_UNIT_MSUN_PER_H: float = 1e10

#: The code-unit values this loader is written against, in Gadget's own
#: `Parameters` spelling: 1 h^-1 Mpc, 1 km/s, 1e10 h^-1 M_sun. Checked rather
#: than assumed -- see `_check_unit_system`.
_EXPECTED_UNIT_LENGTH_CM: float = 3.085678e24
_EXPECTED_UNIT_VELOCITY_CM_S: float = 1.0e5
_EXPECTED_UNIT_MASS_G: float = 1.989e43

#: Fractional tolerance on the unit-system check. Loose enough to accept a
#: differently-rounded Mpc-in-cm (values between 3.0856e24 and 3.0857e24 are
#: all in circulation), tight enough that a factor-of-1000 unit change -- the
#: failure this guard exists for -- cannot pass.
_UNIT_TOLERANCE: float = 1e-3

#: Default thinning of the particle field; see this module's "Subsampling"
#: section. 3 brings n_bar from 0.62 to 0.21 (h^-1 Mpc)^-3 -- still ~1.6x the
#: MDPL2 full catalog, so shot noise stays below where it was already
#: negligible -- at a third of the estimator runtime. Set 1 to disable.
DEFAULT_SUBSAMPLE_FACTOR: int = 3

#: Seed for that draw. A distinct stream from every seed in
#: `dvcorr.pipeline.velocity_centered.RunConfig`'s ladder (42-48), so thinning
#: the field never reuses a center-selection or null randomness stream.
DEFAULT_SUBSAMPLE_SEED: int = 101


# ---------------------------------------------------------------------------
# Box metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotBox:
    """The snapshot's own box and cosmology, read from its header.

    The `dvcorr.conventions` counterpart for a box that is not MDPL2: it holds
    what `BOX_SIZE`, `MAX_ANALYSIS_RADIUS`, `PARTICLE_MASS` and
    `OBSERVER_POSITION` hold for MDPL2, but as *data read from the file* rather
    than as frozen module constants. Frozen as a dataclass because it is
    simulation metadata, not a tunable -- the same reason
    `dvcorr.config.cosmology.CosmologyConfig` is.

    Nothing here is typed in by hand. That is deliberate: a hand-copied box
    size or particle mass is a silent factor on every normalized amplitude, and
    the file already states both.

    Attributes
    ----------
    box_size : float
        Comoving side length, h^-1 Mpc (`Header/BoxSize`).
    n_particles : int
        Particles in the whole box (`Header/NumPart_Total[PARTICLE_TYPE_INDEX]`).
    particle_mass : float
        Mass per particle, h^-1 M_sun (`Header/MassTable`, converted by
        `GADGET_MASS_UNIT_MSUN_PER_H`). Uniform -- this is a single-species
        collisionless run.
    redshift : float
        `Header/Redshift`.
    scale_factor : float
        `Header/Time`. Carried separately from `redshift` rather than derived,
        because it is what Gadget actually wrote and what
        `peculiar_velocity_scale` reads.
    hubble_param : float
        Reduced Hubble parameter h (`Parameters/HubbleParam`).
    omega_m : float
        `Parameters/Omega0`.
    omega_lambda : float
        `Parameters/OmegaLambda`.
    omega_baryon : float
        `Parameters/OmegaBaryon`.
    softening : float
        Comoving gravitational softening, h^-1 Mpc
        (`Parameters/SofteningComovingClass0`). The floor below which a
        separation is not resolved -- the analogue of MDPL2's 5 h^-1 kpc, and
        two orders of magnitude coarser.
    initial_conditions : str
        `Parameters/InitCondFile`. Retained because for this snapshot it is
        the only record of what the run IS: the filename identifies it as a
        constrained realization built from 2MRS, not a random box, which
        changes what the observer at the box center means.
    """

    box_size: float
    n_particles: int
    particle_mass: float
    redshift: float
    scale_factor: float
    hubble_param: float
    omega_m: float
    omega_lambda: float
    omega_baryon: float
    softening: float
    initial_conditions: str

    @property
    def max_analysis_radius(self) -> float:
        """Largest separation for which minimum-image geometry is unambiguous.

        `box_size / 2`, this box's own version of
        `dvcorr.conventions.MAX_ANALYSIS_RADIUS` (which is MDPL2's 500 and is
        five-thirds too permissive here). The factor 2 is the definition of the
        minimum-image ceiling, not a tunable (CLAUDE.md hard rule 4's
        pure-mathematics exemption). Enforced by `check_run_geometry`.
        """
        return self.box_size / 2.0

    @property
    def mean_interparticle_spacing(self) -> float:
        """`box_size / N^(1/3)`, h^-1 Mpc -- the field's own resolution floor.

        The scale below which "the density field" is not a field but a list of
        discrete particles, and therefore the smallest shell radius a run
        should read as physical. It is this snapshot's analogue of MDPL2's
        `RESOLVED_PARTICLE_COUNT` -- a completeness floor stated in the units
        the estimator actually bins in.

        It is far coarser than the force softening (0.2 h^-1 Mpc here), so it,
        not the softening, is the binding constraint: at 1.17 h^-1 Mpc it sits
        ABOVE `dvcorr.pipeline.velocity_centered`'s default
        `shells.min_radius` of 0.5, which is why a run in this box must raise
        that floor rather than inherit it.
        """
        return self.box_size / self.n_particles ** (1.0 / 3.0)

    @property
    def number_density(self) -> float:
        """Mean particle number density of the FULL snapshot, (h^-1 Mpc)^-3.

        The un-subsampled value. A run that thins the field must normalize
        with the thinned density instead -- use `snapshot_number_density` with
        `CarvedParticles.n_total`, never this, which is why the thinned figure
        is not reachable as a property here.
        """
        return self.n_particles / self.box_size**3


def _check_unit_system(parameters: h5py.AttributeManager, path: Path) -> None:
    """Fail loudly if the snapshot's code units are not this loader's.

    The unit system is the one assumption in this module that a wrong file
    would not otherwise announce: positions in kpc/h rather than Mpc/h load
    without error, carve to nothing (or to everything), and produce a plausible
    curve at the wrong scale. Comparing against the file's own `Parameters` is
    a free check, so it is not left to the caller.

    Parameters
    ----------
    parameters : h5py.AttributeManager
        The snapshot's `Parameters` group attributes.
    path : Path
        Only for the error message.

    Raises
    ------
    ValueError
        If any of the three code units differs from this loader's expectation
        by more than `_UNIT_TOLERANCE` (fractionally).
    """
    for key, expected, meaning in (
        ("UnitLength_in_cm", _EXPECTED_UNIT_LENGTH_CM, "1 h^-1 Mpc"),
        ("UnitVelocity_in_cm_per_s", _EXPECTED_UNIT_VELOCITY_CM_S, "1 km/s"),
        ("UnitMass_in_g", _EXPECTED_UNIT_MASS_G, "1e10 h^-1 M_sun"),
    ):
        if key not in parameters:
            raise ValueError(
                f"{path.name}: Parameters/{key} is missing, so the snapshot's "
                "code units cannot be verified. dvcorr.pipeline.dm_snapshot "
                f"assumes {key} = {expected:g} ({meaning})."
            )
        found = float(parameters[key])
        if abs(found / expected - 1.0) > _UNIT_TOLERANCE:
            raise ValueError(
                f"{path.name}: Parameters/{key} = {found:g}, but "
                f"dvcorr.pipeline.dm_snapshot is written for {expected:g} "
                f"({meaning}). Loading it anyway would rescale every position, "
                "velocity or mass silently."
            )


def read_snapshot_header(path: Path) -> SnapshotBox:
    """Read the snapshot's box and cosmology without touching the particles.

    Cheap (attributes only) and the first thing a run should call: it is what
    `check_run_geometry` validates the run's radii against, so a
    `sub_volume_radius` too large for the box is caught before the ~470 MB of
    particle data is read.

    Parameters
    ----------
    path : Path
        The Gadget-4 HDF5 snapshot.

    Returns
    -------
    SnapshotBox

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file has no `PARTICLE_TYPE_GROUP`, or its code units are not
        this loader's (`_check_unit_system`).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    with h5py.File(path, "r") as snapshot:
        if PARTICLE_TYPE_GROUP not in snapshot:
            raise ValueError(
                f"{path.name} has no {PARTICLE_TYPE_GROUP!r} group; it holds "
                f"{sorted(snapshot.keys())}. This loader reads collisionless "
                "dark-matter particles only."
            )
        header = snapshot[_HEADER_GROUP].attrs
        parameters = snapshot[_PARAMETERS_GROUP].attrs
        _check_unit_system(parameters, path)

        return SnapshotBox(
            box_size=float(header["BoxSize"]),
            n_particles=int(header["NumPart_Total"][PARTICLE_TYPE_INDEX]),
            particle_mass=(
                float(header["MassTable"][PARTICLE_TYPE_INDEX])
                * GADGET_MASS_UNIT_MSUN_PER_H
            ),
            redshift=float(header["Redshift"]),
            scale_factor=float(header["Time"]),
            hubble_param=float(parameters["HubbleParam"]),
            omega_m=float(parameters["Omega0"]),
            omega_lambda=float(parameters["OmegaLambda"]),
            omega_baryon=float(parameters["OmegaBaryon"]),
            softening=float(parameters["SofteningComovingClass0"]),
            initial_conditions=str(parameters["InitCondFile"], "utf-8")
            if isinstance(parameters["InitCondFile"], bytes)
            else str(parameters["InitCondFile"]),
        )


def snapshot_observer(box: SnapshotBox) -> np.ndarray:
    """The single observer for a run in this box: its center.

    This box's counterpart to `dvcorr.conventions.OBSERVER_POSITION`, which is
    MDPL2's center and is 350 h^-1 Mpc outside this box. Returned as a value to
    be threaded into the `observer` argument every downstream stage already
    takes -- the convention (one observer per run, at the box center) is
    unchanged; only the box is.

    For this particular snapshot the center is not an arbitrary choice: the
    initial conditions are a constrained realization built from 2MRS
    (`SnapshotBox.initial_conditions`), whose reconstruction is centered on the
    Local Group, so the box center is intended to be *us*. That is a claim
    about the file's provenance, not something the file states, and it is worth
    confirming before any comparison to real CF4 data leans on it.

    Parameters
    ----------
    box : SnapshotBox

    Returns
    -------
    ndarray, shape (3,), float64
        Box coordinates, comoving h^-1 Mpc.
    """
    return np.full(3, box.box_size / 2.0, dtype=float)


def snapshot_number_density(n_particles: int, box: SnapshotBox) -> float:
    """Mean tracer number density n_bar over the WHOLE box, (h^-1 Mpc)^-3.

    The `dvcorr.pipeline.velocity_centered.box_number_density` of this box --
    identical arithmetic (count over box volume), against `box.box_size`
    instead of the frozen `dvcorr.conventions.BOX_SIZE`. That function's
    docstring carries the full argument for why n_bar is the cosmic mean of the
    box rather than the density of the carved sub-volume; it applies here
    unchanged, and the reason this is a separate function rather than a call
    into it is only the box size.

    Pass the count that actually entered the tracer field: for a thinned run
    that is `CarvedParticles.n_total` (the RETAINED particles in the whole box,
    post-subsample and pre-carve), not `box.n_particles`. Using the full count
    against a thinned field would inflate n_bar by `subsample_factor` and
    suppress every ζ̂₁ by the same flat factor, with no change of shape to
    reveal it.

    Parameters
    ----------
    n_particles : int
        Tracer count over the whole box, post-subsample, pre-carve.
    box : SnapshotBox

    Returns
    -------
    float
    """
    return n_particles / box.box_size**3


def peculiar_velocity_scale(box: SnapshotBox) -> float:
    """Factor converting the stored velocity block to peculiar km/s.

    Gadget-4 writes `v_snapshot = v_peculiar / sqrt(a)` -- the `a_scaling = 0.5`
    attribute on the `Velocities` dataset -- so the conversion is `sqrt(a)`.
    At this snapshot's a = 1 it is exactly 1 and multiplying is a no-op; it is
    applied regardless so that pointing the loader at a z > 0 snapshot gives
    peculiar velocities rather than a silent 1/sqrt(a) rescaling of every
    amplitude.

    Parameters
    ----------
    box : SnapshotBox

    Returns
    -------
    float
    """
    return float(np.sqrt(box.scale_factor))


def check_run_geometry(cfg: RunConfig, box: SnapshotBox, outer_radius: float) -> None:
    """Assert a run's radii fit inside this box's minimum-image ceiling.

    The guard `dvcorr.conventions.MAX_ANALYSIS_RADIUS` provides for MDPL2, at
    this box's own size. The estimators' internal checks are against the frozen
    500 and every radius here passes them trivially, so without this the 300
    h^-1 Mpc box would run with NO effective ceiling: a carve wider than
    `box.max_analysis_radius` wraps around the periodic faces, and the plain
    Euclidean geometry the estimators use inside the carve stops being the
    minimum image -- silently, with no exception and no NaN.

    `outer_radius` is passed rather than read off `cfg` because the radius that
    must fit is not always `cfg.sub_volume_radius`: on the redshift-space path
    the tracer field is carved at the BUFFERED radius
    `sub_volume_radius + v_margin`, and that is the number the box has to hold.
    Making the caller name it is what stops the wrong one being checked.

    Parameters
    ----------
    cfg : RunConfig
        Read for `cfg.shells.max_radius` only.
    box : SnapshotBox
    outer_radius : float
        The largest radius the run actually carves at, h^-1 Mpc.

    Raises
    ------
    ValueError
        If `outer_radius` or `cfg.shells.max_radius` exceeds
        `box.max_analysis_radius`, or if the carve leaves no core -- a
        `sub_volume_radius` at or below `shells.max_radius` admits no center
        whose outermost shell fits inside the carve, so the run would produce
        zero centers after an expensive load.
    """
    ceiling = box.max_analysis_radius
    if outer_radius > ceiling:
        raise ValueError(
            f"outer carve radius {outer_radius:.2f} h^-1 Mpc exceeds this box's "
            f"minimum-image ceiling {ceiling:.2f} (= box_size / 2 = "
            f"{box.box_size:.0f} / 2). Beyond it the carve wraps the periodic "
            "faces and plain Euclidean separations stop being minimum-image "
            "(CLAUDE.md hard rule 3)."
        )
    if cfg.shells.max_radius > ceiling:
        raise ValueError(
            f"shells.max_radius {cfg.shells.max_radius:.2f} h^-1 Mpc exceeds "
            f"this box's minimum-image ceiling {ceiling:.2f}."
        )
    if cfg.sub_volume_radius <= cfg.shells.max_radius:
        raise ValueError(
            f"sub_volume_radius ({cfg.sub_volume_radius:.2f}) must exceed "
            f"shells.max_radius ({cfg.shells.max_radius:.2f}): the core cut "
            "keeps only centers whose outermost shell fits inside the carve, "
            "and at this setting no center can."
        )


# ---------------------------------------------------------------------------
# Stage 1: load + carve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarvedParticles:
    """Particles inside the analysis sub-volume, plus the box counts.

    The `dvcorr.pipeline.velocity_centered.CarvedHalos` of this pipeline,
    minus the two fields that have no meaning for particles: there is no
    `mvir` (every particle has `SnapshotBox.particle_mass`) and no
    `is_distinct` (a particle is not a halo, so "is it a subhalo" does not
    apply). They are omitted rather than filled with plausible constants --
    a uniform `mvir` column would make the mass-funnel diagnostics run and
    return a meaningless histogram.

    Attributes
    ----------
    pos : ndarray, shape (N_carved, 3), float64
        Positions inside the carve, comoving h^-1 Mpc, folded into
        [0, box_size).
    vel : ndarray, shape (N_carved, 3), float64
        Peculiar velocities, km/s, same row order.
    n_carved : int
        `pos.shape[0]`.
    n_total : int
        Tracers in the WHOLE box before the carve and after any subsample --
        the count `snapshot_number_density` must be given. Equal to
        `n_snapshot` only when `subsample_factor` is 1.
    n_snapshot : int
        Particles the file holds, before the subsample. Kept so a run reports
        the thinning it applied rather than only its result.
    subsample_factor : int
        The thinning actually used; 1 means the full field.
    """

    pos: np.ndarray
    vel: np.ndarray
    n_carved: int
    n_total: int
    n_snapshot: int
    subsample_factor: int


def _subsample_mask(n_snapshot: int, factor: int, seed: int) -> np.ndarray:
    """Seeded random 1-in-`factor` mask over the snapshot's particles.

    A random mask, never a stride -- see this module's "Subsampling" section
    for why `pos[::factor]` and an ID stride are both spatially structured
    subsets rather than fair thinnings of the density field.

    Parameters
    ----------
    n_snapshot : int
        Particles in the file.
    factor : int
        Keep 1 in this many. Must be >= 1; 1 returns an all-True mask.
    seed : int

    Returns
    -------
    ndarray, shape (n_snapshot,), bool

    Raises
    ------
    ValueError
        If `factor` < 1.
    """
    if factor < 1:
        raise ValueError(f"subsample_factor must be >= 1 (1 = keep all), got {factor}.")
    if factor == 1:
        return np.ones(n_snapshot, dtype=bool)
    rng = np.random.default_rng(seed)
    n_keep = n_snapshot // factor
    mask = np.zeros(n_snapshot, dtype=bool)
    mask[rng.choice(n_snapshot, size=n_keep, replace=False)] = True
    return mask


def _read_particles(
    path: Path, box: SnapshotBox, subsample_factor: int, subsample_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read positions and peculiar velocities, folded and optionally thinned.

    Held at float32 (the file's own width) through the read and the thinning,
    and widened to float64 only by the carve -- the same memory strategy
    `dvcorr.pipeline.velocity_centered.HaloArrays` documents, for the same
    reason: the whole-box arrays are large and only the carved subset needs to
    meet the estimators' float64 contract.

    Positions are folded with `np.mod(pos, box_size)` into the half-open
    [0, box_size), matching what `dvcorr.pipeline.catalog_conversion` does to
    the MDPL2 catalogs. Gadget emits a handful of coordinates at exactly
    `box_size` (one, in this snapshot); the fold sends them to 0.0, which is
    the same point under PBC.

    Parameters
    ----------
    path : Path
    box : SnapshotBox
    subsample_factor, subsample_seed : int
        Passed to `_subsample_mask`.

    Returns
    -------
    pos : ndarray, shape (N_total, 3), float32
    vel : ndarray, shape (N_total, 3), float32
        Comoving h^-1 Mpc and peculiar km/s, row-aligned.
    """
    keep = _subsample_mask(box.n_particles, subsample_factor, subsample_seed)
    with h5py.File(path, "r") as snapshot:
        particles = snapshot[PARTICLE_TYPE_GROUP]
        pos = particles[_COORDINATES_DATASET][:]
        vel = particles[_VELOCITIES_DATASET][:]
    if subsample_factor > 1:
        pos = pos[keep]
        vel = vel[keep]
    # In place: both arrays are freshly read (or freshly masked) and unaliased,
    # and at this size a copy of each is ~200 MB of pure waste.
    np.mod(pos, np.float32(box.box_size), out=pos)
    vel *= np.float32(peculiar_velocity_scale(box))
    return pos, vel


def load_and_carve_particles(
    cfg: RunConfig,
    box: SnapshotBox,
    path: Path,
    *,
    subsample_factor: int = DEFAULT_SUBSAMPLE_FACTOR,
    subsample_seed: int = DEFAULT_SUBSAMPLE_SEED,
) -> CarvedParticles:
    """Load the snapshot and carve a sphere around the box center.

    The `dvcorr.pipeline.velocity_centered.load_and_carve` of this pipeline:
    same contract (load a tracer population, keep what lies within
    `cfg.sub_volume_radius` of the observer, widen to float64), against a
    particle snapshot and this box's own observer instead of a halo Parquet and
    `dvcorr.conventions.OBSERVER_POSITION`. Plain Euclidean distance IS the
    minimum image for the carve itself, because `check_run_geometry` has
    already required `cfg.sub_volume_radius <= box.max_analysis_radius`.

    Parameters
    ----------
    cfg : RunConfig
        Supplies `sub_volume_radius` and `shells.max_radius`. `cfg.catalog` is
        NOT read -- there is no halo catalog on this path, and no mass or
        subhalo cut a particle field could take.
    box : SnapshotBox
        From `read_snapshot_header`.
    path : Path
        The snapshot, i.e. `PathsConfig().dm_snapshot`.
    subsample_factor : int, keyword-only
        Keep a seeded random 1-in-this-many of the particles; 1 keeps all. See
        this module's "Subsampling" section.
    subsample_seed : int, keyword-only

    Returns
    -------
    CarvedParticles

    Raises
    ------
    ValueError
        Via `check_run_geometry`, if the run's radii do not fit this box.
    RuntimeError
        If no particle survives the carve -- checked before any percentage is
        printed from the count, so the failure reads as a diagnosis rather
        than as a ZeroDivisionError inside an f-string.
    """
    check_run_geometry(cfg, box, cfg.sub_volume_radius)
    observer = snapshot_observer(box)

    pos_all, vel_all = _read_particles(path, box, subsample_factor, subsample_seed)
    n_total = pos_all.shape[0]

    d_obs = np.linalg.norm(pos_all - observer.astype(np.float32), axis=1)
    in_sub = d_obs <= cfg.sub_volume_radius
    pos = pos_all[in_sub].astype(float)
    vel = vel_all[in_sub].astype(float)
    n_carved = pos.shape[0]
    if n_carved == 0:
        raise RuntimeError(
            f"no particles survived the sub_volume_radius = "
            f"{cfg.sub_volume_radius} carve around {observer}; check "
            "sub_volume_radius against the box size."
        )
    print(
        f"carved {n_carved} of {n_total} particles within sub_volume_radius = "
        f"{cfg.sub_volume_radius} h^-1 Mpc ({100.0 * n_carved / n_total:.2f}%), "
        f"subsample 1-in-{subsample_factor} of {box.n_particles}"
    )
    return CarvedParticles(
        pos=pos,
        vel=vel,
        n_carved=n_carved,
        n_total=n_total,
        n_snapshot=box.n_particles,
        subsample_factor=subsample_factor,
    )


def load_and_carve_particles_buffered(
    cfg: RedshiftSpaceRunConfig,
    box: SnapshotBox,
    path: Path,
    *,
    subsample_factor: int = DEFAULT_SUBSAMPLE_FACTOR,
    subsample_seed: int = DEFAULT_SUBSAMPLE_SEED,
) -> BufferedCarve:
    """Load the snapshot once; carve twice (plain, then buffered).

    The `dvcorr.pipeline.redshift_space_comparison.load_and_carve_buffered` of
    this pipeline, and a `BufferedCarve` is what it returns -- so
    `build_tracer_spaces`, `select_redshift_shared_centers` and
    `run_both_spaces` are called downstream exactly as notebook 07 calls them.

    Pass 1 carves at the plain `cfg.sub_volume_radius` and derives `v_margin`
    from that population's |v_r|; pass 2 carves at
    `sub_volume_radius + v_margin`, the tracer field that gets displaced. Both
    filter the same in-memory arrays, so this is one file read.

    The buffered radius is the one `check_run_geometry` must protect, and it is
    not known until after pass 1 -- so the check runs TWICE: once on
    `sub_volume_radius` before reading anything, and again on the buffered
    radius as soon as `v_margin` exists. A box this size makes that second
    check load-bearing rather than ceremonial: at
    `v_margin_statistic = "max"` the margin on a particle field is ~29 h^-1 Mpc
    (the fastest particle in the carve moves at ~2900 km/s), so a
    `sub_volume_radius` above ~120 buffers straight through
    `box.max_analysis_radius`. A percentile statistic is the affordable choice
    here; see `RedshiftSpaceRunConfig.v_margin_statistic` for what it costs.

    `BufferedCarve` requires `mvir_core`, `is_distinct_core` and `catalog_mvir`,
    none of which a particle field has. They are filled with the uniform
    `box.particle_mass` and an all-True flag so the dataclass is well formed;
    nothing on the 07 path reads them (the two center selectors take their mass
    and distinctness arguments as optional keywords, left at `None` by the
    notebook), and the mass-funnel diagnostics must not be pointed at this
    pipeline's output.

    Parameters
    ----------
    cfg : RedshiftSpaceRunConfig
    box : SnapshotBox
    path : Path
    subsample_factor : int, keyword-only
    subsample_seed : int, keyword-only

    Returns
    -------
    BufferedCarve

    Raises
    ------
    ValueError
        Via `check_run_geometry`, if either the plain or the BUFFERED radius
        does not fit this box.
    RuntimeError
        If no particle survives the plain carve, so no `v_margin` can be
        derived.
    """
    check_run_geometry(cfg, box, cfg.sub_volume_radius)
    observer = snapshot_observer(box)

    pos_all, vel_all = _read_particles(path, box, subsample_factor, subsample_seed)
    n_total = pos_all.shape[0]
    d_obs = np.linalg.norm(pos_all - observer.astype(np.float32), axis=1)

    in_core = d_obs <= cfg.sub_volume_radius
    pos_core = pos_all[in_core].astype(float)
    vel_core = vel_all[in_core].astype(float)
    n_core = pos_core.shape[0]
    if n_core == 0:
        raise RuntimeError(
            f"no particles survived the plain sub_volume_radius = "
            f"{cfg.sub_volume_radius} carve; cannot derive v_margin."
        )

    v_r_core = radial_velocity(pos_core, vel_core, observer=observer)
    v_margin_kms = v_margin_from_statistic(
        v_r_core, cfg.v_margin_statistic, cfg.v_margin_percentile
    )
    v_margin_mpc = v_margin_kms / 100.0  # H0 = 100 h km/s/Mpc -> h^-1 Mpc

    buffered_radius = cfg.sub_volume_radius + v_margin_mpc
    check_run_geometry(cfg, box, buffered_radius)

    in_buffer = d_obs <= buffered_radius
    pos_buffer = pos_all[in_buffer].astype(float)
    vel_buffer = vel_all[in_buffer].astype(float)
    n_buffer = pos_buffer.shape[0]

    print(
        f"v_margin ({cfg.v_margin_statistic}) = {v_margin_kms:.1f} km/s = "
        f"{v_margin_mpc:.2f} h^-1 Mpc; buffered carve: {n_buffer} particles "
        f"within {buffered_radius:.2f} h^-1 Mpc (ceiling "
        f"{box.max_analysis_radius:.0f}), {100.0 * (n_buffer / n_core - 1.0):.1f}% "
        f"more than the plain {cfg.sub_volume_radius:.0f} h^-1 Mpc carve "
        f"({n_core} particles)"
    )

    uniform_mass_core = np.full(n_core, box.particle_mass, dtype=np.float32)
    uniform_mass_total = np.full(n_total, box.particle_mass, dtype=np.float32)
    return BufferedCarve(
        pos_buffer=pos_buffer,
        vel_buffer=vel_buffer,
        pos_core=pos_core,
        vel_core=vel_core,
        n_core=n_core,
        n_buffer=n_buffer,
        n_total=n_total,
        v_margin_kms=v_margin_kms,
        v_margin_mpc=v_margin_mpc,
        buffered_radius=buffered_radius,
        mvir_core=uniform_mass_core,
        is_distinct_core=np.ones(n_core, dtype=bool),
        catalog_mvir=uniform_mass_total,
    )
