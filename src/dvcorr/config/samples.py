"""
samples.py
----------
`SampleConfig` -- the cuts that turn the observed catalogs into the two
analysis samples of Stage 2 (`docs/real_data_roadmap.md`): the Tempel group
density tracer and the CF4 group velocity tracer. The builders that apply
them are in `dvcorr.pipeline.real_catalogs`.

Also the names of the CF4 velocity definitions a sample can take as u
(`VALID_VELOCITY_DEFINITIONS`). They live here, not next to the formulas,
because `config` may import only `conventions` (the import-layering contract
in `docs/architecture.md`).

Part of the `dvcorr.config` package -- the tunable-settings counterpart to
`dvcorr.conventions`'s frozen conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

#: EDD's `Vpec` recipe with the natural log restored: w Vpds + (1 - w) v_WF,
#: w = clip(1 - Vls/3000, 0, 1) (w = 1 where fV < 0), v_WF the Watkins &
#: Feldman (2015) estimator with ln. The primary choice (decided 2026-10-18,
#: notebook 11 §7 and §13).
VELOCITY_VPEC_LN = "vpec_ln"
#: The Watkins & Feldman estimator with ln at every depth, no ramp.
VELOCITY_VPWF_LN = "vpwf_ln"
#: EDD's published `Vpec`: ln 10 low beyond Vls = 3,000 km/s. A Stage 12
#: velocity-definition variant only.
VELOCITY_VPEC_PUBLISHED = "vpec_published"
#: EDD's published `Vpds` (Davis & Scrimgeour 2014): the right scale, skewed
#: errors at depth.
VELOCITY_VPDS = "vpds"
VALID_VELOCITY_DEFINITIONS = frozenset(
    {VELOCITY_VPEC_LN, VELOCITY_VPWF_LN, VELOCITY_VPEC_PUBLISHED, VELOCITY_VPDS}
)


@dataclass(frozen=True)
class SampleConfig:
    """Cuts that define the Stage 2 analysis samples.

    Attributes
    ----------
    velocity_cz_max : float
        km/s. CF4 groups are kept at CMB-frame `Vcmb` below this: Nusser
        (2017)'s cz < 10,000 km/s. Applied to the velocity tracer only; the
        density tracer is not cut in depth (it must reach past the deepest
        velocity center by the largest shell radius).
    min_richness : int
        Tempel tracers are kept at `Ng` >= this. `Ng` is the Tempel+14
        parent richness, constant within a group and 1 for every isolated
        galaxy, so the default 1 keeps all groups and every singleton --
        the primary sample (decided 2026-10-15), the closest analogue of
        Nusser's all-galaxy 2MRS field. 2 and 6 are the planned splits.
    footprint_nside : int
        HEALPix resolution of the SDSS footprint used for CF4 membership:
        the pixels holding at least one Tempel galaxy. 32 (3.4 deg^2
        pixels, ~135 Tempel galaxies each) leaves no interior pixel empty;
        it is slightly generous at the edges. Stage 4 builds the real
        angular mask.
    velocity_definition : str
        Which velocity becomes u; one of `VALID_VELOCITY_DEFINITIONS`.
        Default `VELOCITY_VPEC_LN`.
    """

    velocity_cz_max: float = 10000.0
    min_richness: int = 1
    footprint_nside: int = 32
    velocity_definition: str = VELOCITY_VPEC_LN

    def __post_init__(self) -> None:
        if self.velocity_definition not in VALID_VELOCITY_DEFINITIONS:
            raise ValueError(
                f"SampleConfig: velocity_definition must be one of "
                f"{sorted(VALID_VELOCITY_DEFINITIONS)}, got {self.velocity_definition!r}."
            )
        if self.min_richness < 1:
            raise ValueError(
                f"SampleConfig: min_richness must be >= 1 (Ng of a singleton), got "
                f"{self.min_richness}."
            )
        if self.velocity_cz_max <= 0:
            raise ValueError(
                f"SampleConfig: velocity_cz_max must be positive, got {self.velocity_cz_max}."
            )
