"""
cosmology.py
------------
MDPL2 simulation cosmology metadata.

Part of the `dvcorr.config` package -- the tunable-settings counterpart to
`dvcorr.conventions`'s frozen conventions. `CosmologyConfig` is simulation
*metadata*, not a `dvcorr.conventions`-style sign or normalization convention,
which is why it lives here rather than in `dvcorr.conventions`; but `h` (see
below) must still agree with `dvcorr.conventions.HUBBLE_PARAM`, so
`__post_init__` checks that explicitly, imported from `dvcorr.conventions`,
never re-typed.
"""

from __future__ import annotations

from dataclasses import dataclass

from dvcorr import conventions

#: Tolerance for the CosmologyConfig <-> dvcorr.conventions.HUBBLE_PARAM
#: consistency check. The two are computed from the same literal (H0 = 67.77)
#: via two different paths (a stored constant vs. H0/100), so any real
#: disagreement is an editing mistake, not floating-point noise; this only
#: needs to be looser than double-precision round-off.
_H_CONSISTENCY_TOL: float = 1e-9


@dataclass(frozen=True)
class CosmologyConfig:
    """MDPL2 simulation cosmology.

    Fixed by the simulation the halo catalog was drawn from, not a
    user-editable analysis knob -- hence `frozen=True` rather than a plain
    dataclass. It is simulation *metadata*, not a `dvcorr.conventions`-style
    sign or normalization convention, which is why it lives here rather than
    in `dvcorr.conventions`; but `h` (see below) must still agree with
    `dvcorr.conventions.HUBBLE_PARAM`, so `__post_init__` checks that
    explicitly.

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
        if abs(self.h - conventions.HUBBLE_PARAM) > _H_CONSISTENCY_TOL:
            raise ValueError(
                f"CosmologyConfig.h = {self.h} (from H0 = {self.H0}) does not "
                f"match dvcorr.conventions.HUBBLE_PARAM = {conventions.HUBBLE_PARAM}. "
                "h and dvcorr.conventions.HUBBLE_PARAM are the same reduced "
                "Hubble parameter and must agree; if the cosmology genuinely "
                "changed, update dvcorr.conventions.HUBBLE_PARAM explicitly "
                "(see CLAUDE.md hard rule 1) rather than letting the two "
                "drift apart."
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
