"""
cosmology.py
------------
MDPL2 simulation cosmology metadata, plus the catalog metadata of the
observed tables (CF4's H0 scale, Tempel's flat Planck 2015) and the one
chi(z) of the data arm (`comoving_distance_h_units`).

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

import numpy as np

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
    H0_CF4 : float
        The Hubble constant CF4's published distances and peculiar velocities
        are on (Tully et al. 2023), km/s/Mpc. NOT a cosmology this project
        adopts: it is the scale the catalog's `D` column was written on, and
        the one its `Vpds` / `Vpwf` columns subtract. Verified against the
        catalog, not taken on trust: `dvcorr.pipeline.real_catalogs`
        reproduces both columns from `fV` and `DMzp` with this value to
        within the file's rounding (`notebooks/11_real_catalog_audit.ipynb`).
        Because of it, CF4 `D` is never used as a position.
    H0_TEMPEL : float
        The Hubble constant Tempel et al. (2017, §1) wrote their distances on,
        km/s/Mpc: flat Planck 2015. Catalog metadata like `H0_CF4`, NOT an
        adopted cosmology. Tempel's `dc` is in Mpc on this scale, so it enters
        h^-1 Mpc through `h_tempel` and never through `h` (MDPL2's 0.6777,
        a +4.4e-4 fractional offset; `docs/real_data_roadmap.md` Stage 3).
    Om0_TEMPEL : float
        Tempel's matter density (flat, Omega_Lambda = 1 - Om0_TEMPEL = 0.692).
        The Omega_m of the one chi(z) that places all three catalogs
        (`comoving_distance_h_units`), so that `dc` is rebuilt to rounding.
    """

    H0: float = 67.77
    Om0: float = 0.307115
    Ode0: float = 0.692885
    Ob0: float = 0.048206
    sigma8: float = 0.8228
    ns: float = 0.96
    growth_index: float = 0.55
    flat: bool = True
    H0_CF4: float = 74.6
    H0_TEMPEL: float = 67.8
    Om0_TEMPEL: float = 0.308

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

    @property
    def h_tempel(self) -> float:
        """Tempel's reduced Hubble parameter, `H0_TEMPEL / 100` = 0.678.

        The factor that turns Tempel's `dc` (Mpc) into h^-1 Mpc. Not `h`: that
        is MDPL2's and is tied to `dvcorr.conventions.HUBBLE_PARAM`.
        """
        return self.H0_TEMPEL / 100.0

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


# --------------------------------------------------------------------------
# Derived background quantities
# --------------------------------------------------------------------------
#
# Free functions rather than methods: they transform the held metadata into a
# number and need nothing of `self` beyond reading it (CLAUDE.md's "prefer
# functions over class methods"). They live here, next to the parameters they
# read, so that there is exactly ONE definition of each -- see
# `hubble_h_units` for why that matters more than usual.


def matter_density_parameter(cosmology: CosmologyConfig, z: float) -> float:
    """`Omega_m(z)` for a flat LCDM background, dimensionless.

    Parameters
    ----------
    cosmology : CosmologyConfig
    z : float
        Redshift.

    Returns
    -------
    float
    """
    matter = cosmology.Om0 * (1.0 + z) ** 3
    return matter / (matter + cosmology.Ode0)


def growth_rate(cosmology: CosmologyConfig, z: float) -> float:
    """`f(z) = Omega_m(z)^gamma`, the linear growth rate, dimensionless.

    `gamma` is `CosmologyConfig.growth_index` (0.55 for GR). Callers wanting a
    growth rate DECOUPLED from `Omega_m` -- the modified-gravity reading, or a
    parameter scan over `f` at fixed transfer function -- pass their own `f`
    downstream instead of calling this.

    Parameters
    ----------
    cosmology : CosmologyConfig
    z : float
        Redshift.

    Returns
    -------
    float
    """
    return matter_density_parameter(cosmology, z) ** cosmology.growth_index


def hubble_h_units(cosmology: CosmologyConfig, z: float) -> float:
    """`H(z) / h`, km/s per h^-1 Mpc, for a flat LCDM background.

    The units conversion that every velocity prediction in this project passes
    through, and the most dangerous line in any of them: getting it upside
    down (multiplying by `h` instead of dividing) rescales every prediction by
    `h**2 ~ 0.46` with NO change of shape, so it cannot be caught by looking at
    a plot. It has a one-line check instead -- at z = 0 this must return
    EXACTLY 100, which is what h^-1 Mpc units are for.

    Divide, do not multiply: one h^-1 Mpc is `1/h` Mpc, so
    `H [km/s/Mpc] * (1/h) [Mpc per h^-1 Mpc] = H/h`.

    Parameters
    ----------
    cosmology : CosmologyConfig
    z : float
        Redshift.

    Returns
    -------
    float
        km/s per h^-1 Mpc.
    """
    hubble_z = cosmology.H0 * np.sqrt(cosmology.Om0 * (1.0 + z) ** 3 + cosmology.Ode0)
    return float(hubble_z / cosmology.h)


def scale_factor_hubble(cosmology: CosmologyConfig, z: float) -> float:
    """`a H(z)`, km/s per COMOVING h^-1 Mpc -- the "aH" of the linear theory.

    Linear theory relates a comoving separation to a peculiar velocity through
    exactly one factor of `a H`, and every prediction in `dvcorr.theory`
    carries exactly one. This is that factor, defined once so no module has to
    re-derive it: `a = 1 / (1 + z)` times `hubble_h_units`. At z = 0 it is
    exactly 100.

    Parameters
    ----------
    cosmology : CosmologyConfig
    z : float
        Redshift.

    Returns
    -------
    float
        km/s per comoving h^-1 Mpc.
    """
    return hubble_h_units(cosmology, z) / (1.0 + z)


#: Gauss-Legendre nodes for the chi(z) integral. 1/E(z) is analytic and
#: slowly varying on [0, z] for the z <= 0.2 of every catalog here, so 16
#: nodes are exact to double precision (checked against adaptive quadrature
#: in `tests/test_sky_positions.py` out to z = 1).
_CHI_QUADRATURE_NODES: int = 16


def flat_lcdm_expansion_rate(z: np.ndarray, om0: float) -> np.ndarray:
    """`E(z) = H(z) / H0` for a flat LCDM background, dimensionless.

    Parameters
    ----------
    z : array_like, shape (N,)
        Redshift.
    om0 : float
        Matter density today; Omega_Lambda = 1 - om0 (flat).

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.sqrt(om0 * (1.0 + z) ** 3 + (1.0 - om0))


def comoving_distance_h_units(z: np.ndarray, om0: float) -> np.ndarray:
    """Line-of-sight comoving distance chi(z), h^-1 Mpc, flat LCDM.

    chi = (c / 100) * integral_0^z dz' / E(z'). In h^-1 Mpc the Hubble
    constant cancels -- the 100 is the definition of h (km/s per h^-1 Mpc,
    the same one `hubble_h_units` returns at z = 0) -- so chi depends on
    `om0` alone. The one chi(z) of the data arm: Stage 3 places Tempel
    (through `dc`), CF4 and 2MRS with it at `CosmologyConfig.Om0_TEMPEL`.

    Parameters
    ----------
    z : array_like, shape (N,)
        Redshift, >= 0. A negative redshift has no distance; the caller
        decides what to do with it (`dvcorr.pipeline.sky_positions` leaves the
        position undefined), so this raises rather than returning a negative
        chi.
    om0 : float

    Returns
    -------
    numpy.ndarray, shape (N,)
        h^-1 Mpc. Shape (N,) even for N = 1.

    Raises
    ------
    ValueError
        If any z is negative or not finite.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError(
            "comoving_distance_h_units: z must be finite and >= 0; mask negative "
            "or missing redshifts before asking for a distance."
        )
    nodes, weights = np.polynomial.legendre.leggauss(_CHI_QUADRATURE_NODES)
    # Map the nodes from [-1, 1] onto [0, z] for every z at once: (N, nodes).
    half = 0.5 * z[:, None]
    zp = half * (nodes[None, :] + 1.0)
    integral = np.sum(weights[None, :] / flat_lcdm_expansion_rate(zp, om0), axis=1) * half[:, 0]
    return conventions.SPEED_OF_LIGHT / 100.0 * integral


def comoving_distance_derivative_h_units(z: np.ndarray, om0: float) -> np.ndarray:
    """d chi / dz = (c / 100) / E(z), h^-1 Mpc per unit redshift.

    What a redshift error becomes in distance: the rounding allowance of a
    chi built from a printed redshift is this times the redshift's half-ulp.

    Parameters
    ----------
    z : array_like, shape (N,)
    om0 : float

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    return conventions.SPEED_OF_LIGHT / 100.0 / flat_lcdm_expansion_rate(z, om0)
