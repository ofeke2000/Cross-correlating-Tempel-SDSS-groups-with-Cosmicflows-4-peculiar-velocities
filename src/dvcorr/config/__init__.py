"""
dvcorr.config
-------------
Tunable settings, stored in classes -- the counterpart to `dvcorr.conventions`'s
frozen conventions. See `dvcorr/config/settings.py` for the full split
rationale.

One dataclass per file (`paths.py`, `cosmology.py`, `shells.py`,
`selection.py`, `catalog.py`, `samples.py`), aggregated by `Settings` (`settings.py`).
`shells.py` also carries the pure shell-edge and volume-weighted-abscissa
builders (`linear_shell_edges`, `log_shell_edges`,
`volume_weighted_shell_radii`) that back `ShellConfig`'s properties, and
`catalog.py` the catalog-name constants (`CATALOG_FULL`, `CATALOG_MVIR12`)
that `PathsConfig.halo_catalog` resolves to files, and `cosmology.py` the
derived-background free functions (`matter_density_parameter`, `growth_rate`,
`hubble_h_units`, `scale_factor_hubble`) that turn those parameters into the
`f` and `a H` every velocity prediction needs. Re-exported here for a
clean import surface: `from dvcorr.config import Settings, default_settings,
PathsConfig, CosmologyConfig, ShellConfig, SelectionConfig, CatalogConfig,
SampleConfig, CATALOG_FULL, CATALOG_MVIR12, VALID_CATALOGS, SPACING_LINEAR, SPACING_LOG,
linear_shell_edges, log_shell_edges, volume_weighted_shell_radii,
matter_density_parameter, growth_rate, hubble_h_units, scale_factor_hubble`.
"""

from __future__ import annotations

from dvcorr.config.catalog import (
    CATALOG_FULL,
    CATALOG_MVIR12,
    VALID_CATALOGS,
    CatalogConfig,
    add_catalog_arguments,
    catalog_from_args,
)
from dvcorr.config.cosmology import (
    CosmologyConfig,
    growth_rate,
    hubble_h_units,
    matter_density_parameter,
    scale_factor_hubble,
)
from dvcorr.config.paths import PathsConfig
from dvcorr.config.samples import (
    VALID_VELOCITY_DEFINITIONS,
    VELOCITY_VPDS,
    VELOCITY_VPEC_LN,
    VELOCITY_VPEC_PUBLISHED,
    VELOCITY_VPWF_LN,
    SampleConfig,
)
from dvcorr.config.selection import SelectionConfig
from dvcorr.config.settings import Settings, default_settings
from dvcorr.config.shells import (
    SPACING_LINEAR,
    SPACING_LOG,
    ShellConfig,
    linear_shell_edges,
    log_shell_edges,
    volume_weighted_shell_radii,
)

__all__ = [
    "Settings",
    "default_settings",
    "PathsConfig",
    "CosmologyConfig",
    "growth_rate",
    "hubble_h_units",
    "matter_density_parameter",
    "scale_factor_hubble",
    "ShellConfig",
    "SelectionConfig",
    "CatalogConfig",
    "SampleConfig",
    "VALID_VELOCITY_DEFINITIONS",
    "VELOCITY_VPEC_LN",
    "VELOCITY_VPWF_LN",
    "VELOCITY_VPEC_PUBLISHED",
    "VELOCITY_VPDS",
    "CATALOG_FULL",
    "CATALOG_MVIR12",
    "VALID_CATALOGS",
    "add_catalog_arguments",
    "catalog_from_args",
    "SPACING_LINEAR",
    "SPACING_LOG",
    "linear_shell_edges",
    "log_shell_edges",
    "volume_weighted_shell_radii",
]
