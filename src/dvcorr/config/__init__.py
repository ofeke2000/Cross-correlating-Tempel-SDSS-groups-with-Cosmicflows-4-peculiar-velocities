"""
dvcorr.config
-------------
Tunable settings, stored in classes -- the counterpart to `dvcorr.conventions`'s
frozen conventions. See `dvcorr/config/settings.py` for the full split
rationale.

One dataclass per file (`paths.py`, `cosmology.py`, `shells.py`,
`selection.py`, `catalog.py`), aggregated by `Settings` (`settings.py`).
`shells.py` also carries the pure shell-edge and volume-weighted-abscissa
builders (`linear_shell_edges`, `log_shell_edges`,
`volume_weighted_shell_radii`) that back `ShellConfig`'s properties, and
`catalog.py` the catalog-name constants (`CATALOG_FULL`, `CATALOG_MVIR12`)
that `PathsConfig.halo_catalog` resolves to files. Re-exported here for a
clean import surface: `from dvcorr.config import Settings, default_settings,
PathsConfig, CosmologyConfig, ShellConfig, SelectionConfig, CatalogConfig,
CATALOG_FULL, CATALOG_MVIR12, VALID_CATALOGS, SPACING_LINEAR, SPACING_LOG,
linear_shell_edges, log_shell_edges, volume_weighted_shell_radii`.
"""

from __future__ import annotations

from dvcorr.config.catalog import (
    CATALOG_FULL,
    CATALOG_MVIR12,
    VALID_CATALOGS,
    CatalogConfig,
)
from dvcorr.config.cosmology import CosmologyConfig
from dvcorr.config.paths import PathsConfig
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
    "ShellConfig",
    "SelectionConfig",
    "CatalogConfig",
    "CATALOG_FULL",
    "CATALOG_MVIR12",
    "VALID_CATALOGS",
    "SPACING_LINEAR",
    "SPACING_LOG",
    "linear_shell_edges",
    "log_shell_edges",
    "volume_weighted_shell_radii",
]
