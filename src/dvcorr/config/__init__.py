"""
dvcorr.config
-------------
Tunable settings, stored in classes -- the counterpart to `dvcorr.conventions`'s
frozen conventions. See `dvcorr/config/settings.py` for the full split
rationale.

One dataclass per file (`paths.py`, `cosmology.py`, `shells.py`,
`selection.py`), aggregated by `Settings` (`settings.py`). Re-exported here for
a clean import surface: `from dvcorr.config import Settings, default_settings,
PathsConfig, CosmologyConfig, ShellConfig, SelectionConfig`.
"""

from __future__ import annotations

from dvcorr.config.cosmology import CosmologyConfig
from dvcorr.config.paths import PathsConfig
from dvcorr.config.selection import SelectionConfig
from dvcorr.config.settings import Settings, default_settings
from dvcorr.config.shells import ShellConfig

__all__ = [
    "Settings",
    "default_settings",
    "PathsConfig",
    "CosmologyConfig",
    "ShellConfig",
    "SelectionConfig",
]
