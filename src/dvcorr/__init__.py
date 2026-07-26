"""
dvcorr
------
Density-velocity cross-correlation dipole estimator for Tempel et al. (2017)
SDSS groups and Cosmicflows-4 peculiar velocities (simulation-validation arm:
MDPL2 halos). See the project README and docs/architecture.md for the full
per-file map.

Deliberately minimal: this module imports nothing beyond the standard library,
so `import dvcorr` never pulls in scipy, matplotlib, or pandas as a side
effect. Import submodules explicitly, e.g. `from dvcorr import conventions`,
`from dvcorr.estimators.shell_dipole import shell_dipole`.
"""

from __future__ import annotations

__version__ = "0.1.0"
