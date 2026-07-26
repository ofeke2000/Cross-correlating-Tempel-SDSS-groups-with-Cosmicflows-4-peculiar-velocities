"""
paths.py
--------
Catalog and output file locations, derived from the repo root.

Part of the `dvcorr.config` package -- the tunable-settings counterpart to
`dvcorr.conventions`'s frozen conventions (see `dvcorr/config/__init__.py`
and `dvcorr/conventions.py` for the split rationale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    """Catalog and output file locations, derived from the repo root.

    Unlike the old bulk-flow repo's `paths_config.py`, nothing here is a
    hardcoded absolute path: every field is derived from `project_root` so the
    settings are portable across machines and checkouts. Only `project_root`
    itself has a real default (the repository root, resolved from this file's
    location under `src/dvcorr/config/`); every other field defaults to `None`
    and is resolved relative to `project_root` / `data_dir` in `__post_init__`,
    since one dataclass default cannot cleanly reference another.

    The MDPL2 halo catalog (`mdpl2_catalog`) is ~4M rows / ~600 MB and is
    gitignored; this class only names its path, it does not load it -- see
    CLAUDE.md's "don't load it fully unless the task needs it".

    Attributes
    ----------
    project_root : Path
        Repository root. Derived as
        `Path(__file__).resolve().parents[3]` -- this file lives at
        `<repo-root>/src/dvcorr/config/paths.py`, so climbing three parents
        (config -> dvcorr -> src -> repo-root) reaches the repo root even
        though the module itself sits three levels deep. This holds under an
        editable install, since `__file__` still points at the in-place
        source tree rather than a site-packages copy.
    data_dir : Path
        Input catalog directory, default `project_root / "data"`.
    output_dir : Path
        Output directory for derived products, default `project_root / "output"`.
    mdpl2_catalog : Path
        MDPL2/Rockstar halo catalog, default
        `data_dir / "mdpl2_rockstar_125_pid-1_mvir12.csv"`.
    cf4_groups_catalog : Path
        Cosmicflows-4 group catalog, default `data_dir / "CF4_Groups.csv"`.
    cf4_velocities_catalog : Path
        Cosmicflows-4 group peculiar velocities, default
        `data_dir / "CF4_Groups_Velocities.csv"`.
    sdss_tempel_catalog : Path
        Tempel et al. (2017) SDSS group catalog, default
        `data_dir / "SDSS_Temple.csv"`.
    """

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )
    data_dir: Path | None = None
    output_dir: Path | None = None
    mdpl2_catalog: Path | None = None
    cf4_groups_catalog: Path | None = None
    cf4_velocities_catalog: Path | None = None
    sdss_tempel_catalog: Path | None = None

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        if self.mdpl2_catalog is None:
            self.mdpl2_catalog = self.data_dir / "mdpl2_rockstar_125_pid-1_mvir12.csv"
        if self.cf4_groups_catalog is None:
            self.cf4_groups_catalog = self.data_dir / "CF4_Groups.csv"
        if self.cf4_velocities_catalog is None:
            self.cf4_velocities_catalog = self.data_dir / "CF4_Groups_Velocities.csv"
        if self.sdss_tempel_catalog is None:
            self.sdss_tempel_catalog = self.data_dir / "SDSS_Temple.csv"

    def ensure_output_dir(self) -> None:
        """Create `output_dir` (and any missing parents) if it does not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
