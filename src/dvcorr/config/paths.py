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

from dvcorr.config.catalog import CATALOG_FULL, CATALOG_MVIR12, VALID_CATALOGS


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

    Both MDPL2 halo catalogs are gitignored and large (`_full` is ~127M rows /
    11.2 GB as CSV, `_mvir12` ~4M rows / 610 MB); this class only names their
    paths, it does not load them -- see CLAUDE.md's "don't load it fully unless
    the task needs it". Runs address them by NAME through `halo_catalog`, never
    by field, so the choice travels in a `CatalogConfig` rather than in
    whichever path a caller happened to reach for.

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
    mdpl2_catalog_full : Path
        Raw MDPL2/Rockstar snapshot-125 catalog (~127M halos, subhalos
        included), default `data_dir / "mdpl2_rockstar_snapnum125.csv"`.
        Resolved by name via `halo_catalog(CATALOG_FULL)`.
    mdpl2_catalog_mvir12 : Path
        Pre-cut subset (4,093,751 distinct halos at mvir >= 1e12), default
        `data_dir / "mdpl2_rockstar_125_pid-1_mvir12.csv"`. Resolved by name
        via `halo_catalog(CATALOG_MVIR12)`.
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
    mdpl2_catalog_full: Path | None = None
    mdpl2_catalog_mvir12: Path | None = None
    cf4_groups_catalog: Path | None = None
    cf4_velocities_catalog: Path | None = None
    sdss_tempel_catalog: Path | None = None

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        if self.mdpl2_catalog_full is None:
            self.mdpl2_catalog_full = self.data_dir / "mdpl2_rockstar_snapnum125.csv"
        if self.mdpl2_catalog_mvir12 is None:
            self.mdpl2_catalog_mvir12 = self.data_dir / "mdpl2_rockstar_125_pid-1_mvir12.csv"
        if self.cf4_groups_catalog is None:
            self.cf4_groups_catalog = self.data_dir / "CF4_Groups.csv"
        if self.cf4_velocities_catalog is None:
            self.cf4_velocities_catalog = self.data_dir / "CF4_Groups_Velocities.csv"
        if self.sdss_tempel_catalog is None:
            self.sdss_tempel_catalog = self.data_dir / "SDSS_Temple.csv"

    def halo_catalog(self, name: str, *, parquet: bool = True) -> Path:
        """Resolve a catalog NAME to a file path.

        The single place a `dvcorr.config.catalog.CatalogConfig.name` becomes a
        file. Callers pass the name, never a path field, so a run's catalog
        choice travels in its config rather than in whichever attribute the
        caller happened to reach for.

        Parameters
        ----------
        name : str
            One of `dvcorr.config.catalog.VALID_CATALOGS`.
        parquet : bool, keyword-only
            Return the `.parquet` sibling (the default -- what the pipeline
            reads, written by `dvcorr.pipeline.catalog_conversion`) rather than
            the source `.csv`. The CSV remains the source of truth; only the
            converter and `notebooks/04_first_mdpl2_run.ipynb` ask for it.

        Returns
        -------
        Path
            The catalog file. NOT checked for existence -- a Parquet path is a
            legitimate request before the conversion has been run, and the
            converter needs to name a file that does not exist yet.

        Raises
        ------
        ValueError
            If `name` is not a known catalog. Raised here rather than returning
            a plausible-looking default, so a typo cannot silently read the
            wrong catalog and be mistaken for a physical result.
        """
        if name not in VALID_CATALOGS:
            raise ValueError(
                f"PathsConfig.halo_catalog: unknown catalog {name!r}; "
                f"expected one of {sorted(VALID_CATALOGS)}."
            )
        csv_path = {
            CATALOG_FULL: self.mdpl2_catalog_full,
            CATALOG_MVIR12: self.mdpl2_catalog_mvir12,
        }[name]
        return csv_path.with_suffix(".parquet") if parquet else csv_path

    def ensure_output_dir(self) -> None:
        """Create `output_dir` (and any missing parents) if it does not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
