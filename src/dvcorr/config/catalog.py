"""
catalog.py
----------
Which halo catalog a run reads, and which halos it keeps from it.

Part of the `dvcorr.config` package -- the tunable-settings counterpart to
`dvcorr.conventions`'s frozen conventions.

Two catalogs, both first-class
------------------------------
The project carries two MDPL2 snapshot-125 catalogs and neither supersedes the
other. `CATALOG_FULL` is the raw snapshot; `CATALOG_MVIR12` is the pre-cut
subset every result before this module was measured on. Both stay available so
runs can be compared against each other and so the earlier catalog can be
returned to at any time. `PathsConfig.halo_catalog` resolves the name to a
file; this module only names the choice and the cuts applied on top of it.

`CATALOG_FULL` is a STRICT SUPERSET of `CATALOG_MVIR12`: filtering it at
`mvir >= 1e12` with `include_subhalos = False` reproduces the smaller catalog
exactly -- 4,093,751 rows, 4,093,751 unique `rockstarid`, verified by
`tests/test_catalog_equivalence.py`. So the older catalog is reachable two
ways, and the agreement of those two paths is the strongest available check on
this whole layer.

Why the cuts live here and not in the file name
-----------------------------------------------
`CATALOG_MVIR12`'s selection (`mvir >= 1e12`, distinct halos only) is baked
into its file and recorded nowhere the code can read. That is exactly what
made `SelectionConfig.mass_min` dead for as long as it existed: the number was
in a filename, so nothing needed to read it, so nothing did. Here the cuts are
data, applied by `dvcorr.pipeline.velocity_centered._load_all_halos`, printed
on every run, and plotted by `dvcorr.pipeline.mass_diagnostics` -- so what a
run actually selected is visible rather than implied.

Resolution floor
----------------
`CATALOG_FULL` reaches down to 2 particles (`2 * conventions.PARTICLE_MASS`),
and about half of it sits below 33. Rockstar masses and especially peculiar
VELOCITIES are unreliable there, so `mass_min = None` (the default -- take
everything) is a deliberate choice to include that population, not an
oversight. `conventions.RESOLVED_PARTICLE_COUNT` and
`conventions.CONVERGED_PARTICLE_COUNT` are the thresholds to set `mass_min`
against when excluding it instead.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The raw MDPL2 snapshot-125 Rockstar catalog: ~127M halos, reaching down to
#: 2 * conventions.PARTICLE_MASS, subhalos included.
CATALOG_FULL: str = "full"

#: The pre-cut subset: 4,093,751 distinct halos at mvir >= 1e12. Every result
#: measured before the full catalog was wired in used this one.
CATALOG_MVIR12: str = "mvir12"

#: The mass floor `CATALOG_MVIR12` was cut at, h^-1 M_sun. Recorded here
#: because the number exists nowhere the code can otherwise read it -- it is
#: baked into that catalog's file and spelled out only in its filename.
#: Applying it to `CATALOG_FULL` is what reproduces `CATALOG_MVIR12`.
MVIR12_CATALOG_FLOOR: float = 1e12

#: Row count of `CATALOG_MVIR12`, and equally the number of distinct halos in
#: `CATALOG_FULL` above `MVIR12_CATALOG_FLOOR`. Measured, and asserted by
#: `tests/test_catalog_equivalence.py`: if the two ever disagree, the superset
#: relation this module's docstring claims has stopped holding.
MVIR12_CATALOG_ROWS: int = 4_093_751

#: Every value `CatalogConfig.name` may take. Named constants rather than bare
#: string literals so a typo'd "ful" raises in the constructor instead of
#: silently selecting the wrong catalog -- the same guard
#: `SelectionConfig._ALLOWED_OBSERVER_SELECTIONS` provides.
VALID_CATALOGS: frozenset[str] = frozenset({CATALOG_FULL, CATALOG_MVIR12})


@dataclass
class CatalogConfig:
    """Which halo catalog to read, and which halos to keep from it.

    Nested into `dvcorr.pipeline.velocity_centered.RunConfig` as its `catalog`
    field, mirroring how `ShellConfig` is nested as `shells`. Switching
    catalogs or cuts is therefore one line at the call site and no pipeline
    signature changes:

        RunConfig()                                                  # full, uncut
        RunConfig(catalog=CatalogConfig(name=CATALOG_MVIR12))        # older catalog
        RunConfig(catalog=CatalogConfig(mass_min=1e12,               # same cut,
                                        include_subhalos=False))     # full file

    The second and third forms select the same halos by construction (see this
    module's docstring); that they also produce the same measurement is what
    `tests/test_catalog_equivalence.py` asserts.

    Attributes
    ----------
    name : str
        Which catalog to read; one of `VALID_CATALOGS`. Resolved to a file by
        `dvcorr.config.paths.PathsConfig.halo_catalog`.
    mass_min : float | None
        Minimum `mvir`, h^-1 M_sun, inclusive. None means no floor -- take the
        catalog as it comes. Pushed down as a Parquet row-group filter, so on
        the mvir-sorted full catalog a high floor skips most of the file
        rather than reading and discarding it.
    mass_max : float | None
        Maximum `mvir`, h^-1 M_sun, inclusive, or None for no ceiling.
    include_subhalos : bool
        Whether to keep halos with `pid != -1`. `CATALOG_MVIR12` contains none
        either way; `CATALOG_FULL` is ~10-15% subhalos, whose velocities are
        orbital rather than large-scale flow.
    """

    name: str = CATALOG_FULL
    mass_min: float | None = None
    mass_max: float | None = None
    include_subhalos: bool = True

    def __post_init__(self) -> None:
        if self.name not in VALID_CATALOGS:
            raise ValueError(
                f"CatalogConfig: name must be one of {sorted(VALID_CATALOGS)}, "
                f"got {self.name!r}."
            )
        if self.mass_min is not None and self.mass_min <= 0.0:
            raise ValueError(
                f"CatalogConfig: mass_min must be > 0 or None (no floor), "
                f"got {self.mass_min}."
            )
        if self.mass_max is not None and self.mass_max <= 0.0:
            raise ValueError(
                f"CatalogConfig: mass_max must be > 0 or None (no ceiling), "
                f"got {self.mass_max}."
            )
        if (
            self.mass_min is not None
            and self.mass_max is not None
            and self.mass_max <= self.mass_min
        ):
            raise ValueError(
                f"CatalogConfig: mass_max ({self.mass_max}) must be greater than "
                f"mass_min ({self.mass_min})."
            )

    def describe_cuts(self) -> str:
        """One-line human-readable summary of the cuts, for run logs and plot titles.

        Returns
        -------
        str
            E.g. `"full, no mass cut, subhalos included"` or
            `"full, mvir >= 1e+12, distinct halos only"`.
        """
        parts = [self.name]
        if self.mass_min is None and self.mass_max is None:
            parts.append("no mass cut")
        else:
            if self.mass_min is not None:
                parts.append(f"mvir >= {self.mass_min:.3g}")
            if self.mass_max is not None:
                parts.append(f"mvir <= {self.mass_max:.3g}")
        parts.append("subhalos included" if self.include_subhalos else "distinct halos only")
        return ", ".join(parts)
