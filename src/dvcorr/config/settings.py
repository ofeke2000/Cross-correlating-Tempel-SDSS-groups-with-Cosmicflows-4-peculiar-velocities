"""
settings.py
------------
`Settings` -- the aggregator over every tunable settings group in
`dvcorr.config` -- and `default_settings()`, its factory.

`dvcorr.conventions` is the single source of truth for anything that carries a
sign or a normalization (pair orientation, the observer, the box, catalog
column names): change it and the science answer changes, silently, if you are
not careful. Nothing in `dvcorr.config` may redefine or restate one of those
conventions; where a setting here needs one (e.g. the shell-edge ceiling, or
the cosmology's self-consistency check against
`dvcorr.conventions.HUBBLE_PARAM`), it is imported from `dvcorr.conventions`,
never re-typed.

Everything in `dvcorr.config` is a first-pass, tunable knob: file locations,
the MDPL2 cosmology parameters (fixed by the simulation, but not sign-bearing),
the radial binning used by `dvcorr.estimators.shell_dipole.shell_dipole`, and
the not-yet-built mocks/selection arm's placeholder parameters. These are
exactly the "numbers and stable configuration ... stored in classes" that
CLAUDE.md's coding conventions ask for, and the mechanism behind hard rule 4
(no bare numbers): every literal lives on a dataclass field, not inline in
analysis code.

Four small dataclasses, one per file, plus this aggregator:

    PathsConfig      (paths.py)     -- catalog and output file locations,
                                        repo-root-relative
    CosmologyConfig  (cosmology.py) -- MDPL2 cosmology (frozen/immutable, but
                                        not a `dvcorr.conventions` convention
                                        -- it is simulation metadata, not a
                                        sign)
    ShellConfig      (shells.py)    -- radial shell binning for the
                                        shell-dipole estimator
    SelectionConfig  (selection.py) -- placeholder knobs for the not-yet-built
                                        mocks/ and selection/ arms

    Settings         (here)         -- aggregates one instance of each; use
                                        `default_settings()` to obtain a
                                        fresh, independent instance rather
                                        than a shared module-level singleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dvcorr.config.cosmology import CosmologyConfig
from dvcorr.config.paths import PathsConfig
from dvcorr.config.selection import SelectionConfig
from dvcorr.config.shells import ShellConfig


@dataclass
class Settings:
    """Aggregates one instance of every tunable settings group.

    Each field uses `default_factory` so that two `Settings` instances (e.g.
    two `default_settings()` calls) own independent sub-configs; mutating one
    instance's `shells` does not affect another's.

    Attributes
    ----------
    paths : PathsConfig
    cosmology : CosmologyConfig
    shells : ShellConfig
    selection : SelectionConfig
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    shells: ShellConfig = field(default_factory=ShellConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)


def default_settings() -> Settings:
    """Return a fresh `Settings` with independent, default-valued sub-configs.

    Deliberately a factory rather than a shared module-level singleton: callers
    that need to override a field (e.g. a test pointing `paths.output_dir` at
    a tmp directory) get their own instance to mutate, without risk of
    clobbering another caller's settings.
    """
    return Settings()
