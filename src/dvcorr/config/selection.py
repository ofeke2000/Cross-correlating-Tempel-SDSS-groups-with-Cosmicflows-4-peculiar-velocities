"""
selection.py
------------
Placeholder selection knobs for the not-yet-built `dvcorr.mocks` and
`dvcorr.selection` arms.

Part of the `dvcorr.config` package -- the tunable-settings counterpart to
`dvcorr.conventions`'s frozen conventions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelectionConfig:
    """Placeholder selection knobs for the not-yet-built `mocks/` and `selection/` arms.

    Kept deliberately lean: only the fields that are already clearly needed
    (an observer count and placement strategy for the MDPL2 mock-observer arm)
    rather than the old repo's full `OriginConfig` surface. Expect this to grow
    once `mocks/` and `selection/` are implemented.

    HALO selection does NOT live here. Which catalog a run reads and which
    halos it keeps from it are `dvcorr.config.catalog.CatalogConfig`'s job,
    which is where `mass_min` / `mass_max` moved to. They sat here unread for
    as long as they existed -- the catalog's mass floor was baked into its
    filename, so nothing needed to consult a config field for it. Keeping a
    second `mass_min` in the codebase after the live one landed would invite
    setting the wrong one and seeing no effect.

    Attributes
    ----------
    number_of_observers : int
        Number of mock observers to place in the box.
    observer_selection : str
        Observer placement strategy; one of "random" or "virgo".
    """

    #: Allowed values of `observer_selection`. Not a dataclass field (no type
    #: annotation), just a class-level constant the validator checks against.
    _ALLOWED_OBSERVER_SELECTIONS = frozenset({"random", "virgo"})

    number_of_observers: int = 1000
    observer_selection: str = "random"

    def __post_init__(self) -> None:
        if self.number_of_observers < 1:
            raise ValueError(
                "SelectionConfig: number_of_observers must be >= 1, got "
                f"{self.number_of_observers}."
            )
        if self.observer_selection not in self._ALLOWED_OBSERVER_SELECTIONS:
            raise ValueError(
                f"SelectionConfig: observer_selection must be one of "
                f"{sorted(self._ALLOWED_OBSERVER_SELECTIONS)}, got "
                f"{self.observer_selection!r}."
            )
