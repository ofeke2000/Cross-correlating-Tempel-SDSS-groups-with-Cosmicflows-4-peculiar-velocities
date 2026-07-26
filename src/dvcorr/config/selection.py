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
    (a mass floor matching the catalog, an observer count and placement
    strategy for the MDPL2 mock-observer arm) rather than the old repo's full
    `OriginConfig` surface. Expect this to grow once `mocks/` and `selection/`
    are implemented.

    Attributes
    ----------
    mass_min : float
        Minimum halo mass, h^-1 M_sun. Defaults to 1e12, matching the
        catalog floor (`mdpl2_rockstar_125_pid-1_mvir12.csv` is already cut
        at mvir >= 1e12).
    mass_max : float | None
        Maximum halo mass, h^-1 M_sun, or None for no upper cut.
    number_of_observers : int
        Number of mock observers to place in the box.
    observer_selection : str
        Observer placement strategy; one of "random" or "virgo".
    """

    #: Allowed values of `observer_selection`. Not a dataclass field (no type
    #: annotation), just a class-level constant the validator checks against.
    _ALLOWED_OBSERVER_SELECTIONS = frozenset({"random", "virgo"})

    mass_min: float = 1e12
    mass_max: float | None = None
    number_of_observers: int = 1000
    observer_selection: str = "random"

    def __post_init__(self) -> None:
        if self.mass_min <= 0.0:
            raise ValueError(f"SelectionConfig: mass_min must be > 0, got {self.mass_min}.")
        if self.mass_max is not None and self.mass_max <= self.mass_min:
            raise ValueError(
                f"SelectionConfig: mass_max ({self.mass_max}) must be greater "
                f"than mass_min ({self.mass_min})."
            )
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
