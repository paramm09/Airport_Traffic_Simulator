"""Airspace waypoint model.

A waypoint marks a named point of the continuous airspace an airborne aircraft
flies toward: an identifier plus a flat (x, y) position and a target altitude.
It is just data; movement along it is implemented in
``backend/simulation/airspace``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Waypoint:
    """A named point in the airspace with a target altitude.

    Attributes:
        id: Unique waypoint name, e.g. ``"DEP_EAST"``. Must be non-empty.
        x: Horizontal position along the east axis (distance units).
        y: Horizontal position along the north axis (distance units).
        altitude: Target altitude, must be non-negative.
    """

    id: str
    x: float
    y: float
    altitude: float

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("waypoint id must not be empty")
        if self.altitude < 0:
            raise ValueError(
                f"waypoint altitude must not be negative, got {self.altitude!r}"
            )