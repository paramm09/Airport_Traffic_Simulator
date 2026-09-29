"""Shared airspace waypoints for a simple deterministic flight pattern.

The airspace is a flat plane. Each waypoint carries a target altitude the
aircraft climbs to / descends from at ``ALTITUDE_RATE`` while it flies toward
the waypoint's (x, y) position.

Naming convention:
    DEP_*      departure phase
    CRUISE_*   cruise phase
    ARR_*      arrival phase
"""

from backend.models.waypoint import Waypoint

DEP_EAST = Waypoint("DEP_EAST", 0.0, 0.0, 1000.0)
CRUISE_NORTH = Waypoint("CRUISE_NORTH", 0.0, 10.0, 2000.0)
CRUISE_SOUTH = Waypoint("CRUISE_SOUTH", 10.0, 10.0, 3000.0)
ARR_EAST = Waypoint("ARR_EAST", 10.0, 0.0, 2000.0)
ARR_WEST = Waypoint("ARR_WEST", 20.0, 0.0, 800.0)

ALL_WAYPOINTS: list[Waypoint] = [
    DEP_EAST,
    CRUISE_NORTH,
    CRUISE_SOUTH,
    ARR_EAST,
    ARR_WEST,
]

WAYPOINTS: dict[str, Waypoint] = {waypoint.id: waypoint for waypoint in ALL_WAYPOINTS}