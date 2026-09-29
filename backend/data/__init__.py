"""Airport network data: the nodes and connections of the airport itself.

This module only describes *what the airport looks like*. Algorithms such as
Dijkstra deliberately live in ``backend/algorithms`` so that the data and the
logic stay separate.

The airspace waypoints live here too: they are *data* describing the flight
pattern, consumed by the airspace movement in ``backend/simulation/airspace``.
"""

from backend.data.airspace_waypoints import (
    ALL_WAYPOINTS,
    ARR_EAST,
    ARR_WEST,
    CRUISE_NORTH,
    CRUISE_SOUTH,
    DEP_EAST,
    WAYPOINTS,
)

__all__ = [
    "ALL_WAYPOINTS",
    "WAYPOINTS",
    "DEP_EAST",
    "CRUISE_NORTH",
    "CRUISE_SOUTH",
    "ARR_EAST",
    "ARR_WEST",
]