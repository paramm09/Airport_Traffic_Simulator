"""Discrete taxi movement of an aircraft along its Dijkstra-generated route.

This module answers *"how far has the aircraft travelled along its route?"*
and is deliberately separate from Dijkstra, which answers *"what is the
shortest route?"*. The relationship is:

    Dijkstra  ->  aircraft.taxi_route  ->  taxi_step()  ->  route_index /
                  distance_on_edge  ->  WAITING_FOR_RUNWAY

``taxi_step`` advances an aircraft by ``TAXI_SPEED * dt`` distance units each
call, edge by edge, using only the public Graph API. It is a deterministic
iteration, not a DAA algorithm.
"""

from __future__ import annotations

import enum

from backend.models.aircraft import Aircraft, AircraftState
from backend.models.graph import Graph

# One distance unit per simulation tick. Intentionally simple; no metres and
# no acceleration yet.
TAXI_SPEED = 1.0


class TaxiStepOutcome(enum.Enum):
    """Result of one :func:`taxi_step` call."""

    MOVED = enum.auto()
    """The aircraft advanced. It may still be on an edge or at a node."""

    ROUTE_COMPLETE = enum.auto()
    """The final route node was reached and the state became
    ``WAITING_FOR_RUNWAY`` (departure) or ``COMPLETED`` (arrival)."""

    BLOCKED = enum.auto()
    """The aircraft ended the call unable to enter its next edge (blocked). It
    may have finished a previous edge earlier in the same call; rerouting is a
    later phase."""


# Which lifecycle state a completed route leads to, per taxi direction.
_ROUTE_COMPLETE_STATE: dict[AircraftState, AircraftState] = {
    AircraftState.TAXIING_TO_RUNWAY: AircraftState.WAITING_FOR_RUNWAY,
    AircraftState.TAXIING_TO_GATE: AircraftState.COMPLETED,
}


def taxi_step(
    aircraft: Aircraft, graph: Graph, dt: float = 1.0
) -> TaxiStepOutcome:
    """Advance ``aircraft`` by ``TAXI_SPEED * dt`` distance units along its route.

    Args:
        aircraft: The moving aircraft. It must be taxiing (``TAXIING_TO_RUNWAY``
            or ``TAXIING_TO_GATE``) and have a non-empty ``taxi_route``.
        graph: The airport graph, used only through its public API.
        dt: Time step in simulation ticks. Must be positive; movement is
            ``TAXI_SPEED * dt`` distance units.

    Returns:
        :attr:`~TaxiStepOutcome.MOVED` if the aircraft advanced,
        :attr:`~TaxiStepOutcome.ROUTE_COMPLETE` when the final route node is
        reached, or :attr:`~TaxiStepOutcome.BLOCKED` if the aircraft cannot
        enter its next edge because that edge is blocked.

    Raises:
        ValueError: if ``dt`` is not positive, the aircraft is not taxiing, or
            the aircraft has no taxi route.

    Movement rules (Phase 2B):
        - ``current node = taxi_route[route_index]``; the next node is
          ``taxi_route[route_index + 1]``.
        - A new edge is entered only if ``graph.has_edge`` reports it
          available; its weight is then stored in ``edge_length`` so that a
          mid-edge blockage cannot stop the aircraft from finishing it.
        - On arrival the route index increments outwards once and
          ``distance_on_edge`` resets to zero; leftover distance from a large
          ``dt`` carries into the next edge.
        - Completing the route ends the taxi: a departure becomes
          ``WAITING_FOR_RUNWAY`` and an arrival becomes ``COMPLETED``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt!r}")
    if aircraft.state not in _ROUTE_COMPLETE_STATE:
        raise ValueError(
            f"aircraft {aircraft.id!r} is in {aircraft.state.name}, expected "
            "TAXIING_TO_RUNWAY or TAXIING_TO_GATE to taxi"
        )
    if not aircraft.taxi_route:
        raise ValueError(f"aircraft {aircraft.id!r} has no taxi route")

    complete_state = _ROUTE_COMPLETE_STATE[aircraft.state]

    # Already at the final node: the journey is complete without moving.
    if aircraft.route_index >= len(aircraft.taxi_route) - 1:
        aircraft.transition_to(complete_state)
        return TaxiStepOutcome.ROUTE_COMPLETE

    remaining = TAXI_SPEED * dt

    while remaining > 0.0:
        current = aircraft.taxi_route[aircraft.route_index]
        next_node = aircraft.taxi_route[aircraft.route_index + 1]

        if aircraft.distance_on_edge == 0.0:
            # Not yet on this edge: it must be available, and its length is
            # recorded so a mid-edge blockage cannot interrupt the crossing.
            if not graph.has_edge(current, next_node):
                return TaxiStepOutcome.BLOCKED
            aircraft.edge_length = graph.get_edge_weight(current, next_node)

        remaining_on_edge = aircraft.edge_length - aircraft.distance_on_edge

        if remaining < remaining_on_edge:
            aircraft.distance_on_edge += remaining
            remaining = 0.0
        else:
            # Complete this edge, arrive at its far node, and put any leftover
            # movement toward the following edge.
            aircraft.distance_on_edge += remaining_on_edge
            remaining -= remaining_on_edge
            aircraft.route_index += 1
            aircraft.distance_on_edge = 0.0
            aircraft.edge_length = 0.0

            if aircraft.route_index == len(aircraft.taxi_route) - 1:
                aircraft.transition_to(complete_state)
                return TaxiStepOutcome.ROUTE_COMPLETE

    return TaxiStepOutcome.MOVED