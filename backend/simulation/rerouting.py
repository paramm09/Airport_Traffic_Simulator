"""Phase 2G: dynamic rerouting around a blocked taxi edge.

Phase 2B plans a taxi route once with Dijkstra and Phase 2C walks it with
:func:`~backend.simulation.taxi.taxi_step`. The weakness of "plan once" is that
the airport network can change *after* the plan: an edge can be blocked by
another aircraft, maintenance, or an incident. This module reacts to that by
re-planning **from where the aircraft actually is**, reusing the existing
Dijkstra implementation. No new shortest-path algorithm is introduced.

The rules that keep the reroute physically sensible
---------------------------------------------------
1. **Never interrupt a crossing.** An aircraft that is already part-way along
   an edge (``distance_on_edge > 0``) is allowed to finish that edge. A
   mid-edge reversal or teleport would be a modelling error, not a shortcut,
   so the current edge is always completed first.
2. **Reroute from the node, not from the middle.** The re-plan starts at
   ``taxi_route[route_index]`` — the node the aircraft has reached or is
   heading to — and ends at the *unchanged* ``destination``. The destination is
   never rewritten to make a path exist.
3. **Reset the progress fields.** A successful re-plan rewrites ``taxi_route``
   and resets ``route_index = 0``, ``distance_on_edge = 0.0`` and
   ``edge_length = 0.0``, because the new path starts at the current node and
   no partial progress carries over.
4. **No path is a real outcome.** If the destination is genuinely unreachable,
   a taxiing aircraft enters :attr:`AircraftState.HOLDING_NO_ROUTE` instead of
   waiting forever with a stale route. This applies to *both* taxi directions: a
   departure that cannot reach a runway and an arrival that cannot reach its
   gate are in the same position, so they are given the same state rather than
   one of them being left to sit in an ordinary taxiing state that looks, in the
   event log, exactly like it is still trying. Recovering is *event driven*: a
   blocked aircraft is not retried on every tick (that would run Dijkstra
   repeatedly for a situation that has not changed). The caller retries when
   something has actually changed, typically after
   :meth:`~backend.models.graph.Graph.unblock_edge`.

Which state to resume into
--------------------------
``HOLDING_NO_ROUTE`` is shared by both taxi directions, so on its own it does not
say whether the aircraft was heading for a runway or for a gate. The state
machine records that on the way in, in
:attr:`~backend.models.aircraft.Aircraft.holding_from`, and :func:`resume_state`
reads it back out. Keeping the record inside
:meth:`~backend.models.aircraft.Aircraft.transition_to` is deliberate: it is
derived from ``state`` at the moment of the transition and cleared on the way
out, so it can never drift out of step with the state machine and is not a
second source of truth.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from backend.algorithms.dijkstra import dijkstra
from backend.models.aircraft import Aircraft, AircraftState
from backend.models.graph import Graph

# The lifecycle states an aircraft can be in while it still needs a taxi route.
# This is also exactly the set of states an aircraft may be resumed into from
# HOLDING_NO_ROUTE, so it stays in step with the transition table in
# backend/models/aircraft.py.
TAXIING_STATES: frozenset[AircraftState] = frozenset(
    {
        AircraftState.TAXIING_TO_RUNWAY,
        AircraftState.TAXIING_TO_GATE,
    }
)


class RerouteResult(enum.Enum):
    """What one reroute attempt did."""

    NOT_NEEDED = enum.auto()
    """The aircraft is not blocked, or is not taxiing: nothing to do."""

    REROUTED = enum.auto()
    """A new route was found and installed."""

    NO_PATH = enum.auto()
    """No route to the destination exists, so the aircraft is holding."""


@dataclass(frozen=True)
class RerouteAttempt:
    """A description of one reroute attempt, for logging and for tests."""

    result: RerouteResult
    aircraft_id: str
    start: str | None = None
    destination: str | None = None
    old_route: tuple[str, ...] = ()
    new_route: tuple[str, ...] = ()
    cost: float = math.inf

    @property
    def rerouted(self) -> bool:
        """``True`` when a new route was actually installed."""
        return self.result is RerouteResult.REROUTED


def resume_state(aircraft: Aircraft) -> AircraftState | None:
    """Return the state an aircraft should hold in while it has no route.

    A taxiing aircraft waiting in ``HOLDING_NO_ROUTE`` must go back to whichever
    taxi state it came from, so the value is read from
    :attr:`~backend.models.aircraft.Aircraft.holding_from` rather than guessed.
    That is the only thing that distinguishes a departure resuming towards a
    runway from an arrival resuming towards a gate.

    Returns ``None`` for an aircraft that is not in a "waiting for a route"
    state at all.

    ``holding_from`` is guaranteed to be set whenever the state is
    ``HOLDING_NO_ROUTE`` -- :class:`Aircraft` enforces that on construction and
    :meth:`~backend.models.aircraft.Aircraft.transition_to` records it on the way
    in -- so no fallback is needed here.
    """
    if aircraft.state is AircraftState.HOLDING_NO_ROUTE:
        return aircraft.holding_from
    if aircraft.state in TAXIING_STATES:
        return aircraft.state
    return None


def current_node(aircraft: Aircraft) -> str | None:
    """Return the node a re-plan must start from, or ``None``.

    That is ``taxi_route[route_index]``: the node the aircraft is standing at,
    or the last one it reached. A mid-edge aircraft (``distance_on_edge > 0``)
    has already left this node, which is why rule 1 forbids interrupting it.
    """
    if not aircraft.taxi_route:
        return None
    if aircraft.route_index >= len(aircraft.taxi_route):
        return None
    return aircraft.taxi_route[aircraft.route_index]


def is_blocked(aircraft: Aircraft, graph: Graph) -> bool:
    """Return ``True`` if the aircraft is stopped at a node by a blocked edge.

    Mirrors exactly the check in :func:`~backend.simulation.taxi.taxi_step`: the
    aircraft is at a node (``distance_on_edge == 0``) and the edge to the next
    route node is not available. A mid-edge aircraft is never reported as
    blocked, so it is left alone until it finishes the crossing.
    """
    if aircraft.state not in TAXIING_STATES:
        return False
    if aircraft.distance_on_edge > 0.0:
        # Mid-edge: let it complete the crossing it already committed to.
        return False
    index = aircraft.route_index
    if index + 1 >= len(aircraft.taxi_route):
        return False
    return not graph.has_edge(
        aircraft.taxi_route[index], aircraft.taxi_route[index + 1]
    )


def _install_route(aircraft: Aircraft, route: list[str]) -> None:
    """Install ``route`` and clear all progress that belonged to the old one."""
    aircraft.taxi_route = list(route)
    aircraft.route_index = 0
    aircraft.distance_on_edge = 0.0
    aircraft.edge_length = 0.0


def try_reroute(aircraft: Aircraft, graph: Graph) -> RerouteAttempt:
    """Re-plan ``aircraft``'s route from its current node using Dijkstra.

    Only called for an aircraft that is actually blocked, so this function does
    not re-check the blockage. The destination is never changed.

    Returns a :class:`RerouteAttempt` describing what happened:

        * :attr:`RerouteResult.REROUTED` when a path was found — the aircraft's
          ``taxi_route`` is replaced and the progress fields are zeroed;
        * :attr:`RerouteResult.NO_PATH` when the destination is unreachable, in
          which case any taxiing aircraft — departure or arrival — is moved to
          ``HOLDING_NO_ROUTE`` and waits to be retried.

    Raises:
        ValueError: if ``aircraft`` has no ``destination`` to head for.
    """
    if aircraft.destination is None:
        raise ValueError(
            f"aircraft {aircraft.id!r} has no destination to reroute to"
        )

    start = current_node(aircraft)
    destination = aircraft.destination
    old_route = tuple(aircraft.taxi_route)

    if start is None:
        # Nothing to plan from: fall back to the destination itself, which
        # makes Dijkstra report whether the destination is even in the network.
        start = destination

    path, cost = dijkstra(graph, start, destination)
    if not path:
        if aircraft.state in TAXIING_STATES:
            # Any taxiing aircraft with no route holds, departure or arrival.
            # The state machine records which taxi state to resume into.
            aircraft.transition_to(AircraftState.HOLDING_NO_ROUTE)
        return RerouteAttempt(
            result=RerouteResult.NO_PATH,
            aircraft_id=aircraft.id,
            start=start,
            destination=destination,
            old_route=old_route,
        )

    _install_route(aircraft, path)
    return RerouteAttempt(
        result=RerouteResult.REROUTED,
        aircraft_id=aircraft.id,
        start=start,
        destination=destination,
        old_route=old_route,
        new_route=tuple(path),
        cost=cost,
    )


def reroute_if_blocked(aircraft: Aircraft, graph: Graph) -> RerouteAttempt:
    """Re-plan ``aircraft``'s route only if it is currently blocked.

    This is the entry point for a simulation tick: it is cheap when nothing is
    blocked (one state check, one edge lookup) and only runs Dijkstra when a
    blocked edge is actually in the way.

    A ``HOLDING_NO_ROUTE`` aircraft is *not* retried here. It is already
    holding by design; retrying every tick would re-run Dijkstra continuously
    for a situation that has not changed. Call :func:`try_reroute` on it when
    the network has actually changed.
    """
    if aircraft.state not in TAXIING_STATES:
        return RerouteAttempt(
            result=RerouteResult.NOT_NEEDED, aircraft_id=aircraft.id
        )
    if not is_blocked(aircraft, graph):
        return RerouteAttempt(
            result=RerouteResult.NOT_NEEDED, aircraft_id=aircraft.id
        )
    return try_reroute(aircraft, graph)
