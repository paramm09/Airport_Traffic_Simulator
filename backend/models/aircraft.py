"""Aircraft data model and lifecycle state machine.

This module models ONE aircraft: its identity, its current lifecycle state,
and the fields that the later phases will drive (taxi route, runway request,
holding, airspace position).

It deliberately contains **no algorithm**. The enum and the transition table
are a deterministic *supporting structure* for the simulation, not a DAA
algorithm. The lifecycle is a strictly ordered chain of states with a terminal
state, so an explicit table is the simplest correct representation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from backend.models.waypoint import Waypoint


class AircraftState(enum.Enum):
    """The legal lifecycle states, in the order an aircraft moves through them.

    Each state is defined as ``enum.auto()`` so the definition order matches the
    lifecycle order — useful for tests and for walking the full chain.
    """

    AT_GATE = enum.auto()
    TAXIING_TO_RUNWAY = enum.auto()
    WAITING_FOR_RUNWAY = enum.auto()
    LINE_UP = enum.auto()
    TAKEOFF = enum.auto()
    CLIMB = enum.auto()
    CRUISE = enum.auto()
    DESCENT = enum.auto()
    APPROACH = enum.auto()
    LANDING = enum.auto()
    TAXIING_TO_GATE = enum.auto()
    COMPLETED = enum.auto()

    # A side state, deliberately declared last because it is NOT part of the
    # straight lifecycle chain. It is entered when a taxiing aircraft -- a
    # departure or an arrival -- has no route to its destination at all
    # (Phase 2G), and left again once a route exists. Putting it last keeps
    # walking the enum in definition order equal to walking the lifecycle chain.
    HOLDING_NO_ROUTE = enum.auto()



class IllegalTransitionError(ValueError):
    """Raised when an aircraft is asked to make an illegal state transition."""


# Explicit, visible transition table (viva-friendly).
# A state not present as a key (COMPLETED) is terminal: no outgoing transitions.
_TRANSITIONS: dict[AircraftState, tuple[AircraftState, ...]] = {
    AircraftState.AT_GATE: (AircraftState.TAXIING_TO_RUNWAY,),
    AircraftState.TAXIING_TO_RUNWAY: (
        AircraftState.WAITING_FOR_RUNWAY,
        AircraftState.HOLDING_NO_ROUTE,
    ),
    AircraftState.WAITING_FOR_RUNWAY: (AircraftState.LINE_UP,),
    AircraftState.LINE_UP: (AircraftState.TAKEOFF,),
    AircraftState.TAKEOFF: (AircraftState.CLIMB,),
    AircraftState.CLIMB: (AircraftState.CRUISE,),
    AircraftState.CRUISE: (AircraftState.DESCENT,),
    AircraftState.DESCENT: (AircraftState.APPROACH,),
    AircraftState.APPROACH: (AircraftState.LANDING,),
    AircraftState.LANDING: (AircraftState.TAXIING_TO_GATE,),
    AircraftState.TAXIING_TO_GATE: (
        AircraftState.COMPLETED,
        AircraftState.HOLDING_NO_ROUTE,
    ),
    # The two exits from the rerouting side state. Which one applies depends on
    # whether the aircraft was heading for a runway or for a gate, which is
    # recorded in ``Aircraft.holding_from`` on the way in.
    AircraftState.HOLDING_NO_ROUTE: (
        AircraftState.TAXIING_TO_RUNWAY,
        AircraftState.TAXIING_TO_GATE,
    ),
}

# The states an aircraft may legally be resumed into from HOLDING_NO_ROUTE, and
# therefore the only values ``holding_from`` is allowed to take.
_HOLDING_ENTRY_STATES: frozenset[AircraftState] = frozenset(
    {
        AircraftState.TAXIING_TO_RUNWAY,
        AircraftState.TAXIING_TO_GATE,
    }
)


@dataclass
class Aircraft:
    """A single aircraft inside the simulation.

    Fields:
        id: Unique identifier, e.g. ``"AC001"``. Must be non-empty.
        state: Current lifecycle state.
        taxi_route: Planned list of taxiway/gate/runway nodes.
        route_index: Index of the node the aircraft is currently heading to.
        destination: Final node the aircraft is taxiing to (e.g. a gate).
        distance_on_edge: How far the aircraft has travelled along its current
            edge. Zero means it is at a route node and not yet on an edge.
        edge_length: Distance of the edge currently being traversed, recorded
            when the aircraft enters the edge so a later mid-edge blockage
            cannot prevent finishing it.
        request_time: Tick at which the aircraft requested the runway.
        holding_until: Tick before which the aircraft may not move; ``None`` if
            not holding.
        holding_from: The taxi state this aircraft held immediately before it
            entered ``HOLDING_NO_ROUTE`` -- ``TAXIING_TO_RUNWAY`` for a
            departure, ``TAXIING_TO_GATE`` for an arrival. It is ``None``
            whenever the aircraft is not in ``HOLDING_NO_ROUTE``.
        position: Continuous airspace position.
        altitude: Altitude, must be non-negative.
        speed: Speed, must be non-negative.
        heading: Heading in degrees.
        next_waypoint: Next continuous waypoint, if in the airspace.
        air_route: Ordered list of airspace waypoints the aircraft flies toward
            while airborne.
        air_route_index: Index of the waypoint the aircraft is currently
            heading to.

    Only simple, useful invariants are enforced: identity and the fields the
    later phases verify numerically. Realistic aviation properties are out of
    scope by design.
    """

    id: str
    state: AircraftState = AircraftState.AT_GATE

    # Taxi-related fields.
    taxi_route: list[str] = field(default_factory=list)
    route_index: int = 0
    destination: str | None = None
    distance_on_edge: float = 0.0
    edge_length: float = 0.0

    # Runway scheduling field.
    request_time: int = 0

    # Holding fields.
    holding_until: int | None = None
    holding_from: AircraftState | None = None

    # Airspace fields.
    position: tuple[float, float] = (0.0, 0.0)
    altitude: float = 0.0
    speed: float = 0.0
    heading: float = 0.0
    next_waypoint: tuple[float, float] | None = None
    air_route: list[Waypoint] = field(default_factory=list)
    air_route_index: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("aircraft id must not be empty")
        if not isinstance(self.state, AircraftState):
            raise ValueError(
                f"state must be an AircraftState, got {self.state!r}"
            )
        if self.speed < 0:
            raise ValueError(f"speed must not be negative, got {self.speed!r}")
        if self.altitude < 0:
            raise ValueError(
                f"altitude must not be negative, got {self.altitude!r}"
            )
        if self.route_index < 0:
            raise ValueError(
                f"route_index must not be negative, got {self.route_index!r}"
            )
        if self.distance_on_edge < 0:
            raise ValueError(
                f"distance_on_edge must not be negative, got {self.distance_on_edge!r}"
            )
        if self.edge_length < 0:
            raise ValueError(
                f"edge_length must not be negative, got {self.edge_length!r}"
            )
        if self.request_time < 0:
            raise ValueError(
                f"request_time must not be negative, got {self.request_time!r}"
            )
        if self.holding_until is not None and not isinstance(
            self.holding_until, int
        ):
            raise ValueError(
                "holding_until must be None or an integer tick, "
                f"got {self.holding_until!r}"
            )
        if self.air_route_index < 0:
            raise ValueError(
                f"air_route_index must not be negative, got {self.air_route_index!r}"
            )
        if self.holding_from is not None and not isinstance(
            self.holding_from, AircraftState
        ):
            raise ValueError(
                "holding_from must be None or an AircraftState, "
                f"got {self.holding_from!r}"
            )
        # holding_from is a continuation record, not a second source of truth:
        # it is meaningful only while the aircraft is actually holding, and it
        # only ever names a taxi state. Enforcing that here means the field can
        # never drift away from `state`.
        if self.state is AircraftState.HOLDING_NO_ROUTE:
            if self.holding_from not in _HOLDING_ENTRY_STATES:
                raise ValueError(
                    "an aircraft in HOLDING_NO_ROUTE must record holding_from "
                    "as TAXIING_TO_RUNWAY or TAXIING_TO_GATE, "
                    f"got {self.holding_from!r}"
                )
        elif self.holding_from is not None:
            raise ValueError(
                "holding_from must be None unless the state is "
                f"HOLDING_NO_ROUTE, got state {self.state.name} with "
                f"holding_from {self.holding_from.name}"
            )
        for waypoint in self.air_route:
            if not isinstance(waypoint, Waypoint):
                raise ValueError(
                    "air_route entries must be Waypoint instances, "
                    f"got {waypoint!r}"
                )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def can_transition_to(self, next_state: AircraftState) -> bool:
        """Return ``True`` if a direct transition from the current state to
        ``next_state`` is legal.
        """
        return next_state in _TRANSITIONS.get(self.state, ())

    def transition_to(self, next_state: AircraftState) -> None:
        """Move the aircraft to ``next_state`` if that transition is legal.

        Entering ``HOLDING_NO_ROUTE`` records the taxi state being left in
        :attr:`holding_from`, and leaving it clears that record. Doing this here,
        in the one method that changes ``state``, is what keeps the record
        consistent with the state machine by construction rather than by
        convention.

        Raises:
            IllegalTransitionError: if there is no legal transition from the
                current state to ``next_state``. The aircraft's state is
                left unchanged.
        """
        if not self.can_transition_to(next_state):
            raise IllegalTransitionError(
                f"{self.state.name} -> {next_state.name} is not a legal "
                "transition"
            )
        if next_state is AircraftState.HOLDING_NO_ROUTE:
            self.holding_from = self.state
        else:
            self.holding_from = None
        self.state = next_state
