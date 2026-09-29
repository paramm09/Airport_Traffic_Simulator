"""Phase 2H: the fixed-timestep simulation that drives every earlier phase.

Phases 2A-2G built correct pieces; none of them owns time. This module is the
orchestrator that advances a whole airport by exactly one tick at a time and
wires the pieces together in a fixed, documented order.

Determinism
-----------
There is no wall clock, no ``random``, no set iteration and no thread here. The
tick counter is an ``int`` that only ever increases by one, and the aircraft
are held in a list in insertion order, so the same starting configuration always
produces exactly the same event log. That is what makes the simulation testable
and reproducible.

The order of one tick
---------------------
The order is part of the specification, not an implementation detail, because
each stage consumes what the previous one produced::

    1. scheduled graph changes   block/unblock edges due at this tick
    2. taxi movement              ground movement, with a reroute if blocked
    3. runway requests            submit each aircraft's request exactly once
    4. runway scheduling          let the heap give the runway to the next one
    5. airborne movement          move the air, unless the aircraft is holding
    6. conflict detection         analytical CPA over all airborne pairs
    7. conflict resolution        issue HOLD instructions, most urgent first
    8. event recording            append to the event log
    9. advance the tick

Two consequences are worth stating explicitly:

* **Holds gate only airborne movement (stage 5).** A HOLD stops an aircraft in
  the air, so it must not stop it taxiing to the runway; that would quietly
  change the answer of the runway scheduler as a side effect of an air-traffic
  conflict.
* **Detection runs after movement (stages 6-7, not before 5).** The prediction
  is therefore made about the positions the aircraft will hold *after* this
  tick's movement, which is the honest thing to detect on.

Who owns what
-------------
This module deliberately owns only the *timing and the wiring*:

============================  ==========================================
:mod:`~backend.simulation.taxi`            how far an aircraft taxis
:mod:`~backend.simulation.runway_scheduler` which aircraft gets the runway
:mod:`~backend.simulation.airspace`        how far an aircraft flies
:mod:`~backend.simulation.conflict`        whether two aircraft conflict
:mod:`~backend.simulation.conflict_resolution` who yields
:mod:`~backend.simulation.rerouting`      what to do about a blocked taxiway
:mod:`~backend.simulation.simulation`      the tick, the order, the event log
============================  ==========================================

The scheduler keeps ownership of the runway queue and its priority policy; the
simulation only submits requests and never picks a winner itself. Likewise the
conflict detector stays a pure function and the resolver stays the only thing
that writes ``holding_until``.

Air routes and lifecycle progress
---------------------------------
:func:`backend.simulation.airspace.airspace_step` needs a non-empty air route
and, when a route is completed, advances the aircraft exactly one step along
``TAKEOFF -> CLIMB -> CRUISE -> DESCENT -> APPROACH -> LANDING``. Supplying the
next leg is therefore a job of whoever owns the flight plan, which is this
class: :attr:`Simulation.leg_provider` turns the current lifecycle state into
the next list of waypoints. The default plan is the deterministic pattern in
:mod:`backend.data.airspace_waypoints`, and tests can replace it to create
conflicts on demand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from backend.data.airspace_waypoints import (
    ARR_EAST,
    ARR_WEST,
    CRUISE_NORTH,
    CRUISE_SOUTH,
    DEP_EAST,
)
from backend.models.aircraft import Aircraft, AircraftState
from backend.models.graph import Graph
from backend.models.waypoint import Waypoint
from backend.simulation.airspace import (
    AIRBORNE_STATES,
    AirspaceStepOutcome,
    airspace_step,
)
from backend.simulation.conflict import (
    HORIZON,
    HORIZONTAL_SEPARATION,
    VERTICAL_SEPARATION,
    detect_all_conflicts,
)
from backend.simulation.conflict_resolution import (
    ConflictResolver,
    ResolutionAction,
    is_holding,
)
from backend.simulation.rerouting import (
    RerouteResult,
    reroute_if_blocked,
    resume_state,
    try_reroute,
)
from backend.simulation.runway_scheduler import (
    RunwayOperation,
    RunwayScheduler,
)
from backend.simulation.taxi import TaxiStepOutcome, taxi_step

# The lifecycle state an aircraft enters when it leaves the runway, per the
# operation it was given the runway for.
_DEPARTURE_RUNWAY_EXIT: dict[RunwayOperation, AircraftState] = {
    RunwayOperation.TAKEOFF: AircraftState.TAKEOFF,
    RunwayOperation.LANDING: AircraftState.TAXIING_TO_GATE,
}

# Default number of ticks :meth:`Simulation.run_until_complete` will run before
# giving up, so a mis-built scenario cannot hang a test suite forever.
DEFAULT_MAX_TICKS = 5_000

# The only lifecycle states that mean "this aircraft will not move again".
# Everything else -- including an airborne aircraft with an empty air route --
# means the simulation is still running. See :meth:`Simulation.is_complete`.
_TERMINAL_STATES: frozenset[AircraftState] = frozenset(
    {AircraftState.COMPLETED, AircraftState.HOLDING_NO_ROUTE}
)


@dataclass(frozen=True)
class SimulationEvent:
    """One entry in the simulation's event log.

    Attributes:
        tick: The tick the event happened on.
        kind: A short machine-readable tag, e.g. ``"HOLD"`` or ``"REROUTE"``.
        message: A human-readable description.
        aircraft_id: The aircraft the event is about, if any.
    """

    tick: int
    kind: str
    message: str
    aircraft_id: str | None = None


@dataclass(frozen=True)
class ScheduledChange:
    """A graph modification that becomes due at a fixed tick.

    Used to model incidents deterministically — a taxiway closing at tick 12
    and reopening at tick 40 — without any randomness or wall-clock time.
    """

    tick: int
    action: str
    node1: str
    node2: str


class LegProvider:
    """Default flight plan: a fixed, deterministic leg per lifecycle state.

    Each airborne state gets its own leg, so when
    :func:`~backend.simulation.airspace.airspace_step` completes a route and
    advances the lifecycle one step, the simulation simply hands over the next
    leg. The pattern flies a departure out to the east, north to cruise, on to
    a second cruise point and then back west for the arrival, which is enough
    for two aircraft on the same plan to be brought into conflict on purpose.
    """

    def __init__(self) -> None:
        self._legs: dict[AircraftState, list[Waypoint]] = {
            AircraftState.TAKEOFF: [DEP_EAST],
            AircraftState.CLIMB: [CRUISE_NORTH],
            AircraftState.CRUISE: [CRUISE_SOUTH],
            AircraftState.DESCENT: [ARR_EAST],
            AircraftState.APPROACH: [ARR_WEST],
        }

    def leg_for(
        self, aircraft: Aircraft, state: AircraftState
    ) -> list[Waypoint]:
        """Return a fresh copy of the waypoints for ``state``.

        A copy is returned so that an aircraft advancing through its air route
        can never mutate the shared plan for every other aircraft.
        """
        return list(self._legs.get(state, []))


def default_leg_provider() -> LegProvider:
    """Return the default :class:`LegProvider`."""
    return LegProvider()


def build_departure(
    aircraft_id: str,
    gate: str,
    runway_access: str,
    position: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: float = 10.0,
    altitude: float = 0.0,
) -> Aircraft:
    """Create a departure sitting at ``gate``, heading for ``runway_access``.

    The aircraft starts at ``AT_GATE`` standing on ``gate``, and its taxi route
    is seeded with that single node so the simulation knows where the re-plan
    starts from. Handing it to :class:`Simulation` then gives it a Dijkstra
    route on construction and a full departure through to landing and back to a
    gate.
    """
    return Aircraft(
        id=aircraft_id,
        state=AircraftState.AT_GATE,
        taxi_route=[gate],
        destination=runway_access,
        position=position,
        heading=heading,
        speed=speed,
        altitude=altitude,
    )


def build_arrival(
    aircraft_id: str,
    gate: str,
    runway_access: str,
    position: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: float = 10.0,
    altitude: float = 2000.0,
) -> Aircraft:
    """Create an arrival that is already in ``LANDING`` and taxiing to ``gate``.

    An arrival starts on the runway side of the lifecycle: the simulation gives
    it an air route for the landing leg, queues a ``LANDING`` request for the
    shared runway, and then taxis it from ``runway_access`` to ``gate`` once the
    runway is released.
    """
    return Aircraft(
        id=aircraft_id,
        state=AircraftState.LANDING,
        taxi_route=[runway_access],
        destination=gate,
        position=position,
        heading=heading,
        speed=speed,
        altitude=altitude,
    )


class Simulation:
    """A deterministic, fixed-timestep airport simulation."""

    def __init__(
        self,
        aircraft: Iterable[Aircraft] | None = None,
        graph: Graph | None = None,
        scheduler: RunwayScheduler | None = None,
        resolver: ConflictResolver | None = None,
        leg_provider: Callable[[Aircraft, AircraftState], list[Waypoint]]
        | None = None,
        scheduled_changes: Sequence[ScheduledChange] = (),
        dt: float = 1.0,
        horizon: float = HORIZON,
        horizontal_separation: float = HORIZONTAL_SEPARATION,
        vertical_separation: float = VERTICAL_SEPARATION,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        self.aircraft: list[Aircraft] = list(aircraft or ())
        self._check_unique_ids()
        self.graph = graph if graph is not None else Graph()
        self.scheduler = scheduler if scheduler is not None else RunwayScheduler()
        self.resolver = resolver if resolver is not None else ConflictResolver()
        self.leg_provider = (
            leg_provider
            if leg_provider is not None
            else default_leg_provider().leg_for
        )
        self._scheduled = sorted(
            scheduled_changes, key=lambda change: change.tick
        )
        self._dt = dt
        self._horizon = horizon
        self._horizontal_separation = horizontal_separation
        self._vertical_separation = vertical_separation

        self._current_tick = 0
        self._events: list[SimulationEvent] = []
        # Per-aircraft bookkeeping. These are the simulation's own facts about
        # what it has already done, so that each one happens exactly once.
        # A request is tracked per (aircraft, operation), NOT per aircraft: a
        # departure legitimately uses the shared runway twice, once to take off
        # and once to land.
        self._requested: set[tuple[str, RunwayOperation]] = set()
        self._operation: dict[str, RunwayOperation] = {}
        self._runway_occupant: str | None = None
        self._departures: set[str] = set()

        # Every aircraft starts on the ground at its gate.
        for aircraft_object in self.aircraft:
            self._departures.add(aircraft_object.id)
            if aircraft_object.state is AircraftState.AT_GATE:
                aircraft_object.transition_to(AircraftState.TAXIING_TO_RUNWAY)
                self._assign_taxi_route(aircraft_object)

    def _check_unique_ids(self) -> None:
        """Reject a scenario in which two aircraft share an ID.

        The ID is the key used throughout the orchestration layer: the runway
        scheduler's heap, the ``_requested`` set that stops a runway request
        being sent twice, the ``_operation`` map, the ``_runway_occupant``
        handle and every event-log entry. Two aircraft with one ID would make
        all of those silently refer to the wrong aircraft, so the mistake is
        caught at construction rather than being diagnosed much later as
        baffling scheduling behaviour.

        Raises:
            ValueError: naming every ID that appears more than once.
        """
        seen: set[str] = set()
        duplicates: list[str] = []
        for aircraft_object in self.aircraft:
            if aircraft_object.id in seen and aircraft_object.id not in duplicates:
                duplicates.append(aircraft_object.id)
            seen.add(aircraft_object.id)
        if duplicates:
            raise ValueError(
                "Duplicate aircraft ID: " + ", ".join(duplicates)
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def current_tick(self) -> int:
        """The tick the next :meth:`step` will act on."""
        return self._current_tick

    @property
    def events(self) -> list[SimulationEvent]:
        """The event log so far, in order. A copy is returned."""
        return list(self._events)

    def find(self, aircraft_id: str) -> Aircraft:
        """Return the aircraft with ``aircraft_id``.

        Raises:
            KeyError: if no such aircraft is in the simulation.
        """
        for aircraft_object in self.aircraft:
            if aircraft_object.id == aircraft_id:
                return aircraft_object
        raise KeyError(f"Unknown aircraft: {aircraft_id!r}")

    def events_of_kind(self, kind: str) -> list[SimulationEvent]:
        """Return every logged event whose ``kind`` equals ``kind``."""
        return [event for event in self._events if event.kind == kind]

    def is_complete(self) -> bool:
        """Return ``True`` when the scenario has stopped and nothing can advance.

        Exactly two lifecycle states are terminal, and both mean "this aircraft
        will not move again on its own":

        * ``COMPLETED`` -- the flight finished its journey; and
        * ``HOLDING_NO_ROUTE`` — a taxiing aircraft stopped by an incident,
          which by Phase 2G design only resumes when the network changes.

        Every other state means the simulation is **not** complete, including
        an airborne aircraft whose ``air_route`` is empty. Such an aircraft is
        mid-lifecycle: it still owns a lifecycle state to advance through and a
        runway request to make, so treating an empty route as "finished" would
        end a scenario while aircraft were still in the air. Progress is
        measured by state, never by "has nothing left to do right now".

        A scenario that can never finish cannot hang the caller: the
        ``max_ticks`` bound on :meth:`run_until_complete` stops it instead.
        """
        return all(
            aircraft_object.state in _TERMINAL_STATES
            for aircraft_object in self.aircraft
        )

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _assign_taxi_route(self, aircraft: Aircraft) -> None:
        """Give a taxiing aircraft a route to its current destination.

        The planning is delegated to :func:`~backend.simulation.rerouting.try_reroute`
        rather than re-implemented here, so the "what happens when there is no
        route" policy is decided in exactly one place. That is what makes
        construction time safe: if the aircraft's starting node is already cut
        off by a closed taxiway, Dijkstra returns no path and the Phase 2G
        planner moves the taxiing aircraft into ``HOLDING_NO_ROUTE`` to wait for
        the retry mechanism. Previously the stale one-node route was left in place,
        which ``taxi_step`` would report as "already arrived" -- so an aircraft
        whose gate access was shut could take off without ever reaching a
        runway.
        """
        if aircraft.destination is None:
            return
        attempt = try_reroute(aircraft, self.graph)
        if attempt.result is RerouteResult.REROUTED:
            return
        self._log_no_route(aircraft)

    def _log_no_route(self, aircraft: Aircraft) -> None:
        """Record that ``aircraft`` cannot currently reach its destination."""
        self._log(
            "NO_ROUTE",
            f"{aircraft.id} has no route to {aircraft.destination} and "
            "is holding",
            aircraft.id,
        )

    def _assign_air_route(self, aircraft: Aircraft) -> None:
        """Install the next air leg for ``aircraft``'s current state."""
        route = self.leg_provider(aircraft, aircraft.state)
        if route:
            aircraft.air_route = route
            aircraft.air_route_index = 0
            aircraft.next_waypoint = (route[0].x, route[0].y)

    # ------------------------------------------------------------------
    # One tick
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance the simulation by exactly one tick.

        The stages run in the fixed order documented in the module docstring.
        """
        tick = self._current_tick

        # 1. Scheduled graph changes come first, so everything below this tick
        #    sees the new network.
        self._apply_scheduled_changes(tick)

        # 2. Ground movement, with a reroute whenever a taxi edge is blocked.
        for aircraft_object in self.aircraft:
            self._taxi_phase(aircraft_object, tick)

        # 3. Runway requests: submitted once per aircraft, never re-sent.
        for aircraft_object in self.aircraft:
            self._request_phase(aircraft_object, tick)

        # 4. The scheduler owns the runway and picks the next occupant.
        self._runway_phase(tick)

        # 5. Airborne movement. A HOLD gates this stage and nothing else.
        for aircraft_object in self.aircraft:
            self._airspace_phase(aircraft_object, tick)

        # 6-7. Detect predicted conflicts, then issue HOLD instructions.
        self._conflict_phase(tick)

        # 8. Events are appended as the stages above run; nothing else.
        # 9. Advance the clock.
        self._current_tick += 1

    def run(self, ticks: int) -> None:
        """Run ``ticks`` ticks, one :meth:`step` each.

        Raises:
            ValueError: if ``ticks`` is negative.
        """
        if ticks < 0:
            raise ValueError(f"ticks must not be negative, got {ticks!r}")
        for _ in range(ticks):
            self.step()

    def run_until_complete(self, max_ticks: int = DEFAULT_MAX_TICKS) -> bool:
        """Run until every aircraft completes, or until ``max_ticks``.

        The bound is what keeps a scenario that cannot finish from hanging the
        caller: this is a teaching simulator, not a daemon.

        Returns:
            ``True`` if the simulation completed, ``False`` if it ran out of
            ticks first.
        """
        for _ in range(max_ticks):
            if self.is_complete():
                return True
            self.step()
        return self.is_complete()

    # ------------------------------------------------------------------
    # Tick stages
    # ------------------------------------------------------------------
    def _log(self, kind: str, message: str, aircraft_id: str | None = None
             ) -> None:
        self._events.append(
            SimulationEvent(
                tick=self._current_tick,
                kind=kind,
                message=message,
                aircraft_id=aircraft_id,
            )
        )

    def _apply_scheduled_changes(self, tick: int) -> None:
        """Apply every graph change whose tick has arrived."""
        while self._scheduled and self._scheduled[0].tick <= tick:
            change = self._scheduled.pop(0)
            if change.action == "block":
                self.graph.block_edge(change.node1, change.node2)
                self._log(
                    "BLOCK",
                    f"edge {change.node1}-{change.node2} closed",
                )
            elif change.action == "unblock":
                self.graph.unblock_edge(change.node1, change.node2)
                self._log(
                    "UNBLOCK",
                    f"edge {change.node1}-{change.node2} reopened",
                )
                # A reopened edge is the signal to retry aircraft that gave up.
                for aircraft_object in self.aircraft:
                    if (
                        aircraft_object.state
                        is AircraftState.HOLDING_NO_ROUTE
                    ):
                        self._retry_no_route(aircraft_object)
            else:
                raise ValueError(
                    f"unknown scheduled change: {change.action!r}"
                )

    def _retry_no_route(self, aircraft: Aircraft) -> None:
        """Re-plan a held aircraft now that the network has changed.

        The resumed state is whichever taxi state the aircraft was holding for,
        read from the state machine's own record: a departure resumes towards a
        runway, an arrival towards its gate.
        """
        attempt = try_reroute(aircraft, self.graph)
        if attempt.result is RerouteResult.REROUTED:
            resume = resume_state(aircraft)
            if resume is not None and aircraft.can_transition_to(resume):
                aircraft.transition_to(resume)
            self._log(
                "ROUTE_FOUND",
                f"{aircraft.id} found a route to "
                f"{aircraft.destination}: {' -> '.join(attempt.new_route)}",
                aircraft.id,
            )
        else:
            self._log(
                "STILL_NO_ROUTE",
                f"{aircraft.id} still has no route to {aircraft.destination}",
                aircraft.id,
            )

    def _taxi_phase(self, aircraft: Aircraft, tick: int) -> None:
        """Stage 2: reroute if blocked, then move on the ground."""
        if aircraft.state not in (
            AircraftState.TAXIING_TO_RUNWAY,
            AircraftState.TAXIING_TO_GATE,
        ):
            return

        attempt = reroute_if_blocked(aircraft, self.graph)
        if attempt.result is RerouteResult.REROUTED:
            self._log(
                "REROUTE",
                f"{aircraft.id} rerouted from {attempt.start} to "
                f"{attempt.destination}: {' -> '.join(attempt.new_route)} "
                f"(cost {attempt.cost})",
                aircraft.id,
            )
        elif attempt.result is RerouteResult.NO_PATH:
            self._log_no_route(aircraft)
            return

        # The two cases below are *state* conditions, not errors, so they are
        # handled explicitly here instead of by catching a ValueError. Any other
        # ValueError raised by taxi_step is a genuine bug and is left to
        # surface.
        if aircraft.destination is None or not aircraft.taxi_route:
            # Nothing to plan towards and nothing to walk: the aircraft is
            # stationary. (Phase 2G has no spare state for an arrival, so it
            # simply stays where it is.)
            self._log(
                "NO_ROUTE",
                f"{aircraft.id} has no destination and no taxi route, "
                "so it cannot move",
                aircraft.id,
            )
            return
        if aircraft.taxi_route[-1] != aircraft.destination:
            # The route in hand does not actually reach the destination -- a
            # stale route from a failed plan. It may even be a single node,
            # which taxi_step would report as "arrived"; refusing to walk it is
            # what stops an unreachable aircraft being declared finished.
            self._log_no_route(aircraft)
            return

        outcome = taxi_step(aircraft, self.graph, self._dt)

        if outcome is TaxiStepOutcome.ROUTE_COMPLETE:
            self._log(
                "TAXI_COMPLETE",
                f"{aircraft.id} finished taxiing and is now "
                f"{aircraft.state.name}",
                aircraft.id,
            )
        elif outcome is TaxiStepOutcome.BLOCKED:
            self._log(
                "BLOCKED",
                f"{aircraft.id} is blocked at {aircraft.taxi_route[aircraft.route_index]}",
                aircraft.id,
            )

    def _request_phase(self, aircraft: Aircraft, tick: int) -> None:
        """Stage 3: submit this aircraft's runway request, once per operation."""
        if aircraft.state is AircraftState.WAITING_FOR_RUNWAY:
            operation = RunwayOperation.TAKEOFF
        elif aircraft.state is AircraftState.LANDING:
            operation = RunwayOperation.LANDING
        else:
            return

        if (aircraft.id, operation) in self._requested:
            return

        self._requested.add((aircraft.id, operation))
        self._operation[aircraft.id] = operation
        self.scheduler.request(aircraft, operation, tick)
        self._log(
            "RUNWAY_REQUEST",
            f"{aircraft.id} requested the runway for {operation.name}",
            aircraft.id,
        )

    def _runway_phase(self, tick: int) -> None:
        """Stage 4: advance the scheduler and apply the runway outcome.

        The release has to be recognised *around* the scheduler call rather
        than after it: :meth:`RunwayScheduler.step` releases a finished
        occupant and then immediately hands the runway to the next one, so the
        runway is already busy again by the time it returns. The release tick
        is therefore captured first and compared against the current tick.
        """
        previous_occupant = self._runway_occupant
        release_tick = (
            self.scheduler.release_tick() if self.scheduler.is_busy() else None
        )
        selected = self.scheduler.step(tick)

        if (
            previous_occupant is not None
            and release_tick is not None
            and tick >= release_tick
        ):
            self._runway_occupant = None
            released_aircraft = self.find(previous_occupant)
            operation = self._operation.get(previous_occupant)
            exit_state = _DEPARTURE_RUNWAY_EXIT.get(operation)
            if exit_state is not None and released_aircraft.can_transition_to(
                exit_state
            ):
                released_aircraft.transition_to(exit_state)
                self._log(
                    "RUNWAY_RELEASED",
                    f"{previous_occupant} left the runway and is now "
                    f"{released_aircraft.state.name}",
                    previous_occupant,
                )
                if exit_state is AircraftState.TAXIING_TO_GATE:
                    self._assign_taxi_route(released_aircraft)
                else:
                    self._assign_air_route(released_aircraft)

        if selected is not None:
            self._runway_occupant = selected.id
            self._log(
                "RUNWAY_ASSIGNED",
                f"{selected.id} got the runway and is now "
                f"{selected.state.name} "
                f"(until tick {self.scheduler.release_tick()})",
                selected.id,
            )

    def _airspace_phase(self, aircraft: Aircraft, tick: int) -> None:
        """Stage 5: move the air, unless a HOLD is in force."""
        if aircraft.state not in AIRBORNE_STATES:
            return

        if is_holding(aircraft, tick):
            self._log(
                "HOLDING",
                f"{aircraft.id} is holding until tick {aircraft.holding_until}",
                aircraft.id,
            )
            return

        if not aircraft.air_route:
            self._assign_air_route(aircraft)
        if not aircraft.air_route:
            return

        outcome = airspace_step(aircraft, self._dt)
        if outcome is AirspaceStepOutcome.ROUTE_COMPLETE:
            self._log(
                "AIR_LEG_COMPLETE",
                f"{aircraft.id} completed its "
                f"{aircraft.air_route[-1].id} leg and is now "
                f"{aircraft.state.name}",
                aircraft.id,
            )
            # The next lifecycle state needs its own leg before it can move.
            if aircraft.state in AIRBORNE_STATES:
                self._assign_air_route(aircraft)

    def _conflict_phase(self, tick: int) -> None:
        """Stages 6 and 7: detect conflicts, then resolve them."""
        airborne = [
            aircraft_object
            for aircraft_object in self.aircraft
            if aircraft_object.state in AIRBORNE_STATES
        ]
        if len(airborne) < 2:
            return

        conflicts = detect_all_conflicts(
            airborne,
            horizon=self._horizon,
            horizontal_separation=self._horizontal_separation,
            vertical_separation=self._vertical_separation,
        )
        real = [conflict for conflict in conflicts if conflict.conflict]
        if not real:
            return

        resolutions = self.resolver.resolve_all(real, tick)
        for resolution in resolutions:
            if resolution.action is not ResolutionAction.HOLD_LOWER_PRIORITY:
                continue
            held = resolution.held_aircraft
            assert held is not None
            self._log(
                "HOLD",
                f"ATC: {held.id} HOLD (until tick {resolution.holding_until}) "
                f"- {resolution.reason}",
                held.id,
            )
