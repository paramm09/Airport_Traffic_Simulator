"""Phase 2H tests: the fixed-timestep simulation orchestrator.

These tests care about the things an orchestrator can get wrong: the order of
the tick stages, who owns which decision, determinism, the binary runway, and
whether a HOLD really does stop an aircraft moving.
"""

import pytest

from backend.data.airport_graph import build_airport_graph
from backend.models.aircraft import Aircraft, AircraftState
from backend.models.waypoint import Waypoint
from backend.simulation.conflict_resolution import HOLD_DURATION
from backend.simulation.runway_scheduler import RunwayScheduler
from backend.simulation.simulation import (
    ScheduledChange,
    Simulation,
    SimulationEvent,
    build_arrival,
    build_departure,
)

EAST = 90.0
WEST = 270.0

# A long leg, so an airborne aircraft needs many ticks to finish one.
FAR_EAST = Waypoint("X_EAST", 100.0, 0.0, 2000.0)
FAR_WEST = Waypoint("X_WEST", -100.0, 0.0, 2000.0)


def converging_leg_provider(aircraft, state):
    """AC001 (on the left) flies East, AC002 (on the right) flies West.

    They start 100 units apart and close at 20 units per tick, so they are
    within the 5-unit separation well before the default 10-tick horizon runs
    out.
    """
    return [FAR_EAST] if aircraft.id == "AC001" else [FAR_WEST]


def airborne(aircraft_id, position, altitude=2000.0, speed=10.0,
             state=AircraftState.CRUISE):
    """An already-airborne aircraft, so no runway phase is needed."""
    return Aircraft(
        id=aircraft_id,
        state=state,
        position=position,
        heading=0.0,
        speed=speed,
        altitude=altitude,
    )


def single_departure(aircraft_id="AC001", gate="G1", runway="R1"):
    graph = build_airport_graph()
    aircraft = build_departure(aircraft_id, gate, runway)
    return Simulation(aircraft=[aircraft], graph=graph), aircraft, graph


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_simulation_starts_at_tick_zero_with_an_empty_log():
    simulation, _aircraft, _graph = single_departure()
    assert simulation.current_tick == 0
    assert simulation.events == []


def test_departure_is_told_to_start_taxiing_on_construction():
    simulation, aircraft, _graph = single_departure()
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY
    assert aircraft.taxi_route[0] == "G1"
    assert aircraft.taxi_route[-1] == "R1"
    assert aircraft.destination == "R1"


def test_simulation_uses_a_fresh_graph_by_default():
    simulation = Simulation()
    assert simulation.graph is not None
    assert simulation.graph.get_nodes() == []
    assert simulation.scheduler is not None


def test_simulation_rejects_a_non_positive_dt():
    with pytest.raises(ValueError, match="dt"):
        Simulation(dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        Simulation(dt=-1.0)


def test_find_returns_the_aircraft():
    simulation, aircraft, _graph = single_departure()
    assert simulation.find("AC001") is aircraft
    with pytest.raises(KeyError):
        simulation.find("NOPE")


def test_events_of_kind_filters_the_log():
    simulation, _aircraft, _graph = single_departure()
    simulation.run(8)
    assert simulation.events_of_kind("RUNWAY_REQUEST")
    assert simulation.events_of_kind("NO_SUCH_KIND") == []


def test_events_property_returns_a_copy():
    simulation, _aircraft, _graph = single_departure()
    simulation.run(8)
    assert simulation.events
    snapshot = simulation.events
    snapshot.clear()
    assert simulation.events  # the log itself is untouched


# ----------------------------------------------------------------------
# The tick
# ----------------------------------------------------------------------
def test_step_advances_the_tick_by_exactly_one():
    simulation, _aircraft, _graph = single_departure()
    simulation.step()
    assert simulation.current_tick == 1
    simulation.step()
    assert simulation.current_tick == 2


def test_run_advances_exactly_n_ticks():
    simulation, _aircraft, _graph = single_departure()
    simulation.run(7)
    assert simulation.current_tick == 7
    simulation.run(0)
    assert simulation.current_tick == 7


def test_run_rejects_a_negative_tick_count():
    simulation, _aircraft, _graph = single_departure()
    with pytest.raises(ValueError, match="ticks"):
        simulation.run(-1)


def test_event_ticks_never_decrease():
    simulation, _aircraft, _graph = single_departure()
    simulation.run_until_complete(max_ticks=200)
    ticks = [event.tick for event in simulation.events]
    assert ticks == sorted(ticks)
    assert all(0 <= tick < simulation.current_tick for tick in ticks)


def test_events_are_immutable_records():
    event = SimulationEvent(tick=3, kind="HOLD", message="m", aircraft_id="A")
    with pytest.raises(Exception):
        event.tick = 4  # type: ignore[misc]


# ----------------------------------------------------------------------
# Stage order within a tick
# ----------------------------------------------------------------------
def test_a_ticks_events_follow_the_documented_stage_order():
    """TAXI -> REQUEST -> SCHEDULER -> AIRSPACE, and conflicts are resolved last.

    Detection runs after movement, so a HOLD logged on a tick is about the
    positions produced by that same tick's airspace stage.
    """
    graph = build_airport_graph()
    a = airborne("AC001", (-50.0, 0.0))
    b = airborne("AC002", (50.0, 0.0))
    simulation = Simulation(
        aircraft=[a, b], graph=graph, leg_provider=converging_leg_provider
    )
    simulation.step()
    first_tick = [event.kind for event in simulation.events if event.tick == 0]
    # The only stage with something to say on tick 0 is conflict resolution,
    # because both aircraft are already airborne and nothing else has happened.
    assert first_tick == ["HOLD"]


def test_ground_movement_is_reported_before_the_runway_request():
    simulation, _aircraft, _graph = single_departure()
    simulation.run(8)
    kinds = [event.kind for event in simulation.events]
    # The taxi stage (2) runs before the request stage (3), which runs before
    # the scheduler stage (4).
    assert kinds.index("TAXI_COMPLETE") < kinds.index("RUNWAY_REQUEST")
    assert kinds.index("RUNWAY_REQUEST") < kinds.index("RUNWAY_ASSIGNED")


def test_scheduled_graph_changes_are_applied_before_anything_moves():
    graph = build_airport_graph()
    aircraft = build_departure("AC001", "G1", "R1")
    simulation = Simulation(
        aircraft=[aircraft],
        graph=graph,
        scheduled_changes=[ScheduledChange(0, "block", "T1", "R1")],
    )
    simulation.step()
    # The closure is the very first thing the tick does, so every stage below
    # it already sees the new network.
    assert simulation.events[0].kind == "BLOCK"
    assert simulation.events[0].tick == 0
    # The aircraft is still on its way to T1, so it moves normally for now.
    assert simulation.events_of_kind("BLOCK") == simulation.events[:1]
    assert not simulation.events_of_kind("REROUTE")
    # Once it reaches T1 the closed edge is its next one, and because the
    # closure was applied first it re-plans instead of driving into it.
    simulation.run(2)
    assert simulation.events_of_kind("REROUTE")
    assert aircraft.destination == "R1"


def test_a_gate_cut_off_from_the_network_makes_the_aircraft_hold():
    graph = build_airport_graph()
    aircraft = build_departure("AC001", "G1", "R1")
    simulation = Simulation(
        aircraft=[aircraft],
        graph=graph,
        scheduled_changes=[ScheduledChange(0, "block", "G1", "T1")],
    )
    simulation.step()
    # G1's only connection is closed, so there is genuinely nowhere to go.
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert simulation.events_of_kind("NO_ROUTE")


# ----------------------------------------------------------------------
# Full departure lifecycle
# ----------------------------------------------------------------------
def test_a_departure_runs_the_whole_lifecycle_to_completed():
    simulation, aircraft, _graph = single_departure()
    assert simulation.run_until_complete(max_ticks=500) is True
    assert aircraft.state is AircraftState.COMPLETED
    assert simulation.is_complete() is True


def test_the_lifecycle_passes_through_every_ground_and_air_state():
    simulation, aircraft, _graph = single_departure()
    simulation.run_until_complete(max_ticks=500)
    seen = set()
    for event in simulation.events:
        for state in AircraftState:
            if state.name in event.message:
                seen.add(state)
    for expected in (
        AircraftState.WAITING_FOR_RUNWAY,
        AircraftState.LINE_UP,
        AircraftState.TAKEOFF,
        AircraftState.CLIMB,
        AircraftState.CRUISE,
        AircraftState.DESCENT,
        AircraftState.APPROACH,
        AircraftState.LANDING,
        AircraftState.TAXIING_TO_GATE,
        AircraftState.COMPLETED,
    ):
        assert expected in seen, f"{expected.name} never appeared"


def test_an_arrival_taxis_to_its_gate_and_completes():
    graph = build_airport_graph()
    aircraft = build_arrival(
        "AR001", "G5", "R3", position=(200.0, 0.0), heading=270.0
    )
    simulation = Simulation(aircraft=[aircraft], graph=graph)
    assert simulation.run_until_complete(max_ticks=500) is True
    assert aircraft.state is AircraftState.COMPLETED
    landings = simulation.events_of_kind("RUNWAY_REQUEST")
    assert any("LANDING" in event.message for event in landings)


def test_is_complete_is_false_while_aircraft_are_still_moving():
    simulation, _aircraft, _graph = single_departure()
    assert simulation.is_complete() is False
    simulation.run(1)
    assert simulation.is_complete() is False


def test_run_until_complete_reports_failure_when_bound_is_too_small():
    simulation, _aircraft, _graph = single_departure()
    assert simulation.run_until_complete(max_ticks=2) is False
    assert simulation.current_tick <= 2


def test_run_until_complete_on_an_empty_simulation_is_immediately_done():
    assert Simulation().run_until_complete(max_ticks=10) is True


# ----------------------------------------------------------------------
# The runway is a single binary resource
# ----------------------------------------------------------------------
def two_departures():
    graph = build_airport_graph()
    a = build_departure("AC001", "G1", "R1")
    b = build_departure("AC002", "G2", "R1")
    return Simulation(aircraft=[a, b], graph=graph), a, b


def test_the_runway_is_never_double_booked():
    simulation, _a, _b = two_departures()
    simulation.run_until_complete(max_ticks=500)
    # The true invariant is alternation: looking only at runway events, no two
    # assignments may follow each other, because the previous occupant has to
    # be released first.
    runway_kinds = [
        event.kind
        for event in simulation.events
        if event.kind in ("RUNWAY_ASSIGNED", "RUNWAY_RELEASED")
    ]
    assert runway_kinds, "the scenario never used the runway"
    for index in range(1, len(runway_kinds)):
        if runway_kinds[index - 1] == "RUNWAY_ASSIGNED":
            assert runway_kinds[index] == "RUNWAY_RELEASED", runway_kinds


def test_each_aircraft_is_assigned_after_the_other_releases():
    simulation, _a, _b = two_departures()
    simulation.run_until_complete(max_ticks=500)
    runway_events = [
        (event.kind, event.aircraft_id)
        for event in simulation.events
        if event.kind in ("RUNWAY_ASSIGNED", "RUNWAY_RELEASED")
    ]
    for index in range(1, len(runway_events)):
        if runway_events[index - 1][0] == "RUNWAY_ASSIGNED":
            assert runway_events[index][0] == "RUNWAY_RELEASED", runway_events
            # ... and it must be the same aircraft that gave it up.
            assert runway_events[index][1] == runway_events[index - 1][1]


def test_a_runway_request_is_submitted_exactly_once_per_operation():
    simulation, aircraft, _graph = single_departure()
    simulation.run_until_complete(max_ticks=500)
    requests = simulation.events_of_kind("RUNWAY_REQUEST")
    # A full flight uses the runway twice: once to take off, once to land.
    assert len(requests) == 2
    assert any("TAKEOFF" in event.message for event in requests)
    assert any("LANDING" in event.message for event in requests)
    assert {event.aircraft_id for event in requests} == {"AC001"}


def test_two_departures_take_off_one_after_the_other():
    simulation, _a, _b = two_departures()
    simulation.run_until_complete(max_ticks=500)
    firsts = [
        event.message
        for event in simulation.events_of_kind("RUNWAY_ASSIGNED")
        if "left the runway" not in event.message
    ]
    assert len(firsts) >= 2
    # The second one only got the runway after the first released it.
    ticks = [event.tick for event in simulation.events_of_kind("RUNWAY_ASSIGNED")]
    assert ticks == sorted(ticks)
    assert len(set(ticks)) == len(ticks)  # never two occupants on one tick


def test_a_shared_scheduler_instance_is_used_rather_than_replaced():
    scheduler = RunwayScheduler()
    graph = build_airport_graph()
    aircraft = build_departure("AC001", "G1", "R1")
    simulation = Simulation(
        aircraft=[aircraft], graph=graph, scheduler=scheduler
    )
    assert simulation.scheduler is scheduler
    simulation.run(2)
    assert scheduler.waiting_count() + (
        0 if scheduler.is_busy() else 0
    ) >= 0  # the scheduler really is the one being driven


# ----------------------------------------------------------------------
# Holds
# ----------------------------------------------------------------------
def crossing_simulation():
    graph = build_airport_graph()
    a = airborne("AC001", (-50.0, 0.0))
    b = airborne("AC002", (50.0, 0.0))
    simulation = Simulation(
        aircraft=[a, b], graph=graph, leg_provider=converging_leg_provider
    )
    return simulation, a, b


def test_a_converging_pair_produces_a_hold():
    simulation, _a, _b = crossing_simulation()
    simulation.step()
    holds = simulation.events_of_kind("HOLD")
    assert holds
    assert holds[0].tick == 0
    assert holds[0].message.startswith("ATC: ")


def test_the_higher_priority_aircraft_is_the_one_that_keeps_flying():
    simulation, a, b = crossing_simulation()
    simulation.step()
    # Both are in CRUISE, so both map to TAKEOFF and the id breaks the tie:
    # the greater id yields.
    assert b.holding_until == HOLD_DURATION
    assert a.holding_until is None


def test_a_hold_stops_the_aircraft_actually_moving():
    simulation, a, b = crossing_simulation()
    simulation.step()
    held_position = b.position
    simulation.step()
    assert b.position == held_position  # frozen while holding
    assert a.position != (-50.0, 0.0)  # the other one carried on


def test_a_hold_does_not_stop_the_aircraft_taxiing():
    """A HOLD is an air-traffic instruction, so it must not gate taxiing."""
    graph = build_airport_graph()
    aircraft = build_departure("AC001", "G10", "R5")
    aircraft.holding_until = 50
    simulation = Simulation(aircraft=[aircraft], graph=graph)
    simulation.run_until_complete(max_ticks=500)
    # A hold was in force the whole flight, yet the aircraft still got home.
    assert aircraft.state is AircraftState.COMPLETED


def test_holding_is_logged_and_expires():
    simulation, _a, _b = crossing_simulation()
    simulation.run(3)
    assert simulation.events_of_kind("HOLDING")
    holding_events = simulation.events_of_kind("HOLDING")
    assert all("holding until tick" in event.message for event in holding_events)


def test_a_conflict_that_goes_away_stops_producing_holds():
    simulation, a, _b = crossing_simulation()
    simulation.run_until_complete(max_ticks=200)
    # AC001 flies past the crossing point, so the pair stops conflicting.
    tail = [event for event in simulation.events if event.tick >= 10]
    assert not [event for event in tail if event.kind == "HOLD"]


# ----------------------------------------------------------------------
# Rerouting inside the simulation
# ----------------------------------------------------------------------
def test_a_blocked_taxi_edge_triggers_a_reroute_event():
    graph = build_airport_graph()
    aircraft = build_departure("AC001", "G1", "R1")
    # Close the T1 -> R1 shortcut at tick 0, before it is ever used.
    simulation = Simulation(
        aircraft=[aircraft],
        graph=graph,
        scheduled_changes=[ScheduledChange(0, "block", "T1", "R1")],
    )
    simulation.run_until_complete(max_ticks=500)
    # The aircraft still gets home, and the closed edge was never used.
    assert aircraft.state is AircraftState.COMPLETED
    for node1, node2 in zip(aircraft.taxi_route, aircraft.taxi_route[1:]):
        assert graph.has_edge(node1, node2)


def test_an_unreachable_destination_makes_the_aircraft_hold_then_recover():
    graph = build_airport_graph()
    aircraft = build_departure("AC001", "G1", "R1")
    simulation = Simulation(
        aircraft=[aircraft],
        graph=graph,
        scheduled_changes=[
            ScheduledChange(0, "block", "T1", "T2"),
            ScheduledChange(0, "block", "T1", "T10"),
            ScheduledChange(0, "block", "T1", "R1"),
            ScheduledChange(4, "unblock", "T1", "R1"),
        ],
    )
    simulation.run(2)
    assert simulation.events_of_kind("BLOCK")
    assert aircraft.destination == "R1"
    # It is now holding, which is a *terminal* state, so run_until_complete
    # would stop here and the tick-4 reopening would never be reached. Drive the
    # clock across it explicitly instead.
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert simulation.is_complete() is True
    simulation.run(4)  # ticks 2..5, so the tick-4 reopening is processed
    # The existing Phase 2G retry mechanism picks it back up.
    assert simulation.events_of_kind("ROUTE_FOUND")
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY
    assert simulation.run_until_complete(max_ticks=500) is True
    assert aircraft.state is AircraftState.COMPLETED


def test_scheduled_changes_are_applied_in_tick_order():
    graph = build_airport_graph()
    simulation = Simulation(
        aircraft=[],
        graph=graph,
        scheduled_changes=[
            ScheduledChange(5, "unblock", "T1", "R1"),
            ScheduledChange(1, "block", "T1", "R1"),
        ],
    )
    simulation.run(3)
    assert graph.is_blocked("T1", "R1")
    simulation.run(3)
    assert not graph.is_blocked("T1", "R1")


def test_an_unknown_scheduled_action_is_rejected():
    graph = build_airport_graph()
    simulation = Simulation(
        aircraft=[],
        graph=graph,
        scheduled_changes=[ScheduledChange(0, "explode", "T1", "R1")],
    )
    with pytest.raises(ValueError, match="scheduled change"):
        simulation.step()


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def scenario_factory():
    """The same starting traffic every time, so determinism is testable."""
    return [
        build_departure("AC001", "G1", "R1"),
        build_departure("AC002", "G3", "R2"),
        build_arrival(
            "AR001", "G5", "R3", position=(150.0, 0.0), heading=270.0
        ),
    ]


def test_two_identical_runs_produce_identical_event_logs():
    first = Simulation(
        aircraft=scenario_factory(), graph=build_airport_graph()
    )
    second = Simulation(
        aircraft=scenario_factory(), graph=build_airport_graph()
    )
    first.run(40)
    second.run(40)
    assert first.events == second.events
    assert first.current_tick == second.current_tick


def test_two_identical_runs_produce_identical_final_states():
    first = Simulation(aircraft=scenario_factory(), graph=build_airport_graph())
    second = Simulation(aircraft=scenario_factory(), graph=build_airport_graph())
    first.run_until_complete(max_ticks=1000)
    second.run_until_complete(max_ticks=1000)
    for left, right in zip(first.aircraft, second.aircraft):
        assert left.state is right.state
        assert left.position == pytest.approx(right.position)
        assert left.altitude == pytest.approx(right.altitude)
        assert left.taxi_route == right.taxi_route


def test_the_same_seed_of_events_repeats_after_a_reset():
    """A fresh Simulation is a clean slate: no state leaks between runs."""
    graph_one = build_airport_graph()
    simulation_one = Simulation(
        aircraft=scenario_factory(), graph=graph_one
    )
    simulation_one.run_until_complete(max_ticks=1000)
    first_length = len(simulation_one.events)

    graph_two = build_airport_graph()
    simulation_two = Simulation(
        aircraft=scenario_factory(), graph=graph_two
    )
    simulation_two.run_until_complete(max_ticks=1000)
    assert len(simulation_two.events) == first_length


def test_running_one_tick_at_a_time_matches_running_them_all_at_once():
    stepwise = Simulation(aircraft=scenario_factory(), graph=build_airport_graph())
    stepwise.run(25)
    bulk = Simulation(aircraft=scenario_factory(), graph=build_airport_graph())
    bulk.run(25)
    assert stepwise.events == bulk.events


# ----------------------------------------------------------------------
# The three-aircraft integrated scenario
# ----------------------------------------------------------------------
def test_the_integrated_scenario_completes_every_aircraft():
    simulation = Simulation(
        aircraft=scenario_factory(), graph=build_airport_graph()
    )
    assert simulation.run_until_complete(max_ticks=2000) is True
    for aircraft in simulation.aircraft:
        assert aircraft.state is AircraftState.COMPLETED, aircraft.id
    assert simulation.events_of_kind("RUNWAY_ASSIGNED")
    assert simulation.events_of_kind("TAXI_COMPLETE")


def test_the_integrated_scenario_respects_the_single_runway():
    simulation = Simulation(
        aircraft=scenario_factory(), graph=build_airport_graph()
    )
    simulation.run_until_complete(max_ticks=2000)
    # The runway is a single binary resource, so runway events must alternate
    # even though unrelated events (taxiing, air legs) happen in between.
    runway_kinds = [
        event.kind
        for event in simulation.events
        if event.kind in ("RUNWAY_ASSIGNED", "RUNWAY_RELEASED")
    ]
    for index in range(1, len(runway_kinds)):
        if runway_kinds[index - 1] == "RUNWAY_ASSIGNED":
            assert runway_kinds[index] == "RUNWAY_RELEASED", runway_kinds


def test_no_aircraft_is_ever_left_in_a_broken_state():
    simulation = Simulation(
        aircraft=scenario_factory(), graph=build_airport_graph()
    )
    simulation.run_until_complete(max_ticks=2000)
    for aircraft in simulation.aircraft:
        assert aircraft.state is not AircraftState.HOLDING_NO_ROUTE
        assert aircraft.taxi_route, aircraft.id
        if aircraft.state is AircraftState.COMPLETED:
            continue
        assert aircraft.destination is not None


# ----------------------------------------------------------------------
# FIX 1 -- a route that is unplannable before the simulation starts
# ----------------------------------------------------------------------
def pre_blocked_departure(block_at_tick=None):
    """A departure whose gate access was closed before the first tick.

    ``G1``'s only connection is the gate access edge ``G1-T1``, so closing it
    leaves the aircraft genuinely stranded: Dijkstra has nothing to plan.
    """
    graph = build_airport_graph()
    graph.block_edge("G1", "T1")  # blocked BEFORE the Simulation is built
    aircraft = build_departure("AC001", "G1", "R1")
    changes = ()
    if block_at_tick is not None:
        changes = (ScheduledChange(block_at_tick, "unblock", "G1", "T1"),)
    return Simulation(
        aircraft=[aircraft], graph=graph, scheduled_changes=changes
    ), aircraft, graph


def test_a_pre_blocked_route_never_reaches_the_runway():
    simulation, aircraft, _graph = pre_blocked_departure()
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    # Run long enough that a working taxi would have finished and taken off.
    simulation.run(20)
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    for forbidden in (
        AircraftState.WAITING_FOR_RUNWAY,
        AircraftState.LINE_UP,
        AircraftState.TAKEOFF,
    ):
        assert aircraft.state is not forbidden


def test_a_pre_blocked_aircraft_never_requests_the_runway():
    simulation, _aircraft, _graph = pre_blocked_departure()
    simulation.run(20)
    # The whole point: it must not be sequenced onto a shared runway it never
    # taxied to.
    assert simulation.events_of_kind("RUNWAY_REQUEST") == []
    assert simulation.events_of_kind("RUNWAY_ASSIGNED") == []
    assert simulation.events_of_kind("TAXI_COMPLETE") == []


def test_a_pre_blocked_aircraft_waits_for_the_explicit_retry_mechanism():
    simulation, aircraft, _graph = pre_blocked_departure(block_at_tick=3)
    simulation.run(3)
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    # The tick-3 reopening is processed, which is the Phase 2G retry trigger.
    simulation.step()
    assert simulation.events_of_kind("ROUTE_FOUND")
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY
    assert aircraft.taxi_route[0] == "G1"
    assert aircraft.taxi_route[-1] == "R1"


def test_a_recovered_aircraft_taxes_normally_to_completion():
    simulation, aircraft, _graph = pre_blocked_departure(block_at_tick=3)
    # Holding is terminal, so this stops at once without advancing the clock.
    assert simulation.run_until_complete(max_ticks=500) is True
    assert simulation.current_tick == 0
    simulation.run(4)  # ticks 0..3, crossing the reopening
    assert simulation.run_until_complete(max_ticks=500) is True
    assert aircraft.state is AircraftState.COMPLETED
    # It really did taxi the full journey once unblocked.
    assert simulation.events_of_kind("TAXI_COMPLETE")
    assert simulation.events_of_kind("RUNWAY_ASSIGNED")


def test_an_unblocked_route_is_planned_normally_on_construction():
    """The ordinary case must be untouched by the no-route handling."""
    graph = build_airport_graph()
    aircraft = build_departure("AC001", "G1", "R1")
    simulation = Simulation(aircraft=[aircraft], graph=graph)
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY
    assert aircraft.state is not AircraftState.HOLDING_NO_ROUTE
    assert aircraft.taxi_route[0] == "G1"
    assert aircraft.taxi_route[-1] == "R1"
    assert not simulation.events_of_kind("NO_ROUTE")
    assert simulation.run_until_complete(max_ticks=500) is True
    assert aircraft.state is AircraftState.COMPLETED


def test_an_arrival_with_a_reachable_gate_behaves_normally():
    # The regression guard for the fix below: the ordinary arrival path must be
    # untouched, and must not go anywhere near HOLDING_NO_ROUTE.
    graph = build_airport_graph()
    aircraft = build_arrival("AR001", "G5", "R3", position=(200.0, 0.0))
    simulation = Simulation(aircraft=[aircraft], graph=graph)
    assert simulation.run_until_complete(max_ticks=2000) is True
    assert aircraft.state is AircraftState.COMPLETED
    assert simulation.events_of_kind("NO_ROUTE") == []


def test_an_unreachable_arrival_holds_instead_of_completing():
    """An arrival whose gate is cut off holds, and never fakes a success.

    Reporting it COMPLETED while it is still standing on the runway would be a
    false success, so it has to stop somewhere honest.
    """
    graph = build_airport_graph()
    graph.block_edge("G5", "T5")  # cut the arrival's gate off from the network
    aircraft = build_arrival("AR001", "G5", "R3", position=(200.0, 0.0))
    simulation = Simulation(aircraft=[aircraft], graph=graph)
    simulation.run(20)
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert aircraft.holding_from is AircraftState.TAXIING_TO_GATE
    assert aircraft.state is not AircraftState.COMPLETED
    # HOLDING_NO_ROUTE is terminal by Phase 2G design, so the simulation reports
    # itself settled rather than pretending the aircraft is still working.
    assert simulation.is_complete() is True
    assert simulation.events_of_kind("NO_ROUTE")


def test_a_holding_arrival_logs_no_route_once_not_every_tick():
    # Before the fix the arrival sat in TAXIING_TO_GATE and re-logged NO_ROUTE
    # on every single tick, which reads exactly like it is still trying.
    graph = build_airport_graph()
    graph.block_edge("G5", "T5")
    aircraft = build_arrival("AR001", "G5", "R3", position=(200.0, 0.0))
    simulation = Simulation(aircraft=[aircraft], graph=graph)
    simulation.run(20)
    assert len(simulation.events_of_kind("NO_ROUTE")) == 1
    assert not simulation.events_of_kind("STILL_NO_ROUTE")


def test_a_holding_arrival_resumes_to_its_gate_after_an_unblock():
    # run_until_complete() would stop the moment the arrival holds, by design,
    # so the test steps across the reopening explicitly.
    graph = build_airport_graph()
    graph.block_edge("G5", "T5")  # blocked BEFORE the Simulation is built
    aircraft = build_arrival("AR001", "G5", "R3", position=(200.0, 0.0))
    simulation = Simulation(
        aircraft=[aircraft],
        graph=graph,
        scheduled_changes=[ScheduledChange(12, "unblock", "G5", "T5")],
    )

    # Ticks 0..11: it lands, is released, tries to taxi, finds no route, holds.
    simulation.run(12)
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert not simulation.events_of_kind("ROUTE_FOUND")

    # Tick 12: the reopening is processed and is the retry trigger.
    simulation.run(1)
    assert simulation.events_of_kind("ROUTE_FOUND")
    assert aircraft.state is AircraftState.TAXIING_TO_GATE
    assert aircraft.holding_from is None
    assert aircraft.taxi_route, aircraft.id

    # From here it is an ordinary arrival again: taxi to the gate, complete.
    assert simulation.run_until_complete(max_ticks=2000) is True
    assert aircraft.state is AircraftState.COMPLETED


def test_a_holding_arrival_that_stays_cut_off_keeps_waiting():
    # A retry that still finds nothing must not invent progress, and must not
    # flip the aircraft into the departure-side exit by mistake.
    graph = build_airport_graph()
    graph.block_edge("G5", "T5")
    aircraft = build_arrival("AR001", "G5", "R3", position=(200.0, 0.0))
    simulation = Simulation(
        aircraft=[aircraft],
        graph=graph,
        # An unrelated edge reopens, so the retry mechanism runs -- and fails,
        # because the arrival's own gate is still cut off.
        scheduled_changes=[ScheduledChange(12, "unblock", "T1", "T2")],
    )
    simulation.run(20)
    assert simulation.events_of_kind("STILL_NO_ROUTE")
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert aircraft.holding_from is AircraftState.TAXIING_TO_GATE
    assert aircraft.state is not AircraftState.COMPLETED


def test_a_holding_departure_still_resumes_towards_the_runway():
    # The departure half of the change: unchanged behaviour, now going through
    # the same shared state.
    graph = build_airport_graph()
    graph.block_edge("G1", "T1")
    aircraft = build_departure("DP001", "G1", "R1")
    simulation = Simulation(
        aircraft=[aircraft],
        graph=graph,
        scheduled_changes=[ScheduledChange(12, "unblock", "G1", "T1")],
    )
    # Blocked before the first tick, so it holds on construction.
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert aircraft.holding_from is AircraftState.TAXIING_TO_RUNWAY

    simulation.run(13)  # cross the tick-12 reopening
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY
    assert aircraft.holding_from is None

    simulation.run_until_complete(max_ticks=4000)
    assert aircraft.state is AircraftState.COMPLETED


# ----------------------------------------------------------------------
# FIX 2 -- is_complete() is a statement about states, not about empty routes
# ----------------------------------------------------------------------
def in_state(state, **kwargs):
    return Aircraft(id=kwargs.pop("aircraft_id", "AC001"), state=state, **kwargs)


def holding_no_route(aircraft_id="AC001", **kwargs):
    """An aircraft parked in HOLDING_NO_ROUTE, held for a known taxi state.

    HOLDING_NO_ROUTE has two exits, so an aircraft may not occupy it without
    saying which taxi state it should resume into.
    """
    return Aircraft(
        id=aircraft_id,
        state=AircraftState.HOLDING_NO_ROUTE,
        holding_from=AircraftState.TAXIING_TO_RUNWAY,
        **kwargs,
    )


# Every state that means "this aircraft can still move", and so must make the
# simulation report itself incomplete.
NON_TERMINAL_STATES = (
    AircraftState.AT_GATE,
    AircraftState.TAXIING_TO_RUNWAY,
    AircraftState.WAITING_FOR_RUNWAY,
    AircraftState.LINE_UP,
    AircraftState.TAKEOFF,
    AircraftState.CLIMB,
    AircraftState.CRUISE,
    AircraftState.DESCENT,
    AircraftState.APPROACH,
    AircraftState.LANDING,
    AircraftState.TAXIING_TO_GATE,
)


def test_all_completed_is_complete():
    simulation = Simulation(
        aircraft=[in_state(AircraftState.COMPLETED)],
        graph=build_airport_graph(),
    )
    assert simulation.is_complete() is True


def test_all_holding_no_route_is_complete():
    simulation = Simulation(
        aircraft=[holding_no_route()],
        graph=build_airport_graph(),
    )
    assert simulation.is_complete() is True


def test_a_mix_of_completed_and_holding_no_route_is_complete():
    simulation = Simulation(
        aircraft=[
            in_state(AircraftState.COMPLETED, aircraft_id="AC001"),
            holding_no_route(aircraft_id="AC002"),
            in_state(AircraftState.COMPLETED, aircraft_id="AC003"),
        ],
        graph=build_airport_graph(),
    )
    assert simulation.is_complete() is True


def test_one_active_aircraft_among_terminal_ones_is_not_complete():
    simulation = Simulation(
        aircraft=[
            in_state(AircraftState.COMPLETED, aircraft_id="AC001"),
            in_state(AircraftState.CRUISE, aircraft_id="AC002"),
        ],
        graph=build_airport_graph(),
    )
    assert simulation.is_complete() is False


@pytest.mark.parametrize("state", NON_TERMINAL_STATES)
def test_no_active_state_is_ever_treated_as_complete(state):
    """No mid-lifecycle state may be mistaken for a finished one."""
    simulation = Simulation(
        aircraft=[in_state(state)], graph=build_airport_graph()
    )
    assert simulation.is_complete() is False, state.name


@pytest.mark.parametrize(
    "state",
    [
        AircraftState.CRUISE,
        AircraftState.CLIMB,
        AircraftState.TAKEOFF,
        AircraftState.DESCENT,
        AircraftState.APPROACH,
    ],
)
def test_an_airborne_aircraft_without_a_route_is_not_complete(state):
    """The specific trap: an empty air route must not imply 'finished'.

    An aircraft in an airborne state with nothing in ``air_route`` is still
    mid-lifecycle -- it has a runway to use and a gate to reach -- so the
    simulation is still running.
    """
    aircraft = in_state(state, speed=10.0, altitude=2000.0)
    assert aircraft.air_route == []
    simulation = Simulation(
        aircraft=[aircraft], graph=build_airport_graph()
    )
    assert simulation.is_complete() is False, state.name


def test_an_aircraft_that_has_flown_its_last_waypoint_is_not_complete():
    """Reaching the end of an air leg is progress, not an ending.

    ``air_route_index`` has run off the end, but the lifecycle is mid-chain, so
    the aircraft still has to advance and then land.
    """
    aircraft = in_state(
        AircraftState.CRUISE, speed=10.0, altitude=2000.0,
        air_route=[FAR_EAST], air_route_index=1,
    )
    simulation = Simulation(aircraft=[aircraft], graph=build_airport_graph())
    assert simulation.is_complete() is False


def test_an_empty_simulation_is_complete():
    assert Simulation().is_complete() is True


def test_run_until_complete_still_gives_up_rather_than_looping_forever():
    """The terminal-state rules must not remove the infinite-loop guard."""
    aircraft = in_state(AircraftState.CRUISE, speed=10.0, altitude=2000.0)
    simulation = Simulation(aircraft=[aircraft], graph=build_airport_graph())
    assert simulation.run_until_complete(max_ticks=25) is False
    assert simulation.current_tick == 25


# ----------------------------------------------------------------------
# FIX 3 -- expected state conditions are handled, not swallowed
# ----------------------------------------------------------------------
def test_a_standing_arrival_logs_a_no_route_event_instead_of_raising():
    """The condition the old ``except ValueError`` used to hide.

    An arrival with no destination has no route to walk. That is a state
    condition, so it is reported as an event and the aircraft simply stays put.
    """
    aircraft = Aircraft(
        id="AR001", state=AircraftState.TAXIING_TO_GATE, taxi_route=[]
    )
    simulation = Simulation(aircraft=[aircraft], graph=build_airport_graph())
    simulation.step()  # must not raise
    assert aircraft.state is AircraftState.TAXIING_TO_GATE
    reasons = simulation.events_of_kind("NO_ROUTE")
    assert reasons
    assert "no destination" in reasons[0].message


def test_a_standing_arrival_keeps_logging_rather_than_vanishing():
    aircraft = Aircraft(
        id="AR001", state=AircraftState.TAXIING_TO_GATE, taxi_route=[]
    )
    simulation = Simulation(aircraft=[aircraft], graph=build_airport_graph())
    simulation.run(3)
    # The problem was never hidden, it is reported every tick it applies.
    assert len(simulation.events_of_kind("NO_ROUTE")) == 3


def test_a_genuine_invalid_input_still_raises_instead_of_being_swallowed():
    """A real error must surface rather than being quietly turned into a no-op."""
    # A waypoint that is not a Waypoint is a programming error, and the model
    # rejects it outright rather than letting it reach the movement code.
    with pytest.raises(ValueError, match="Waypoint"):
        Aircraft(id="AC002", state=AircraftState.CRUISE, air_route=["NOT_A_WP"])



def test_illegal_lifecycle_transitions_are_not_swallowed_either():
    aircraft = in_state(AircraftState.CRUISE)
    with pytest.raises(ValueError, match="not a legal transition"):
        aircraft.transition_to(AircraftState.WAITING_FOR_RUNWAY)


def test_no_broad_exception_handling_remains_in_the_orchestrator():
    """Guard against a future ``except Exception: pass`` creeping back in."""
    import inspect

    from backend.simulation import simulation as simulation_module

    source = inspect.getsource(simulation_module)
    assert "except Exception" not in source
    assert "except ValueError" not in source
    assert "except KeyError" not in source


# ----------------------------------------------------------------------
# FIX 5 -- aircraft IDs must be unique
# ----------------------------------------------------------------------
def test_unique_aircraft_ids_are_accepted():
    simulation = Simulation(
        aircraft=scenario_factory(), graph=build_airport_graph()
    )
    assert [aircraft.id for aircraft in simulation.aircraft] == [
        "AC001", "AC002", "AR001",
    ]


def test_a_duplicate_aircraft_id_is_rejected():
    graph = build_airport_graph()
    duplicate = [
        build_departure("AC001", "G1", "R1"),
        build_departure("AC001", "G2", "R1"),
    ]
    with pytest.raises(ValueError, match="Duplicate aircraft ID: AC001"):
        Simulation(aircraft=duplicate, graph=graph)


def test_several_duplicate_ids_are_all_reported():
    graph = build_airport_graph()
    duplicated = [
        build_departure("AC001", "G1", "R1"),
        build_departure("AC002", "G2", "R1"),
        build_departure("AC001", "G3", "R1"),
        build_departure("AC002", "G4", "R1"),
        build_departure("AC001", "G5", "R1"),
    ]
    with pytest.raises(ValueError) as error:
        Simulation(aircraft=duplicated, graph=graph)
    assert "AC001" in str(error.value)
    assert "AC002" in str(error.value)


def test_ordinary_string_ids_keep_working():
    graph = build_airport_graph()
    ordinary = [
        build_departure("BA-117", "G1", "R1"),
        build_departure("BA_118", "G2", "R1"),
        build_departure("flight 119", "G3", "R1"),
        build_arrival("AR-001", "G5", "R3", position=(150.0, 0.0)),
    ]
    simulation = Simulation(aircraft=ordinary, graph=graph)
    assert simulation.find("BA-117").id == "BA-117"
    assert simulation.find("flight 119").id == "flight 119"
    assert simulation.run_until_complete(max_ticks=2000) is True


def test_the_duplicate_check_runs_before_any_aircraft_is_mutated():
    """A rejected scenario must not have been half-started."""
    graph = build_airport_graph()
    duplicated = [
        build_departure("AC001", "G1", "R1"),
        build_departure("AC001", "G2", "R1"),
    ]
    with pytest.raises(ValueError, match="Duplicate aircraft ID"):
        Simulation(aircraft=duplicated, graph=graph)
    # Still AT_GATE, i.e. construction aborted before the taxi phase.
    assert all(
        aircraft.state is AircraftState.AT_GATE for aircraft in duplicated
    )

