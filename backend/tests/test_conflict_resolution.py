"""Phase 2F tests: HOLD-based conflict resolution.

The resolver must be deterministic, must only ever write ``holding_until``,
must never shorten an existing hold, and must resolve the most urgent conflict
first when several are predicted in the same tick.
"""

import pytest

from backend.models.aircraft import Aircraft, AircraftState
from backend.models.waypoint import Waypoint
from backend.simulation.conflict import detect_all_conflicts, detect_conflict
from backend.simulation.conflict_resolution import (
    HOLD_DURATION,
    ARRIVAL_STATES,
    ConflictResolver,
    ResolutionAction,
    choose_aircraft_to_hold,
    hold,
    is_holding,
    operation_for,
    priority_key,
    resolve,
    urgency_key,
)
from backend.simulation.runway_scheduler import RunwayOperation

EAST = 90.0
WEST = 270.0


def flying(
    aircraft_id,
    position=(0.0, 0.0),
    heading=EAST,
    speed=10.0,
    altitude=2000.0,
    state=AircraftState.CRUISE,
    request_time=0,
):
    """An airborne aircraft, with ``request_time`` already set."""
    waypoint = Waypoint(f"{aircraft_id}_WP", position[0], position[1], altitude)
    return Aircraft(
        id=aircraft_id,
        state=state,
        position=position,
        heading=heading,
        speed=speed,
        altitude=altitude,
        request_time=request_time,
        air_route=[waypoint],
    )


def airborne_snapshot(aircraft):
    """Everything a resolver must NOT change."""
    return (
        aircraft.state,
        aircraft.position,
        aircraft.heading,
        aircraft.speed,
        aircraft.altitude,
        aircraft.taxi_route,
        aircraft.route_index,
        aircraft.destination,
        aircraft.destination,
        aircraft.distance_on_edge,
        aircraft.edge_length,
        aircraft.request_time,
        aircraft.air_route_index,
        aircraft.air_route,
    )


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
def test_hold_duration_default():
    assert HOLD_DURATION == 5


def test_resolver_exposes_its_hold_duration():
    assert ConflictResolver().hold_duration == HOLD_DURATION
    assert ConflictResolver(hold_duration=3).hold_duration == 3


# ----------------------------------------------------------------------
# Operation inference
# ----------------------------------------------------------------------
@pytest.mark.parametrize("state", sorted(ARRIVAL_STATES, key=lambda s: s.name))
def test_arrival_states_map_to_landing(state):
    assert operation_for(flying("AC001", state=state)) is RunwayOperation.LANDING


@pytest.mark.parametrize(
    "state",
    [
        AircraftState.AT_GATE,
        AircraftState.TAXIING_TO_RUNWAY,
        AircraftState.WAITING_FOR_RUNWAY,
        AircraftState.LINE_UP,
        AircraftState.TAKEOFF,
        AircraftState.CLIMB,
        AircraftState.CRUISE,
    ],
)
def test_departure_states_map_to_takeoff(state):
    assert operation_for(flying("AC001", state=state)) is RunwayOperation.TAKEOFF


def test_landing_outranks_takeoff():
    arrival = flying("AC001", state=AircraftState.APPROACH, request_time=99)
    departure = flying("AC002", state=AircraftState.CRUISE, request_time=0)
    assert priority_key(arrival) < priority_key(departure)


def test_earlier_request_outranks_later_within_a_class():
    first = flying("AC001", request_time=1)
    second = flying("AC002", request_time=2)
    assert priority_key(first) < priority_key(second)


def test_id_is_the_final_tie_breaker():
    low = flying("AC001", request_time=5)
    high = flying("AC002", request_time=5)
    assert priority_key(low) < priority_key(high)


# ----------------------------------------------------------------------
# choose_aircraft_to_hold
# ----------------------------------------------------------------------
def test_lower_priority_class_is_held():
    arrival = flying("AC001", state=AircraftState.APPROACH, request_time=99)
    departure = flying("AC002", state=AircraftState.CRUISE, request_time=0)
    to_hold, to_continue = choose_aircraft_to_hold(arrival, departure)
    assert to_hold is departure
    assert to_continue is arrival


def test_later_request_is_held_within_a_class():
    early = flying("AC001", request_time=1)
    late = flying("AC002", request_time=7)
    to_hold, to_continue = choose_aircraft_to_hold(early, late)
    assert to_hold is late
    assert to_continue is early


def test_full_tie_holds_the_greater_id():
    a = flying("AC001", request_time=3)
    b = flying("AC002", request_time=3)
    to_hold, to_continue = choose_aircraft_to_hold(a, b)
    assert to_hold is b
    assert to_continue is a


def test_choice_is_symmetric_in_argument_order():
    a = flying("AC001", request_time=3)
    b = flying("AC002", request_time=3)
    assert choose_aircraft_to_hold(a, b) == choose_aircraft_to_hold(b, a)
    assert choose_aircraft_to_hold(a, b)[0] is choose_aircraft_to_hold(b, a)[0]


def test_choice_does_not_mutate_either_aircraft():
    a = flying("AC001", state=AircraftState.APPROACH, request_time=0)
    b = flying("AC002", request_time=0)
    before_a, before_b = airborne_snapshot(a), airborne_snapshot(b)
    choose_aircraft_to_hold(a, b)
    assert airborne_snapshot(a) == before_a
    assert airborne_snapshot(b) == before_b


# ----------------------------------------------------------------------
# hold()
# ----------------------------------------------------------------------
def test_hold_sets_holding_until_to_now_plus_duration():
    aircraft = flying("AC001")
    assert hold(aircraft, current_tick=10) == 15
    assert aircraft.holding_until == 15


def test_hold_from_zero_tick():
    aircraft = flying("AC001")
    assert hold(aircraft, current_tick=0) == HOLD_DURATION


def test_hold_extends_a_longer_existing_hold():
    aircraft = flying("AC001")
    aircraft.holding_until = 100
    assert hold(aircraft, current_tick=10) == 100
    assert aircraft.holding_until == 100


def test_hold_shortens_nothing_but_overrides_a_shorter_hold():
    aircraft = flying("AC001")
    aircraft.holding_until = 12
    assert hold(aircraft, current_tick=10) == 15
    assert aircraft.holding_until == 15


def test_repeated_holds_never_shorten_the_deadline():
    aircraft = flying("AC001")
    deadlines = [hold(aircraft, current_tick=tick) for tick in range(5)]
    # The deadline may only move forward, and each tick's instruction extends
    # it to at most that tick + HOLD_DURATION.
    assert deadlines == sorted(deadlines)
    assert deadlines == [tick + HOLD_DURATION for tick in range(5)]
    assert aircraft.holding_until == 4 + HOLD_DURATION


def test_holds_at_the_same_tick_do_not_stack():
    aircraft = flying("AC001")
    first = hold(aircraft, current_tick=3)
    second = hold(aircraft, current_tick=3)
    third = hold(aircraft, current_tick=3)
    assert first == second == third == 3 + HOLD_DURATION


def test_custom_hold_duration_is_honoured():
    aircraft = flying("AC001")
    assert hold(aircraft, current_tick=10, hold_duration=2) == 12
    assert aircraft.holding_until == 12


# ----------------------------------------------------------------------
# is_holding()
# ----------------------------------------------------------------------
def test_is_holding_false_when_never_held():
    assert is_holding(flying("AC001"), current_tick=0) is False


def test_is_holding_true_before_the_deadline():
    aircraft = flying("AC001")
    aircraft.holding_until = 5
    assert is_holding(aircraft, current_tick=0) is True
    assert is_holding(aircraft, current_tick=4) is True


def test_is_holding_false_at_and_after_the_deadline():
    aircraft = flying("AC001")
    aircraft.holding_until = 5
    assert is_holding(aircraft, current_tick=5) is False
    assert is_holding(aircraft, current_tick=6) is False


# ----------------------------------------------------------------------
# resolve()
# ----------------------------------------------------------------------
def test_resolve_holds_the_lower_priority_aircraft():
    arrival = flying(
        "AC001", (0.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH
    )
    departure = flying("AC002", (100.0, 0.0), WEST, 10.0,
                       state=AircraftState.CRUISE)
    conflict = detect_conflict(arrival, departure)
    result = resolve(conflict, current_tick=7)
    assert result.action is ResolutionAction.HOLD_LOWER_PRIORITY
    assert result.held_aircraft is departure
    assert result.holding_until == 12
    assert departure.holding_until == 12
    assert arrival.holding_until is None


def test_resolve_writes_only_holding_until():
    arrival = flying(
        "AC001", (0.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH
    )
    departure = flying("AC002", (100.0, 0.0), WEST, 10.0,
                       state=AircraftState.CRUISE)
    before = airborne_snapshot(departure)
    resolve(detect_conflict(arrival, departure), current_tick=0)
    # Only holding_until may differ, and it is the only field the resolver adds.
    assert airborne_snapshot(departure)[:1] == before[:1]
    assert departure.position == before[1]
    assert departure.heading == before[2]
    assert departure.speed == before[3]
    assert departure.altitude == before[4]
    assert departure.state is before[0]


def test_resolve_does_not_move_the_continuing_aircraft():
    arrival = flying(
        "AC001", (0.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH
    )
    departure = flying("AC002", (100.0, 0.0), WEST, 10.0,
                       state=AircraftState.CRUISE)
    before = airborne_snapshot(arrival)
    resolve(detect_conflict(arrival, departure), current_tick=0)
    assert airborne_snapshot(arrival) == before


def test_resolve_of_a_non_conflict_does_nothing():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (0.0, 500.0), EAST, 10.0)
    result = resolve(detect_conflict(a, b), current_tick=3)
    assert result.action is ResolutionAction.NO_ACTION
    assert result.held_aircraft is None
    assert result.holding_until is None
    assert a.holding_until is None
    assert b.holding_until is None


def test_resolve_never_shortens_an_earlier_hold():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0,
               state=AircraftState.APPROACH)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    b.holding_until = 50
    result = resolve(detect_conflict(a, b), current_tick=1)
    assert result.holding_until == 50
    assert b.holding_until == 50


def test_resolve_produces_a_human_readable_reason():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0,
               state=AircraftState.APPROACH)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    reason = resolve(detect_conflict(a, b), current_tick=0).reason
    assert "AC002" in reason and "AC001" in reason
    assert "ATC" not in reason  # the ATC message is the caller's job


def test_resolve_respects_a_custom_hold_duration():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0,
               state=AircraftState.APPROACH)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    result = resolve(detect_conflict(a, b), current_tick=10, hold_duration=2)
    assert result.holding_until == 12
    assert b.holding_until == 12


# ----------------------------------------------------------------------
# urgency_key()
# ----------------------------------------------------------------------
def test_urgency_key_orders_by_time_to_conflict():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0,
               state=AircraftState.APPROACH)
    near = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    far = flying("AC003", (0.0, 0.0), EAST, 10.0, state=AircraftState.CRUISE)
    # AC002/AC003 diverge, so force a timing difference with a second pair.
    slow = flying("AC004", (400.0, 0.0), WEST, 10.0,
                  state=AircraftState.CRUISE)
    near_conflict = detect_conflict(a, near)
    far_conflict = detect_conflict(flying("AC005", (0.0, 0.0), EAST, 10.0,
                                         state=AircraftState.APPROACH), slow)
    assert urgency_key(near_conflict) < urgency_key(far_conflict)
    # A non-conflict has no urgency and sorts last.
    no_conflict = detect_conflict(far, flying("AC006", (0.0, 900.0)))
    assert urgency_key(no_conflict) > urgency_key(near_conflict)
    assert urgency_key(no_conflict)[0] == float("inf")


def test_urgency_key_breaks_ties_on_the_sorted_id_pair():
    a = flying("AC002", (0.0, 0.0), EAST, 10.0)
    b = flying("AC001", (100.0, 0.0), WEST, 10.0)
    conflict = detect_conflict(a, b)
    assert urgency_key(conflict) == (conflict.time_to_conflict, "AC001", "AC002")


# ----------------------------------------------------------------------
# ConflictResolver.resolve_all
# ----------------------------------------------------------------------
def build_three_way_conflict():
    """AC002 (cruise) is caught between two higher-priority arrivals."""
    lead = flying("AC001", (0.0, 0.0), EAST, 10.0,
                  state=AircraftState.APPROACH)
    middle = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    return lead, middle


def test_resolve_all_reports_no_action_for_clear_aircraft():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0,
               state=AircraftState.APPROACH)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    c = flying("AC003", (0.0, 900.0), EAST, 10.0, state=AircraftState.CRUISE)
    conflicts = detect_all_conflicts([a, b, c])
    results = ConflictResolver().resolve_all(conflicts, current_tick=0)
    actions = [r.action for r in results]
    assert ResolutionAction.HOLD_LOWER_PRIORITY in actions
    assert ResolutionAction.NO_ACTION in actions


def test_resolve_all_holds_the_weaker_aircraft_in_every_real_conflict():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0,
               state=AircraftState.APPROACH)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    results = ConflictResolver().resolve_all(detect_all_conflicts([a, b]), 0)
    held = [r for r in results if r.action is ResolutionAction.HOLD_LOWER_PRIORITY]
    assert len(held) == 1
    assert held[0].held_aircraft is b


def test_resolve_all_processes_the_most_urgent_conflict_first():
    # Two independent head-on pairs with different closing times.
    a1 = flying("AC001", (0.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH)
    b1 = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    a2 = flying("AC003", (0.0, 500.0), EAST, 10.0, state=AircraftState.APPROACH)
    b2 = flying("AC004", (500.0, 500.0), WEST, 10.0, state=AircraftState.CRUISE)
    conflicts = detect_all_conflicts([a1, b1, a2, b2])
    resolver = ConflictResolver()
    results = resolver.resolve_all(conflicts, current_tick=0)
    held_pairs = [
        r.conflict for r in results
        if r.action is ResolutionAction.HOLD_LOWER_PRIORITY
    ]
    # The closer pair conflicts sooner, so it must be resolved first.
    first = held_pairs[0]
    second = held_pairs[-1]
    assert first.time_to_conflict <= second.time_to_conflict


def test_resolve_all_is_order_independent():
    a1 = flying("AC001", (0.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH)
    b1 = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    a2 = flying("AC003", (0.0, 500.0), EAST, 10.0, state=AircraftState.APPROACH)
    b2 = flying("AC004", (500.0, 500.0), WEST, 10.0, state=AircraftState.CRUISE)
    forward = detect_all_conflicts([a1, b1, a2, b2])
    resolver = ConflictResolver()
    resolver.resolve_all(forward, current_tick=4)
    expected = {ac.id: ac.holding_until for ac in (a1, b1, a2, b2)}

    a1 = flying("AC001", (0.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH)
    b1 = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    a2 = flying("AC003", (0.0, 500.0), EAST, 10.0, state=AircraftState.APPROACH)
    b2 = flying("AC004", (500.0, 500.0), WEST, 10.0, state=AircraftState.CRUISE)
    shuffled = detect_all_conflicts([a2, b2, a1, b1])
    resolver.resolve_all(shuffled, current_tick=4)
    assert {ac.id: ac.holding_until for ac in (a1, b1, a2, b2)} == expected


def test_resolve_all_accumulates_holds_on_a_multi_conflict_aircraft():
    # AC003 is caught by two higher-priority arrivals: one arriving head-on and
    # one already on top of it. AC001/AC002 fly together and never conflict.
    a1 = flying("AC001", (0.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH)
    a2 = flying("AC002", (100.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH)
    victim = flying("AC003", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    conflicts = [
        c for c in detect_all_conflicts([a1, a2, victim]) if c.conflict
    ]
    assert len(conflicts) == 2
    results = ConflictResolver().resolve_all(conflicts, current_tick=6)
    assert sum(
        1 for r in results if r.held_aircraft is victim
    ) == 2
    # Both instructions ask for the same deadline; the max() rule makes the
    # result independent of the order they were applied in.
    assert victim.holding_until == 11
    assert a1.holding_until is None
    assert a2.holding_until is None


def test_resolve_all_on_an_empty_list_is_empty():
    assert ConflictResolver().resolve_all([], current_tick=0) == []


def test_resolver_rejects_a_non_positive_hold_duration():
    with pytest.raises(ValueError, match="hold_duration"):
        ConflictResolver(hold_duration=0)
    with pytest.raises(ValueError, match="hold_duration"):
        ConflictResolver(hold_duration=-1)


def test_resolver_does_not_modify_the_detector_results():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0, state=AircraftState.APPROACH)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0, state=AircraftState.CRUISE)
    conflicts = detect_all_conflicts([a, b])
    before = [
        (c.conflict, c.time_to_conflict, c.aircraft_a.id, c.aircraft_b.id)
        for c in conflicts
    ]
    ConflictResolver().resolve_all(conflicts, current_tick=0)
    after = [
        (c.conflict, c.time_to_conflict, c.aircraft_a.id, c.aircraft_b.id)
        for c in conflicts
    ]
    assert before == after
