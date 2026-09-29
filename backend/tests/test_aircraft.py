"""Unit tests for the Aircraft model and its state machine (Phase 2A).

Only the aircraft data model, the AircraftState enum and the legal transition
table are tested here. Routing, runway scheduling, conflict detection and the
simulation clock come in later phases and are deliberately not touched.
"""

import pytest

from backend.models.aircraft import (
    Aircraft,
    AircraftState,
    IllegalTransitionError,
)

# The full legal lifecycle, in order.
FE = AircraftState  # short alias used below


def test_aircraft_initializes_in_at_gate() -> None:
    aircraft = Aircraft(id="AC001")
    assert aircraft.state is AircraftState.AT_GATE


def test_aircraft_id_is_stored() -> None:
    aircraft = Aircraft(id="AC001")
    assert aircraft.id == "AC001"


def test_default_values_are_correct() -> None:
    aircraft = Aircraft(id="AC001")
    assert aircraft.taxi_route == []
    assert aircraft.route_index == 0
    assert aircraft.destination is None
    assert aircraft.distance_on_edge == 0.0
    assert aircraft.edge_length == 0.0
    assert aircraft.request_time == 0
    assert aircraft.holding_until is None
    assert aircraft.position == (0.0, 0.0)
    assert aircraft.altitude == 0.0
    assert aircraft.speed == 0.0
    assert aircraft.heading == 0.0
    assert aircraft.next_waypoint is None


def test_valid_transition_at_gate_to_taxiing() -> None:
    aircraft = Aircraft(id="AC001")
    aircraft.transition_to(AircraftState.TAXIING_TO_RUNWAY)
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY


def test_full_valid_lifecycle_chain() -> None:
    aircraft = Aircraft(id="AC001")
    for state in [
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
        AircraftState.COMPLETED,
    ]:
        aircraft.transition_to(state)
    assert aircraft.state is AircraftState.COMPLETED


def test_invalid_transition_at_gate_to_takeoff_raises() -> None:
    aircraft = Aircraft(id="AC001")
    with pytest.raises(IllegalTransitionError):
        aircraft.transition_to(AircraftState.TAKEOFF)
    # The failed attempt must not have moved the aircraft.
    assert aircraft.state is AircraftState.AT_GATE


def test_invalid_transition_completed_to_taxiing_raises() -> None:
    aircraft = Aircraft(id="AC001")
    aircraft.state = AircraftState.COMPLETED
    with pytest.raises(IllegalTransitionError):
        aircraft.transition_to(AircraftState.TAXIING_TO_RUNWAY)


def test_invalid_transition_skips_states() -> None:
    # AT_GATE -> WAITING_FOR_RUNWAY skips TAXIING_TO_RUNWAY and must fail.
    aircraft = Aircraft(id="AC001")
    with pytest.raises(IllegalTransitionError):
        aircraft.transition_to(AircraftState.WAITING_FOR_RUNWAY)


def test_negative_speed_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="AC001", speed=-1.0)


def test_negative_altitude_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="AC001", altitude=-100.0)


def test_negative_route_index_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="AC001", route_index=-1)


def test_negative_distance_on_edge_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="AC001", distance_on_edge=-0.5)


def test_negative_edge_length_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="AC001", edge_length=-1.0)


def test_two_aircraft_keep_independent_data() -> None:
    first = Aircraft(id="AC001")
    second = Aircraft(id="AC002")

    first.transition_to(AircraftState.TAXIING_TO_RUNWAY)
    first.taxi_route = ["G1", "T1", "R1"]
    first.request_time = 5
    first.speed = 10.0

    assert second.state is AircraftState.AT_GATE
    assert second.taxi_route == []
    assert second.request_time == 0
    assert second.speed == 0.0
    assert first.id == "AC001"
    assert second.id == "AC002"


def test_empty_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="")
    with pytest.raises(ValueError):
        Aircraft(id=None)  # type: ignore[arg-type]


def test_invalid_state_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="AC001", state="TAXIING_TO_RUNWAY")  # type: ignore[arg-type]


def test_non_integer_holding_until_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="AC001", holding_until=5.5)


def test_holding_until_accepts_an_integer_tick() -> None:
    aircraft = Aircraft(id="AC001", holding_until=42)
    assert aircraft.holding_until == 42


def test_negative_request_time_is_rejected() -> None:
    with pytest.raises(ValueError):
        Aircraft(id="AC001", request_time=-1)


def test_can_transition_to_reports_legality() -> None:
    aircraft = Aircraft(id="AC001")
    assert aircraft.can_transition_to(AircraftState.TAXIING_TO_RUNWAY) is True
    assert aircraft.can_transition_to(AircraftState.CRUISE) is False
    assert aircraft.can_transition_to(AircraftState.TAXIING_TO_RUNWAY) is True


def test_completed_state_has_no_outgoing_transitions() -> None:
    aircraft = Aircraft(id="AC001", state=AircraftState.COMPLETED)
    for state in AircraftState:
        assert aircraft.can_transition_to(state) is False


def test_every_state_is_reachable_in_lifecycle_order() -> None:
    # The enum definition order must match the lifecycle chain, so that
    # walking list(AircraftState) exercises every legal transition.
    # HOLDING_NO_ROUTE is appended after COMPLETED because it is a side state
    # rather than a link in the chain, so the chain is walked up to COMPLETED.
    chain = [
        state
        for state in AircraftState
        if state is not AircraftState.HOLDING_NO_ROUTE
    ]
    aircraft = Aircraft(id="AC001")
    for state in chain[1:]:
        aircraft.transition_to(state)
    assert aircraft.state is AircraftState.COMPLETED


def test_holding_no_route_is_a_side_state_off_the_chain() -> None:
    # It is reachable from TAXIING_TO_RUNWAY and returns there, but it is not
    # the next state in the straight lifecycle.
    taxiing = Aircraft(id="AC001", state=AircraftState.TAXIING_TO_RUNWAY)
    assert taxiing.can_transition_to(AircraftState.HOLDING_NO_ROUTE) is True
    taxiing.transition_to(AircraftState.HOLDING_NO_ROUTE)
    assert taxiing.can_transition_to(AircraftState.TAXIING_TO_RUNWAY) is True
    taxiing.transition_to(AircraftState.TAXIING_TO_RUNWAY)
    # ... and it does not interrupt the normal chain: walking every state in
    # order still ends at COMPLETED.
    assert list(AircraftState)[-1] is AircraftState.HOLDING_NO_ROUTE


def test_holding_no_route_is_reachable_from_arrivals_too() -> None:
    # An arrival whose gate is cut off is in exactly the same position as a
    # departure whose runway is cut off, so it holds too.
    arriving = Aircraft(id="AC001", state=AircraftState.TAXIING_TO_GATE)
    assert arriving.can_transition_to(AircraftState.HOLDING_NO_ROUTE) is True
    arriving.transition_to(AircraftState.HOLDING_NO_ROUTE)
    # Only the arrival side of the lifecycle is left, so it can return to the
    # gate and then complete.
    assert arriving.can_transition_to(AircraftState.TAXIING_TO_GATE) is True
    assert arriving.can_transition_to(AircraftState.TAXIING_TO_RUNWAY) is True
    arriving.transition_to(AircraftState.TAXIING_TO_GATE)
    arriving.transition_to(AircraftState.COMPLETED)
    assert arriving.state is AircraftState.COMPLETED


def test_entering_holding_no_route_records_where_to_resume() -> None:
    for state in (AircraftState.TAXIING_TO_RUNWAY, AircraftState.TAXIING_TO_GATE):
        aircraft = Aircraft(id="AC001", state=state)
        assert aircraft.holding_from is None
        aircraft.transition_to(AircraftState.HOLDING_NO_ROUTE)
        assert aircraft.holding_from is state


def test_the_resume_record_is_cleared_on_leaving_holding_no_route() -> None:
    # The record is derived from the transition, not a parallel piece of state
    # that has to be kept in step by hand.
    aircraft = Aircraft(
        id="AC001",
        state=AircraftState.TAXIING_TO_GATE,
        route_index=1,
    )
    aircraft.transition_to(AircraftState.HOLDING_NO_ROUTE)
    aircraft.transition_to(AircraftState.TAXIING_TO_GATE)
    assert aircraft.holding_from is None
    aircraft.transition_to(AircraftState.COMPLETED)
    assert aircraft.holding_from is None


def test_a_resume_record_must_be_a_state() -> None:
    with pytest.raises(ValueError, match="holding_from"):
        Aircraft(
            id="AC001",
            state=AircraftState.HOLDING_NO_ROUTE,
            holding_from="TAXIING_TO_GATE",  # type: ignore[arg-type]
        )


def test_a_resume_record_without_holding_is_rejected() -> None:
    with pytest.raises(ValueError, match="holding_from"):
        Aircraft(
            id="AC001",
            state=AircraftState.TAXIING_TO_GATE,
            holding_from=AircraftState.TAXIING_TO_RUNWAY,
        )


def test_a_holding_aircraft_must_record_a_resumable_state() -> None:
    with pytest.raises(ValueError, match="holding_from"):
        Aircraft(
            id="AC001",
            state=AircraftState.HOLDING_NO_ROUTE,
            holding_from=AircraftState.CRUISE,
        )
    with pytest.raises(ValueError, match="holding_from"):
        Aircraft(id="AC001", state=AircraftState.HOLDING_NO_ROUTE)


def test_landing_aircraft_cannot_enter_holding_no_route() -> None:
    # HOLDING_NO_ROUTE only exists for aircraft that are taxiing; a landing has
    # already been allocated the runway and must not be diverted.
    landing = Aircraft(id="AC001", state=AircraftState.LANDING)
    assert landing.can_transition_to(AircraftState.HOLDING_NO_ROUTE) is False
