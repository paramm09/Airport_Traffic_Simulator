"""Phase 2B tests: discrete taxi movement along a precomputed Dijkstra route.

Rooted entirely in the public Graph API. The example route
``G1 -> T1 -> T2 -> T9 -> R3`` uses real airport edges of weights
1, 2, 4, 5 (totalling 12 distance units) and is written out explicitly so the
tests stay deterministic even though :func:`dijkstra` also finds a tying
12-unit route via T10.
"""

import pytest

from backend.algorithms.dijkstra import dijkstra
from backend.data.airport_graph import build_airport_graph
from backend.models.aircraft import Aircraft, AircraftState
from backend.simulation.taxi import TAXI_SPEED, TaxiStepOutcome, taxi_step

G1_TO_R3 = ["G1", "T1", "T2", "T9", "R3"]
G1_TO_T9 = ["G1", "T1", "T2", "T9"]


@pytest.fixture
def airport():
    return build_airport_graph()


def taxiing(route, aircraft_id="AC001") -> Aircraft:
    """A departure-ready aircraft holding a taxi route."""
    return Aircraft(
        id=aircraft_id,
        state=AircraftState.TAXIING_TO_RUNWAY,
        taxi_route=list(route),
    )


def taxi(aircraft, airport, steps, dt=1.0) -> list[TaxiStepOutcome]:
    """Run ``steps`` taxi ticks and collect every outcome."""
    return [taxi_step(aircraft, airport, dt) for _ in range(steps)]


def test_taxi_speed_is_one_constant():
    assert TAXI_SPEED == 1.0


def test_aircraft_starts_at_first_route_node():
    aircraft = taxiing(G1_TO_R3)
    assert aircraft.taxi_route[aircraft.route_index] == "G1"


def test_one_tick_travels_one_distance_unit(airport):
    aircraft = taxiing(G1_TO_T9)
    taxi(aircraft, airport, 1)
    assert aircraft.route_index == 1
    taxi(aircraft, airport, 1)
    # On T1-T2 (weight 2), one tick covers 1.0 of the 2.0 units.
    assert aircraft.distance_on_edge == 1.0
    assert aircraft.taxi_route[aircraft.route_index] == "T1"


def test_does_not_advance_before_the_edge_is_complete(airport):
    aircraft = taxiing(G1_TO_R3)
    taxi(aircraft, airport, 1, dt=0.5)
    assert aircraft.route_index == 0
    assert aircraft.distance_on_edge == 0.5


def test_edge_of_two_units_takes_two_ticks_to_cross(airport):
    aircraft = taxiing(G1_TO_R3)  # G1-T1 is weight 1, T1-T2 weight 2.
    taxi(aircraft, airport, 1)
    assert aircraft.route_index == 1  # T1
    taxi(aircraft, airport, 1)
    assert aircraft.route_index == 1  # still crossing T1-T2
    assert aircraft.distance_on_edge == 1.0
    taxi(aircraft, airport, 1)
    assert aircraft.route_index == 2  # now at T2


def test_route_index_increments_exactly_when_an_edge_is_completed(airport):
    aircraft = taxiing(G1_TO_R3)
    assert taxi(aircraft, airport, 1) == [TaxiStepOutcome.MOVED]
    assert aircraft.route_index == 1


def test_distance_resets_to_zero_after_arriving(airport):
    aircraft = taxiing(G1_TO_R3)
    taxi(aircraft, airport, 1)
    assert aircraft.route_index == 1
    assert aircraft.distance_on_edge == 0.0


def test_complete_route_reaches_the_final_node(airport):
    aircraft = taxiing(G1_TO_R3)
    taxi(aircraft, airport, 12)  # total route length is 12 units.
    assert aircraft.route_index == len(G1_TO_R3) - 1
    assert aircraft.taxi_route[aircraft.route_index] == "R3"


def test_reaching_the_final_node_transitions_to_waiting(airport):
    aircraft = taxiing(G1_TO_R3)
    outcomes = taxi(aircraft, airport, 12)
    assert outcomes[-1] is TaxiStepOutcome.ROUTE_COMPLETE
    assert aircraft.state is AircraftState.WAITING_FOR_RUNWAY


def test_route_is_never_skipped(airport):
    aircraft = taxiing(G1_TO_R3)
    visited: list[str] = [aircraft.taxi_route[aircraft.route_index]]
    for _ in range(12):
        taxi(aircraft, airport, 1)
        visited.append(aircraft.taxi_route[aircraft.route_index])
    # Collapse consecutive repeats into the distinct node sequence.
    distinct = [
        node for i, node in enumerate(visited) if i == 0 or node != visited[i - 1]
    ]
    assert distinct == G1_TO_R3


def test_overshooting_the_route_stops_at_the_final_node(airport):
    aircraft = taxiing(G1_TO_R3)
    outcome = taxi_step(aircraft, airport, dt=13.0)
    assert outcome is TaxiStepOutcome.ROUTE_COMPLETE
    assert aircraft.taxi_route[aircraft.route_index] == "R3"
    assert aircraft.distance_on_edge == 0.0


def test_single_node_route_is_immediately_complete(airport):
    aircraft = taxiing(["R3"])
    outcome = taxi_step(aircraft, airport)
    assert outcome is TaxiStepOutcome.ROUTE_COMPLETE
    assert aircraft.state is AircraftState.WAITING_FOR_RUNWAY


def test_blocked_next_edge_prevents_entry(airport):
    airport.block_edge("T1", "T2")
    aircraft = taxiing(G1_TO_R3)
    taxi(aircraft, airport, 1)  # arrive at T1.
    outcome = taxi_step(aircraft, airport)
    assert outcome is TaxiStepOutcome.BLOCKED
    assert aircraft.taxi_route[aircraft.route_index] == "T1"
    assert aircraft.distance_on_edge == 0.0
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY


def test_unblocking_resumes_a_stalled_aircraft(airport):
    airport.block_edge("T1", "T2")
    aircraft = taxiing(G1_TO_R3)
    taxi(aircraft, airport, 1)
    assert taxi_step(aircraft, airport) is TaxiStepOutcome.BLOCKED
    airport.unblock_edge("T1", "T2")
    assert taxi_step(aircraft, airport) is TaxiStepOutcome.MOVED
    assert aircraft.distance_on_edge == 1.0


def test_mid_edge_blockage_allows_finishing_the_current_edge(airport):
    # Route ends at T2, so finishing T1-T2 also completes the journey.
    short_route = ["G1", "T1", "T2"]
    aircraft = taxiing(short_route)
    taxi(aircraft, airport, 1)  # arrive at T1.
    taxi(aircraft, airport, 1)  # halfway across T1-T2 (weight 2).
    assert aircraft.distance_on_edge == 1.0
    airport.block_edge("T1", "T2")  # blocked now, while the aircraft is on it.
    outcome = taxi_step(aircraft, airport)
    assert outcome is TaxiStepOutcome.ROUTE_COMPLETE
    assert aircraft.taxi_route[aircraft.route_index] == "T2"
    assert aircraft.state is AircraftState.WAITING_FOR_RUNWAY


def test_mid_edge_blockage_does_not_stop_the_next_open_edge(airport):
    aircraft = taxiing(G1_TO_T9)
    taxi(aircraft, airport, 2)  # arrive at T1, then cross 1.0 of T1-T2.
    airport.block_edge("T1", "T2")
    taxi(aircraft, airport, 2)  # 1.0 finishes T1-T2 then waits at T9-free check.
    assert aircraft.taxi_route[aircraft.route_index] == "T2"
    assert aircraft.distance_on_edge == 1.0  # now crossing T2-T9.
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY


def test_dt_must_be_positive(airport):
    aircraft = taxiing(G1_TO_R3)
    with pytest.raises(ValueError, match="positive"):
        taxi_step(aircraft, airport, dt=0.0)
    with pytest.raises(ValueError, match="positive"):
        taxi_step(aircraft, airport, dt=-1.0)


def test_empty_route_is_rejected_safely(airport):
    aircraft = Aircraft(
        id="AC001", state=AircraftState.TAXIING_TO_RUNWAY, taxi_route=[]
    )
    with pytest.raises(ValueError, match="no taxi route"):
        taxi_step(aircraft, airport)


def test_wrong_state_does_not_move_the_aircraft(airport):
    aircraft = taxiing(G1_TO_R3)
    aircraft.state = AircraftState.AT_GATE
    with pytest.raises(ValueError, match="TAXIING_TO_RUNWAY"):
        taxi_step(aircraft, airport)
    assert aircraft.route_index == 0
    assert aircraft.distance_on_edge == 0.0


def test_leftover_movement_carries_into_the_next_edge(airport):
    aircraft = taxiing(G1_TO_T9)
    taxi(aircraft, airport, 1, dt=1.5)  # 1.0 finishes G1-T1, 0.5 enters T1-T2.
    assert aircraft.taxi_route[aircraft.route_index] == "T1"
    assert aircraft.distance_on_edge == 0.5
    taxi(aircraft, airport, 1)
    assert aircraft.distance_on_edge == 1.5


def test_a_large_dt_crosses_several_edges(airport):
    aircraft = taxiing(G1_TO_T9)
    taxi_step(aircraft, airport, dt=4.0)  # 1 + 2 = 3, leftover 1 into T2-T9.
    assert aircraft.taxi_route[aircraft.route_index] == "T2"
    assert aircraft.distance_on_edge == 1.0
    assert aircraft.taxi_route[aircraft.route_index + 1] == "T9"


def test_fractional_steps_accumulate_exactly(airport):
    aircraft = taxiing(G1_TO_R3)
    taxi(aircraft, airport, 4, dt=0.25)  # four quarter ticks = one full unit.
    assert aircraft.taxi_route[aircraft.route_index] == "T1"
    assert aircraft.distance_on_edge == 0.0


def test_dijkstra_route_from_a_gate_to_a_runway_is_followed():
    airport = build_airport_graph()
    path, cost = dijkstra(airport, "G2", "R2")
    assert (path, cost) == (["G2", "T2", "T3", "T4", "R2"], 11)
    aircraft = taxiing(path)
    outcomes = taxi(aircraft, airport, 11)
    assert outcomes[-1] is TaxiStepOutcome.ROUTE_COMPLETE
    assert aircraft.taxi_route[aircraft.route_index] == "R2"
    assert aircraft.state is AircraftState.WAITING_FOR_RUNWAY


def test_taxiing_aircraft_keeps_its_position_in_the_fleet(airport):
    first = taxiing(G1_TO_R3, aircraft_id="AC001")
    second = taxiing(G1_TO_R3, aircraft_id="AC002")
    taxi_step(first, airport, dt=3.0)  # 1 + 2, leftover 0 into T2-T9.
    assert first.taxi_route[first.route_index] == "T2"
    assert first.distance_on_edge == 0.0
    assert second.route_index == 0
    assert second.distance_on_edge == 0.0
    assert first.id == "AC001"
    assert second.id == "AC002"