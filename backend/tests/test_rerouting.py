"""Phase 2G tests: dynamic rerouting around a blocked taxi edge."""

import math

import pytest

from backend.algorithms.dijkstra import dijkstra
from backend.data.airport_graph import build_airport_graph
from backend.models.aircraft import Aircraft, AircraftState
from backend.models.graph import Graph
from backend.simulation.rerouting import (
    TAXIING_STATES,
    RerouteResult,
    current_node,
    is_blocked,
    reroute_if_blocked,
    resume_state,
    try_reroute,
)


def taxiing(aircraft_id="AC001", route=None, destination="R1",
            route_index=0, distance_on_edge=0.0, edge_length=0.0,
            state=AircraftState.TAXIING_TO_RUNWAY):
    """A taxiing aircraft with an explicit route, so reroutes are observable."""
    return Aircraft(
        id=aircraft_id,
        state=state,
        taxi_route=list(route if route is not None else ["G1", "T1", "R1"]),
        route_index=route_index,
        destination=destination,
        distance_on_edge=distance_on_edge,
        edge_length=edge_length,
    )


def simple_graph() -> Graph:
    """A network with a direct path G1-T1-T2-G2 and detours via T3.

    T1-T2 and T2-G2 are the "direct" edges; T1-T3-R1 and T3-G2 are the
    detours, so blocking either direct edge still leaves a way through.
    """
    graph = Graph()
    for node in ("G1", "G2", "T1", "T2", "T3", "R1"):
        graph.add_node(node)
    graph.add_edge("G1", "T1", 1)
    graph.add_edge("T1", "T2", 1)
    graph.add_edge("T2", "G2", 1)
    graph.add_edge("T1", "T3", 5)
    graph.add_edge("T3", "R1", 1)
    graph.add_edge("T3", "G2", 2)
    return graph


# ----------------------------------------------------------------------
# current_node()
# ----------------------------------------------------------------------
def test_current_node_is_the_route_index_node():
    aircraft = taxiing(route=["G1", "T1", "R1"], route_index=1)
    assert current_node(aircraft) == "T1"


def test_current_node_is_none_without_a_route():
    aircraft = taxiing(route=[])
    assert current_node(aircraft) is None


def test_current_node_is_none_when_the_index_is_past_the_route():
    aircraft = taxiing(route=["G1", "T1", "R1"], route_index=3)
    assert current_node(aircraft) is None


# ----------------------------------------------------------------------
# is_blocked()
# ----------------------------------------------------------------------
def test_is_not_blocked_when_the_next_edge_is_open():
    graph = simple_graph()
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="G2")
    assert is_blocked(aircraft, graph) is False


def test_is_blocked_when_the_next_edge_is_blocked():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    # Standing on T1 with T2 next: that is the edge that is blocked.
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="G2",
                       route_index=1)
    assert is_blocked(aircraft, graph) is True


def test_a_blocked_edge_further_ahead_is_not_yet_a_blockage():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    # Still standing on G1 heading for open G1-T1, so nothing is blocking it
    # yet: the aircraft must reach T1 first.
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="G2",
                       route_index=0)
    assert is_blocked(aircraft, graph) is False


def test_a_mid_edge_aircraft_is_never_reported_as_blocked():
    # Mid-edge the aircraft must finish the crossing, even if the edge it is
    # on has been blocked behind it.
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = taxiing(
        route=["G1", "T1", "T2", "G2"],
        destination="G2",
        route_index=0,
        distance_on_edge=0.5,
        edge_length=1.0,
    )
    assert is_blocked(aircraft, graph) is False


def test_a_completed_route_is_not_blocked():
    graph = simple_graph()
    aircraft = taxiing(route=["G1", "T1"], destination="T1", route_index=1)
    assert is_blocked(aircraft, graph) is False


def test_an_airborne_aircraft_is_not_blocked():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = Aircraft(id="AC001", state=AircraftState.CRUISE)
    assert is_blocked(aircraft, graph) is False


def test_a_holding_aircraft_is_not_blocked():
    graph = simple_graph()
    aircraft = taxiing(route=["G1"], destination="R1")
    aircraft.transition_to(AircraftState.HOLDING_NO_ROUTE)
    assert is_blocked(aircraft, graph) is False


def test_taxiing_states_cover_both_directions():
    assert TAXIING_STATES == {
        AircraftState.TAXIING_TO_RUNWAY,
        AircraftState.TAXIING_TO_GATE,
    }


# ----------------------------------------------------------------------
# try_reroute(): a new path exists
# ----------------------------------------------------------------------
def test_reroute_installs_a_new_path_avoiding_the_blocked_edge():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="R1")
    attempt = try_reroute(aircraft, graph)
    assert attempt.result is RerouteResult.REROUTED
    assert attempt.rerouted is True
    assert aircraft.taxi_route == ["G1", "T1", "T3", "R1"]
    assert attempt.destination == "R1"


def test_reroute_resets_every_progress_field():
    graph = simple_graph()
    aircraft = taxiing(
        route=["G1", "T1", "T2", "G2"],
        destination="R1",
        route_index=1,
        distance_on_edge=0.0,
        edge_length=0.0,
    )
    graph.block_edge("T1", "T2")
    try_reroute(aircraft, graph)
    assert aircraft.route_index == 0
    assert aircraft.distance_on_edge == 0.0
    assert aircraft.edge_length == 0.0


def test_reroute_never_changes_the_destination():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="R1")
    try_reroute(aircraft, graph)
    assert aircraft.destination == "R1"


def test_rerouted_route_actually_avoids_the_blocked_edge():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="R1",
                       route_index=1)
    attempt = try_reroute(aircraft, graph)
    for node1, node2 in zip(aircraft.taxi_route, aircraft.taxi_route[1:]):
        assert graph.has_edge(node1, node2), f"{node1}->{node2} is blocked"
    assert attempt.new_route[0] == "T1"
    assert aircraft.taxi_route[-1] == "R1"


def test_reroute_starts_from_the_node_the_aircraft_reached():
    graph = simple_graph()
    graph.block_edge("T2", "G2")
    aircraft = taxiing(
        route=["G1", "T1", "T2", "G2"],
        destination="R1",
        route_index=2,  # has reached T2
    )
    attempt = try_reroute(aircraft, graph)
    assert attempt.start == "T2"
    assert attempt.new_route[0] == "T2"


def test_reroute_reports_the_dijkstra_cost():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="R1",
                       route_index=1)
    attempt = try_reroute(aircraft, graph)
    expected_path, expected_cost = dijkstra(graph, "T1", "R1")
    assert attempt.cost == pytest.approx(expected_cost)
    assert attempt.new_route == tuple(expected_path)


def test_reroute_does_not_teleport_a_mid_edge_aircraft():
    """A mid-edge aircraft is never rerouted by reroute_if_blocked."""
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = taxiing(
        route=["G1", "T1", "T2", "G2"],
        destination="R1",
        route_index=0,
        distance_on_edge=0.5,
        edge_length=1.0,
    )
    before = list(aircraft.taxi_route)
    attempt = reroute_if_blocked(aircraft, graph)
    assert attempt.result is RerouteResult.NOT_NEEDED
    assert aircraft.taxi_route == before
    assert aircraft.distance_on_edge == 0.5


def test_reroute_preserves_the_aircraft_state():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = taxiing(
        route=["G1", "T1", "T2", "G2"],
        destination="R1",
        state=AircraftState.TAXIING_TO_GATE,
    )
    try_reroute(aircraft, graph)
    assert aircraft.state is AircraftState.TAXIING_TO_GATE


def test_reroute_works_for_arrivals_too():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    # An arriving aircraft on its way to a gate, blocked on T1 -> T2.
    aircraft = taxiing(
        route=["T1", "T2", "G2"],
        destination="G2",
        route_index=0,
        state=AircraftState.TAXIING_TO_GATE,
    )
    attempt = try_reroute(aircraft, graph)
    assert attempt.result is RerouteResult.REROUTED
    assert aircraft.taxi_route == ["T1", "T3", "G2"]
    # Crucially it stays in the arrival pipeline: finding a route is not a
    # reason to enter HOLDING_NO_ROUTE at all.
    assert aircraft.state is AircraftState.TAXIING_TO_GATE


def test_reroute_to_the_node_the_aircraft_is_already_on():
    graph = simple_graph()
    aircraft = taxiing(route=["G1", "T1"], destination="T1", route_index=1)
    attempt = try_reroute(aircraft, graph)
    assert attempt.result is RerouteResult.REROUTED
    assert aircraft.taxi_route == ["T1"]
    assert attempt.cost == pytest.approx(0.0)


def test_reroute_rejects_an_aircraft_with_no_destination():
    graph = simple_graph()
    aircraft = taxiing(route=["G1", "T1"], destination=None)
    with pytest.raises(ValueError, match="destination"):
        try_reroute(aircraft, graph)


# ----------------------------------------------------------------------
# try_reroute(): no path exists -> HOLDING_NO_ROUTE
# ----------------------------------------------------------------------
def isolated_graph() -> Graph:
    """G1, T2 and R1 all exist but nothing connects them."""
    graph = Graph()
    for node in ("G1", "T2", "R1"):
        graph.add_node(node)
    return graph


def test_no_path_puts_a_taxiing_departure_into_holding_no_route():
    graph = isolated_graph()
    aircraft = taxiing(route=["G1"], destination="R1")
    attempt = try_reroute(aircraft, graph)
    assert attempt.result is RerouteResult.NO_PATH
    assert attempt.rerouted is False
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE


def test_no_path_does_not_invent_a_route():
    graph = isolated_graph()
    aircraft = taxiing(route=["G1"], destination="R1")
    try_reroute(aircraft, graph)
    assert aircraft.taxi_route == ["G1"]


def test_no_path_puts_a_taxiing_arrival_into_holding_no_route():
    graph = isolated_graph()
    aircraft = taxiing(
        route=["T2"], destination="G1", state=AircraftState.TAXIING_TO_GATE
    )
    attempt = try_reroute(aircraft, graph)
    assert attempt.result is RerouteResult.NO_PATH
    # An arrival has no separate spare state, so it holds alongside departures
    # and records that it must resume towards the gate, not a runway.
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert aircraft.holding_from is AircraftState.TAXIING_TO_GATE


def test_no_path_records_a_departures_own_state_for_resuming():
    graph = isolated_graph()
    aircraft = taxiing(route=["G1"], destination="R1")
    try_reroute(aircraft, graph)
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert aircraft.holding_from is AircraftState.TAXIING_TO_RUNWAY


def test_no_path_when_every_route_to_the_destination_is_blocked():
    graph = simple_graph()
    for node1, node2 in (("T1", "T3"), ("T3", "R1")):
        graph.block_edge(node1, node2)
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="R1")
    attempt = try_reroute(aircraft, graph)
    assert attempt.result is RerouteResult.NO_PATH
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE


# ----------------------------------------------------------------------
# resume_state()
# ----------------------------------------------------------------------
def test_resume_state_of_a_holding_departure_is_taxiing_to_runway():
    # Built by really running out of route, so this covers the recording too.
    graph = isolated_graph()
    aircraft = taxiing(route=["G1"], destination="R1")
    assert try_reroute(aircraft, graph).result is RerouteResult.NO_PATH
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE
    assert resume_state(aircraft) is AircraftState.TAXIING_TO_RUNWAY


def test_resume_state_of_a_holding_arrival_is_taxiing_to_gate():
    # The same state, the other way round: only the recorded taxi state
    # distinguishes the two.
    aircraft = Aircraft(
        id="AC001",
        state=AircraftState.HOLDING_NO_ROUTE,
        holding_from=AircraftState.TAXIING_TO_GATE,
    )
    assert resume_state(aircraft) is AircraftState.TAXIING_TO_GATE


def test_resume_state_of_a_taxiing_arrival_is_itself():
    aircraft = Aircraft(id="AC001", state=AircraftState.TAXIING_TO_GATE)
    assert resume_state(aircraft) is AircraftState.TAXIING_TO_GATE


def test_resume_state_is_none_for_an_unrelated_state():
    aircraft = Aircraft(id="AC001", state=AircraftState.CRUISE)
    assert resume_state(aircraft) is None


# ----------------------------------------------------------------------
# Retry after an unblock
# ----------------------------------------------------------------------
def test_retry_after_unblock_recovers_the_departure():
    graph = simple_graph()
    graph.block_edge("T1", "T3")
    graph.block_edge("T3", "R1")
    graph.block_edge("T3", "G2")
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="R1")
    assert try_reroute(aircraft, graph).result is RerouteResult.NO_PATH
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE

    # The network improves; the caller retries explicitly.
    graph.unblock_edge("T1", "T3")
    graph.unblock_edge("T3", "R1")
    attempt = try_reroute(aircraft, graph)
    assert attempt.result is RerouteResult.REROUTED
    assert aircraft.taxi_route == ["G1", "T1", "T3", "R1"]
    # The aircraft resumes the lifecycle where it left off.
    aircraft.transition_to(resume_state(aircraft))
    assert aircraft.state is AircraftState.TAXIING_TO_RUNWAY


def test_a_holding_aircraft_is_not_retried_every_tick():
    graph = isolated_graph()
    aircraft = taxiing(route=["G1"], destination="R1")
    aircraft.transition_to(AircraftState.HOLDING_NO_ROUTE)
    for _ in range(5):
        attempt = reroute_if_blocked(aircraft, graph)
        assert attempt.result is RerouteResult.NOT_NEEDED
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE


def test_retry_after_unblock_recovers_the_arrival_to_its_gate():
    graph = simple_graph()
    # G2 hangs off T2 and T3, so closing both strands the gate completely.
    for node1, node2 in (("T2", "G2"), ("T3", "G2")):
        graph.block_edge(node1, node2)
    aircraft = taxiing(
        route=["G1", "T1", "T2", "G2"],
        destination="G2",
        state=AircraftState.TAXIING_TO_GATE,
    )
    assert try_reroute(aircraft, graph).result is RerouteResult.NO_PATH
    assert aircraft.state is AircraftState.HOLDING_NO_ROUTE

    graph.unblock_edge("T3", "G2")  # the detour via T3 reopens
    attempt = try_reroute(aircraft, graph)
    assert attempt.result is RerouteResult.REROUTED
    assert aircraft.taxi_route == ["G1", "T1", "T3", "G2"]
    # It resumes the arrival pipeline, so the next thing that can happen to it
    # is reaching the gate and completing.
    aircraft.transition_to(resume_state(aircraft))
    assert aircraft.state is AircraftState.TAXIING_TO_GATE
    assert aircraft.holding_from is None
    aircraft.transition_to(AircraftState.COMPLETED)
    assert aircraft.state is AircraftState.COMPLETED


# ----------------------------------------------------------------------
# reroute_if_blocked()
# ----------------------------------------------------------------------
def test_reroute_if_blocked_reroutes_a_blocked_aircraft():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = taxiing(route=["G1", "T1", "T2", "G2"], destination="R1",
                       route_index=1)
    assert reroute_if_blocked(aircraft, graph).result is RerouteResult.REROUTED
    # The new route starts at the node the aircraft is actually standing on,
    # so the already-travelled G1 -> T1 leg is not replayed.
    assert aircraft.taxi_route == ["T1", "T3", "R1"]
    assert aircraft.route_index == 0


def test_reroute_if_blocked_does_nothing_when_clear():
    graph = simple_graph()
    aircraft = taxiing(route=["G1", "T1", "R1"], destination="R1")
    attempt = reroute_if_blocked(aircraft, graph)
    assert attempt.result is RerouteResult.NOT_NEEDED
    assert aircraft.taxi_route == ["G1", "T1", "R1"]


def test_reroute_if_blocked_does_nothing_for_an_airborne_aircraft():
    graph = simple_graph()
    graph.block_edge("T1", "T2")
    aircraft = Aircraft(id="AC001", state=AircraftState.CRUISE)
    assert (
        reroute_if_blocked(aircraft, graph).result is RerouteResult.NOT_NEEDED
    )


# ----------------------------------------------------------------------
# Works on the real airport network
# ----------------------------------------------------------------------
def test_reroute_on_the_real_airport_graph_avoids_a_blocked_taxiway():
    graph = build_airport_graph()
    # The aircraft is standing on T1 and needs to go on to T2, which is blocked.
    aircraft = taxiing(route=["G1", "T1", "T2", "R4"], destination="R1",
                       route_index=1)
    graph.block_edge("T1", "T2")
    assert is_blocked(aircraft, graph) is True
    attempt = reroute_if_blocked(aircraft, graph)
    assert attempt.result is RerouteResult.REROUTED
    for node1, node2 in zip(aircraft.taxi_route, aircraft.taxi_route[1:]):
        assert graph.has_edge(node1, node2)
    assert aircraft.taxi_route[0] == "T1"
    assert aircraft.taxi_route[-1] == "R1"


def test_reroute_on_the_real_graph_cuts_cost_when_the_shortcut_is_blocked():
    graph = build_airport_graph()
    # Without the blockage the cheapest way from G1 to R1 is G1-T1-R1 (cost 7).
    aircraft = taxiing(route=["G1", "T1", "R1"], destination="R1")
    assert dijkstra(graph, "G1", "R1")[1] == pytest.approx(7.0)
    assert is_blocked(aircraft, graph) is False

    graph.block_edge("T1", "R1")
    # The aircraft is standing on T1 and R1 is its next (blocked) node.
    aircraft = taxiing(route=["G1", "T1", "R1"], destination="R1",
                       route_index=1)
    assert is_blocked(aircraft, graph) is True
    attempt = reroute_if_blocked(aircraft, graph)
    assert attempt.result is RerouteResult.REROUTED
    assert ("T1", "R1") not in list(
        zip(attempt.new_route, attempt.new_route[1:])
    )
    assert attempt.new_route[0] == "T1"
    # The detour is strictly more expensive than the blocked direct route.
    assert attempt.cost > 5.0
    assert math.isfinite(attempt.cost)


def test_reroute_is_deterministic():
    graph = build_airport_graph()
    graph.block_edge("T1", "R1")
    routes = []
    for _ in range(3):
        aircraft = taxiing(route=["G1", "T1", "R1"], destination="R1",
                           route_index=1)
        reroute_if_blocked(aircraft, graph)
        routes.append(tuple(aircraft.taxi_route))
    assert routes[0] == routes[1] == routes[2]
