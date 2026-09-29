"""Unit tests for Dijkstra's shortest-path algorithm.

The expected values used here were calculated directly from
``backend/data/airport_graph.py`` by enumerating *every* simple path and taking
the cheapest one, then cross-checked against the implementation. They are not
guessed.
"""

import math

import pytest

from backend.algorithms.dijkstra import dijkstra
from backend.data.airport_graph import build_airport_graph
from backend.models.graph import Graph


@pytest.fixture
def small_graph() -> Graph:
    """A small open network used by the non-airport tests.

        A --2-- B --3-- C
        |      /
        5     4
        |    /
        D
    """
    graph = Graph()
    for node in ("A", "B", "C", "D"):
        graph.add_node(node)
    graph.add_edge("A", "B", 2)
    graph.add_edge("B", "C", 3)
    graph.add_edge("A", "D", 5)
    graph.add_edge("B", "D", 4)
    return graph


@pytest.fixture
def unreachable_graph() -> Graph:
    """Two disconnected components: A--B and C--D."""
    graph = Graph()
    for node in ("A", "B", "C", "D"):
        graph.add_node(node)
    graph.add_edge("A", "B", 1)
    graph.add_edge("C", "D", 1)
    return graph


def path_cost(graph: Graph, path: list[str]) -> float:
    """Re-add the weights along ``path`` to prove it is a real route.

    Also fails if two consecutive nodes are not actually connected, so it
    catches a bogus path that happens to sum to the right number.
    """
    if not path:
        return math.inf
    total: float = 0
    for node1, node2 in zip(path, path[1:]):
        total += graph.get_neighbors(node1)[node2]
    return total


# ----------------------------------------------------------------------
# 1. / 2. Basic shortest path and correct total distance
# ----------------------------------------------------------------------
def test_shortest_path_from_a_to_c(small_graph):
    path, cost = dijkstra(small_graph, "A", "C")
    # A->B->C costs 5, which beats A->D (dead end) and any detour.
    assert path == ["A", "B", "C"]
    assert cost == 5


def test_direct_cheap_edge_beats_a_multi_hop_detour(small_graph):
    # A->D costs 5, while detouring A->B->D costs 2 + 4 = 6.
    path, cost = dijkstra(small_graph, "A", "D")
    assert path == ["A", "D"]
    assert cost == 5


def test_multi_hop_route_wins_when_the_detour_is_cheaper():
    # A->D is 10, but A->B->D is only 2 + 4 = 6, so more hops is cheaper.
    graph = Graph()
    for node in ("A", "B", "D"):
        graph.add_node(node)
    graph.add_edge("A", "B", 2)
    graph.add_edge("B", "D", 4)
    graph.add_edge("A", "D", 10)
    assert dijkstra(graph, "A", "D") == (["A", "B", "D"], 6)


def test_reported_cost_matches_the_weights_along_the_path(small_graph):
    path, cost = dijkstra(small_graph, "A", "C")
    assert path_cost(small_graph, path) == cost


def test_graph_is_not_modified_by_searching(small_graph):
    dijkstra(small_graph, "A", "C")
    assert small_graph.get_neighbors("A") == {"B": 2, "D": 5}
    assert small_graph.get_nodes() == ["A", "B", "C", "D"]


def test_path_is_reversible_because_the_graph_is_undirected(small_graph):
    forward, forward_cost = dijkstra(small_graph, "A", "C")
    backward, backward_cost = dijkstra(small_graph, "C", "A")
    assert backward == forward[::-1]
    assert backward_cost == forward_cost


# ----------------------------------------------------------------------
# 3. start == destination
# ----------------------------------------------------------------------
def test_start_equals_destination_returns_single_node_path(small_graph):
    assert dijkstra(small_graph, "A", "A") == (["A"], 0)


def test_start_equals_destination_on_the_airport():
    assert dijkstra(build_airport_graph(), "T4", "T4") == (["T4"], 0)


# ----------------------------------------------------------------------
# 4. Unreachable destination
# ----------------------------------------------------------------------
def test_unreachable_destination_returns_empty_path(unreachable_graph):
    path, cost = dijkstra(unreachable_graph, "A", "D")
    assert path == []
    assert cost == math.inf


def test_unreachable_in_the_other_direction(unreachable_graph):
    assert dijkstra(unreachable_graph, "D", "A") == ([], math.inf)


# ----------------------------------------------------------------------
# 5. A blocked edge causes a different route
# ----------------------------------------------------------------------
def test_blocked_edge_forces_a_different_route():
    # Unblocked, G1->R1 is 1 + 6 = 7 through the direct T1->R1 access.
    airport = build_airport_graph()
    assert dijkstra(airport, "G1", "R1") == (["G1", "T1", "R1"], 7)

    # With T1->R1 closed the aircraft must taxi round via T3: 1 + 2 + 2 + 5 = 10.
    airport.block_edge("T1", "R1")
    path, cost = dijkstra(airport, "G1", "R1")
    assert path == ["G1", "T1", "T2", "T3", "R1"]
    assert cost == 10


def test_second_blocked_edge_changes_the_route_again():
    airport = build_airport_graph()
    airport.block_edge("T1", "R1")
    airport.block_edge("T2", "T3")
    path, cost = dijkstra(airport, "G1", "R1")
    assert cost == 19
    assert path_cost(airport, path) == 19


# ----------------------------------------------------------------------
# 6. Blocking enough edges makes the destination unreachable
# ----------------------------------------------------------------------
def test_blocking_every_runway_access_makes_r1_unreachable():
    airport = build_airport_graph()
    airport.block_edge("T1", "R1")
    airport.block_edge("T3", "R1")
    assert dijkstra(airport, "G1", "R1") == ([], math.inf)


def test_blocking_a_gate_makes_its_gate_unreachable():
    airport = build_airport_graph()
    airport.block_edge("G5", "T5")
    assert dijkstra(airport, "G5", "R1") == ([], math.inf)
    # Other gates are unaffected, so the graph is only partly disconnected.
    assert dijkstra(airport, "G6", "R2")[1] == 7


# ----------------------------------------------------------------------
# 7. / 8. Unknown start or destination
# ----------------------------------------------------------------------
def test_unknown_start_node_raises(small_graph):
    with pytest.raises(KeyError):
        dijkstra(small_graph, "Z9", "C")


def test_unknown_destination_node_raises(small_graph):
    with pytest.raises(KeyError):
        dijkstra(small_graph, "A", "Z9")


def test_unknown_nodes_on_the_airport():
    airport = build_airport_graph()
    with pytest.raises(KeyError):
        dijkstra(airport, "G99", "R1")
    with pytest.raises(KeyError):
        dijkstra(airport, "G1", "R99")


# ----------------------------------------------------------------------
# 9. Lazy deletion of stale heap entries
# ----------------------------------------------------------------------
@pytest.fixture
def stale_entry_graph() -> Graph:
    """A network that forces a stale entry to be popped and skipped.

    Edges: A-B 10, A-C 3, C-B 4, C-D 15, B-E 9.

    B is first reached at cost 10 through A, then improved to 7 through C, so
    the heap ends up holding both (10, B) and (7, B). The stale (10, B) entry
    is popped *before* the destination and must be ignored.
    """
    graph = Graph()
    for node in ("A", "B", "C", "D", "E"):
        graph.add_node(node)
    graph.add_edge("A", "B", 10)
    graph.add_edge("A", "C", 3)
    graph.add_edge("C", "B", 4)
    graph.add_edge("C", "D", 15)
    graph.add_edge("B", "E", 9)
    return graph


def test_stale_heap_entry_is_ignored(stale_entry_graph):
    # Pop order: (0,A) (3,C) (7,B) (10,B stale -> skipped) (16,E) (18,D).
    # If the stale entry were re-expanded, the improved distances would be
    # overwritten and the route through B would be lost.
    path, cost = dijkstra(stale_entry_graph, "A", "D")
    assert path == ["A", "C", "D"]
    assert cost == 18


def test_improved_distance_is_used_for_later_nodes(stale_entry_graph):
    # E is only reachable through B, so E's cost proves the improved 7 was used
    # rather than the older 10.
    path, cost = dijkstra(stale_entry_graph, "A", "E")
    assert path == ["A", "C", "B", "E"]
    assert cost == 16


# ----------------------------------------------------------------------
# 10. Real airport routes
#
# Ground truth obtained by enumerating all simple paths of the airport graph.
# Two of these routes have several equal-cost shortest paths, so the cost is
# asserted exactly while the path is checked against every valid answer.
# ----------------------------------------------------------------------
AIRPORT_ROUTES = [
    # (start, destination, expected cost)
    ("G1", "R3", 12),
    ("G2", "R2", 11),
    ("G5", "R1", 11),
    ("G10", "R2", 16),
    ("G6", "R4", 17),
]


@pytest.mark.parametrize("start, destination, expected_cost", AIRPORT_ROUTES)
def test_airport_route_cost(start, destination, expected_cost):
    _, cost = dijkstra(build_airport_graph(), start, destination)
    assert cost == expected_cost


@pytest.mark.parametrize("start, destination, expected_cost", AIRPORT_ROUTES)
def test_airport_route_is_a_real_valid_path(start, destination, expected_cost):
    airport = build_airport_graph()
    path, cost = dijkstra(airport, start, destination)
    assert path[0] == start
    assert path[-1] == destination
    assert cost == expected_cost
    assert path_cost(airport, path) == expected_cost


def test_g1_to_r3_has_cheapest_route_of_cost_12():
    airport = build_airport_graph()
    path, cost = dijkstra(airport, "G1", "R3")
    # Two routes tie at 12: via T2 or via T10.
    assert cost == 12
    assert path in (
        ["G1", "T1", "T2", "T9", "R3"],
        ["G1", "T1", "T10", "T9", "R3"],
    )


def test_g2_to_r2_has_unique_shortest_route():
    assert dijkstra(build_airport_graph(), "G2", "R2") == (
        ["G2", "T2", "T3", "T4", "R2"],
        11,
    )


def test_g5_to_r1_has_unique_shortest_route():
    assert dijkstra(build_airport_graph(), "G5", "R1") == (
        ["G5", "T5", "T4", "T3", "R1"],
        11,
    )


def test_g10_to_r2_has_unique_shortest_route():
    assert dijkstra(build_airport_graph(), "G10", "R2") == (
        ["G10", "T10", "T9", "T8", "T7", "T6", "R2"],
        16,
    )


def test_g6_to_r4_has_cheapest_route_of_cost_17():
    airport = build_airport_graph()
    path, cost = dijkstra(airport, "G6", "R4")
    # Three routes tie at 17.
    assert cost == 17
    assert path in (
        ["G6", "T6", "T5", "T4", "T3", "T2", "R4"],
        ["G6", "T6", "T7", "T8", "T3", "T2", "R4"],
        ["G6", "T6", "T7", "T8", "T9", "T10", "R4"],
    )


def test_gate_to_own_runway_access_via_gate_taxiway():
    # G1->T1->R1 = 1 + 6 = 7 (the direct T1->R1 access).
    assert dijkstra(build_airport_graph(), "G1", "R1") == (
        ["G1", "T1", "R1"],
        7,
    )


# ======================================================================
# Known-answer tests
#
# Every expectation below is worked out by hand in a comment first, then
# compared against Dijkstra. These are the tests that would catch a broken
# priority queue, a missing relaxation step, or an off-by-one in the path
# reconstruction.
# ======================================================================


def brute_force_shortest_path(
    graph: Graph, start: str, destination: str
) -> tuple[list[str], float]:
    """Reference implementation: enumerate every simple path, keep the cheapest.

    Deliberately *not* Dijkstra and not heap-based, so it is an independent
    oracle that the real implementation can be checked against. Only suitable
    for small graphs, since the number of simple paths grows quickly.
    """
    best_path: list[str] = []
    best_cost = math.inf

    def walk(node: str, path: list[str], cost: float) -> None:
        nonlocal best_path, best_cost
        if cost >= best_cost:
            return  # weights are non-negative, so this branch cannot improve
        if node == destination:
            best_path, best_cost = list(path), cost
            return
        for neighbor, weight in graph.get_neighbors(node).items():
            if neighbor in path:
                continue
            path.append(neighbor)
            walk(neighbor, path, cost + weight)
            path.pop()

    walk(start, [start], 0)
    return best_path, best_cost


# ----------------------------------------------------------------------
# Test A - simple graph where the shortest path skips the direct edge
#   A --5-- B
#    \     /
#     2   1
#      \ /
#       C
# ----------------------------------------------------------------------
@pytest.fixture
def test_a_graph() -> Graph:
    graph = Graph()
    for node in ("A", "B", "C"):
        graph.add_node(node)
    graph.add_edge("A", "B", 5)
    graph.add_edge("A", "C", 2)
    graph.add_edge("C", "B", 1)
    return graph


def test_a_finds_the_three_hop_cheaper_route(test_a_graph):
    # By hand: A->C->B = 2 + 1 = 3, A->B = 5. So the detour wins.
    assert dijkstra(test_a_graph, "A", "B") == (["A", "C", "B"], 3)


# ----------------------------------------------------------------------
# Test B - the direct route is not the shortest
#   A --10-- B
#    \       /
#     2     2
#      \   /
#        C
# ----------------------------------------------------------------------
@pytest.fixture
def test_b_graph() -> Graph:
    graph = Graph()
    for node in ("A", "B", "C"):
        graph.add_node(node)
    graph.add_edge("A", "B", 10)
    graph.add_edge("A", "C", 2)
    graph.add_edge("C", "B", 2)
    return graph


def test_b_prefers_the_two_hop_route_over_the_direct_edge(test_b_graph):
    # By hand: A->C->B = 2 + 2 = 4, A->B = 10. The direct edge is 2.5x worse.
    path, cost = dijkstra(test_b_graph, "A", "B")
    assert path == ["A", "C", "B"]
    assert cost == 4
    # Explicitly prove the direct edge was available but not chosen.
    assert test_b_graph.has_edge("A", "B")


# ----------------------------------------------------------------------
# Test C - the airport diagram, with a manual cross-check
#
#   GATE_A --1-- T1 --2-- T2 --5-- T4 --2-- T5 --6-- RUNWAY_09
#                   \--2-- T3 --3--/
#
# Weights chosen to mirror airport-like values (the diagram itself did not
# specify any). Manual calculation:
#   via T2: 1 + 2 + 5 + 2 + 6 = 16
#   via T3: 1 + 2 + 3 + 2 + 6 = 14   <-- smaller
# Manual answer: GATE_A -> T1 -> T3 -> T4 -> T5 -> RUNWAY_09, cost 14
# ----------------------------------------------------------------------
@pytest.fixture
def test_c_graph() -> Graph:
    graph = Graph()
    for node in ("GATE_A", "T1", "T2", "T3", "T4", "T5", "RUNWAY_09"):
        graph.add_node(node)
    graph.add_edge("GATE_A", "T1", 1)
    graph.add_edge("T1", "T2", 2)
    graph.add_edge("T1", "T3", 2)
    graph.add_edge("T2", "T4", 5)
    graph.add_edge("T3", "T4", 3)
    graph.add_edge("T4", "T5", 2)
    graph.add_edge("T5", "RUNWAY_09", 6)
    return graph


def test_c_matches_the_hand_calculated_answer(test_c_graph):
    assert dijkstra(test_c_graph, "GATE_A", "RUNWAY_09") == (
        ["GATE_A", "T1", "T3", "T4", "T5", "RUNWAY_09"],
        14,
    )


def test_c_matches_the_independent_brute_force_oracle(test_c_graph):
    manual_path, manual_cost = ["GATE_A", "T1", "T3", "T4", "T5", "RUNWAY_09"], 14
    oracle_path, oracle_cost = brute_force_shortest_path(
        test_c_graph, "GATE_A", "RUNWAY_09"
    )
    assert (oracle_path, oracle_cost) == (manual_path, manual_cost)
    assert dijkstra(test_c_graph, "GATE_A", "RUNWAY_09") == (oracle_path, oracle_cost)


def test_c_going_backwards_gives_the_reversed_route(test_c_graph):
    path, cost = dijkstra(test_c_graph, "RUNWAY_09", "GATE_A")
    assert path == ["RUNWAY_09", "T5", "T4", "T3", "T1", "GATE_A"]
    assert cost == 14


# ----------------------------------------------------------------------
# Test D - the runway is completely disconnected
#   GATE_A --1-- T1 --1-- T2      RUNWAY_09  (isolated node)
# ----------------------------------------------------------------------
@pytest.fixture
def test_d_graph() -> Graph:
    graph = Graph()
    graph.add_node("GATE_A")
    graph.add_node("T1")
    graph.add_node("T2")
    graph.add_node("RUNWAY_09")
    graph.add_edge("GATE_A", "T1", 1)
    graph.add_edge("T1", "T2", 1)
    return graph


def test_d_unreachable_runway_does_not_crash(test_d_graph):
    assert dijkstra(test_d_graph, "GATE_A", "RUNWAY_09") == ([], math.inf)


def test_d_connected_part_of_the_graph_still_works(test_d_graph):
    assert dijkstra(test_d_graph, "GATE_A", "T2") == (["GATE_A", "T1", "T2"], 2)


def test_d_missing_runway_node_is_an_input_error(test_d_graph):
    # The node itself does not exist, which is different from being unreachable.
    with pytest.raises(KeyError):
        dijkstra(test_d_graph, "GATE_A", "RUNWAY_21")


# ----------------------------------------------------------------------
# Test E - blocking a taxiway dynamically changes the chosen route
#
#   GATE_A --1-- T1
#                /  \
#              1/    \1
#             T2      T3
#              \1000  /2000
#               \    /
#                T4 --1-- RUNWAY
# ----------------------------------------------------------------------
@pytest.fixture
def test_e_graph() -> Graph:
    graph = Graph()
    for node in ("GATE_A", "T1", "T2", "T3", "T4", "RUNWAY"):
        graph.add_node(node)
    graph.add_edge("GATE_A", "T1", 1)
    graph.add_edge("T1", "T2", 1)
    graph.add_edge("T1", "T3", 1)
    graph.add_edge("T2", "T4", 1000)
    graph.add_edge("T3", "T4", 2000)
    graph.add_edge("T4", "RUNWAY", 1)
    return graph


def test_e_picks_the_cheaper_taxiway_to_t4(test_e_graph):
    # By hand: via T2 = 1 + 1 + 1000 + 1 = 1003, via T3 = 1 + 1 + 2000 + 1 = 2003.
    assert dijkstra(test_e_graph, "GATE_A", "RUNWAY") == (
        ["GATE_A", "T1", "T2", "T4", "RUNWAY"],
        1003,
    )


def test_e_blocking_t2_to_t4_reroutes_via_t3(test_e_graph):
    test_e_graph.block_edge("T2", "T4")
    # By hand: T2->T4 is gone, so only via T3: 1 + 1 + 2000 + 1 = 2003.
    assert dijkstra(test_e_graph, "GATE_A", "RUNWAY") == (
        ["GATE_A", "T1", "T3", "T4", "RUNWAY"],
        2003,
    )


def test_e_unblocking_restores_the_original_route(test_e_graph):
    test_e_graph.block_edge("T2", "T4")
    test_e_graph.unblock_edge("T2", "T4")
    assert dijkstra(test_e_graph, "GATE_A", "RUNWAY")[1] == 1003


def test_e_blocking_both_taxiways_makes_the_runway_unreachable(test_e_graph):
    test_e_graph.block_edge("T2", "T4")
    test_e_graph.block_edge("T3", "T4")
    assert dijkstra(test_e_graph, "GATE_A", "RUNWAY") == ([], math.inf)


# ----------------------------------------------------------------------
# Test F - start equals destination
# ----------------------------------------------------------------------
def test_f_start_equals_destination(test_e_graph):
    assert dijkstra(test_e_graph, "T4", "T4") == (["T4"], 0)


def test_f_start_equals_destination_on_an_isolated_node(test_d_graph):
    assert dijkstra(test_d_graph, "RUNWAY_09", "RUNWAY_09") == (["RUNWAY_09"], 0)


# ----------------------------------------------------------------------
# Test G - invalid nodes
#
# Decision (also documented in dijkstra's docstring): an unknown start or
# destination raises KeyError, matching the behaviour of Graph. It is not
# silently treated as unreachable, because "you asked about a node that does
# not exist" is a programming error, while "no route exists" is a real answer.
# ----------------------------------------------------------------------
def test_g_unknown_start_raises_key_error(test_e_graph):
    with pytest.raises(KeyError):
        dijkstra(test_e_graph, "ABC", "RUNWAY")


def test_g_unknown_destination_raises_key_error(test_e_graph):
    with pytest.raises(KeyError):
        dijkstra(test_e_graph, "GATE_A", "RUNWAY_09")


def test_g_unknown_node_is_not_reported_as_unreachable(test_e_graph):
    # A KeyError is raised, so a typo can never look like a valid empty route.
    try:
        result = dijkstra(test_e_graph, "ABC", "RUNWAY")
    except KeyError:
        return
    pytest.fail(f"expected KeyError, got {result!r}")


def test_g_error_message_names_the_offending_node(test_e_graph):
    with pytest.raises(KeyError, match="ABC"):
        dijkstra(test_e_graph, "ABC", "RUNWAY")


# ----------------------------------------------------------------------
# Test H - zero-weight edges
# ----------------------------------------------------------------------
def test_h_zero_weight_edge():
    graph = Graph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B", 0)
    assert dijkstra(graph, "A", "B") == (["A", "B"], 0)


def test_h_zero_weight_edge_does_not_hide_a_longer_route():
    graph = Graph()
    for node in ("A", "B", "C"):
        graph.add_node(node)
    graph.add_edge("A", "B", 0)
    graph.add_edge("B", "C", 5)
    graph.add_edge("A", "C", 100)
    # A->B->C = 0 + 5 = 5 still beats the direct 100.
    assert dijkstra(graph, "A", "C") == (["A", "B", "C"], 5)


def test_h_all_zero_weight_cycle_terminates():
    # A zero-weight cycle must not make the search loop forever.
    graph = Graph()
    for node in ("A", "B", "C"):
        graph.add_node(node)
    graph.add_edge("A", "B", 0)
    graph.add_edge("B", "C", 0)
    graph.add_edge("C", "A", 0)
    assert dijkstra(graph, "A", "C") == (["A", "C"], 0)


# ----------------------------------------------------------------------
# Test I - several equally short routes
#   A --3-- B --7-- D
#    \--4-- C --6--/
# both routes cost 10
# ----------------------------------------------------------------------
@pytest.fixture
def test_i_graph() -> Graph:
    graph = Graph()
    for node in ("A", "B", "C", "D"):
        graph.add_node(node)
    graph.add_edge("A", "B", 3)
    graph.add_edge("B", "D", 7)
    graph.add_edge("A", "C", 4)
    graph.add_edge("C", "D", 6)
    return graph


def test_i_cost_is_ten_whichever_route_is_returned(test_i_graph):
    path, cost = dijkstra(test_i_graph, "A", "D")
    # Requirement is the cost only: no particular tie-break is demanded.
    assert cost == 10
    assert path in (["A", "B", "D"], ["A", "C", "D"])


def test_i_returned_path_is_a_real_route_of_cost_ten(test_i_graph):
    path, cost = dijkstra(test_i_graph, "A", "D")
    assert path_cost(test_i_graph, path) == cost == 10


def test_i_three_way_tie_on_the_airport():
    # G6->R4 has three equal-cost routes, so only the cost may be asserted.
    _, cost = dijkstra(build_airport_graph(), "G6", "R4")
    assert cost == 17


# ----------------------------------------------------------------------
# Cross-check against the brute-force oracle on the real airport
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "start, destination",
    [
        ("G1", "R1"),
        ("G1", "R3"),
        ("G2", "R2"),
        ("G5", "R1"),
        ("G10", "R2"),
        ("G6", "R4"),
        ("G10", "R5"),
        ("G3", "R4"),
    ],
)
def test_airport_routes_match_the_brute_force_oracle(start, destination):
    airport = build_airport_graph()
    _, oracle_cost = brute_force_shortest_path(airport, start, destination)
    path, cost = dijkstra(airport, start, destination)
    assert cost == oracle_cost
    assert path_cost(airport, path) == oracle_cost


def test_airport_matches_oracle_even_with_blocked_taxiways():
    airport = build_airport_graph()
    airport.block_edge("T1", "R1")
    airport.block_edge("T2", "T3")
    airport.block_edge("T7", "T8")
    for start, destination in [
        ("G1", "R1"),
        ("G4", "R3"),
        ("G10", "R2"),
        ("G5", "R4"),
    ]:
        _, oracle_cost = brute_force_shortest_path(airport, start, destination)
        path, cost = dijkstra(airport, start, destination)
        assert cost == oracle_cost, f"{start}->{destination}"
        assert path_cost(airport, path) == oracle_cost

