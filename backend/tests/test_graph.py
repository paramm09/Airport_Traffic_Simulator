"""Unit tests for the :class:`Graph` data structure only.

    Path finding lives in `backend/tests/test_dijkstra.py`; this file covers
    the graph structure that Dijkstra reads.

"""

import pytest

from backend.data.airport_graph import ALL_NODES, build_airport_graph
from backend.models.graph import Graph


@pytest.fixture
def graph() -> Graph:
    """A tiny graph reused by most of the tests: T1 --2-- T2 --3-- T3."""
    g = Graph()
    for node in ("T1", "T2", "T3"):
        g.add_node(node)
    g.add_edge("T1", "T2", 2)
    g.add_edge("T2", "T3", 3)
    return g


# ----------------------------------------------------------------------
# 1. Adding a node
# ----------------------------------------------------------------------
def test_add_node_adds_the_node():
    g = Graph()
    g.add_node("T1")
    assert g.has_node("T1")


def test_new_graph_starts_empty():
    assert Graph().has_node("T1") is False


def test_adding_an_existing_node_keeps_its_edges():
    g = Graph()
    g.add_node("T1")
    g.add_node("T2")
    g.add_edge("T1", "T2", 2)
    g.add_node("T1")  # duplicate add
    assert g.get_neighbors("T1") == {"T2": 2}
    assert g.get_neighbors("T2") == {"T1": 2}


# ----------------------------------------------------------------------
# 2. Checking whether a node exists
# ----------------------------------------------------------------------
def test_has_node_true_for_known_node(graph):
    assert graph.has_node("T1") is True


def test_has_node_false_for_unknown_node(graph):
    assert graph.has_node("T99") is False


# ----------------------------------------------------------------------
# Listing all nodes
# ----------------------------------------------------------------------
def test_get_nodes_lists_every_node(graph):
    assert sorted(graph.get_nodes()) == ["T1", "T2", "T3"]


def test_get_nodes_on_empty_graph():
    assert Graph().get_nodes() == []


def test_get_nodes_includes_isolated_nodes():
    g = Graph()
    g.add_node("G1")
    g.add_node("T1")
    assert sorted(g.get_nodes()) == ["G1", "T1"]


def test_get_nodes_returns_a_copy(graph):
    nodes = graph.get_nodes()
    nodes.append("T99")
    assert graph.get_nodes() == ["T1", "T2", "T3"]


# ----------------------------------------------------------------------
# 3. / 4. Adding an undirected edge and checking both directions
# ----------------------------------------------------------------------
def test_add_edge_is_visible_in_both_directions(graph):
    assert graph.has_edge("T1", "T2")
    assert graph.has_edge("T2", "T1")


def test_has_edge_false_for_unconnected_pair(graph):
    assert graph.has_edge("T1", "T3") is False


def test_has_edge_false_for_unknown_nodes(graph):
    assert graph.has_edge("T1", "T99") is False


# ----------------------------------------------------------------------
# 5. Correct edge weight
# ----------------------------------------------------------------------
def test_edge_weight_is_correct(graph):
    assert graph.get_neighbors("T1") == {"T2": 2}
    assert graph.get_neighbors("T2") == {"T1": 2, "T3": 3}
    assert graph.get_neighbors("T3") == {"T2": 3}


def test_readding_an_edge_updates_the_weight():
    g = Graph()
    g.add_node("T1")
    g.add_node("T2")
    g.add_edge("T1", "T2", 2)
    g.add_edge("T1", "T2", 7)
    assert g.get_neighbors("T1") == {"T2": 7}
    assert g.get_neighbors("T2") == {"T1": 7}


# ----------------------------------------------------------------------
# 6. Getting neighbors
# ----------------------------------------------------------------------
def test_get_neighbors_of_isolated_node():
    g = Graph()
    g.add_node("G1")
    assert g.get_neighbors("G1") == {}


def test_get_neighbors_returns_a_copy(graph):
    neighbors = graph.get_neighbors("T1")
    neighbors["T3"] = 99
    assert graph.get_neighbors("T1") == {"T2": 2}


def test_get_edge_weight_returns_the_stored_weight(graph):
    assert graph.get_edge_weight("T1", "T2") == 2
    assert graph.get_edge_weight("T2", "T3") == 3


def test_get_edge_weight_accepts_either_argument_order(graph):
    assert graph.get_edge_weight("T2", "T1") == 2


def test_get_edge_weight_reports_the_weight_even_when_blocked(graph):
    graph.block_edge("T1", "T2")
    assert graph.has_edge("T1", "T2") is False
    assert graph.get_edge_weight("T1", "T2") == 2


def test_get_edge_weight_of_unconnected_pair_raises(graph):
    with pytest.raises(KeyError):
        graph.get_edge_weight("T1", "T3")


def test_get_edge_weight_of_unknown_node_raises(graph):
    with pytest.raises(KeyError):
        graph.get_edge_weight("T1", "T99")


# ----------------------------------------------------------------------
# 7. / 8. Blocking an edge makes it unavailable in both directions
# ----------------------------------------------------------------------
def test_blocked_edge_is_unavailable_in_both_directions(graph):
    graph.block_edge("T2", "T3")
    assert graph.has_edge("T2", "T3") is False
    assert graph.has_edge("T3", "T2") is False


def test_blocked_edge_disappears_from_get_neighbors(graph):
    graph.block_edge("T1", "T2")
    assert graph.get_neighbors("T1") == {}
    assert graph.get_neighbors("T2") == {"T3": 3}


def test_block_edge_keeps_the_weight_in_the_graph_structure(graph):
    graph.block_edge("T1", "T2")
    # Not deleted: the connection can still be restored with its old weight.
    graph.unblock_edge("T1", "T2")
    assert graph.get_neighbors("T1") == {"T2": 2}


def test_blocking_one_edge_leaves_the_others_untouched(graph):
    graph.block_edge("T1", "T2")
    assert graph.has_edge("T2", "T3") is True


def test_is_blocked_reports_the_blocking_state(graph):
    assert graph.is_blocked("T1", "T2") is False
    graph.block_edge("T1", "T2")
    assert graph.is_blocked("T2", "T1") is True


# ----------------------------------------------------------------------
# 9. / 10. Unblocking restores the original connection
# ----------------------------------------------------------------------
def test_unblock_edge_restores_both_directions(graph):
    graph.block_edge("T1", "T2")
    graph.unblock_edge("T1", "T2")
    assert graph.has_edge("T1", "T2")
    assert graph.has_edge("T2", "T1")
    assert graph.get_neighbors("T1") == {"T2": 2}


def test_unblock_edge_accepts_either_argument_order(graph):
    graph.block_edge("T1", "T2")
    graph.unblock_edge("T2", "T1")
    assert graph.has_edge("T1", "T2")


def test_unblocking_an_unblocked_edge_is_a_noop(graph):
    graph.unblock_edge("T1", "T2")
    assert graph.get_neighbors("T1") == {"T2": 2}


# ----------------------------------------------------------------------
# 11. Rejecting negative edge weights
# ----------------------------------------------------------------------
def test_add_edge_rejects_negative_weight():
    g = Graph()
    g.add_node("T1")
    g.add_node("T2")
    with pytest.raises(ValueError):
        g.add_edge("T1", "T2", -1)


def test_rejected_negative_weight_is_not_stored():
    g = Graph()
    g.add_node("T1")
    g.add_node("T2")
    with pytest.raises(ValueError):
        g.add_edge("T1", "T2", -5)
    assert g.has_edge("T1", "T2") is False
    assert g.get_neighbors("T1") == {}


def test_zero_weight_is_allowed():
    g = Graph()
    g.add_node("T1")
    g.add_node("T2")
    g.add_edge("T1", "T2", 0)
    assert g.get_neighbors("T1") == {"T2": 0}


# ----------------------------------------------------------------------
# 12. Handling unknown nodes
# ----------------------------------------------------------------------
def test_add_edge_with_unknown_node_raises():
    g = Graph()
    g.add_node("T1")
    with pytest.raises(KeyError):
        g.add_edge("T1", "T99", 2)


def test_get_neighbors_of_unknown_node_raises(graph):
    with pytest.raises(KeyError):
        graph.get_neighbors("T99")


def test_block_unknown_node_raises(graph):
    with pytest.raises(KeyError):
        graph.block_edge("T1", "T99")


# ----------------------------------------------------------------------
# Blocking / unblocking a connection that does not exist
# ----------------------------------------------------------------------
def test_block_missing_edge_raises(graph):
    with pytest.raises(KeyError):
        graph.block_edge("T1", "T3")


def test_unblock_missing_edge_raises(graph):
    with pytest.raises(KeyError):
        graph.unblock_edge("T1", "T3")


def test_is_blocked_on_missing_edge_raises(graph):
    with pytest.raises(KeyError):
        graph.is_blocked("T1", "T3")


# ----------------------------------------------------------------------
# The real airport data
# ----------------------------------------------------------------------
def test_airport_graph_has_all_nodes():
    airport = build_airport_graph()
    for node in ALL_NODES:
        assert airport.has_node(node), f"missing node {node}"


def test_airport_gate_taxiway_weight_is_one():
    airport = build_airport_graph()
    for i in range(1, 11):
        gate, taxiway = f"G{i}", f"T{i}"
        assert airport.get_neighbors(gate) == {taxiway: 1}
        assert airport.get_neighbors(taxiway)[gate] == 1


def test_airport_runway_access_weight():
    airport = build_airport_graph()
    assert airport.get_neighbors("T3")["R1"] == 5
    assert airport.get_neighbors("T8")["R5"] == 7
    assert airport.get_neighbors("R1") == {"T1": 6, "T3": 5}


def test_airport_edge_count():
    airport = build_airport_graph()
    # 10 gate + 13 taxiway + 10 runway = 33 undirected connections.
    undirected_edges = {
        frozenset((node, neighbour))
        for node in ALL_NODES
        for neighbour in airport.get_neighbors(node)
    }
    assert len(undirected_edges) == 33


def test_airport_graph_can_be_blocked_and_restored():
    airport = build_airport_graph()
    airport.block_edge("T2", "T3")
    assert "T3" not in airport.get_neighbors("T2")
    airport.unblock_edge("T2", "T3")
    assert airport.get_neighbors("T2")["T3"] == 2
