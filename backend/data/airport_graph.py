"""Builds the airport network as a :class:`Graph` object.

Node types
----------
``G1``-``G10``
    Airport gates.
``T1``-``T10``
    Taxiway intersections / taxiway nodes.
``R1``-``R5``
    Runway access nodes.

Connection types
----------------
gate <-> taxiway
    Gate access connection.
taxiway <-> taxiway
    Taxiway connection.
taxiway <-> runway access
    Runway access connection.

The graph is undirected, so every connection below is traversable in both
directions. Weights are distance / cost units.
"""

from backend.models.graph import Graph

GATES: list[str] = [f"G{i}" for i in range(1, 11)]
TAXIWAYS: list[str] = [f"T{i}" for i in range(1, 11)]
RUNWAY_ACCESS: list[str] = [f"R{i}" for i in range(1, 6)]

ALL_NODES: list[str] = GATES + TAXIWAYS + RUNWAY_ACCESS

# Gate access connections.
GATE_TAXIWAY_EDGES: list[tuple[str, str, int]] = [
    (f"G{i}", f"T{i}", 1) for i in range(1, 11)
]

# Taxiway network.
TAXIWAY_EDGES: list[tuple[str, str, int]] = [
    ("T1", "T2", 2),
    ("T1", "T10", 4),
    ("T2", "T3", 2),
    ("T2", "T9", 4),
    ("T3", "T4", 3),
    ("T3", "T8", 4),
    ("T4", "T5", 2),
    ("T4", "T7", 5),
    ("T5", "T6", 3),
    ("T6", "T7", 2),
    ("T7", "T8", 2),
    ("T8", "T9", 3),
    ("T9", "T10", 2),
]

# Runway access connections.
RUNWAY_ACCESS_EDGES: list[tuple[str, str, int]] = [
    ("T1", "R1", 6),
    ("T3", "R1", 5),
    ("T4", "R2", 5),
    ("T6", "R2", 6),
    ("T7", "R3", 6),
    ("T9", "R3", 5),
    ("T2", "R4", 6),
    ("T10", "R4", 7),
    ("T5", "R5", 8),
    ("T8", "R5", 7),
]

EDGES: list[tuple[str, str, int]] = (
    GATE_TAXIWAY_EDGES + TAXIWAY_EDGES + RUNWAY_ACCESS_EDGES
)


def build_airport_graph() -> Graph:
    """Return a fresh :class:`Graph` containing the whole airport network.

    A new graph is built on every call, so a caller may safely block edges on
    the result without affecting anybody else.
    """
    graph = Graph()
    for node in ALL_NODES:
        graph.add_node(node)
    for node1, node2, weight in EDGES:
        graph.add_edge(node1, node2, weight)
    return graph
