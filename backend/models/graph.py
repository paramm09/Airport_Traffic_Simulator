"""Weighted undirected graph used to model the airport network.

The graph is stored as an adjacency list: one dictionary maps every node to a
dictionary of its neighbours, and the value is the weight of the connection::

    {"T1": {"T2": 2, "T10": 4, "R1": 6}}

The airport network is undirected, so a connection is written into the
adjacency list of *both* endpoints. Each connection is also given a single
canonical key (its two node names in alphabetical order) so that blocking it in
one direction automatically blocks it in the other.
"""

from __future__ import annotations

EdgeKey = tuple[str, str]


class Graph:
    """A weighted undirected graph stored as an adjacency list.

    Example:
        >>> graph = Graph()
        >>> graph.add_node("T1")
        >>> graph.add_node("T2")
        >>> graph.add_edge("T1", "T2", 2)
        >>> graph.get_neighbors("T1")
        {'T2': 2}
        >>> graph.get_neighbors("T2")
        {'T1': 2}
    """

    def __init__(self) -> None:
        # node -> {neighbour: weight}
        self._adjacency: dict[str, dict[str, float]] = {}
        # canonical keys of the connections that are currently blocked
        self._blocked: set[EdgeKey] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _edge_key(node1: str, node2: str) -> EdgeKey:
        """Return one canonical key for an undirected connection.

        ``("T1", "T2")`` and ``("T2", "T1")`` both produce the same key, so a
        connection is only ever recorded once in the blocked set.
        """
        return (node1, node2) if node1 <= node2 else (node2, node1)

    def _require_node(self, node: str) -> None:
        """Raise ``KeyError`` if ``node`` is not part of the graph."""
        if node not in self._adjacency:
            raise KeyError(f"Unknown node: {node!r}")

    def _require_edge(self, node1: str, node2: str) -> None:
        """Raise ``KeyError`` if the undirected connection does not exist."""
        self._require_node(node1)
        self._require_node(node2)
        if node2 not in self._adjacency[node1]:
            raise KeyError(f"No edge between {node1!r} and {node2!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_node(self, node: str) -> None:
        """Add ``node`` to the graph as an isolated node.

        Adding a node that is already present does nothing, so the connections
        that already exist are never lost.
        """
        if node not in self._adjacency:
            self._adjacency[node] = {}

    def add_edge(self, node1: str, node2: str, weight: float) -> None:
        """Connect ``node1`` and ``node2`` in both directions with ``weight``.

        Both nodes must already exist. Re-adding an existing connection simply
        updates its weight.

        Raises:
            ValueError: if ``weight`` is negative, because the graph will later
                be used with Dijkstra, which requires non-negative weights.
            KeyError: if either node has not been added yet.
        """
        if weight < 0:
            raise ValueError(f"Edge weight cannot be negative: {weight!r}")
        self._require_node(node1)
        self._require_node(node2)
        self._adjacency[node1][node2] = weight
        self._adjacency[node2][node1] = weight

    def get_neighbors(self, node: str) -> dict[str, float]:
        """Return ``{neighbour: weight}`` for the connections of ``node``.

        Blocked connections are left out, so the returned dictionary holds
        exactly the nodes a path-finding algorithm may currently travel to. A
        copy is returned so that callers cannot modify the graph by accident.

        Raises:
            KeyError: if ``node`` is not part of the graph.
        """
        self._require_node(node)
        return {
            neighbour: weight
            for neighbour, weight in self._adjacency[node].items()
            if self._edge_key(node, neighbour) not in self._blocked
        }

    def get_edge_weight(self, node1: str, node2: str) -> float:
        """Return the weight of the undirected connection.

        The weight is reported even when the connection is currently blocked,
        because blocking does not delete a connection — it only makes it
        unavailable. Availability is reported by :meth:`has_edge`.

        Returns:
            The weight stored for the connection.

        Raises:
            KeyError: if the connection does not exist, or either node is
                unknown.
        """
        self._require_edge(node1, node2)
        return self._adjacency[node1][node2]

    def block_edge(self, node1: str, node2: str) -> None:
        """Make the undirected connection unavailable without deleting it.

        The weight is kept in the adjacency list, so :meth:`unblock_edge` can
        restore the original connection later. Blocking an already blocked
        connection does nothing.

        Raises:
            KeyError: if the connection does not exist.
        """
        self._require_edge(node1, node2)
        self._blocked.add(self._edge_key(node1, node2))

    def unblock_edge(self, node1: str, node2: str) -> None:
        """Make a previously blocked connection available again.

        Unblocking a connection that is not blocked does nothing.

        Raises:
            KeyError: if the connection does not exist.
        """
        self._require_edge(node1, node2)
        self._blocked.discard(self._edge_key(node1, node2))

    def has_node(self, node: str) -> bool:
        """Return ``True`` if ``node`` has been added to the graph."""
        return node in self._adjacency

    def get_nodes(self) -> list[str]:
        """Return every node in the graph.

        Path-finding algorithms need to set up one entry per node, so this
        gives them the full node list. A copy is returned so that callers
        cannot modify the graph by accident.
        """
        return list(self._adjacency.keys())

    def has_edge(self, node1: str, node2: str) -> bool:
        """Return ``True`` if the connection exists and is not blocked.

        This answers "can an aircraft travel between these two nodes right
        now?", which is what the simulation needs. Use :meth:`is_blocked` to
        tell a missing connection apart from a blocked one.
        """
        if not (self.has_node(node1) and self.has_node(node2)):
            return False
        if node2 not in self._adjacency[node1]:
            return False
        return self._edge_key(node1, node2) not in self._blocked

    def is_blocked(self, node1: str, node2: str) -> bool:
        """Return ``True`` if the connection exists but is currently blocked.

        Raises:
            KeyError: if the connection does not exist.
        """
        self._require_edge(node1, node2)
        return self._edge_key(node1, node2) in self._blocked
