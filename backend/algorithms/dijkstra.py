"""Dijkstra's shortest-path algorithm for the airport network.

Implements the pseudocode in ``docs/dijkstra.md``:

1. Check that ``start`` and ``destination`` exist.
2. Initialise ``distances`` to infinity and ``previous`` to ``None``.
3. Repeatedly pop the closest node from a min-priority queue (``heapq``),
   relax its available connections, and record how it was reached.
4. Rebuild the path by walking backwards through ``previous``.

Blocked taxiways need no special handling here: :meth:`Graph.get_neighbors`
already leaves them out, so "available neighbours" comes for free. The
algorithm only uses the public API of :class:`~backend.models.graph.Graph` and
never touches its private fields.
"""

from __future__ import annotations

import heapq
import math

from backend.models.graph import Graph

INFINITY = math.inf
Path = list[str]


def dijkstra(graph: Graph, start: str, destination: str) -> tuple[Path, float]:
    """Return the shortest path from ``start`` to ``destination``.

    Args:
        graph: The airport network. Connections that are currently blocked are
            treated as unusable, because ``graph.get_neighbors`` omits them.
        start: Node the path begins at.
        destination: Node the path must end at.

    Returns:
        A ``(path, total_cost)`` tuple, where ``path`` is the list of nodes from
        ``start`` to ``destination`` inclusive. If ``start == destination`` the
        result is ``([start], 0)``. If the destination cannot be reached (it is
        in a disconnected part of the network, or every route to it is blocked)
        the result is ``([], math.inf)``.

    Raises:
        KeyError: If ``start`` or ``destination`` is not a node of the graph.

    The graph must have non-negative edge weights, which
    :meth:`Graph.add_edge` enforces.
    """
    if not graph.has_node(start):
        raise KeyError(f"Unknown start node: {start!r}")
    if not graph.has_node(destination):
        raise KeyError(f"Unknown destination node: {destination!r}")

    # One entry per node: best known distance, and how the node was reached.
    distances: dict[str, float] = {node: INFINITY for node in graph.get_nodes()}
    previous: dict[str, str | None] = {node: None for node in graph.get_nodes()}
    distances[start] = 0

    # Min-priority queue of (distance, node) pairs, smallest distance first.
    # heapq is a min-heap and a second entry for a node is simply pushed again
    # (lazy deletion) instead of doing a manual decrease-key.
    queue: list[tuple[float, str]] = [(0, start)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # Lazy deletion: a shorter route to this node was found after this
        # entry was queued, so this entry is stale and must be ignored.
        if current_distance > distances[current_node]:
            continue

        if current_node == destination:
            break

        for neighbor, weight in graph.get_neighbors(current_node).items():
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))

    if distances[destination] == INFINITY:
        return ([], INFINITY)

    # Walk backwards from the destination, then reverse into start -> end order.
    path: Path = []
    node: str | None = destination
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()

    return (path, distances[destination])
