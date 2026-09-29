# Dijkstra's Algorithm — Pseudocode and Implementation (Phase 1)

**Status:** implemented. `backend/algorithms/dijkstra.py` follows the pseudocode
below step for step, and `backend/tests/test_dijkstra.py` covers it. The
implementation is `dijkstra(graph, start, destination) -> (path, cost)`.

## Purpose

Find the shortest path (smallest total distance) from a start node to a
destination node in the weighted, undirected airport graph.

## When it applies

Our graph has **non-negative** edge weights (distance / cost units), which is
exactly the condition Dijkstra requires. This is why `Graph.add_edge` rejects
negative weights.

## Pseudocode

```
DIJKSTRA(graph, start, destination)

    Check that start and destination exist

    Initialize all distances to infinity
    Set start distance to 0

    Set previous of every node to None

    Create empty min-priority queue
    Insert (0, start)

    While priority queue is not empty:

        Remove (current_distance, current_node)

        If current_distance is greater than distances[current_node]:
            continue

        If current_node == destination:
            stop

        For each available neighbor of current_node:

            Calculate new distance

            If new distance is smaller than distances[neighbor]:

                Update distances[neighbor]
                Set previous[neighbor] = current_node
                Insert (new distance, neighbor) into priority queue

    If distance[destination] is infinity:
        return no path

    Reconstruct path using previous

    Return path and total distance
```

## Supporting data structures

| Structure | Meaning | Python equivalent |
|---|---|---|
| `distances` | best known distance from `start` to each node | `dict[str, float]`, filled with `math.inf` |
| `previous` | node visited just before this one, `None` until improved | `dict[str, str \| None]` |
| priority queue | tuples of `(distance, node)`, smallest distance removed first | `heapq` min-heap, with lazy deletion |

## How it maps onto our `Graph` class

- *"Check that start and destination exist"* — `graph.has_node(start)` and
  `graph.has_node(destination)`; `dijkstra` raises `KeyError` for unknown
  nodes.
- *"Initialize all distances to infinity"* / *"Set previous of every node to
  None"* — both tables are filled from `graph.get_nodes()`.
- *"Insert (0, start)"* — the queue stores `(distance, node)` pairs, which is
  the form `heapq` compares correctly when the second element is a string.
- *"If current_distance is greater than distances[current_node]: continue"* —
  this is **lazy deletion**. An improved node is pushed again without removing
  its older entry, so stale entries are skipped here. This is what lets us use
  `heapq` without any decrease-key operation.
- *"For each available neighbor of current_node"* — `graph.get_neighbors(node)`
  already returns `{neighbour: weight}` with **blocked** connections filtered
  out, so this one call handles the "available" part and the simulation's closed
  taxiways for free. No extra blocked-edge logic is needed in the algorithm.
- *"Undirected"* — nothing extra; the algorithm works on whatever
  `get_neighbors` reports in each direction.
- *"If distance[destination] is infinity: return no path"* — covers a genuinely
  unreachable destination and also a disconnected sub-network. The implemented
  choice is an empty list paired with `math.inf`.
- *"Reconstruct path using previous"* — walk backwards from `destination`
  through `previous` until reaching `start`, then reverse. The `None` values set
  during initialisation are the safety net for that walk.

## Expected result shape

```
path        = ["G1", "T1", "T2", "T3", "R1"]
total_cost  = 1 + 2 + 2 + 5 = 10
```

## Resolved decisions

These were the open questions when this document was written. Each is now
settled by the implementation:

1. **Return shape.** One call returns the path to a single destination plus its
   cost, not the whole `distances` / `previous` tables. The tables stay local to
   the call.
2. **No path.** Returns `([], math.inf)`. A `KeyError` is reserved for a
   `start` or `destination` that is not a node of the graph at all, which is an
   input error rather than an unreachable destination.
3. **`start == destination`.** Returns `([start], 0)`.
4. **Blocked taxiways.** A single search reads a fixed network, since the
   algorithm only ever calls `get_neighbors`. Phase 2G re-invokes `dijkstra`
   after an edge is reopened rather than trying to resume a search.
