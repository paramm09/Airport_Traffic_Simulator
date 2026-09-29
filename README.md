# Airport & Air-Traffic Movement Simulator

A small, deterministic simulation of aircraft taxiing, taking off, crossing a
simplified airspace, landing, and taxiing back — built as a DAA (Design and
Analysis of Algorithms) teaching project.

> **This is an educational model, not air-traffic software.** The physics,
> separation minima and procedures are deliberately simplified so the
> algorithms stay explainable. Nothing here should be read as a claim about
> real-world ATC safety, separation standards, or aviation compliance.

## What it does

Aircraft move through a lifecycle:

```
AT_GATE -> TAXIING_TO_RUNWAY -> WAITING_FOR_RUNWAY -> LINE_UP -> TAKEOFF
        -> CLIMB -> CRUISE -> DESCENT -> APPROACH -> LANDING
        -> TAXIING_TO_GATE -> COMPLETED
```

while the simulation sequences them onto a single shared runway, predicts
airspace conflicts and issues HOLD instructions, and re-plans taxi routes when
a taxiway closes.

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Graph | `backend/models/graph.py` | weighted, undirected airport network; edges can be blocked |
| Data | `backend/data/airport_graph.py`, `.../airspace_waypoints.py` | the airport network and the deterministic waypoint set |
| Model | `backend/models/aircraft.py`, `.../waypoint.py` | one aircraft's state, position and explicit transition table |
| Dijkstra | `backend/algorithms/dijkstra.py` | shortest taxi route, ignoring blocked edges |
| Taxi | `backend/simulation/taxi.py` | walks a route edge by edge |
| Runway | `backend/simulation/runway_scheduler.py` | binary-resource scheduler with a deterministic priority policy |
| Airspace | `backend/simulation/airspace.py` | continuous movement toward waypoints; heading stays consistent with position |
| CPA | `backend/simulation/conflict.py` | analytical closest-point-of-approach prediction over a horizon |
| Resolution | `backend/simulation/conflict_resolution.py` | chooses which aircraft yields, and issues a HOLD |
| Rerouting | `backend/simulation/rerouting.py` | re-plans from the current node around a blocked edge |
| Integration | `backend/simulation/simulation.py` | the fixed-timestep clock that wires the above together and owns the event log |

### The tick

One tick, in this fixed order (the order is the specification, not an
implementation detail):

1. scheduled graph changes
2. taxi movement, with a reroute if blocked
3. runway requests, once per aircraft per operation
4. runway scheduling
5. airborne movement, unless the aircraft is holding
6. conflict detection
7. conflict resolution
8. event recording
9. advance the clock

Detection runs *after* movement, so a conflict is predicted about the positions
the aircraft actually hold. A HOLD gates airborne movement only — it never
stops an aircraft taxiing, which would let an air-traffic instruction quietly
change the answer of the runway scheduler.

## Determinism

There is no wall clock, no `random`, no set iteration and no threading. The tick
counter only ever increases by one and aircraft are held in insertion order, so
an identical starting configuration always produces an identical event log.

Two lifecycle states are terminal, meaning the aircraft will not move again:
`COMPLETED` (the journey finished) and `HOLDING_NO_ROUTE` (a departure stopped
by an incident, which only resumes when the network changes). Every other state
means the simulation is still running.

## Running the tests

```bash
pytest -q
```

Coverage, if you want it:

```bash
coverage run -m pytest -q
coverage report -m
```

## Design documents

- `docs/phase1.md` — the airport graph and the Phase 1 rationale
- `docs/dijkstra.md` — Dijkstra pseudocode and how it maps onto the code
- `docs/phase2.md` — the simulation architecture and the design rationale
