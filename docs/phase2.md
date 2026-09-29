# Phase 2 — Algorithmic Architecture

*Airport & Air-Traffic Movement Simulator — DAA project*
*Status: implemented (Phases 2A–2H). This document remains the design rationale;
the code it describes is in `backend/`.*

---

## 1. Phase 2 objective

Add an **aircraft simulation engine** on top of the Phase 1 airport graph and
Dijkstra. Aircraft taxi from gates to runways, take off, move through a
*simplified* airspace, land, and taxi back to a gate — while the simulator:

- sequences aircraft onto a shared runway (priority policy),
- detects predicted conflicts in the airspace with a deterministic algorithm,
- resolves conflicts with deterministic ATC-like instructions,
- re-routes aircraft when a taxiway becomes blocked mid-taxi.

The educational goal is a small set of **genuinely necessary** DAA algorithms that
can each be explained in a viva. We deliberately do **not** force "every DAA topic"
(MST, max flow, topological sort, dynamic programming, …) into the project.

---

## 2. Overall simulation architecture

```
                   ┌──────────────────────────────────────────────┐
  aircraft        │  SIMULATION CLOCK (fixed time step dt)        │
  requests  ─────▶│                                              │
                   │  1. move each aircraft one step              │
                   │  2. process finished taxi edges / waypoints  │
                   │  3. runway requests  -> RunwayScheduler      │
                   │  4. conflict check    -> ConflictDetector    │
                   │  5. conflict found    -> ConflictResolver    │
                   │  6. blocked edge      -> TaxiRerouter        │
                   └──────────────────────────────────────────────┘
                            │                  │
              ┌─────────────▼───────┐  ┌───────▼────────────┐
              │ RunwayScheduler     │  │ ConflictDetector/   │
              │ (priority queue)    │  │ Resolver            │
              └─────────────┬───────┘  └────────────────────┘
                            │
              ┌─────────────▼───────┐
              │ TaxiRouter          │
              │ (Phase 1 Dijkstra)  │
              └─────────────────────┘
```

Components and the existing piece they rest on:

| Component | Responsibility | Module | Uses from Phase 1 |
|---|---|---|---|
| `Aircraft` | one aircraft's state, route, position | `backend/models/aircraft.py` | — |
| lifecycle transition table | legal state changes | `backend/models/aircraft.py` | — |
| `dijkstra` | shortest taxi route | `backend/algorithms/dijkstra.py` | `Graph` |
| `taxi_step` | walk that route edge by edge | `backend/simulation/taxi.py` | `Graph` |
| `airspace_step` | continuous movement along waypoints | `backend/simulation/airspace.py` | — |
| `RunwayScheduler` | decide which waiting aircraft gets the runway | `backend/simulation/runway_scheduler.py` | binary heap |
| `detect_conflict` | analytically predict conflicts over a horizon | `backend/simulation/conflict.py` | — |
| `ConflictResolver` | pick one deterministic ATC instruction | `backend/simulation/conflict_resolution.py` | — |
| `try_reroute` | re-plan around a blocked taxiway | `backend/simulation/rerouting.py` | `dijkstra`, `Graph` |
| `Simulation` | advance time, wire the above together | `backend/simulation/simulation.py` | — |

There is **no database, no frontend, no network**. Output is text (console)
that reads like ATC instructions — produced as *plain strings* by the resolver,
not by any ML or speech system.

> **Scope.** This is an educational, deliberately simplified model. It is not
> validated air-traffic software, and nothing here should be read as a claim
> about real-world ATC safety, separation standards or aviation compliance.

---

## 3. Aircraft data model

Tedious to force into a plain dictionary; a **small class** (e.g. `@dataclass`)
is the honest representation because an aircraft has *many coexisting fields*
and a group of methods that act on them.

```
Aircraft
  id              : unique string, e.g. "AC001"
  state           : StateMachine value
  taxi_route      : list[str]   (Dijkstra output, current plan)
  route_index     : int         (node we are travelling towards)
  gates_visited   : int

  # airspace fields (continuous)
  position        : (x, y)
  altitude        : float
  speed           : float
  heading         : float (degrees)
  next_waypoint   : (x, y)

  # bookkeeping
  request_time    : tick at which runway was requested
  holding_until   : tick; while current_tick < holding_until the
                    aircraft does not advance (HOLD, defined below)
```

### HOLD semantics

`HOLD` is an actual simulation action with a fixed, simple rule.

```
HOLD_DURATION = 5                          # simulation ticks

on HOLD instruction:
    holding_until = current_tick + HOLD_DURATION
    emit "ATC: <AC001> HOLD"

while current_tick < holding_until:
    aircraft does not advance along its current movement

when current_tick >= holding_until:
    aircraft resumes normal movement
```

No real-world ATC holding procedures or holding patterns are modelled. This is
only a deterministic educational rule.

Taxi vs airspace positions are deliberately represented differently (see §7).
A class is chosen over a dictionary because the fields have *invariants* (e.g.
`speed ≥ 0`, one of exactly two coordinate representations) that a viva can
point at; a bare dict has none. It is still **not** an algorithm — just data.

---

## 4. Aircraft state machine

```
AT_GATE → TAXIING_TO_RUNWAY → WAITING_FOR_RUNWAY → LINE_UP → TAKEOFF
TAKEOFF → CLIMB → CRUISE → DESCENT → APPROACH → LANDING
LANDING → TAXIING_TO_GATE → COMPLETED
```

**Choice: `enum` + an explicit transition table**, not a general graph or
dictionary-of-dictionaries.

*Why:* the lifecycle has a **fixed, small, strictly ordered set of states** with
a handful of legal transitions. An `enum` makes every legal value concrete and
cheap to debug. The transition table is a static mapping:

```
Transitions = {
    AT_GATE:            {TAXIING_TO_RUNWAY},
    TAXIING_TO_RUNWAY:  {WAITING_FOR_RUNWAY},            # joined the runway queue
    WAITING_FOR_RUNWAY: {LINE_UP},
    LINE_UP:            {TAKEOFF},                       # runway cleared
    TAKEOFF:            {CLIMB},
    CLIMB:              {CRUISE},
    CRUISE:             {DESCENT},
    DESCENT:            {APPROACH},
    APPROACH:           {LANDING},
    LANDING:            {TAXIING_TO_GATE},
    TAXIING_TO_GATE:    {COMPLETED},
}
```

*Why not a general graph/state-machine library:* the key insight the professor
wants is that this is a **linear ordering plus one loop** — no cycles, no
parallel states — so a general BFS/DFS-friendly graph or a state-machine
framework would be pure machinery for a problem that a table solves.

---

## 5. Taxi routing (reuse of Phase 1 Dijkstra)

Phase 2 uses Dijkstra **unchanged**:

```
dijkstra(graph, gate, runway_access_node)   # departure
dijkstra(graph, runway_access_node, gate)   # arrival (reverse direction)
```

- Every time an aircraft leaves a gate, one Dijkstra call builds its taxi route.
- Blocked taxiways are already invisible to `Graph.get_neighbors`, so a
  re-route needs **no change** to Dijkstra (see §11).

### DAA justification (template applied to Dijkstra)

| Question | Answer |
|---|---|
| 1. Problem solved | Shortest (cheapest) taxi path between two nodes, repeatedly. |
| 2. Why the problem needs an algorithm | Naive alternatives (enumerate all paths) grow exponentially; a taxi path is a shortest-path problem. |
| 3. Input | `Graph`, `start`, `destination`, non-negative integer weights. |
| 4. Output | `(path: list[str], total_cost: float)`; `([], inf)` if unreachable; `([start], 0)` if start == destination. |
| 5. Pseudocode | see `docs/dijkstra.md` (min-heap, lazy deletion). |
| 6. Time | O((V+E) log V). |
| 7. Space | O(V+E) (adjacency list, `distances`, `previous`, heap). |
| 8. Why appropriate | Exact, deterministic, optimal for non-negative weights; already tested against a brute-force oracle. |
| 9. Alternatives | BFS (only unweighted), Bellman–Ford (handles negatives we forbid), A* (needs an admissible heuristic for taxi networks where geometry ≠ cost). |
| 10. Why not selected | BFS gives the wrong answer on weighted edges; Bellman–Ford is slower everywhere; A* adds heuristic-design cost with no benefit on a 25-node graph. |

---

## 6. Priority queue / runway scheduling

**The genuine problem:** several aircraft want one runway. The simulator must
pick the *next* aircraft deterministically. That is exactly "give me the
minimum element of a dynamic set" — a **priority queue** is appropriate, and
this is the second genuine heap usage in the project (Dijkstra being the first).

### Concrete policy (must be defined, not hand-waved)

The queue is a **binary min-heap / priority queue** whose elements are ordered
by a **lexicographic priority ordering** on the key:

    (priority_class, request_time, aircraft_id)

- `priority_class` — `0` = **landing**, `1` = **takeoff**.
- Within the same class, an **earlier `request_time` wins** (first-come-first-served).
- `aircraft_id` (plain string order) is the **final deterministic tie-breaker**.

The heap always pops the smallest key under this ordering. Note that this
*ordering policy* is a design decision about the key — it is **not** itself a
DAA algorithm; the DAA data structure is the **binary min-heap / priority
queue**.

**Stored:** `(priority_class, request_time, aircraft_id)`.

**Selection:** `heapq.heappop` — the aircraft with the smallest
`(priority_class, request_time, aircraft_id)`.

**Insertion:** `heapq.heappush(queue, (priority_class, request_time, aircraft_id))`.

**Removal:** `heapq.heappop(queue)` — only the selected aircraft leaves the
queue; it becomes the current runway occupant.

**Priority changes** (e.g., a holding landing aircraft has exceeded its holding
budget → its class is bumped below `0`): push the entry again under the new
`(priority_class, request_time, id)` key and **lazily discard** the old tuple
when it is later popped — the *exact same lazy-deletion pattern Dijkstra already
uses*, a consistent viva story.

### Runway protocol — no preemption of the current occupant

"Landing priority" means **priority when selecting the *next* runway occupant**,
never an interruption of an aircraft already on the runway.

```
   RUNWAY FREE                 RUNWAY BUSY
        │                           │
        ▼                           ▼
  select minimum heap entry   current operation continues
        │                           │
   ├─ landing   priority 0          │
   ├─ takeoff   priority 1          │
   └─ request_time breaks ties      │
        │                    runway becomes free
        ▼                           │
    new occupant                     ▼
                              select next aircraft
```

In words:

1. While the runway is busy, no selection happens — the current operation
   (`TAKEOFF` / `LANDING`) runs to completion and the runway is released at its
   `release_tick`.
2. Only when the runway is free: pop the minimum key
   `(priority_class, request_time, aircraft_id)` and skip any stale
   (lazy-deleted) entries.
3. The selected aircraft goes `LINE_UP` → `TAKEOFF` / `LANDING`, and the runway
   becomes busy again for a fixed `RUNWAY_TIME`.

```
RUNWAY-SCHEDULER-next()
    if runway busy: wait until release_tick          # no preemption
    if queue empty: do nothing
    entry = HEAP-EXTRACT-MIN(queue)                  # O(log n)
    skip stale entries (lazy deletion)
    assign runway release_tick = now + RUNWAY_TIME
    aircraft  -> LINE_UP
```

| Question | Answer |
|---|---|
| 6. Time | each insert/extract O(log n), n = waiting aircraft. |
| 7. Space | O(n) heap. |
| 8. Why appropriate | "smallest key first" is precisely a min-heap; matches Dijkstra's heap for one shared story. |
| 9. Alternative | plain FIFO queue — correct but *no* priority policy; sorted list — O(n) insert. |
| 10. Why not | FIFO cannot express arrival-over-departure priority or priority bumps; a sorted list makes insert quadratic in the worst case. |

**Separate from taxi routing:** yes. Runway and taxiway are different shared
resources with different scheduling (priority queue vs. per-aircraft shortest
path). Mixing them in one structure would couple two independent problems.

---

## 7. Airspace representation

Deliberately minimal: a **2D plane with an altitude layer**, because conflict
detection separates aircraft by *horizontal* distance (2D) plus a *vertical*
difference. Full 3D adds nothing but cost here.

```
Aircraft in airspace:
    position  (x, y)           metres
    altitude  (alt)            feet
    speed     (v)              m/s
    heading   (heading)        degrees from north
```

Airspace routes are **straight line segments between a small fixed set of
waypoints** (e.g. `DEP_EAST`, `DEP_WEST`, `ARR_EAST` …). This is enough: every
conflict reduces to "two moving points on segments", which is the exact shape
the deterministic detector (§9) solves. No grid, no curved paths, no
aerodynamics.

Positions over the airport *surface* are nodes (`T1`, `R1`, …); positions in the
airspace are continuous `(x, y, alt)`. The two live in different layers and an
aircraft converts coordinates at `TAKEOFF` (node → continuous) and `LANDING`
(continuous → node).

---

## 8. Aircraft movement algorithm

### Taxi (discrete per edge)

```
TAXI-STEP(aircraft, dt)
    if current_tick < holding_until: return          # HOLD: no advance (§3)
    next = aircraft.taxi_route[route_index+1]
    # RULE (§11): the CURRENT edge always finishes; only the NEXT edge is
    # checked before the aircraft enters it.
    if edge node->next currently blocked:            # §11
        re-route and return
    aircraft.distance_on_edge += taxi_speed * dt
    if distance_on_edge >= weight(node, next):       # arrived
        aircraft.route_index++
        aircraft.distance_on_edge = 0
        if route complete: -> WAITING_FOR_RUNWAY (departure)
                             or COMPLETED (arrival)
```

Edge dwell time = `weight / taxi_speed`; the graph weight is *distance in units*
— one calibration constant converts units to metres (§18). Stepping happens per
tick, so the tick–movement link is trivial. This is an **iteration**, not a
sophisticated algorithm: movement is node-to-node interpolation plus arrival
checks.

### Airspace (continuous per tick)

```
UPDATE-POSITION(aircraft, dt)
    if current_tick < holding_until: return          # HOLD: no advance (§3)
    if next_waypoint is None: advance to next segment
    dx, dy = ORIENT(aircraft.speed, aircraft.heading)
    aircraft.position += (dx*dt, dy*dt)
    aircraft.altitude  += climb_rate * dt      # toward target_altitude
    if reached next_waypoint: move to next waypoint or enter
                              approach/landing sequence
```

Two equations, no physics. `heading` is constant per segment; the primary
conflict action is `HOLD` (§10), which pauses movement via the guard above.
`REDUCE_SPEED` (if ever added) is an OPTIONAL extra and is not part of the
minimum viable conflict resolution (§10).

---

## 9. Conflict detection (the core Phase 2 algorithm)

### Problem

Two aircraft in the airspace may become closer than the minimum permitted
separation *within* a prediction horizon. The simulator must know **in
advance**, and for free with all pairs.

### Why an algorithm

Checking current separation alone is wrong (they may be far apart *now* and on
a collision course). Re-simulating physics every pair every tick is wasteful.
The geometric **closest-point-of-approach (CPA)** formula gives the *minimum
possible* separation over the horizon in O(1) — the exact, deterministic answer.

### Formal definitions

- Horizontal separation threshold: `S_h` (e.g. 5 km).
- Vertical separation threshold: `S_v` (e.g. 1000 ft).
- Prediction horizon: `H` seconds.

### Input

- Vehicle A: `posA = (xA, yA)`, `velA = (vxA, vyA)` (derived from `speed`,
  `heading`), `altA`.
- Vehicle B: `posB, velB, altB`.
- Horizon `H`, thresholds `S_h`, `S_v`.

### Output

- `CONFLICT` with `t_cpa` (predicted time), `(x_cpa, y_cpa)` (predicted
  location of the closest approach), and `d_min`, **or** `NO_CONFLICT`.

### Algorithm

```
CPA(A, B, H, S_h, S_v)
    d0 = A.pos - B.pos                 # initial relative position
    rv = A.vel - B.vel                 # relative velocity vector
    if |rv|^2 == 0:                    # parallel equal velocity: constant offset
        t* = 0
    else:
        t* = -(d0 · rv) / |rv|^2       # minimises |d0 + rv·t|^2
        t* = clamp(t*, 0, H)           # only the horizon matters
    d_min = |d0 + rv·t*|

    if d_min <= S_h  and  |altA - altB| <= S_v:
        return CONFLICT at time t*, position A.pos + A.vel·t*
    return NO_CONFLICT
```

**Maths behind it (viva-ready):** separation²(t) = `|d0 + rv·t|²` is a convex
quadratic in `t`. Its minimum is where the derivative is zero, i.e.
`(d0 + rv·t)·rv = 0`. Everything follows from that one line. Because the
distance function is convex, the minimum over a closed interval lies either at
the stationary point or at the endpoints — so clamping `t*` to `[0, H]` is
correct, **no** numerical search is needed.

### DAA justification

| Question | Answer |
|---|---|
| 1. Problem solved | Predict whether two aircraft ever violate the separation minima within the horizon. |
| 2. Why an algorithm | Naive "sample the future every few seconds" is an approximation with tunable error; CPA is exact, closed-form, O(1). |
| 3. Input | A/B state (pos, vel, alt), `H`, `S_h`, `S_v`. |
| 4. Output | `(CONFLICT, t_cpa, loc, d_min)` or none. |
| 5. Pseudocode | above. |
| 6. Time | O(1) per pair → O(A²) for all pairs, A = airborne aircraft. |
| 7. Space | O(1) per pair; O(A²) if results stored, O(1) if discarded after use. |
| 8. Why appropriate | Exact, deterministic, explainable, matches a 2D+altitude model; no randomness to confuse a viva. |
| 9. Alternatives | time-stepped numeric prediction; machine learning; spatial grid spatial-partitioning first. |
| 10. Why not | time-stepping is approximate and needs a magic step; ML is non-deterministic, unexplainable, and contradicts the educational goal; a grid only helps *pair enumeration*, not the per-pair math — revisit as an optimisation in a later phase if aircraft count grows. |

---

## 10. Conflict resolution

**Primary and deterministic mechanism: `HOLD`.** One `REDUCE_SPEED` operation is
*not* guaranteed to make a conflict disappear, so Phase 2 does not assume that.
A held aircraft fully stops moving for a fixed, defined interval (§3), which
deterministically changes its trajectory over time; resolution is verified by
**recomputing CPA** and, if needed, holding again.

Minimum action set for Phase 2: **`HOLD`** plus **`RELEASE`** (issued when the
conflict is gone). `REDUCE_SPEED` is **OPTIONAL** and not part of the minimum
viable implementation. Everything else (TURN_LEFT/RIGHT, CLIMB/DESCEND,
CHANGE_HEADING) is **REMOVED** — each would add geometric re-planning
(waypoints/segments) for little educational value beyond CPA itself.

### Deterministic choice of which aircraft to hold

Hold the **lower-priority** aircraft of the pair, ordered by the same
lexicographic key as the runway heap (§6):

1. Compare runway/traffic priority classes if applicable — the aircraft with
   the **higher `priority_class`** (e.g. takeoff `1` vs. landing `0`) is held.
2. If the classes are equal, hold the aircraft with the **later `request_time`**.
3. If still tied, `aircraft_id` (string order) is the final deterministic
   tie-breaker.

This is the same `(priority_class, request_time, aircraft_id)` ordering used in
§6, so one policy serves both selection and conflict resolution.

### Procedure

```
CONFLICT DETECTED (CPA, §9)
    ↓
select aircraft to delay     (lower priority, rule above)
    ↓
HOLD for HOLD_DURATION ticks (§3)
    ↓
recalculate CPA
    ↓
if conflict remains:  extend HOLD / keep aircraft held
    ↓
if conflict is clear: RELEASE
```

```
RESOLVE(A, B)
    HOLDA = LOWER-PRIORITY(A, B)              # deterministic tie-breaks above
    HOLDA.holding_until = current_tick + HOLD_DURATION
    emit "ATC: <HOLDA.id> HOLD"
    while CPA(A, B, H, S_h, S_v) reports CONFLICT:
        HOLDA.holding_until = current_tick + HOLD_DURATION     # extend HOLD
    emit "ATC: <HOLDA.id> RELEASE"
```

Why HOLD is sufficient for the *educational* scope: aircraft fly straight
segments, so deferring one in time increases their temporal separation without
re-designing its geometry; each HOLD extension defers the held aircraft by a
defined number of ticks, so the loop *converges in practice* for a handful of
aircraft. To keep the rule deterministic and bounded, a **maximum number of
HOLD extensions per conflict** is fixed (a single constant); if that cap is
reached the conflict is reported instead of looping forever (§18).

No claim is made that one instruction always resolves a conflict — only that
the simulator recomputes CPA and keeps the aircraft held deterministically
until the conflict clears. If a viva demands *spatial* separation, `TURN` can be
added later as an OPTIONAL waypoint-insertion action.

---

## 11. Dynamic rerouting

Phase 1 already modelling blocked edges; Phase 2 decides **when** and **how** to
react.

1. **Detection.** Two triggers, both cheap:
   - *On arrival:* at the start of `TAXI-STEP`, verify the **next** planned edge
     `(current_node, next_node)` is still present and unblocked via
     `graph.has_edge`. `has_edge` already returns `False` for blocked edges.
   - *On block event:* the simulator polls `Graph.is_blocked` for every
     taxiing aircraft's current edge once per tick (25 edges max → negligible).
2. **Should Dijkstra re-run immediately?** Yes — from the aircraft's **current
   node**. On this graph a Dijkstra call is microseconds and it is the *same*
   proven function; there is no benefit in re-planning later.
3. **Mid-edge blockage — finish the current edge.** If `T2–T3` becomes blocked
   while an aircraft is already travelling along it, the aircraft **finishes
   the current edge** and reaches `T3`. Blockage takes effect only at the next
   entry decision:
   - The **CURRENT edge** may finish even if it becomes blocked *after* the
     aircraft has already entered it.
   - The **NEXT edge** must be available *before* the aircraft enters it.
   - The aircraft is never interrupted mid-edge and is never teleported.
4. **On arrival / before the next edge.** At the moment the aircraft arrives at
   `T3`, it checks whether the next planned edge `(T3, next)` is blocked. If
   yes, it re-routes; if no, it enters that edge and continues normally (§8's
   `TAXI-STEP` shows exactly this ordering).
5. **New route selection.** `dijkstra(graph, current_node, destination)` with
   the blocked edge already invisible to `get_neighbors`. Replace
   `taxi_route`, reset `route_index`, keep the destination and state.
6. **If no route exists.** Dijkstra returns `([], inf)`. The aircraft is marked
   `HOLDING_NO_ROUTE`, stays at its node, and the simulator continues with other
   aircraft; the incident is reported. Unblock a later edge and a re-route
   attempt is made again automatically.

```
REROUTE(aircraft)                    # called when the NEXT edge is blocked (§11.4)
    new_path = DIJKSTRA(graph, aircraft.node, aircraft.destination)
    if new_path == ([], inf): aircraft -> HOLDING_NO_ROUTE; report; return
    aircraft.taxi_route = new_path; aircraft.route_index = 0
    emit "ATC: <A> reroute via <new_path>"
```

---

## 12. Simulation clock

| Criterion | Fixed time step (chosen) | Event-driven |
|---|---|---|
| Conceptual simplicity for a viva | direct: "every second, do the same thing" | needs an event queue, clock abstraction |
| Ordering determinism | all aircraft updated in a fixed, documented order | must specify tie-breaking manually |
| Visual/educational clarity | progress visible each tick | progress hidden inside events |
| Efficiency | fine: 25 nodes, handful of aircraft | a win only with very many idle entities |
| Fit with Phase 1 | Dijkstra/tick both trivial | extra machinery for no real benefit |

**Decision: fixed time step.** The genuine benefit of event-driven simulation —
not repeatedly polling idle entities — disappears at this scale (dozens of
aircraft, not millions). Introduce it in a later phase only if running a
full-day scenario, which we are not.

---

## 13. Data structures summary

| Problem | Structure | Why |
|---|---|---|
| Airport connectivity | adjacency-list dict (already in `Graph`) | O(deg) neighbour access; no memory blow-up |
| Shortest taxi path | binary min-heap + two dicts (already in Dijkstra) | textbook optimal for weighted graphs |
| Runway queue | binary min-heap `(priority_class, request_time, aircraft_id)` | "smallest key first" + lazy-deletion priority bumps |
| Aircraft lifecycle | `enum` + transition dict | fixed finite states (see §4) |
| Aircraft records | class/dataclass | grouped fields + invariants |
| Airspace | continuous `(x,y,alt)` + waypoint segments | 2D+altitude is enough for CPA |
| All-pairs conflict check | O(A²) nested loop | A is tiny; a spatial grid is postponed optimization (§18) |
| **Explicitly not used** | MST, max-flow, topological sort, DP, tries, etc. | no genuine computational problem in this simulator needs them |

---

## 14. DAA algorithms used (final list)

1. **Dijkstra with binary heap + lazy deletion** — taxi routing and dynamic
   re-routing. *Phase 1, reused unchanged.*
2. **Priority queue (binary heap)** — runway scheduling.
3. **Closest-point-of-approach (analytic minimisation)** — conflict detection.
4. **Lexicographic priority ordering on `(priority_class, request_time,
   aircraft_id)`** — the runway-selection policy shared by conflict resolution.
   This is a deterministic *ordering policy*, **not** a separate DAA algorithm;
   the DAA data structure behind it is the binary min-heap / priority queue
   (item 2).

Everything else is data modelling (enum state machine, waypoint following),
which the viva can classify honestly as *not* algorithmic.

---

## 15. Complexity analysis

| Component | Time | Space |
|---|---|---|
| Taxi route (one Dijkstra) | O((V+E) log V), V=25, E≈66 directed | O(V+E) |
| Dynamic re-route (per triggered aircraft) | O((V+E) log V) | O(V+E) |
| Runway scheduler step | O(log n), n = waiting aircraft | O(n) |
| Conflict detection (all pairs) | O(A²) with O(1) per pair, A = airborne | O(1) per pair |
| Conflict resolution (HOLD + recompute loop) | O(k) with k ≤ MAX_HOLD_EXTENSIONS (a fixed constant) | O(1) |
| One simulation tick | O(A log A + A² + (V+E) log V · reroutes) | O(V + E + A) |

For the planned scale — 25 nodes, ~66 directed edges, a handful of aircraft —
even the "worst" per-tick step is microseconds; complexity discussion is
genuine (the professor can ask about it) rather than a performance crisis.

---

## 16. Features to remove

| # | Feature | Verdict | Reason |
|---|---|---|---|
| 1 | ML-based aircraft prediction | **REMOVE** | Non-deterministic, unexplainable, violates the educational goal; CPA is exact and O(1). |
| 2 | Realistic aircraft physics | **REMOVE** | Would dominate complexity; nothing DAA is learned. |
| 3 | Real weather | **REMOVE** | Adds randomness that obscures deterministic algorithm behaviour. |
| 4 | Wind simulation | **REMOVE** | Same as weather; it changes velocities *externally* and muddies CPA teaching. |
| 5 | Complex 3D flight dynamics | **REMOVE** | 2D + altitude is sufficient; full 3D adds cost without a viva story. |
| 6 | Full real-world ATC rules | **REMOVE** | Huge, domain-heavy, not algorithm teaching. |
| 7 | Real airport maps | **OPTIONAL** | Only replaces the *data*, never the algorithms; revisit as a visual nicety. |
| 8 | Multiple airports | **REMOVE** | Every algorithm is fully exercised on one airport. |
| 9 | Fuel optimisation | **REMOVE** | A separate optimization problem; distracts from core algorithms. |
| 10 | Passenger scheduling | **REMOVE** | Unrelated to the traffic/route algorithms. |
| 11 | Airline scheduling | **REMOVE** | Unrelated; adds a timetable subsystem. |
| 12 | Aircraft maintenance | **REMOVE** | Unrelated to movement/conflict algorithms. |
| 13 | AI-generated ATC voice | **REMOVE** | Not DAA; a gimmick. Plain-text instructions have the same educational value. |
| 14 | Real-time external aviation data | **REMOVE** | Data feeds break determinism and reproducibility of results. |

**What stays (the minimum set):** the Phase 1 graph + Dijkstra, taxiing,
priority-queue runway scheduling, CPA conflict detection, temporal conflict
resolution, dynamic rerouting, a fixed-step clock, and text output.

---

## 17. Phase 2 implementation order

Each step is testable before moving on:

1. `Aircraft` class + `AircraftStateMachine` (enum + transitions table) + tests.
2. `TAXI-STEP`: node-to-node movement along a Dijkstra route, arrival
   detection + tests.
3. `RunwayScheduler`: min-heap over `(priority_class, request_time, aircraft_id)`,
   non-preemptive, landings/takeoff sequencing + tests.
4. Departure pipeline end-to-end: gate → aim at runway request → takeoff.
5. Airspace model + waypoint movement (`UPDATE-POSITION`) + tests.
6. `ConflictDetector` (CPA) + hand-computed cross-checks (collision course,
   parallel, diverging, exactly-at-threshold).
7. `ConflictResolver` (HOLD + RELEASE, HOLD_DURATION) + recompute-after-HOLD
   + HOLD-extension/cap tests.
8. Arrival pipeline: approach → landing → runway occupancy → taxi to gate.
9. `REROUTE` on blocked edge + `HOLDING_NO_ROUTE` handling + tests.
10. Full-clock test: a scripted scenario with a fixed expected end state.

Every milestone reuses the Phase 1 test style (`pytest`, known-answer tests,
brute-force cross-checks where cheap).

---

## 18. Risks and edge cases

| Risk/edge case | Handling |
|---|---|
| `([], inf)` on re-route | aircraft marked `HOLDING_NO_ROUTE`, no teleport, report emitted (§11). |
| Blocked edge mid-edge | the CURRENT edge finishes; the NEXT edge is checked before entry (§8, §11). |
| HOLD never clears a conflict | repeat HOLD until CPA is clear; a fixed maximum extension count caps the loop, then the conflict is reported (§10). |
| `start == destination` in a re-route | Dijkstra already returns `([node], 0)`; treat as "already there", no-op. |
| Zero-weight edges | allowed by `Graph`; CPA/dwell-time math still fine (dwell 0 edges are instantaneous). |
| Ties in Dijkstra path | accepted; tests assert cost-only for tied routes (§9 of tests). |
| Two aircraft with identical velocity (parallel offset) | CPA special-cases `|rv|=0`, distance constant (no call, separation maths ✅). |
| `t*` outside horizon | clamp to `[0, H]`; the quadratic is convex, so this is exact, not heuristic. |
| Runway occupied during landing-takeoff mix | policy: current occupant finishes first; queue is non-blocking. |
| Zero airborne aircraft | O(A²) loop over empty set — trivially fine; tick still advances clock. |
| Weights as "distance units" vs. real metres | introduce ONE calibration constant (`unit → m`) and `taxi_speed`; document it (Phase 1 never used real units, so nothing breaks). |
| Unbounded holding (e.g., no route ever) | simulator continues; scenario ends when all aircraft are `COMPLETED` or `HOLDING_NO_ROUTE`. |

---

## 19. Final recommended architecture (minimum viable Phase 2)

**Phase 2 implementation (keep):**
- Graph + Dijkstra — reused as-is (taxi routing + rerouting). *(exists)*
- `enum`+table aircraft state machine.
- Binary-heap runway scheduler over `(priority_class, request_time,
  aircraft_id)`, non-preemptive, with lazy deletion.
- 2D+altitude airspace with waypoint segments.
- Waypoint/node-stepping movement, held while `current_tick < holding_until`.
- CPA conflict detection (analytic min separation over a horizon).
- Temporal conflict resolution — `HOLD` primary, `REDUCE_SPEED` optional.
- Dynamic re-route on blocked edges.
- Fixed time-step clock.
- Text-only ATC-like output.

**Postponed to later phases (optional and clearly separable):**
- Spatial hash grid for all-pairs conflict checks (only if aircraft count grows).
- `TURN`-based *spatial* conflict resolution (waypoint insertion).
- Real airport map as *data only* (§16 #7).
- A console/animation frontend.
- Event-driven clock (only if full-day batch scenarios matter).

**Explicitly out of scope (removed, §16):** ML prediction, physics, weather,
wind, 3D dynamics, full ATC rules, multi-airport, fuel/passenger/airline/
maintenance scheduling, AI voice, live data feeds.

The result is a simulator where *every* computational decision traces to a
small set of explainable, deterministic algorithms — the strongest position for
a DAA viva — and Phase 2 adds **exactly one new genuinely algorithmic piece**
(CPA) plus deterministic policy structures (heap scheduling, state table) around
the existing Phase 1 core.