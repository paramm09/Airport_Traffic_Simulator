"""HTTP API: `uvicorn backend.api:app --reload` (port 8000)."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.algorithms.dijkstra import dijkstra
from backend.data.airport_graph import EDGES, build_airport_graph
from backend.data.airspace_waypoints import ALL_WAYPOINTS
from backend.models.aircraft import AircraftState
from backend.simulation.conflict_resolution import is_holding
from backend.simulation.simulation import (
    ScheduledChange, Simulation, build_arrival, build_departure,
)

app = FastAPI(title="Airport Traffic Simulator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

_AIR = {AircraftState[n] for n in
        ("TAKEOFF", "CLIMB", "CRUISE", "DESCENT", "APPROACH", "LANDING")}
MAX_TICKS = 1500


def _frame(sim: Simulation) -> list[dict]:
    out = []
    for a in sim.aircraft:
        row = {"id": a.id, "s": a.state.name, "hold": is_holding(a, sim.current_tick)}
        if a.state in _AIR:
            row.update(x=a.position[0], y=a.position[1], alt=a.altitude)
        elif a.taxi_route:
            i = min(a.route_index, len(a.taxi_route) - 1)
            nxt = a.taxi_route[min(i + 1, len(a.taxi_route) - 1)]
            row.update(a=a.taxi_route[i], b=nxt,
                       t=a.distance_on_edge / a.edge_length if a.edge_length else 0)
        out.append(row)
    return out


def _demo() -> Simulation:
    return Simulation(
        aircraft=[
            build_departure("AC001", "G1", "R1"), build_departure("AC002", "G4", "R2"),
            build_departure("AC003", "G7", "R3"), build_departure("AC004", "G10", "R4"),
            build_arrival("AC005", "G5", "R5"), build_departure("AC006", "G2", "R1"),
            build_departure("AC007", "G8", "R3"), build_arrival("AC008", "G9", "R4"),
            build_departure("AC009", "G6", "R2"),
        ],
        graph=build_airport_graph(),
        scheduled_changes=[ScheduledChange(6, "block", "T1", "R1"),
                           ScheduledChange(40, "unblock", "T1", "R1")],
    )


@app.get("/api/network")
def network():
    return {"edges": [list(e) for e in EDGES],
            "waypoints": [[w.id, w.x, w.y, w.altitude] for w in ALL_WAYPOINTS]}


@app.get("/api/replay")
def replay():
    """Run the demo scenario to completion; one frame per tick (deterministic)."""
    sim = _demo()
    frames = [_frame(sim)]
    while not sim.is_complete() and sim.current_tick < MAX_TICKS:
        sim.step()
        frames.append(_frame(sim))
    return {"frames": frames,
            "events": [[e.tick, e.kind, e.message, e.aircraft_id] for e in sim.events]}


@app.get("/api/route")
def route(start: str, end: str):
    """Dijkstra shortest taxi route."""
    graph = build_airport_graph()
    if not (graph.has_node(start) and graph.has_node(end)):
        raise HTTPException(404, "unknown node")
    path, cost = dijkstra(graph, start, end)
    return {"path": path, "cost": cost}
