from fastapi.testclient import TestClient

from backend.api import app

client = TestClient(app)


def test_replay_runs_to_completion():
    r = client.get("/api/replay").json()
    assert r["frames"] and all(a["s"] == "COMPLETED" for a in r["frames"][-1])


def test_route_and_unknown_node():
    assert client.get("/api/route?start=G1&end=R1").json()["path"][0] == "G1"
    assert client.get("/api/route?start=X&end=R1").status_code == 404
