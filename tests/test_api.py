"""Tests for the StellaSentinal FastAPI server."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path so server imports resolve
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """TestClient as context manager so the lifespan handler runs."""
    with TestClient(app) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "version" in data


# ── Workforce ─────────────────────────────────────────────────────────────────

class TestWorkforce:
    def test_get_workforce(self, client):
        r = client.get("/api/workforce")
        assert r.status_code == 200
        data = r.json()
        assert "squad_name" in data
        assert "agents" in data
        assert isinstance(data["agents"], list)

    def test_get_agents(self, client):
        r = client.get("/api/workforce/agents")
        assert r.status_code == 200
        agents = r.json()
        assert isinstance(agents, list)
        assert len(agents) > 0
        assert "id" in agents[0]

    def test_get_agent_by_id(self, client):
        r = client.get("/api/workforce/agents/jim")
        assert r.status_code == 200
        assert r.json()["name"] == "Jim"

    def test_get_agent_not_found(self, client):
        r = client.get("/api/workforce/agents/nonexistent")
        assert r.status_code == 404


# ── Kanban ────────────────────────────────────────────────────────────────────

class TestKanban:
    def test_get_kanban(self, client):
        r = client.get("/api/kanban")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_create_kanban_task(self, client):
        r = client.post("/api/kanban", json={
            "title": "Test task",
            "status": "todo",
            "owner": "jim",
            "project": "SAAP PLUS",
        })
        assert r.status_code == 201
        data = r.json()
        assert data["task"]["title"] == "Test task"
        assert "index" in data

    def test_create_kanban_task_minimal(self, client):
        r = client.post("/api/kanban", json={"title": "Minimal task"})
        assert r.status_code == 201

    def test_create_kanban_task_empty_title_fails(self, client):
        r = client.post("/api/kanban", json={"title": ""})
        assert r.status_code == 422

    def test_update_kanban_task(self, client):
        create = client.post("/api/kanban", json={"title": "Update me"})
        idx = create.json()["index"]
        r = client.patch(f"/api/kanban/{idx}", json={"status": "done"})
        assert r.status_code == 200
        assert r.json()["task"]["status"] == "done"

    def test_update_kanban_out_of_range(self, client):
        r = client.patch("/api/kanban/9999", json={"status": "done"})
        assert r.status_code == 404


# ── Watercooler ───────────────────────────────────────────────────────────────

class TestWatercooler:
    def test_get_watercooler(self, client):
        r = client.get("/api/watercooler")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_post_watercooler_message(self, client):
        r = client.post("/api/watercooler", json={
            "agent_id": "dwight",
            "message": "Bears. Beets. Battlestar Galactica.",
        })
        assert r.status_code == 201
        data = r.json()
        assert data["entry"]["agent_id"] == "dwight"

    def test_post_watercooler_invalid_agent(self, client):
        r = client.post("/api/watercooler", json={
            "agent_id": "toby",
            "message": "Nobody wants Toby here.",
        })
        assert r.status_code == 404


# ── Anomaly/NBA ───────────────────────────────────────────────────────────────

class TestAnomaly:
    def test_predict_with_training_data(self, client):
        training = [[1, 2, 3]] * 50 + [[100, 200, 300]]
        predict_data = [[1, 2, 3], [100, 200, 300]]
        r = client.post("/api/anomaly/predict", json={
            "training_data": training,
            "feature_matrix": predict_data,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 2
        assert len(data["scores"]) == 2
        assert "anomaly_count" in data
        assert data["total"] == 2

    def test_recommend_connectivity(self, client):
        r = client.post("/api/anomaly/recommend", json={
            "type": "connectivity",
            "root_cause_hint": "Profile Misconfiguration",
            "summary": "Connectivity loss post-push",
            "affected_cohort": "Site-A",
        })
        assert r.status_code == 200
        rec = r.json()["recommendation"]
        assert rec["action"] == "Rollback Profile"

    def test_recommend_unknown(self, client):
        r = client.post("/api/anomaly/recommend", json={
            "type": "mystery",
        })
        assert r.status_code == 200
        assert r.json()["recommendation"]["action"] == "Manual Investigation Required"
