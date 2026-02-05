from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from device_anomaly.api.dependencies import get_backend_db
from device_anomaly.api.main import app
from device_anomaly.database import connection as results_connection
from device_anomaly.database.schema import AnomalyResult, DeviceMetadata
from device_anomaly.db.models import Base as BackendBase
from device_anomaly.db.models import Tenant
from device_anomaly.db.models_cost import DeviceTypeCost


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUST_CLIENT_HEADERS", "true")
    monkeypatch.setenv("DEFAULT_USER_ROLE", "admin")
    monkeypatch.setenv("COST_ALERTS_PATH", str(tmp_path / "cost_alerts.json"))
    monkeypatch.setenv("RESULTS_DB_PATH", str(tmp_path / "results.db"))

    results_connection._ENGINE = None
    results_connection._SESSION_FACTORY = None
    results_connection.get_results_db_engine()

    backend_path = tmp_path / "backend.db"
    engine = create_engine(
        f"sqlite:///{backend_path}",
        connect_args={"check_same_thread": False},
    )
    BackendBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    with Session() as session:
        session.add(Tenant(tenant_id="default", name="Default", tier="standard"))
        session.commit()

    def _get_backend_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_backend_db] = _get_backend_db
    app.state.backend_sessionmaker = Session

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
    if hasattr(app.state, "backend_sessionmaker"):
        delattr(app.state, "backend_sessionmaker")


def _seed_results_data():
    session = results_connection.get_results_db_session()
    try:
        session.add_all(
            [
                DeviceMetadata(
                    tenant_id="default",
                    device_id=1,
                    device_model="Zebra TC52",
                    device_name="Device 1",
                    status="online",
                ),
                AnomalyResult(
                    tenant_id="default",
                    device_id=1,
                    timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                    anomaly_score=-0.8,
                    anomaly_label=-1,
                    status="false_positive",
                ),
            ]
        )
        session.commit()
    finally:
        session.close()


def _seed_backend_cost(cost_id: int = 1):
    Session = app.state.backend_sessionmaker
    with Session() as session:
        session.add(
            DeviceTypeCost(
                id=cost_id,
                tenant_id="default",
                device_model="Zebra TC52",
                purchase_cost=50000,
                replacement_cost=60000,
                repair_cost_avg=15000,
                currency_code="USD",
            )
        )
        session.commit()


def test_cost_alert_lifecycle(client):
    headers = {"X-User-Role": "admin", "X-User-Id": "tester"}
    response = client.get("/api/costs/alerts", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"alerts": [], "total": 0}

    create_response = client.post(
        "/api/costs/alerts",
        headers=headers,
        json={
            "name": "Monthly anomaly cost",
            "threshold_type": "anomaly_cost_monthly",
            "threshold_value": 1000,
            "is_active": True,
        },
    )
    assert create_response.status_code == 201
    alert_id = create_response.json()["id"]

    update_response = client.put(
        f"/api/costs/alerts/{alert_id}",
        headers=headers,
        json={"threshold_value": 1200},
    )
    assert update_response.status_code == 200
    assert float(update_response.json()["threshold_value"]) == 1200

    delete_response = client.delete(f"/api/costs/alerts/{alert_id}", headers=headers)
    assert delete_response.status_code == 204


def test_battery_forecast_summary(client):
    _seed_results_data()
    _seed_backend_cost()

    response = client.get("/api/costs/battery-forecast")
    assert response.status_code == 200
    payload = response.json()
    assert "forecasts" in payload
    assert "total_replacements_due_90_days" in payload


def test_nff_summary(client):
    _seed_results_data()

    response = client.get("/api/costs/nff/summary")
    assert response.status_code == 200
    payload = response.json()
    assert "total_nff_count" in payload
    assert "nff_rate_percent" in payload


def test_update_hardware_cost(client):
    _seed_backend_cost(cost_id=42)

    headers = {"X-User-Role": "admin", "X-User-Id": "tester"}
    response = client.put(
        "/api/costs/hardware/42",
        headers=headers,
        json={"purchase_cost": 750.0},
    )
    assert response.status_code == 200
    assert float(response.json()["purchase_cost"]) == 750.0
