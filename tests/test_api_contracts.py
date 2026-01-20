import json
import os
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from device_anomaly.api.main import app
from device_anomaly.data_access.anomaly_persistence import persist_anomaly_results
from device_anomaly.features.device_features import DeviceFeatureBuilder
from device_anomaly.models.anomaly_detector import AnomalyDetectorIsolationForest, AnomalyDetectorConfig
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tiny_telemetry.csv"


def _reset_results_db(tmp_path: Path) -> None:
    os.environ["RESULTS_DB_PATH"] = str(tmp_path / "results.db")
    from device_anomaly.database import connection as db_connection

    db_connection._ENGINE = None
    db_connection._SESSION_FACTORY = None


def _seed_anomalies(tmp_path: Path) -> None:
    _reset_results_db(tmp_path)
    df_raw = pd.read_csv(FIXTURE_PATH)
    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"])

    builder = DeviceFeatureBuilder()
    df_feat = builder.transform(df_raw)

    detector = AnomalyDetectorIsolationForest(
        config=AnomalyDetectorConfig(contamination=0.4, random_state=42)
    )
    detector.fit(df_feat)
    scored = detector.score_dataframe(df_feat)
    persist_anomaly_results(scored, only_anomalies=True)


def _write_baselines(model_dir: Path) -> None:
    payload = {
        "schema_version": "data_driven_v1",
        "baseline_type": "data_driven",
        "generated_at": "2025-01-01T00:00:00Z",
        "baselines": {
            "TotalBatteryLevelDrop": {
                "global": {"median": 5.0, "mad": 1.0, "sample_count": 10},
                "by_device_type": {},
                "by_hour": None,
                "thresholds": {"p95": 9.5, "p99": 12.0},
            }
        },
    }
    (model_dir / "baselines.json").write_text(json.dumps(payload))


def test_baseline_endpoints_contract(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _write_baselines(model_dir)

    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(model_dir))
    _reset_results_db(tmp_path)

    client = TestClient(app)

    response = client.get("/api/baselines/features")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["feature"] == "TotalBatteryLevelDrop"
    assert "baseline" in payload[0]
    assert "observed" in payload[0]
    assert "status" in payload[0]
    assert "drift_percent" in payload[0]
    assert "mad" in payload[0]
    assert "sample_count" in payload[0]

    response = client.get("/api/baselines/suggestions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_baseline_suggestions_empty_without_anomalies(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _write_baselines(model_dir)

    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(model_dir))
    _reset_results_db(tmp_path)

    client = TestClient(app)
    response = client.get("/api/baselines/suggestions")
    assert response.status_code == 200
    assert response.json() == [], "Expected empty suggestions when no anomalies exist"


def test_anomaly_endpoints_contract(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_ARTIFACTS_DIR", raising=False)
    _seed_anomalies(tmp_path)

    client = TestClient(app)
    response = client.get("/api/anomalies?page=1&page_size=10")
    assert response.status_code == 200
    payload = response.json()
    assert "anomalies" in payload
    assert "total" in payload

    response = client.get("/api/anomalies/grouped")
    assert response.status_code == 200
    grouped = response.json()
    assert "groups" in grouped
    assert "total_anomalies" in grouped
