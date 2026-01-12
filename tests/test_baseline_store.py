import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from device_anomaly.api.main import app
from device_anomaly.database.connection import get_results_db_session
from device_anomaly.database.schema import AnomalyResult
from device_anomaly.models.baseline_store import resolve_baselines


def _write_data_driven_baselines(path: Path) -> dict:
    payload = {
        "schema_version": "data_driven_v1",
        "baseline_type": "data_driven",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baselines": {
            "BatteryDrop_roll_mean": {
                "global": {"median": 10.0, "mad": 1.0},
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def _write_legacy_baselines(path: Path) -> dict:
    payload = {
        "global": [
            {
                "__group_key__": "all",
                "feature": "BatteryDrop_roll_mean",
                "median": 2.0,
                "mad": 0.5,
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def _reset_results_db(tmp_path: Path) -> None:
    os.environ["RESULTS_DB_PATH"] = str(tmp_path / "results.db")
    from device_anomaly.database import connection as db_connection

    db_connection._ENGINE = None
    db_connection._SESSION_FACTORY = None


def test_resolve_baselines_prefers_production(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    _write_legacy_baselines(artifacts_dir / "dw_baselines.json")

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    baselines_path = model_dir / "baselines.json"
    _write_data_driven_baselines(baselines_path)

    resolution = resolve_baselines("dw", models_dir=model_dir)
    assert resolution is not None
    assert resolution.kind == "data_driven"
    assert resolution.path == baselines_path


def test_resolve_baselines_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    legacy_path = artifacts_dir / "dw_baselines.json"
    _write_legacy_baselines(legacy_path)

    resolution = resolve_baselines("dw", models_dir=tmp_path / "models")
    assert resolution is not None
    assert resolution.kind == "legacy"
    assert resolution.path == legacy_path


def test_baseline_suggestions_use_production_baselines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    baselines_path = model_dir / "baselines.json"
    _write_data_driven_baselines(baselines_path)

    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(model_dir))
    _reset_results_db(tmp_path)

    session = get_results_db_session()
    try:
        session.add(
            AnomalyResult(
                tenant_id="default",
                device_id=1,
                timestamp=datetime.now(timezone.utc),
                anomaly_score=-1.5,
                anomaly_label=-1,
                feature_values_json=json.dumps({"BatteryDrop_roll_mean": 25.0}),
            )
        )
        session.commit()
    finally:
        session.close()

    client = TestClient(app)
    response = client.get("/api/baselines/suggestions")
    assert response.status_code == 200
    assert response.json()
