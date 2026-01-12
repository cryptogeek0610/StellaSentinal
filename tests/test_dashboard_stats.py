import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from device_anomaly.api.main import app


def _reset_results_db(tmp_path: Path) -> None:
    os.environ["RESULTS_DB_PATH"] = str(tmp_path / "results.db")
    from device_anomaly.database import connection as db_connection

    db_connection._ENGINE = None
    db_connection._SESSION_FACTORY = None


def test_dashboard_stats_uses_trained_config(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    metadata = {
        "config": {
            "contamination": 0.02,
            "n_estimators": 777,
            "random_state": 99,
        },
        "detector_config": {
            "contamination": 0.02,
            "n_estimators": 777,
            "random_state": 99,
            "scale_features": False,
            "min_variance": 0.123,
        },
        "feature_cols": ["BatteryDrop_roll_mean", "Upload_delta"],
        "artifacts": {"model_path": "isolation_forest.pkl"},
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata))

    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(model_dir))
    _reset_results_db(tmp_path)

    client = TestClient(app)
    response = client.get("/api/dashboard/isolation-forest/stats")
    assert response.status_code == 200
    payload = response.json()

    assert payload["config"]["n_estimators"] == 777
    assert payload["config"]["contamination"] == 0.02
    assert payload["config"]["scale_features"] is False
    assert payload["defaults"]["n_estimators"] != 777
