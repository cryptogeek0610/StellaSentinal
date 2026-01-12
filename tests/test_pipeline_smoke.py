import os
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from device_anomaly.data_access.anomaly_persistence import _select_feature_columns, persist_anomaly_results
from device_anomaly.features.device_features import DeviceFeatureBuilder
from device_anomaly.models.anomaly_detector import AnomalyDetectorConfig, AnomalyDetectorIsolationForest
from device_anomaly.models.baseline import compute_data_driven_baselines


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tiny_telemetry.csv"


def load_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def reset_results_db(tmp_path: Path) -> None:
    os.environ["RESULTS_DB_PATH"] = str(tmp_path / "results.db")
    from device_anomaly.database import connection as db_connection

    db_connection._ENGINE = None
    db_connection._SESSION_FACTORY = None


def test_data_driven_baselines_temporal_handling():
    df = load_fixture()
    baselines = compute_data_driven_baselines(
        df=df,
        feature_cols=["TotalBatteryLevelDrop"],
        timestamp_col="Timestamp",
        include_temporal=True,
        min_samples=2,
    )

    baseline = baselines["TotalBatteryLevelDrop"]
    assert baseline.by_hour is not None
    assert 0 in baseline.by_hour

    df_no_ts = df.drop(columns=["Timestamp"])
    baselines_no_ts = compute_data_driven_baselines(
        df=df_no_ts,
        feature_cols=["TotalBatteryLevelDrop"],
        timestamp_col="Timestamp",
        include_temporal=True,
        min_samples=2,
    )
    assert baselines_no_ts["TotalBatteryLevelDrop"].by_hour is None


def test_feature_parity_train_inference():
    df_raw = load_fixture()
    builder = DeviceFeatureBuilder()
    df_feat = builder.transform(df_raw)

    detector = AnomalyDetectorIsolationForest(
        config=AnomalyDetectorConfig(contamination=0.4, random_state=42)
    )
    detector.fit(df_feat)
    assert set(detector.feature_cols).issubset(df_feat.columns)

    scored = detector.score_dataframe(df_feat)
    assert "anomaly_score" in scored.columns
    assert len(scored) == len(df_feat)


def test_model_artifact_roundtrip(tmp_path: Path):
    df_raw = load_fixture()
    builder = DeviceFeatureBuilder()
    df_feat = builder.transform(df_raw)

    detector = AnomalyDetectorIsolationForest(
        config=AnomalyDetectorConfig(contamination=0.4, random_state=42)
    )
    detector.fit(df_feat)

    model_base = tmp_path / "isolation_forest"
    paths = detector.save_model(model_base, export_onnx=False)

    loaded = AnomalyDetectorIsolationForest.load_model(paths["sklearn"])
    scored = loaded.score_dataframe(df_feat.head(2))
    assert "anomaly_score" in scored.columns


def test_persistence_feature_column_selection():
    df_raw = load_fixture()
    builder = DeviceFeatureBuilder()
    df_feat = builder.transform(df_raw)

    feature_cols = _select_feature_columns(df_feat)
    assert any(col.endswith("_roll_mean") for col in feature_cols)


def test_end_to_end_smoke_serves_anomalies(tmp_path: Path):
    reset_results_db(tmp_path)

    df_raw = load_fixture()
    builder = DeviceFeatureBuilder()
    df_feat = builder.transform(df_raw)

    detector = AnomalyDetectorIsolationForest(
        config=AnomalyDetectorConfig(contamination=0.4, random_state=42)
    )
    detector.fit(df_feat)
    scored = detector.score_dataframe(df_feat)

    persisted = persist_anomaly_results(scored, only_anomalies=True)
    assert persisted > 0

    from device_anomaly.api.main import app

    client = TestClient(app)
    response = client.get("/api/anomalies")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 1
