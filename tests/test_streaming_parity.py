import subprocess
import sys
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from device_anomaly.features.cohort_stats import (
    CohortStatsStore,
    apply_cohort_stats,
    compute_cohort_stats,
)
from device_anomaly.features.device_features import DeviceFeatureBuilder, compute_feature_norms
from device_anomaly.models.anomaly_detector import (
    AnomalyDetectorConfig,
    AnomalyDetectorIsolationForest,
)
from device_anomaly.streaming.anomaly_processor import AnomalyStreamProcessor
from device_anomaly.streaming.feature_computer import StreamingFeatureComputer
from device_anomaly.streaming.telemetry_stream import DeviceBuffer, TelemetryBuffer, TelemetryEvent

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "streaming_parity_events.csv"
BUFFER_MAX_AGE_HOURS = 24 * 365 * 10


def make_buffer(device_id: int) -> DeviceBuffer:
    return DeviceBuffer(device_id=device_id, max_age_hours=BUFFER_MAX_AGE_HOURS)


def load_parity_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def build_event(row: pd.Series, metric_cols: list[str]) -> TelemetryEvent:
    metrics: dict[str, float] = {}
    for col in metric_cols:
        value = row[col]
        if pd.isna(value):
            continue
        metrics[col] = float(value)

    return TelemetryEvent(
        device_id=int(row["DeviceId"]),
        timestamp=row["Timestamp"].to_pydatetime(),
        metrics=metrics,
        manufacturer_id=int(row["ManufacturerId"]),
        model_id=int(row["ModelId"]),
        os_version_id=int(row["OsVersionId"]),
        firmware_version=None
        if pd.isna(row.get("FirmwareVersion"))
        else str(row.get("FirmwareVersion")),
        tenant_id="default",
    )


def test_feature_parity_batch_vs_streaming():
    df_raw = load_parity_fixture().sort_values(["DeviceId", "Timestamp"])
    builder = DeviceFeatureBuilder(compute_cohort=False)
    df_base = builder.transform(df_raw)
    feature_norms = compute_feature_norms(df_base)

    builder = DeviceFeatureBuilder(compute_cohort=False, feature_norms=feature_norms)
    df_feat = builder.transform(df_raw)

    cohort_payload = compute_cohort_stats(df_feat, min_samples=2)
    cohort_store = CohortStatsStore(cohort_payload)
    df_feat = apply_cohort_stats(df_feat, cohort_store)

    df_feat_keyed = df_feat.set_index(["DeviceId", "Timestamp"])

    detector = AnomalyDetectorIsolationForest(
        config=AnomalyDetectorConfig(contamination=0.2, random_state=42)
    )
    detector.fit(df_feat)

    processor = AnomalyStreamProcessor(engine=None)
    processor._model = detector

    metric_cols = [
        col
        for col in df_raw.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]

    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=cohort_store,
        feature_spec=feature_spec,
        feature_norms=feature_norms,
        feature_mode="canonical",
    )

    buffers: dict[int, DeviceBuffer] = {}

    for _, row in df_raw.iterrows():
        event = build_event(row, metric_cols)
        buffer = buffers.setdefault(event.device_id, make_buffer(device_id=event.device_id))
        buffer.add_event(event)

        stream_features = computer._compute_features(event, buffer)
        ts_key = pd.Timestamp(event.timestamp)
        batch_row = df_feat_keyed.loc[(event.device_id, ts_key)]
        if isinstance(batch_row, pd.DataFrame):
            batch_row = batch_row.iloc[-1]

        for name, value in stream_features.items():
            batch_value = batch_row.get(name)
            if pd.isna(batch_value):
                continue
            assert value == pytest.approx(float(batch_value), abs=1e-6)

        batch_score = detector.score_dataframe(batch_row.to_frame().T)["anomaly_score"].iloc[0]
        stream_score = processor._score_features(
            device_id=event.device_id,
            timestamp=event.timestamp,
            features=stream_features,
            cohort_id=event.cohort_id,
            tenant_id="default",
        ).anomaly_score
        assert stream_score == pytest.approx(float(batch_score), abs=1e-4)


def test_streaming_state_rehydration():
    df_raw = load_parity_fixture()
    df_device = df_raw[df_raw["DeviceId"] == 1].sort_values("Timestamp").head(5)

    builder = DeviceFeatureBuilder(compute_cohort=False)
    feature_norms = compute_feature_norms(builder.transform(df_device))
    feature_spec = builder.get_feature_spec()

    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_norms=feature_norms,
        feature_mode="canonical",
    )

    metric_cols = [
        col
        for col in df_device.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]

    buffer = TelemetryBuffer()
    device_buffer = make_buffer(device_id=1)
    buffer._buffers[1] = device_buffer

    events = [build_event(row, metric_cols) for _, row in df_device.iterrows()]
    for event in events[:3]:
        device_buffer.add_event(event)

    snapshot = buffer.snapshot()
    restored = TelemetryBuffer.from_snapshot(snapshot)

    next_event = events[3]
    device_buffer.add_event(next_event)
    restored.get_buffer(1).add_event(next_event)

    features_original = computer._compute_features(next_event, device_buffer)
    features_restored = computer._compute_features(next_event, restored.get_buffer(1))

    for name, value in features_original.items():
        if name not in features_restored:
            continue
        assert value == pytest.approx(float(features_restored[name]), abs=1e-6)


def test_verify_command():
    result = subprocess.run(
        [sys.executable, "scripts/verify.py", "--check", "--skip-frontend"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "verify" in result.stdout.lower()


def test_parity_no_events_returns_empty():
    builder = DeviceFeatureBuilder(compute_cohort=False)
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    event = TelemetryEvent(
        device_id=1,
        timestamp=pd.Timestamp("2026-01-07T00:00:00Z").to_pydatetime(),
        metrics={"TotalBatteryLevelDrop": 1.0},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    buffer = make_buffer(device_id=1)
    features = computer._compute_features(event, buffer)
    assert features == {}, "Expected empty features when buffer has no events"


def test_parity_single_event_missing_rollups():
    df_raw = load_parity_fixture().head(1)
    builder = DeviceFeatureBuilder(compute_cohort=False)
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    metric_cols = [
        col
        for col in df_raw.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]
    event = build_event(df_raw.iloc[0], metric_cols)
    buffer = make_buffer(device_id=event.device_id)
    buffer.add_event(event)

    features = computer._compute_features(event, buffer)
    has_rollup = any(name.endswith("_roll_mean") for name in features.keys())
    assert not has_rollup, "Expected no rolling features with a single event"


def test_parity_below_window_missing_rollups():
    df_raw = load_parity_fixture().head(2)
    builder = DeviceFeatureBuilder(
        compute_cohort=False,
        window=3,
        rolling_windows=[3],
        min_periods=3,
    )
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    metric_cols = [
        col
        for col in df_raw.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]
    buffer = make_buffer(device_id=int(df_raw.iloc[0]["DeviceId"]))
    for _, row in df_raw.iterrows():
        buffer.add_event(build_event(row, metric_cols))

    event = build_event(df_raw.iloc[-1], metric_cols)
    features = computer._compute_features(event, buffer)
    has_rollup = any(name.endswith("_roll_mean") for name in features.keys())
    assert not has_rollup, "Expected no rolling features when history < window size"


def test_parity_exact_window_rollups():
    # Use last 3 events for device 1 (recent timestamps to avoid pruning)
    df_full = load_parity_fixture()
    df_raw = df_full[df_full["DeviceId"] == 1].tail(3)
    builder = DeviceFeatureBuilder(
        compute_cohort=False,
        window=3,
        rolling_windows=[3],
        min_periods=3,
    )
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    metric_cols = [
        col
        for col in df_raw.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]
    buffer = make_buffer(device_id=int(df_raw.iloc[0]["DeviceId"]))
    for _, row in df_raw.iterrows():
        buffer.add_event(build_event(row, metric_cols))

    event = build_event(df_raw.iloc[-1], metric_cols)
    features = computer._compute_features(event, buffer)
    rollup_keys = [name for name in features if name.endswith("_roll_mean")]
    assert rollup_keys, "Expected rolling features when window size is satisfied"


def test_parity_over_window_rollups():
    df_raw = load_parity_fixture().head(4)
    builder = DeviceFeatureBuilder(
        compute_cohort=False,
        window=3,
        rolling_windows=[3],
        min_periods=3,
    )
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    metric_cols = [
        col
        for col in df_raw.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]
    buffer = make_buffer(device_id=int(df_raw.iloc[0]["DeviceId"]))
    for _, row in df_raw.iterrows():
        buffer.add_event(build_event(row, metric_cols))

    event = build_event(df_raw.iloc[-1], metric_cols)
    features = computer._compute_features(event, buffer)
    rollup_keys = [name for name in features if name.endswith("_roll_mean")]
    assert rollup_keys, "Expected rolling features when history exceeds window size"


def test_parity_handles_nan_and_inf():
    df_raw = load_parity_fixture().head(2).copy()
    df_raw.loc[df_raw.index[0], "Download"] = float("inf")
    df_raw.loc[df_raw.index[1], "Upload"] = float("nan")

    builder = DeviceFeatureBuilder(compute_cohort=False)
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    metric_cols = [
        col
        for col in df_raw.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]
    buffer = make_buffer(device_id=int(df_raw.iloc[0]["DeviceId"]))
    for _, row in df_raw.iterrows():
        buffer.add_event(build_event(row, metric_cols))

    event = build_event(df_raw.iloc[-1], metric_cols)
    features = computer._compute_features(event, buffer)
    assert all(np.isfinite(list(features.values()))), "Features must not include NaN/inf values"


def test_parity_out_of_order_timestamps():
    df_raw = load_parity_fixture().head(3).copy()
    shuffled = df_raw.iloc[[2, 0, 1]].reset_index(drop=True)
    builder = DeviceFeatureBuilder(compute_cohort=False)
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    metric_cols = [
        col
        for col in df_raw.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]
    buffer = make_buffer(device_id=int(df_raw.iloc[0]["DeviceId"]))
    for _, row in shuffled.iterrows():
        buffer.add_event(build_event(row, metric_cols))

    event = build_event(shuffled.iloc[-1], metric_cols)
    features = computer._compute_features(event, buffer)
    assert features, "Expected features even when events arrive out of order"


def test_parity_selects_event_timestamp_row():
    builder = DeviceFeatureBuilder(compute_cohort=False)
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    event_early = TelemetryEvent(
        device_id=1,
        timestamp=pd.Timestamp("2026-01-07T00:00:00Z").to_pydatetime(),
        metrics={"TotalBatteryLevelDrop": 1.0},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    event_late = TelemetryEvent(
        device_id=1,
        timestamp=pd.Timestamp("2026-01-08T00:00:00Z").to_pydatetime(),
        metrics={"TotalBatteryLevelDrop": 9.0},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    buffer = make_buffer(device_id=1)
    buffer.add_event(event_late)
    buffer.add_event(event_early)

    features = computer._compute_features(event_early, buffer)
    assert features.get("TotalBatteryLevelDrop") == 1.0


def test_parity_ignores_non_numeric_metrics():
    builder = DeviceFeatureBuilder(compute_cohort=False)
    feature_spec = builder.get_feature_spec()
    computer = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_mode="canonical",
    )
    event = TelemetryEvent(
        device_id=1,
        timestamp=pd.Timestamp("2026-01-07T00:00:00Z").to_pydatetime(),
        metrics={"TotalBatteryLevelDrop": 1.0, "NonNumeric": "oops"},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    buffer = make_buffer(device_id=1)
    buffer.add_event(event)

    features = computer._compute_features(event, buffer)
    assert "NonNumeric" not in features


def test_buffer_running_stats_exclude_inf():
    """Ensure inf values don't contaminate buffer running statistics."""
    from datetime import datetime, timedelta

    now = datetime.now(UTC)
    buffer = make_buffer(device_id=1)

    # Add normal event (recent timestamp to avoid pruning)
    event_normal = TelemetryEvent(
        device_id=1,
        timestamp=now - timedelta(hours=1),
        metrics={"Metric": 10.0},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    buffer.add_event(event_normal)

    # Add event with inf
    event_inf = TelemetryEvent(
        device_id=1,
        timestamp=now,
        metrics={"Metric": float("inf")},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    buffer.add_event(event_inf)

    # Running mean should be the first value only (inf excluded)
    mean = buffer.get_rolling_mean("Metric")
    assert mean is not None
    assert np.isfinite(mean)
    assert mean == 10.0, "Inf should be excluded from rolling stats"


def test_buffer_running_stats_exclude_negative_inf():
    """Ensure -inf values don't contaminate buffer running statistics."""
    from datetime import datetime, timedelta

    now = datetime.now(UTC)
    buffer = make_buffer(device_id=1)

    event_normal = TelemetryEvent(
        device_id=1,
        timestamp=now - timedelta(hours=1),
        metrics={"Metric": 20.0},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    buffer.add_event(event_normal)

    event_neg_inf = TelemetryEvent(
        device_id=1,
        timestamp=now,
        metrics={"Metric": float("-inf")},
        manufacturer_id=10,
        model_id=100,
        os_version_id=1000,
    )
    buffer.add_event(event_neg_inf)

    mean = buffer.get_rolling_mean("Metric")
    assert mean is not None
    assert np.isfinite(mean)
    assert mean == 20.0, "-Inf should be excluded from rolling stats"


def test_parity_out_of_order_produces_same_features():
    """Out-of-order arrivals should produce same features as sorted arrivals."""
    df_raw = load_parity_fixture().head(3).copy()
    builder = DeviceFeatureBuilder(compute_cohort=False)
    feature_norms = compute_feature_norms(builder.transform(df_raw))
    feature_spec = builder.get_feature_spec()

    metric_cols = [
        col
        for col in df_raw.columns
        if col
        not in {
            "DeviceId",
            "Timestamp",
            "ManufacturerId",
            "ModelId",
            "OsVersionId",
            "FirmwareVersion",
        }
    ]

    # Sorted order
    computer_sorted = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_norms=feature_norms,
        feature_mode="canonical",
    )
    buffer_sorted = make_buffer(device_id=1)
    for _, row in df_raw.sort_values("Timestamp").iterrows():
        buffer_sorted.add_event(build_event(row, metric_cols))
    event = build_event(df_raw.sort_values("Timestamp").iloc[-1], metric_cols)
    features_sorted = computer_sorted._compute_features(event, buffer_sorted)

    # Out-of-order (reverse)
    computer_ooo = StreamingFeatureComputer(
        engine=None,
        buffer=TelemetryBuffer(),
        cohort_stats=None,
        feature_spec=feature_spec,
        feature_norms=feature_norms,
        feature_mode="canonical",
    )
    buffer_ooo = make_buffer(device_id=1)
    for _, row in df_raw.sort_values("Timestamp", ascending=False).iterrows():
        buffer_ooo.add_event(build_event(row, metric_cols))
    features_ooo = computer_ooo._compute_features(event, buffer_ooo)

    # Features should match for the same event
    for name, value in features_sorted.items():
        if name in features_ooo:
            assert value == pytest.approx(features_ooo[name], abs=1e-6), (
                f"Feature {name} differs: sorted={value} vs ooo={features_ooo[name]}"
            )
