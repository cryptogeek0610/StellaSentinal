from datetime import datetime, timezone

import numpy as np
import pandas as pd

from device_anomaly.features.cohort_stats import (
    CohortStatsStore,
    apply_cohort_stats,
    compute_cohort_stats,
)
from device_anomaly.streaming.feature_computer import StreamingFeatureComputer
from device_anomaly.streaming.telemetry_stream import TelemetryBuffer, TelemetryEvent


def _make_cohort_df(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ManufacturerId": [1] * len(values),
            "ModelId": [1] * len(values),
            "OsVersionId": [1] * len(values),
            "BatteryDrop": values,
        }
    )


def test_cohort_stats_use_train_only():
    train_df = _make_cohort_df([10.0, 12.0, 14.0, 16.0])
    val_df = _make_cohort_df([100.0])

    train_payload = compute_cohort_stats(
        df=train_df,
        feature_cols=["BatteryDrop"],
        min_samples=2,
    )
    full_payload = compute_cohort_stats(
        df=pd.concat([train_df, val_df], ignore_index=True),
        feature_cols=["BatteryDrop"],
        min_samples=2,
    )

    cohort_id = "1_1_1"
    train_median = train_payload["stats"][cohort_id]["BatteryDrop"]["median"]
    full_median = full_payload["stats"][cohort_id]["BatteryDrop"]["median"]
    assert train_median != full_median

    store = CohortStatsStore(train_payload)
    val_scored = apply_cohort_stats(val_df, store)
    expected = (100.0 - train_median) / (train_payload["stats"][cohort_id]["BatteryDrop"]["mad"] + 1e-6)
    assert np.isclose(val_scored["BatteryDrop_cohort_z"].iloc[0], expected)


def test_cohort_stats_parity_streaming_offline():
    df = _make_cohort_df([10.0, 12.0])
    payload = compute_cohort_stats(
        df=df,
        feature_cols=["BatteryDrop"],
        min_samples=1,
    )
    store = CohortStatsStore(payload)

    offline = apply_cohort_stats(df.tail(1), store)
    offline_z = offline["BatteryDrop_cohort_z"].iloc[0]

    computer = StreamingFeatureComputer(engine=None, buffer=TelemetryBuffer(), cohort_stats=store)
    features: dict[str, float] = {}
    event = TelemetryEvent(
        device_id=1,
        timestamp=datetime.now(timezone.utc),
        metrics={"BatteryDrop": 12.0},
        manufacturer_id=1,
        model_id=1,
        os_version_id=1,
    )
    computer._add_cohort_features(event, features)

    assert "BatteryDrop_cohort_z" in features
    assert np.isclose(features["BatteryDrop_cohort_z"], offline_z)
