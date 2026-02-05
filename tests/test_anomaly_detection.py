import numpy as np
import pandas as pd

from device_anomaly.models.anomaly_detector import AnomalyDetectorConfig
from device_anomaly.models.heuristics import apply_heuristics, build_rules_from_dicts
from device_anomaly.models.hybrid import HybridAnomalyDetector, HybridAnomalyDetectorConfig


def _make_timestamped(n: int) -> pd.Series:
    return pd.date_range("2025-01-01", periods=n, freq="D")


def test_hybrid_flags_clear_outlier():
    """
    Build a tiny dataset with mostly normal battery drop + RSSI and one extreme outlier.
    The outlier should be tagged as an anomaly.
    """
    normal_rows = pd.DataFrame(
        {
            "DeviceId": 1,
            "Timestamp": _make_timestamped(30),
            "TotalBatteryLevelDrop": np.random.normal(10, 1, size=30),
            "Rssi": np.random.normal(-60, 2, size=30),
        }
    )
    outlier = pd.DataFrame(
        {
            "DeviceId": [1],
            "Timestamp": [pd.Timestamp("2025-02-01")],
            "TotalBatteryLevelDrop": [120],
            "Rssi": [-110],
        }
    )
    df = pd.concat([normal_rows, outlier], ignore_index=True)

    detector = HybridAnomalyDetector(
        HybridAnomalyDetectorConfig(
            iso_config=AnomalyDetectorConfig(contamination=0.1),
            use_cohort_models=False,
        )
    )
    detector.fit(df)
    scored = detector.score_dataframe(df)

    anomalies = scored[scored["anomaly_label"] == -1]
    # Expect the extreme row to be anomalous and at least one anomaly present
    assert not anomalies.empty
    assert pd.Timestamp("2025-02-01") in anomalies["Timestamp"].values


def test_heuristics_push_bad_device_over_threshold():
    """
    Heuristic flags should contribute to hybrid score and surface reasons.
    """
    df = pd.DataFrame(
        {
            "DeviceId": [101, 102],
            "Timestamp": _make_timestamped(2),
            "TotalBatteryLevelDrop": [50, 10],
            "Rssi": [-95, -60],
        }
    )

    heuristic_cfg = [
        {
            "name": "battery_hot",
            "column": "TotalBatteryLevelDrop",
            "threshold": 40,
            "op": ">=",
            "min_consecutive": 1,
            "severity": 1.0,
            "description": "Battery drain well above normal",
        }
    ]
    heuristic_flags = apply_heuristics(df, build_rules_from_dicts(heuristic_cfg))

    detector = HybridAnomalyDetector(
        HybridAnomalyDetectorConfig(
            iso_config=AnomalyDetectorConfig(contamination=0.1),
            use_cohort_models=False,
        )
    )
    detector.fit(df)
    scored = detector.score_dataframe(df, heuristic_flags=heuristic_flags)

    flagged = scored[scored["DeviceId"] == 101].iloc[0]
    control = scored[scored["DeviceId"] == 102].iloc[0]

    assert flagged["heuristic_score"] > 0
    assert flagged["anomaly_label"] == -1  # penalized by heuristic
    assert control["anomaly_label"] == 1  # stays normal
