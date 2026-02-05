from __future__ import annotations

import json

import pandas as pd

from device_anomaly.config.feature_config import FeatureConfig


def group_anomalies_to_events(
    anomalies_df: pd.DataFrame,
    max_gap_hours: int = 2,
) -> pd.DataFrame:
    """
    Group row-level anomalies into contiguous events per device.

    An event is a sequence of anomalous rows for the same DeviceId where gaps
    between consecutive anomalies are <= max_gap_hours.
    """
    if anomalies_df.empty:
        return anomalies_df.copy()

    df = anomalies_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["DeviceId", "Timestamp"])

    # Compute time difference between consecutive anomalies per device
    df["time_diff"] = df.groupby("DeviceId")["Timestamp"].diff()

    max_gap = pd.Timedelta(hours=max_gap_hours)
    df["new_event"] = df["time_diff"].isna() | (df["time_diff"] > max_gap)

    # event_id is per-device cumulative count of "new_event"
    df["event_id"] = df.groupby("DeviceId")["new_event"].cumsum()

    events: list[dict] = []
    for (device_id, _event_id), group in df.groupby(["DeviceId", "event_id"]):
        event_start = group["Timestamp"].min()
        event_end = group["Timestamp"].max()
        duration_minutes = max(1, int((event_end - event_start).total_seconds() / 60))

        score_min = float(group["anomaly_score"].min())
        score_max = float(group["anomaly_score"].max())
        score_mean = float(group["anomaly_score"].mean())
        row_count = int(len(group))

        # Aggregate metrics (mean per event for now)
        metrics = {}
        for col in FeatureConfig.genericFeatures:
            if col in group.columns:
                metrics[col] = float(group[col].mean())

        events.append(
            {
                "DeviceId": int(device_id),
                "EventStart": event_start,
                "EventEnd": event_end,
                "DurationMinutes": duration_minutes,
                "AnomalyScoreMin": score_min,
                "AnomalyScoreMax": score_max,
                "AnomalyScoreMean": score_mean,
                "RowCount": row_count,
                **metrics,
            }
        )

    return pd.DataFrame(events)


def build_event_results(
    anomalies_df: pd.DataFrame,
    source: str,
    model_version: str,
    max_gap_hours: int = 2,
) -> pd.DataFrame:
    events_df = group_anomalies_to_events(anomalies_df, max_gap_hours=max_gap_hours)
    if events_df.empty:
        return events_df

    records = []
    for _, row in events_df.iterrows():
        metrics = {
            col: row[col] for col in FeatureConfig.genericFeatures if col in events_df.columns
        }

        records.append(
            {
                "Source": source,
                "DeviceId": int(row["DeviceId"]),
                "EventStart": row["EventStart"],
                "EventEnd": row["EventEnd"],
                "DurationMinutes": int(row["DurationMinutes"]),
                "AnomalyScoreMin": float(row["AnomalyScoreMin"]),
                "AnomalyScoreMax": float(row["AnomalyScoreMax"]),
                "AnomalyScoreMean": float(row["AnomalyScoreMean"]),
                "RowCount": int(row["RowCount"]),
                "ModelVersion": model_version,
                "MetricsJson": json.dumps(metrics, default=float),
                # explanation is generated later in the UI
                "Explanation": None,
            }
        )

    return pd.DataFrame.from_records(records)


def select_top_anomalous_devices(
    df_scored: pd.DataFrame,
    top_n: int = 10,
    min_total_points: int = 50,
    min_anomalies: int = 3,
) -> list[int]:
    """
    Rank devices by anomaly rate and return the top_n device IDs.

    - df_scored is the full scored dataframe (normal + anomalies)
    - anomaly_label == -1 => anomaly

    We require:
      - each device has at least min_total_points rows
      - at least min_anomalies anomalies
    """
    if df_scored.empty:
        return []

    total_per_device = df_scored.groupby("DeviceId").size()
    anomalies_per_device = df_scored[df_scored["anomaly_label"] == -1].groupby("DeviceId").size()

    stats = pd.DataFrame(
        {
            "total": total_per_device,
            "anomalies": anomalies_per_device,
        }
    ).fillna(0)
    stats["anomaly_rate"] = stats["anomalies"] / stats["total"].clip(lower=1)

    # apply minimum filters
    stats = stats[(stats["total"] >= min_total_points) & (stats["anomalies"] >= min_anomalies)]

    if stats.empty:
        return []

    # sort by anomaly_rate desc, then by anomalies count
    stats = stats.sort_values(
        ["anomaly_rate", "anomalies"],
        ascending=[False, False],
    )

    return [int(did) for did in stats.head(top_n).index]
