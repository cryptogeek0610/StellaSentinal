from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd

from device_anomaly.llm.client import get_default_llm_client
from device_anomaly.config.feature_config import (
    FeatureConfig
)

def _build_device_pattern_payload(
    device_id: int,
    df_scored_device: pd.DataFrame,
    events_device: pd.DataFrame,
    period_start,
    period_end,
) -> dict:
    """
    Build a compact JSON-like payload describing this device over the period.
    """

    device_id = int(device_id)

    total_points = int(len(df_scored_device))
    total_anomalies = int((df_scored_device["anomaly_label"] == -1).sum())
    anomaly_rate = float(total_anomalies / max(total_points, 1))

    anomaly_rows = df_scored_device[df_scored_device["anomaly_label"] == -1]

    if anomaly_rows.empty:
        worst_score = 0.0
        mean_score = 0.0
    else:
        worst_score = float(anomaly_rows["anomaly_score"].min())
        mean_score = float(anomaly_rows["anomaly_score"].mean())

    event_count = int(len(events_device))

    # Per-metric summary: typical vs anomalous means
    metric_summary = {}
    for col in FeatureConfig.genericFeatures:
        if col not in df_scored_device.columns:
            continue

        typical_mean = float(df_scored_device[col].mean())
        anomaly_mean = float(anomaly_rows[col].mean()) if not anomaly_rows.empty else typical_mean

        metric_summary[col] = {
            "typical_mean": typical_mean,
            "anomaly_mean": anomaly_mean,
        }

    # Optionally include top few events (by severity)
    top_events = []
    if not events_device.empty:
        ev_sorted = events_device.sort_values("AnomalyScoreMin").head(3)
        for _, e in ev_sorted.iterrows():
            start = pd.to_datetime(e["EventStart"])
            end = pd.to_datetime(e["EventEnd"])
            duration_days = 1
            if pd.notna(start) and pd.notna(end):
                duration_days = max(1, (end - start).days + 1)
            top_events.append(
                {
                    "event_start": str(e["EventStart"]),
                    "event_end": str(e["EventEnd"]),
                    "duration_days": int(duration_days),
                    "row_count": int(e["RowCount"]),
                    "worst_score": float(e["AnomalyScoreMin"]),
                }
            )

    payload = {
        "device_id": device_id,
        "period_start": str(period_start),
        "period_end": str(period_end),
        "total_points": total_points,
        "total_anomalies": total_anomalies,
        "anomaly_rate": anomaly_rate,
        "event_count": event_count,
        "worst_anomaly_score": worst_score,
        "mean_anomaly_score": mean_score,
        "metric_summary": metric_summary,
        "top_events": top_events,
    }
    return payload


def _build_pattern_prompt(payload: dict) -> str:
    def _default(o):
        try:
            # try to treat numeric-like stuff as float
            return float(o)
        except Exception:
            # fallback: string representation (e.g. timestamps)
            return str(o)
    
    json_str = json.dumps(payload, indent=2, default=_default)

    prompt = f"""
        You are an assistant summarizing anomaly patterns in **daily** device telemetry
        for a non-technical audience.

        You will receive JSON describing one device over a given time period:
        - overall anomaly rate
        - number of anomaly events
        - worst anomaly scores
        - average daily values for important metrics in normal vs anomalous periods
        - a few of the most severe events

        JSON:
        ```json
        {json_str}
        Please:

        1-Give a short summary of how "healthy" this device looks over the period.
        2-Highlight any clear patterns (for example: frequent network issues, persistent battery drain, repeated offline days).
        3-Mention 2-3 metrics where anomalous daily behavior is most different from normal.
        4-Suggest 1-2 possible follow-up actions (at a high level).
        5-Keep it under 250 words.
    """
    return prompt.strip()

def build_device_pattern_results(
    df_scored: pd.DataFrame,
    events_df: pd.DataFrame,
    source: str,
    model_version: str,
    period_start,
    period_end,
    device_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    if df_scored.empty or events_df.empty:
        return pd.DataFrame()

    if device_ids is None:
        device_ids = sorted(events_df["DeviceId"].unique())

    records = []
    for device_id in device_ids:
        df_dev = df_scored[df_scored["DeviceId"] == device_id]
        if df_dev.empty:
            continue

        events_dev = events_df[events_df["DeviceId"] == device_id]
        if events_dev.empty:
            continue

        payload = _build_device_pattern_payload(
            device_id=device_id,
            df_scored_device=df_dev,
            events_device=events_dev,
            period_start=period_start,
            period_end=period_end,
        )

        records.append(
            {
                "Source": source,
                "DeviceId": int(device_id),
                "PeriodStart": period_start,
                "PeriodEnd": period_end,
                "TotalPoints": int(payload["total_points"]),
                "TotalAnomalies": int(payload["total_anomalies"]),
                "AnomalyRate": float(payload["anomaly_rate"]),
                "EventCount": int(payload["event_count"]),
                "WorstAnomalyScore": float(payload["worst_anomaly_score"]),
                "MeanAnomalyScore": float(payload["mean_anomaly_score"]),
                "PatternJson": json.dumps(payload, default=float),
                # explanation deferred to UI / on-demand
                "Explanation": None,
                "ModelVersion": model_version,
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)
