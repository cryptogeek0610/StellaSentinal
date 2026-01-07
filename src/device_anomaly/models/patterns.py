from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd

from device_anomaly.llm.client import get_default_llm_client
from device_anomaly.llm.prompt_utils import (
    get_health_status,
    NO_THINKING_INSTRUCTION,
)
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
    """Build prompt for summarizing device health patterns over a time period."""
    def _default(o):
        try:
            return float(o)
        except Exception:
            return str(o)

    json_str = json.dumps(payload, indent=2, default=_default)

    # Calculate health indicators
    anomaly_rate = payload.get("anomaly_rate", 0)
    event_count = payload.get("event_count", 0)
    health_status, health_emoji = get_health_status(anomaly_rate)

    prompt = f"""<role>
You are a device fleet analyst creating health summaries for operations managers who oversee hundreds of mobile devices in warehouses and retail locations.
</role>

<output_format>
{NO_THINKING_INSTRUCTION}

Structure your response EXACTLY as:

DEVICE HEALTH: {health_status} {health_emoji}

OVERVIEW
[2-3 sentences summarizing this device's behavior over the analysis period]

KEY FINDINGS
- [Most important observation about this device]
- [Second observation]
- [Third observation if relevant]

PATTERNS DETECTED
[Describe any recurring issues: "This device consistently shows..." or "No concerning patterns detected"]

COMPARISON TO FLEET
[How does this device compare to similar devices? Better/worse/average?]

RECOMMENDATION
[Single clear action: "Continue monitoring" / "Schedule for inspection" / "Replace battery" / etc.]
</output_format>

<device_analysis_data>
Analysis Period: {payload.get("period_start", "N/A")} to {payload.get("period_end", "N/A")}
Days Analyzed: {payload.get("total_days", "N/A")}
Anomalous Days: {payload.get("total_anomalies", "N/A")} ({anomaly_rate:.1%} of period)
Number of Anomaly Events: {event_count}

Device Metrics (Normal Days vs Anomalous Days):
{json_str}
```

<metric_categories>
When discussing metrics, translate technical names:
- Battery metrics (TotalBatteryLevelDrop, TotalDischargeTime): "battery performance"
- Connectivity (DisconnectCount, OfflineMinutes, AvgSignalStrength): "network reliability"
- Usage (AppForegroundTime, AppVisitCount): "device utilization"
- Stability (CrashCount, ANRCount): "app stability"
- Physical (TotalDropCnt): "physical handling"
</metric_categories>

<health_thresholds>
Reference for assessment:
- Anomaly rate <5%: Normal device behavior
- Anomaly rate 5-15%: Some issues, worth monitoring
- Anomaly rate >15%: Significant problems, needs attention
- Multiple events with worst_score < -0.7: Severe issues detected
</health_thresholds>

<instructions>
1. Write for an operations manager, not a data scientist
2. Use plain language: "battery drains quickly" not "elevated TotalBatteryLevelDrop"
3. Compare anomalous vs normal periods - what's different?
4. Be specific in recommendations (what action, not just "investigate")
5. Keep total response under 200 words
6. If device is healthy, say so clearly - don't manufacture concerns
</instructions>"""
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
