from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from device_anomaly.llm.client import get_default_llm_client, strip_thinking_tags
from device_anomaly.config.feature_config import FeatureConfig


def build_event_llm_payload(row: pd.Series) -> dict[str, Any]:
    """
    Build a compact payload for LLM from an ml_AnomalyEvents row.
    Expects columns: DeviceId, EventStart, EventEnd, DurationMinutes,
    AnomalyScoreMin/Mean, RowCount, MetricsJson.
    """
    metrics = json.loads(row["MetricsJson"]) if row.get("MetricsJson") else {}

    event_start = pd.to_datetime(row["EventStart"])
    event_end = pd.to_datetime(row["EventEnd"])
    duration_days = 1
    if pd.notna(event_start) and pd.notna(event_end):
        duration_days = max(1, (event_end - event_start).days + 1)

    payload = {
        "device_id": int(row["DeviceId"]),
        "event_start": str(row["EventStart"]),
        "event_end": str(row["EventEnd"]),
        "duration_days": int(duration_days),
        "row_count": int(row["RowCount"]),
        "worst_anomaly_score": float(row["AnomalyScoreMin"]),
        "mean_anomaly_score": float(row["AnomalyScoreMean"]),
        "metrics": metrics,
    }
    return payload


def build_event_prompt(payload: dict[str, Any]) -> str:
    json_str = json.dumps(payload, indent=2)

    prompt = f"""
You are an assistant helping explain anomaly *events* in daily device telemetry.

You receive JSON describing a device event composed of consecutive anomalous days.
Fields include:
- event window (start/end are dates)
- duration in days / row counts
- anomaly scores
- aggregated daily metrics captured during the event

JSON:
```json
{json_str}
Please:

Briefly summarize what is unusual about this stretch of days.

Mention 2-3 key metrics that look abnormal for those days (state if higher/lower than
normal daily values).

Suggest 1-2 possible next steps (high-level, non-technical). Keep it under 200 words.
"""
    return prompt.strip()

def generate_and_save_event_explanation(
    engine: Engine,
    event_id: int,
    ) -> str:
    """
    Load a single event by Id, generate an explanation via LLM,
    store it back in dbo.ml_AnomalyEvents.Explanation, and return it.
    """
    with engine.begin() as conn:
        df = pd.read_sql(
        text(
        """
        SELECT *
        FROM dbo.ml_AnomalyEvents
        WHERE Id = :event_id
        """
        ),
        conn,
        params={"event_id": event_id},
        )

        if df.empty:
            raise ValueError(f"No event found with Id={event_id}")

        row = df.iloc[0]

        # If already present, you can short-circuit if you want
        if row.get("Explanation"):
            return row["Explanation"]

        payload = build_event_llm_payload(row)
        prompt = build_event_prompt(payload)

        llm = get_default_llm_client()
        raw_explanation = llm.generate(prompt)
        explanation = strip_thinking_tags(raw_explanation)

        conn.execute(
            text(
                """
                UPDATE dbo.ml_AnomalyEvents
                SET Explanation = :exp
                WHERE Id = :event_id
                """
            ),
            {"exp": explanation, "event_id": event_id},
        )

    return explanation
