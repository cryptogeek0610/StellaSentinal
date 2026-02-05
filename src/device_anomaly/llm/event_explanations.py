from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from device_anomaly.llm.client import get_default_llm_client, strip_thinking_tags
from device_anomaly.llm.prompt_utils import (
    NO_THINKING_INSTRUCTION,
    get_duration_interpretation,
    get_severity_emoji,
    get_severity_word,
)


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
    """Build prompt for explaining multi-day anomaly events."""
    json_str = json.dumps(payload, indent=2)

    duration = payload.get("duration_days", 1)
    worst_score = payload.get("worst_anomaly_score", 0)
    mean_score = payload.get("mean_anomaly_score", 0)

    severity_word = get_severity_word(worst_score)
    severity_emoji = get_severity_emoji(worst_score)
    duration_interpretation = get_duration_interpretation(duration)

    prompt = f"""<role>
You are a device health analyst explaining multi-day anomaly events to IT operations staff managing enterprise mobile devices in warehouses and retail operations.
</role>

<output_format>
{NO_THINKING_INSTRUCTION}

Structure your response EXACTLY as:

EVENT SUMMARY
Duration: {duration} day(s) | Severity: {severity_word.title()} {severity_emoji}
[1-2 sentences: What happened over this period]

PATTERN OBSERVED
[Describe the trend - was it getting worse, stable, or improving?]

MOST AFFECTED AREAS
1. [Metric category]: [Brief description of the issue]
2. [Another category]: [Description]

ROOT CAUSE HYPOTHESIS
[Most likely explanation for this sustained anomaly]

RECOMMENDED RESPONSE
Priority: [High/Medium/Low]
- [Specific action to take]
- [Follow-up action if first doesn't resolve]
</output_format>

<event_data>
Device: {payload.get("device_id")}
Event Window: {payload.get("event_start")} to {payload.get("event_end")}
Duration: {duration} consecutive anomalous days
Worst Anomaly Score: {worst_score:.3f} (more negative = more severe)
Average Anomaly Score: {mean_score:.3f}

Aggregated Metrics During Event:
{json_str}
```

<duration_context>
{duration_interpretation}
</duration_context>

<common_multi_day_patterns>
Multi-day events suggest persistent issues:
- Gradual battery degradation → aging battery needs replacement
- Persistent connectivity issues → device location or hardware problem
- Sustained high usage → either legitimate heavy use or potential misuse
- Recurring crashes → software bug or compatibility issue
</common_multi_day_patterns>

<instructions>
1. Consider whether the anomaly trend is worsening, stable, or improving
2. Multi-day events suggest persistent causes, not random glitches
3. Prioritize actionable recommendations
4. Keep response under 200 words
5. If metrics suggest heavy but legitimate usage, note that
</instructions>"""
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
