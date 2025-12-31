from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json
import pandas as pd
from sqlalchemy import text

from device_anomaly.llm.client import BaseLLMClient, get_default_llm_client, strip_thinking_tags
from device_anomaly.config.feature_config import (
    FeatureConfig
)
from device_anomaly.models.drift_monitor import load_stats


@dataclass
class MetricStats:
    mean: float
    std: float


@dataclass
class ExplanationContext:
    """Holds typical stats for metrics so we can talk about what is 'abnormal'."""

    metric_stats: Dict[str, MetricStats]
    llm_client: BaseLLMClient


def build_explanation_context(df_reference: pd.DataFrame) -> ExplanationContext:
    """
    Compute mean/std for each metric from a reference DataFrame (e.g. training slice).
    """
    metric_stats: Dict[str, MetricStats] = {}
    for col in FeatureConfig.genericFeatures:
        if col not in df_reference.columns:
            continue
        series = df_reference[col].astype(float)
        metric_stats[col] = MetricStats(
            mean=float(series.mean()),
            std=float(series.std(ddof=0) or 1.0),  # avoid 0 std
        )

    llm_client = get_default_llm_client()
    return ExplanationContext(metric_stats=metric_stats, llm_client=llm_client)


def _build_anomaly_payload(row: pd.Series, ctx: ExplanationContext, top_k: int = 5) -> Dict:
    """
    Build a compact payload describing the anomaly for the LLM.
    Select top-k metrics by z-score magnitude.
    """
    metrics_payload = []
    for col, stats in ctx.metric_stats.items():
        if col not in row.index:
            continue

        value = float(row[col])
        z = (value - stats.mean) / stats.std if stats.std > 0 else 0.0

        metrics_payload.append(
            {
                "name": col,
                "value": value,
                "typical_mean": stats.mean,
                "typical_std": stats.std,
                "z_score": z,
            }
        )
        delta_col = f"{col}_delta"
        if delta_col in row.index:
            try:
                metrics_payload[-1]["day_over_day_delta"] = float(row[delta_col])
            except (TypeError, ValueError):
                metrics_payload[-1]["day_over_day_delta"] = None

    # Sort by |z| descending and keep top_k
    metrics_payload.sort(key=lambda m: abs(m["z_score"]), reverse=True)
    metrics_payload = metrics_payload[:top_k]

    payload = {
        "device_id": int(row.get("DeviceId", -1)),
        "timestamp": str(row.get("Timestamp", "")),
        "anomaly_score": float(row.get("anomaly_score", 0.0)),
        "metrics": metrics_payload,
    }
    return payload


def _build_prompt(payload: Dict) -> str:
    """
    Build a natural-language prompt for the LLM based on the anomaly payload.
    """
    json_str = json.dumps(payload, indent=2)

    prompt = f"""
        You are an assistant helping explain anomalies in **daily** device telemetry.

        You will receive a JSON object describing one device and an anomalous day
        (the timestamp marks the day). Each metric contains:
        - name
        - value (daily aggregate)
        - typical_mean
        - typical_std
        - z_score (how many standard deviations away from typical this value is)
        - optional day_over_day_delta (change from the prior day if available)

        JSON:
        ```json
        {json_str}
        Please:

        Briefly summarize what is unusual about this device on that day.

        Mention the top 2-3 most abnormal metrics, whether they are higher or lower
        than what devices typically report for a day, and relate them to the metric name.

        Suggest 1-2 possible interpretations or next steps (high-level, not too technical).

        Keep the explanation under 200 words.
        """
    return prompt.strip()

def explain_anomaly_row(row: pd.Series, ctx: ExplanationContext) -> Dict[str, str]:
    """
    Given a scored anomaly row and an ExplanationContext, return:
    - 'explanation': LLM-generated text (or dummy)
    - 'prompt': the prompt we sent (for debugging)
    """
    payload = _build_anomaly_payload(row, ctx)
    prompt = _build_prompt(payload)
    raw_text = ctx.llm_client.generate(prompt)
    text = strip_thinking_tags(raw_text)

    return {
        "explanation": text,
        "prompt": prompt,
    }


def _build_anomaly_payload_from_stats(
    metrics: dict,
    stats: dict,
    device_id: int,
    timestamp: str,
    anomaly_score: float,
    top_k: int = 5,
) -> dict:
    metrics_payload = []
    feature_stats = (stats or {}).get("features", {})

    for name, value in metrics.items():
        if name not in feature_stats:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        typical_median = float(feature_stats[name].get("median", 0.0))
        typical_mad = float(feature_stats[name].get("mad", 1.0) or 1.0)
        z = (numeric_value - typical_median) / typical_mad

        metrics_payload.append(
            {
                "name": name,
                "value": numeric_value,
                "typical_median": typical_median,
                "typical_mad": typical_mad,
                "z_score": z,
            }
        )

    metrics_payload.sort(key=lambda m: abs(m["z_score"]), reverse=True)
    metrics_payload = metrics_payload[:top_k]

    return {
        "device_id": int(device_id),
        "timestamp": timestamp,
        "anomaly_score": float(anomaly_score),
        "metrics": metrics_payload,
    }


def _build_prompt_from_stats(payload: dict) -> str:
    json_str = json.dumps(payload, indent=2)
    prompt = f"""
You are an assistant helping explain anomalies detected by Isolation Forest in daily device telemetry.

Each metric includes:
- name
- value (daily aggregate)
- typical_median
- typical_mad (robust deviation)
- z_score (robust z-score based on median/MAD)

JSON:
```json
{json_str}

Please:

Briefly summarize what is unusual about this device on that day.

Mention the top 2-3 most abnormal metrics, whether they are higher or lower
than typical for this device population.

Suggest 1-2 possible interpretations or next steps (high-level, not too technical).

Keep the explanation under 200 words.
"""
    return prompt.strip()


def generate_and_save_row_explanation(engine, result_id: int) -> str:
    """
    Load a single anomaly result by Id, generate an explanation via LLM,
    store it back in dbo.ml_AnomalyResults.Explanation, and return it.
    """
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT *
                FROM dbo.ml_AnomalyResults
                WHERE Id = :result_id
                """
            ),
            conn,
            params={"result_id": result_id},
        )

        if df.empty:
            raise ValueError(f"No anomaly result found with Id={result_id}")

        row = df.iloc[0]

        if row.get("Explanation"):
            return row["Explanation"]

        source = str(row.get("Source") or "dw").lower()
        stats_path = Path("artifacts/synthetic_stats.json" if source == "synthetic" else "artifacts/dw_stats.json")
        stats = load_stats(stats_path) or {}

        metrics = {}
        metrics_json = row.get("MetricsJson")
        if isinstance(metrics_json, str) and metrics_json.strip():
            try:
                metrics = json.loads(metrics_json)
            except json.JSONDecodeError:
                metrics = {}

        payload = _build_anomaly_payload_from_stats(
            metrics=metrics,
            stats=stats,
            device_id=int(row.get("DeviceId", -1)),
            timestamp=str(row.get("Timestamp", "")),
            anomaly_score=float(row.get("AnomalyScore", row.get("anomaly_score", 0.0))),
        )
        prompt = _build_prompt_from_stats(payload)

        llm = get_default_llm_client()
        raw_explanation = llm.generate(prompt)
        explanation = strip_thinking_tags(raw_explanation)

        conn.execute(
            text(
                """
                UPDATE dbo.ml_AnomalyResults
                SET Explanation = :exp
                WHERE Id = :result_id
                """
            ),
            {"exp": explanation, "result_id": result_id},
        )

    return explanation
