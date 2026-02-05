from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from device_anomaly.config.feature_config import FeatureConfig
from device_anomaly.llm.client import BaseLLMClient, get_default_llm_client, strip_thinking_tags
from device_anomaly.llm.prompt_utils import (
    NO_THINKING_INSTRUCTION,
    get_metric_info,
    get_severity_word,
    get_z_score_description,
)
from device_anomaly.costs.calculator import CostCalculator
from device_anomaly.costs.models import CostContext
from device_anomaly.models.drift_monitor import load_stats


@dataclass
class MetricStats:
    mean: float
    std: float


@dataclass
class ExplanationContext:
    """Holds typical stats for metrics so we can talk about what is 'abnormal'."""

    metric_stats: dict[str, MetricStats]
    llm_client: BaseLLMClient


def build_explanation_context(df_reference: pd.DataFrame) -> ExplanationContext:
    """
    Compute mean/std for each metric from a reference DataFrame (e.g. training slice).
    """
    metric_stats: dict[str, MetricStats] = {}
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


def _build_anomaly_payload(row: pd.Series, ctx: ExplanationContext, top_k: int = 5) -> dict:
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


def _build_prompt(payload: dict) -> str:
    """
    Build a natural-language prompt for the LLM based on the anomaly payload.
    """
    json_str = json.dumps(payload, indent=2)

    # Build human-readable metric summary
    metrics = payload.get("metrics", [])
    metric_lines = []
    for m in sorted(metrics, key=lambda x: abs(x.get("z_score", 0)), reverse=True)[:5]:
        name = m.get("name", "")
        z = m.get("z_score", 0)
        human_name, unit, desc = get_metric_info(name)
        severity_desc = get_z_score_description(z)
        metric_lines.append(f"- {human_name}: {severity_desc} (z={z:.1f})")

    metric_summary = "\n".join(metric_lines) if metric_lines else "No significant deviations"
    anomaly_score = payload.get("anomaly_score", 0)
    severity = get_severity_word(anomaly_score)

    prompt = f"""<role>
You are a device health advisor explaining anomaly alerts to warehouse supervisors and IT staff managing mobile devices (Android scanners, tablets) in enterprise operations.
</role>

<output_format>
{NO_THINKING_INSTRUCTION}

Structure your response as:

WHAT HAPPENED
[2-3 sentences describing the unusual behavior in plain English]

KEY CONCERNS
- [Most significant metric and what it means]
- [Second most significant if applicable]

LIKELY EXPLANATION
[1-2 sentences suggesting what might have caused this]

SUGGESTED ACTION
[1 specific, actionable recommendation]
</output_format>

<device_context>
Device ID: {payload.get("device_id")}
Date: {payload.get("timestamp")}
Anomaly Severity: {severity.title()}
</device_context>

<metric_summary>
Most unusual metrics on this day:
{metric_summary}
</metric_summary>

<raw_data>
{json_str}
```

<z_score_guide>
z-score interpretation:
- |z| < 2: Within normal variation
- |z| 2-3: Notably unusual, worth attention
- |z| > 3: Significantly unusual, likely a real issue
</z_score_guide>

<instructions>
1. Write for someone who manages devices but isn't a data scientist
2. Focus on the 2-3 most unusual metrics (highest |z-score|)
3. Translate technical metric names to plain language
4. Keep explanation under 150 words
5. Don't be alarmist - some anomalies are benign (e.g., unusually busy day)
</instructions>"""
    return prompt.strip()

def explain_anomaly_row(row: pd.Series, ctx: ExplanationContext) -> dict[str, str]:
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
    """Build prompt using median/MAD based statistics."""
    json_str = json.dumps(payload, indent=2)

    # Build human-readable metric summary
    metrics = payload.get("metrics", [])
    metric_lines = []
    for m in sorted(metrics, key=lambda x: abs(x.get("z_score", 0)), reverse=True)[:5]:
        name = m.get("name", "")
        z = m.get("z_score", 0)
        human_name, unit, desc = get_metric_info(name)
        severity_desc = get_z_score_description(z)
        metric_lines.append(f"- {human_name}: {severity_desc} (z={z:.1f})")

    metric_summary = "\n".join(metric_lines) if metric_lines else "No significant deviations"
    anomaly_score = payload.get("anomaly_score", 0)
    severity = get_severity_word(anomaly_score)

    prompt = f"""<role>
You are a device health advisor explaining anomaly alerts to warehouse supervisors and IT staff managing enterprise mobile devices.
</role>

<output_format>
{NO_THINKING_INSTRUCTION}

Structure your response as:

WHAT HAPPENED
[2-3 sentences describing the unusual behavior in plain English]

KEY CONCERNS
- [Most significant metric and what it means]
- [Second most significant if applicable]

LIKELY EXPLANATION
[1-2 sentences suggesting what might have caused this]

SUGGESTED ACTION
[1 specific, actionable recommendation]
</output_format>

<device_context>
Device ID: {payload.get("device_id")}
Date: {payload.get("timestamp")}
Anomaly Severity: {severity.title()}
</device_context>

<metric_summary>
Most unusual metrics (compared to device population):
{metric_summary}
</metric_summary>

<raw_data>
{json_str}
```

<statistics_note>
These metrics use robust statistics (median/MAD) which are less affected by outliers than mean/std.
z-score interpretation: |z| < 2 = normal, |z| 2-3 = notable, |z| > 3 = significant
</statistics_note>

<instructions>
1. Write for someone who manages devices but isn't a data scientist
2. Focus on the 2-3 most unusual metrics
3. Translate technical metric names to plain language
4. Keep explanation under 150 words
5. Be balanced - note if high values might be legitimate heavy usage
</instructions>"""
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


# =============================================================================
# COST-AWARE EXPLANATION FUNCTIONS
# =============================================================================

def explain_anomaly_with_cost(
    metrics: dict,
    stats: dict,
    device_id: int,
    timestamp: str,
    anomaly_score: float,
    cost_context: CostContext,
    cost_calculator: CostCalculator,
) -> dict[str, any]:
    """Generate a cost-aware explanation for an anomaly.

    This function combines technical anomaly explanation with pre-calculated
    financial impact data. The LLM never calculates costs - all figures are
    pre-computed and injected into the prompt.

    Args:
        metrics: Dictionary of metric values.
        stats: Reference statistics for comparison.
        device_id: Device identifier.
        timestamp: Anomaly timestamp.
        anomaly_score: Anomaly score from detection model.
        cost_context: Full cost context for the anomaly.
        cost_calculator: Cost calculator service instance.

    Returns:
        Dictionary containing:
        - 'explanation': LLM-generated text with financial context
        - 'prompt': The prompt sent to the LLM
        - 'financial_impact': The FinancialImpactSummary object
        - 'validation_result': Result of anti-hallucination validation
    """
    from device_anomaly.llm.cost_prompts import build_cost_aware_anomaly_prompt
    from device_anomaly.llm.cost_validator import validate_financial_output

    # Build technical payload
    payload = _build_anomaly_payload_from_stats(
        metrics=metrics,
        stats=stats,
        device_id=device_id,
        timestamp=timestamp,
        anomaly_score=anomaly_score,
    )

    # Calculate financial impact
    cost_result = cost_calculator.calculate_anomaly_impact(cost_context)

    # Build cost-aware prompt
    anomaly_json = json.dumps(payload, indent=2)
    prompt = build_cost_aware_anomaly_prompt(
        anomaly_json=anomaly_json,
        financial_data=cost_result.financial_data,
    )

    # Generate explanation via LLM
    llm = get_default_llm_client()
    raw_explanation = llm.generate(prompt)
    explanation = strip_thinking_tags(raw_explanation)

    # Validate financial output (anti-hallucination)
    allowed_amounts = cost_calculator.get_allowed_amounts(cost_result)
    sanitized_text, validation_result = validate_financial_output(
        explanation,
        allowed_amounts,
        auto_sanitize=True,
    )

    return {
        "explanation": sanitized_text,
        "prompt": prompt,
        "financial_impact": cost_result.impact,
        "financial_data": cost_result.financial_data,
        "validation_result": validation_result,
    }


def generate_cost_aware_explanation(
    engine,
    result_id: int,
    tenant_id: str,
    device_model: str = None,
    device_value_usd: float = None,
) -> dict[str, any]:
    """Generate and save a cost-aware explanation for an anomaly result.

    Loads the anomaly result, calculates financial impact based on tenant
    cost configuration, and generates an explanation that includes business
    impact context.

    Args:
        engine: SQLAlchemy engine.
        result_id: Anomaly result ID.
        tenant_id: Tenant identifier for cost lookup.
        device_model: Optional device model for cost lookup.
        device_value_usd: Optional device value override.

    Returns:
        Dictionary with explanation and financial data.
    """
    from decimal import Decimal

    from device_anomaly.costs.calculator import CostCalculator
    from device_anomaly.costs.config import get_cost_config
    from device_anomaly.costs.models import CostContext, DeviceCostContext

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

        # Load stats
        source = str(row.get("Source") or "dw").lower()
        stats_path = Path("artifacts/synthetic_stats.json" if source == "synthetic" else "artifacts/dw_stats.json")
        stats = load_stats(stats_path) or {}

        # Parse metrics
        metrics = {}
        metrics_json = row.get("MetricsJson")
        if isinstance(metrics_json, str) and metrics_json.strip():
            try:
                metrics = json.loads(metrics_json)
            except json.JSONDecodeError:
                metrics = {}

        # Build cost context
        config = get_cost_config()
        device_value = Decimal(str(device_value_usd)) if device_value_usd else config.average_device_cost_usd

        device_context = DeviceCostContext(
            device_id=int(row.get("DeviceId", -1)),
            device_model=device_model,
            purchase_cost_usd=device_value,
            current_value_usd=device_value,  # Simplified - could calculate depreciation
        )

        cost_context = CostContext(
            tenant_id=tenant_id,
            device_context=device_context,
            anomaly_id=result_id,
            anomaly_severity=_score_to_severity(float(row.get("AnomalyScore", 0.0))),
            estimated_resolution_hours=0.5,  # Default 30 min investigation
            worker_hourly_rate_usd=config.worker_hourly_rate_usd,
            it_support_hourly_rate_usd=config.it_support_hourly_rate_usd,
            downtime_cost_per_hour_usd=config.downtime_cost_per_hour_usd,
        )

        calculator = CostCalculator(config)

        # Generate cost-aware explanation
        result = explain_anomaly_with_cost(
            metrics=metrics,
            stats=stats,
            device_id=int(row.get("DeviceId", -1)),
            timestamp=str(row.get("Timestamp", "")),
            anomaly_score=float(row.get("AnomalyScore", row.get("anomaly_score", 0.0))),
            cost_context=cost_context,
            cost_calculator=calculator,
        )

        # Optionally save explanation back to database
        # (keeping original field for backward compatibility)
        conn.execute(
            text(
                """
                UPDATE dbo.ml_AnomalyResults
                SET Explanation = :exp
                WHERE Id = :result_id
                """
            ),
            {"exp": result["explanation"], "result_id": result_id},
        )

    return result


def _score_to_severity(anomaly_score: float) -> str:
    """Convert anomaly score to severity level.

    Args:
        anomaly_score: Anomaly score (typically -1 to 0 for Isolation Forest).

    Returns:
        Severity level: 'critical', 'high', 'medium', or 'low'.
    """
    # Isolation Forest: more negative = more anomalous
    if anomaly_score < -0.5:
        return "critical"
    elif anomaly_score < -0.3:
        return "high"
    elif anomaly_score < -0.1:
        return "medium"
    return "low"
