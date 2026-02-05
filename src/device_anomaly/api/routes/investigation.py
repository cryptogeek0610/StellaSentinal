"""API routes for the Investigation Panel - enhanced anomaly analysis."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_current_user, get_db, get_tenant_id, require_role
from device_anomaly.api.models import (
    AIAnalysisFeedbackRequest,
    AIAnalysisResponse,
    AnomalyExplanation,
    BaselineComparison,
    BaselineConfig,
    BaselineMetric,
    EvidenceEvent,
    EvidenceHypothesis,
    FeatureContribution,
    HistoricalTimelineResponse,
    InvestigationPanelResponse,
    LearnFromFixRequest,
    RemediationOutcomeRequest,
    RemediationSuggestion,
    RootCauseHypothesis,
    SimilarCase,
    TimeSeriesDataPoint,
)
from device_anomaly.database.schema import (
    AnomalyExplanationCache,
    AnomalyResult,
    DeviceMetadata,
    LearnedRemediation,
    RemediationOutcome,
)
from device_anomaly.llm.client import get_default_llm_client, strip_thinking_tags
from device_anomaly.llm.prompt_utils import (
    NO_THINKING_INSTRUCTION,
    get_common_root_causes,
    get_severity_emoji,
    get_severity_word,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/anomalies", tags=["investigation"])


# ============================================
# Feature Display Names and Units
# ============================================

FEATURE_DISPLAY_NAMES = {
    "total_battery_level_drop": "Battery Drop",
    "total_free_storage_kb": "Free Storage",
    "download": "Download",
    "upload": "Upload",
    "offline_time": "Offline Time",
    "disconnect_count": "Disconnects",
    "wifi_signal_strength": "WiFi Signal",
    "connection_time": "Connection Time",
}

FEATURE_UNITS = {
    "total_battery_level_drop": "%",
    "total_free_storage_kb": "KB",
    "download": "MB",
    "upload": "MB",
    "offline_time": "min",
    "disconnect_count": "",
    "wifi_signal_strength": "dBm",
    "connection_time": "s",
}

FEATURE_EXPLANATIONS = {
    "total_battery_level_drop": "Battery drain over the measurement period",
    "total_free_storage_kb": "Available storage space on the device",
    "download": "Data downloaded over the network",
    "upload": "Data uploaded over the network",
    "offline_time": "Time the device was disconnected",
    "disconnect_count": "Number of network disconnections",
    "wifi_signal_strength": "WiFi signal quality (higher is better)",
    "connection_time": "Duration of network connectivity",
}

# Metrics where HIGH values are GOOD (benign) - should not be flagged as anomalous
# when deviating upward from baseline. Only low values would be problematic.
BENIGN_HIGH_METRICS = {
    "total_free_storage_kb",
    "wifi_signal_strength",
}


def _get_severity(score: float) -> str:
    """Convert anomaly score to severity level."""
    if score <= -0.7:
        return "critical"
    if score <= -0.5:
        return "high"
    if score <= -0.3:
        return "medium"
    return "low"


def _format_value(feature_name: str, value: float) -> str:
    """Format a feature value for display."""
    if feature_name == "total_free_storage_kb":
        if value >= 1024 * 1024:
            return f"{value / (1024 * 1024):.1f} GB"
        elif value >= 1024:
            return f"{value / 1024:.1f} MB"
        return f"{value:.0f} KB"

    unit = FEATURE_UNITS.get(feature_name, "")
    if isinstance(value, float):
        return f"{value:.1f} {unit}".strip()
    return f"{value} {unit}".strip()


def _filter_benign_evidence(text: str) -> str:
    """Remove lines from LLM response that mention benign-high features as problems.

    High free storage, high battery level, and strong signal are GOOD, not problems.
    This filter removes evidence lines that incorrectly flag these as issues.
    """
    if not text:
        return text

    # Patterns for benign-high features being mentioned as anomalies
    # We want to remove lines that mention these with positive deviation context
    benign_patterns = [
        # Free Storage mentioned as high/above baseline (benign)
        r'Free\s*Storage.*(?:above|higher|increased|significant|large|high).*(?:baseline|normal|typical)',
        r'Free\s*Storage.*\d+\.?\d*\s*(?:GB|MB|TB).*(?:above|higher|deviation)',
        r'(?:High|Large|Significant|Elevated).*[Ff]ree\s*[Ss]torage',
        r'Free\s*Storage.*(?:\+|\bpositive\b).*deviation',
        r'Free\s*Storage.*(?:11\.1|[5-9]\d*\.?\d*)\s*σ',  # High positive sigma
        # Memory mentioned as high (benign)
        r'(?:Free|Available)\s*Memory.*(?:above|higher|high)',
        # Battery level high (benign)
        r'Battery\s*Level.*(?:high|above|good)',
        # Signal strength strong (benign)
        r'Signal\s*Strength.*(?:strong|high|good|above)',
    ]

    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        is_benign = False
        for pattern in benign_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_benign = True
                logger.debug(f"Filtering benign evidence line: {line[:100]}")
                break
        if not is_benign:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def _calculate_baseline_stats(db: Session, tenant_id: str, device_id: int, days: int = 30) -> dict:
    """Calculate baseline statistics for a device over the past N days."""
    import statistics as stats_module

    # Default baseline values (industry-typical for mobile devices)
    default_baselines = {
        "total_battery_level_drop": {"mean": 15.0, "std": 8.0, "min": 2.0, "max": 40.0, "count": 0},
        "total_free_storage_kb": {"mean": 8000000.0, "std": 3000000.0, "min": 1000000.0, "max": 20000000.0, "count": 0},
        "download": {"mean": 150.0, "std": 100.0, "min": 5.0, "max": 500.0, "count": 0},
        "upload": {"mean": 50.0, "std": 40.0, "min": 1.0, "max": 200.0, "count": 0},
        "offline_time": {"mean": 5.0, "std": 10.0, "min": 0.0, "max": 60.0, "count": 0},
        "disconnect_count": {"mean": 2.0, "std": 3.0, "min": 0.0, "max": 15.0, "count": 0},
        "wifi_signal_strength": {"mean": -55.0, "std": 15.0, "min": -85.0, "max": -30.0, "count": 0},
        "connection_time": {"mean": 2.0, "std": 1.5, "min": 0.5, "max": 10.0, "count": 0},
    }

    cutoff = datetime.now(UTC) - timedelta(days=days)

    # Get historical normal readings for this device
    results = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.device_id == device_id)
        .filter(AnomalyResult.timestamp >= cutoff)
        .filter(AnomalyResult.anomaly_label == 1)  # Normal readings only
        .all()
    )

    # Start with defaults
    baseline_stats = dict(default_baselines)

    if not results:
        logger.info(f"No historical data for device {device_id}, using default baselines")
        return baseline_stats

    # Calculate stats for each feature, overwriting defaults where we have data
    features = [
        "total_battery_level_drop", "total_free_storage_kb", "download", "upload",
        "offline_time", "disconnect_count", "wifi_signal_strength", "connection_time"
    ]

    for feature in features:
        values = [getattr(r, feature) for r in results if getattr(r, feature) is not None]
        if values:
            mean = stats_module.mean(values)
            std = stats_module.stdev(values) if len(values) > 1 else mean * 0.1
            baseline_stats[feature] = {
                "mean": mean,
                "std": std if std > 0 else mean * 0.1,  # Avoid zero std
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

    return baseline_stats


def _build_feature_contributions(anomaly: AnomalyResult, baseline_stats: dict) -> list[FeatureContribution]:
    """Build feature contribution breakdown for an anomaly."""
    contributions = []

    features = [
        "total_battery_level_drop", "total_free_storage_kb", "download", "upload",
        "offline_time", "disconnect_count", "wifi_signal_strength", "connection_time"
    ]

    # Calculate z-scores for each feature that has data
    z_scores = []
    for feature in features:
        value = getattr(anomaly, feature)
        if feature in baseline_stats:
            stats = baseline_stats[feature]
            if value is not None:
                z = (value - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0
                z_scores.append((feature, abs(z), z, value, stats, True))
            else:
                # Include feature even without value, for display purposes
                z_scores.append((feature, 0, 0, stats["mean"], stats, False))

    # Sort by absolute z-score (features with data come first)
    z_scores.sort(key=lambda x: (not x[5], -x[1]))  # has_value=False sorts last, then by -abs_z

    # Calculate total contribution for percentage (only from features with values)
    total_z = sum(x[1] for x in z_scores if x[5]) or 1

    for feature, abs_z, z, value, stats, has_value in z_scores:
        # Skip benign-high metrics with positive deviation - they're not problems
        # (e.g., high free storage, strong WiFi signal)
        if feature in BENIGN_HIGH_METRICS and z > 0:
            continue

        contribution_pct = (abs_z / total_z) * 100 if total_z > 0 and has_value else 0

        # Determine direction
        direction = "positive" if z > 0 else "negative"

        # Calculate percentile (simplified)
        percentile = min(99.9, max(0.1, 50 + (z * 34))) if has_value else 50.0

        # Generate plain text explanation
        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        if not has_value:
            explanation = f"{display_name}: No data available for comparison"
        elif abs(z) >= 3:
            severity_word = "extremely"
            direction_word = "higher" if z > 0 else "lower"
            explanation = f"{display_name} is {severity_word} {direction_word} than typical ({_format_value(feature, stats['mean'])} baseline)"
        elif abs(z) >= 2:
            severity_word = "significantly"
            direction_word = "higher" if z > 0 else "lower"
            explanation = f"{display_name} is {severity_word} {direction_word} than typical ({_format_value(feature, stats['mean'])} baseline)"
        elif abs(z) >= 1:
            severity_word = "moderately"
            direction_word = "higher" if z > 0 else "lower"
            explanation = f"{display_name} is {severity_word} {direction_word} than typical ({_format_value(feature, stats['mean'])} baseline)"
        else:
            explanation = f"{display_name} is within normal range (baseline: {_format_value(feature, stats['mean'])})"

        contributions.append(FeatureContribution(
            feature_name=feature,
            feature_display_name=display_name,
            contribution_percentage=round(contribution_pct, 1),
            contribution_direction=direction,
            current_value=value if has_value else 0,
            current_value_display=_format_value(feature, value) if has_value else "N/A",
            baseline_value=stats["mean"],
            baseline_value_display=_format_value(feature, stats["mean"]),
            deviation_sigma=round(z, 2) if has_value else 0.0,
            percentile=round(percentile, 1),
            plain_text_explanation=explanation,
        ))

    return contributions


def _build_explanation(anomaly: AnomalyResult, contributions: list[FeatureContribution]) -> AnomalyExplanation:
    """Build the anomaly explanation from contributions."""
    if not contributions:
        return AnomalyExplanation(
            summary_text="Insufficient baseline data for detailed explanation",
            detailed_explanation="This anomaly was detected but there is not enough historical data to provide a detailed breakdown.",
            feature_contributions=[],
            top_contributing_features=[],
            explanation_method="z_score",
            explanation_generated_at=datetime.now(UTC),
        )

    # Filter to contributions that have actual values
    contributions_with_data = [c for c in contributions if c.current_value_display != "N/A"]

    # Build summary from top contributors
    top_features = contributions_with_data[:3] if contributions_with_data else contributions[:3]
    summary_parts = []
    for fc in top_features:
        if fc.current_value_display != "N/A" and abs(fc.deviation_sigma) >= 2:
            summary_parts.append(f"{fc.feature_display_name} ({fc.current_value_display})")

    if summary_parts:
        summary_text = f"Unusual values detected in: {', '.join(summary_parts)}"
    elif contributions_with_data:
        summary_text = "Anomalous pattern detected across multiple metrics"
    else:
        summary_text = "Anomaly detected based on overall device behavior pattern"

    # Build detailed explanation
    num_with_data = len(contributions_with_data)
    if num_with_data > 0:
        detailed_parts = [
            f"This anomaly was detected based on deviations across {num_with_data} monitored metrics.",
            "",
            "Key findings:",
        ]
        for i, fc in enumerate(contributions_with_data[:5], 1):
            detailed_parts.append(
                f"{i}. {fc.plain_text_explanation} (deviation: {fc.deviation_sigma:.1f})"
            )
    else:
        detailed_parts = [
            "This anomaly was detected by the ML model based on overall behavior patterns.",
            "",
            "Limited telemetry data is available for this anomaly. The detection was based on:",
            "- Historical device behavior patterns",
            "- Comparison with similar devices in the fleet",
            "- Isolation Forest anomaly detection algorithm",
        ]

    detailed_explanation = "\n".join(detailed_parts)

    return AnomalyExplanation(
        summary_text=summary_text,
        detailed_explanation=detailed_explanation,
        feature_contributions=contributions,
        top_contributing_features=[c.feature_name for c in contributions_with_data[:5]] if contributions_with_data else [],
        explanation_method="z_score",
        explanation_generated_at=datetime.now(UTC),
    )


def _build_baseline_comparison(anomaly: AnomalyResult, baseline_stats: dict) -> BaselineComparison | None:
    """Build baseline comparison data."""
    if not baseline_stats:
        return None

    metrics = []
    total_deviation = 0

    features = [
        "total_battery_level_drop", "total_free_storage_kb", "download", "upload",
        "offline_time", "disconnect_count", "wifi_signal_strength", "connection_time"
    ]

    for feature in features:
        value = getattr(anomaly, feature)
        if feature in baseline_stats:
            stats = baseline_stats[feature]

            # Handle NULL values - use baseline mean as current value for display
            current_value = value if value is not None else stats["mean"]
            z = (current_value - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0

            # Skip benign-high metrics with positive deviation - they're not problems
            if feature in BENIGN_HIGH_METRICS and z > 0:
                continue

            # Determine if anomalous (outside 2σ)
            is_anomalous = abs(z) >= 2 if value is not None else False
            direction = "above" if z > 0 else "below" if z < 0 else "normal"

            deviation_pct = ((current_value - stats["mean"]) / stats["mean"]) * 100 if stats["mean"] != 0 else 0
            percentile = min(99.9, max(0.1, 50 + (z * 34)))

            metrics.append(BaselineMetric(
                metric_name=feature,
                metric_display_name=FEATURE_DISPLAY_NAMES.get(feature, feature),
                metric_unit=FEATURE_UNITS.get(feature, ""),
                current_value=current_value,
                current_value_display=_format_value(feature, current_value) if value is not None else "N/A",
                baseline_mean=stats["mean"],
                baseline_std=stats["std"],
                baseline_min=stats["min"],
                baseline_max=stats["max"],
                deviation_sigma=round(z, 2) if value is not None else 0.0,
                deviation_percentage=round(deviation_pct, 1) if value is not None else 0.0,
                percentile_rank=round(percentile, 1) if value is not None else 50.0,
                is_anomalous=is_anomalous,
                anomaly_direction=direction if value is not None else "normal",
            ))

            if value is not None:
                total_deviation += abs(z)

    if not metrics:
        return None

    metrics_with_values = [m for m in metrics if m.current_value_display != "N/A"]
    return BaselineComparison(
        baseline_config=BaselineConfig(
            baseline_type="rolling_average" if any(s.get("count", 0) > 0 for s in baseline_stats.values()) else "default",
            baseline_period_days=30,
            comparison_window_hours=24,
            statistical_method="z_score",
            baseline_calculated_at=datetime.now(UTC),
        ),
        metrics=metrics,
        overall_deviation_score=round(total_deviation / len(metrics_with_values), 2) if metrics_with_values else 0,
    )


def _generate_evidence_events(anomaly: AnomalyResult) -> list[EvidenceEvent]:
    """Generate synthetic evidence events based on anomaly data."""
    events = []
    timestamp = anomaly.timestamp

    # Add anomaly detection event
    events.append(EvidenceEvent(
        event_id=str(uuid.uuid4()),
        timestamp=timestamp,
        event_type="anomaly_detected",
        event_category="system",
        severity="high" if anomaly.anomaly_score <= -0.5 else "medium",
        title="Anomaly Detected",
        description=f"Isolation Forest model detected anomaly with score {anomaly.anomaly_score:.4f}",
        is_contributing_event=True,
        contribution_note="Primary detection event",
    ))

    # Add events based on anomalous metrics
    if anomaly.total_battery_level_drop and anomaly.total_battery_level_drop > 30:
        events.append(EvidenceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp - timedelta(minutes=15),
            event_type="battery_drain",
            event_category="battery",
            severity="high" if anomaly.total_battery_level_drop > 50 else "medium",
            title="Excessive Battery Drain",
            description=f"Battery dropped {anomaly.total_battery_level_drop:.1f}% in measurement period",
            is_contributing_event=True,
            contribution_note="High battery drain contributing to anomaly",
        ))

    if anomaly.total_free_storage_kb and anomaly.total_free_storage_kb < 500000:
        events.append(EvidenceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp - timedelta(minutes=10),
            event_type="storage_low",
            event_category="storage",
            severity="critical" if anomaly.total_free_storage_kb < 100000 else "high",
            title="Low Storage Space",
            description=f"Free storage at {_format_value('total_free_storage_kb', anomaly.total_free_storage_kb)}",
            is_contributing_event=True,
            contribution_note="Critical storage level",
        ))

    if anomaly.offline_time and anomaly.offline_time > 60:
        events.append(EvidenceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp - timedelta(minutes=int(anomaly.offline_time / 2)),
            event_type="extended_offline",
            event_category="network",
            severity="medium",
            title="Extended Offline Period",
            description=f"Device was offline for {anomaly.offline_time:.0f} minutes",
            is_contributing_event=True,
            contribution_note="Unusual offline duration",
        ))

    if anomaly.disconnect_count and anomaly.disconnect_count > 5:
        events.append(EvidenceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp - timedelta(minutes=20),
            event_type="frequent_disconnects",
            event_category="network",
            severity="medium",
            title="Frequent Network Disconnections",
            description=f"{int(anomaly.disconnect_count)} disconnections recorded",
            is_contributing_event=True,
            contribution_note="Network instability detected",
        ))

    # Sort by timestamp descending
    events.sort(key=lambda e: e.timestamp, reverse=True)
    return events


def _generate_remediation_suggestions(anomaly: AnomalyResult, contributions: list[FeatureContribution]) -> list[RemediationSuggestion]:
    """Generate remediation suggestions based on anomaly characteristics."""
    suggestions = []
    priority = 1

    # Storage-related remediation
    if anomaly.total_free_storage_kb and anomaly.total_free_storage_kb < 500000:
        suggestions.append(RemediationSuggestion(
            remediation_id=str(uuid.uuid4()),
            title="Clear Device Storage",
            description="Free up storage space by removing unnecessary files and apps",
            detailed_steps=[
                "Review and remove unused applications",
                "Clear application caches",
                "Remove old downloads and media files",
                "Consider offloading data to cloud storage",
            ],
            priority=priority,
            confidence_score=0.85,
            confidence_level="high",
            source="policy",
            source_details="Standard procedure for low storage",
            estimated_impact="Could recover 500MB-2GB of storage",
            is_automated=False,
        ))
        priority += 1

    # Battery-related remediation
    if anomaly.total_battery_level_drop and anomaly.total_battery_level_drop > 30:
        suggestions.append(RemediationSuggestion(
            remediation_id=str(uuid.uuid4()),
            title="Investigate Battery Drain",
            description="Identify and address causes of excessive battery consumption",
            detailed_steps=[
                "Check battery usage by app in device settings",
                "Identify background apps consuming power",
                "Review location services and sync settings",
                "Consider disabling power-hungry features",
            ],
            priority=priority,
            confidence_score=0.75,
            confidence_level="medium",
            source="ai_generated",
            source_details="Based on battery drain pattern",
            estimated_impact="Could improve battery life by 20-40%",
            is_automated=False,
        ))
        priority += 1

    # Network-related remediation
    if anomaly.disconnect_count and anomaly.disconnect_count > 5:
        suggestions.append(RemediationSuggestion(
            remediation_id=str(uuid.uuid4()),
            title="Diagnose Network Connectivity",
            description="Investigate and resolve network disconnection issues",
            detailed_steps=[
                "Check WiFi signal strength at device location",
                "Verify network credentials are current",
                "Test device connectivity in different locations",
                "Consider network infrastructure review",
            ],
            priority=priority,
            confidence_score=0.70,
            confidence_level="medium",
            source="ai_generated",
            source_details="Based on disconnect frequency",
            estimated_impact="Could stabilize connectivity",
            is_automated=False,
        ))
        priority += 1

    # General recommendation
    suggestions.append(RemediationSuggestion(
        remediation_id=str(uuid.uuid4()),
        title="Contact Device User",
        description="Reach out to the device user for more context",
        detailed_steps=[
            "Send notification to device user",
            "Ask about recent changes or issues",
            "Gather additional context about device usage",
            "Document findings in investigation notes",
        ],
        priority=priority,
        confidence_score=0.50,
        confidence_level="low",
        source="policy",
        source_details="Standard investigation procedure",
        is_automated=False,
    ))

    return suggestions


def _find_similar_cases(db: Session, tenant_id: str, anomaly: AnomalyResult, limit: int = 5) -> list[SimilarCase]:
    """Find similar historical anomaly cases."""
    # Find resolved anomalies with similar characteristics, joined with device metadata
    similar = (
        db.query(AnomalyResult, DeviceMetadata.device_name)
        .outerjoin(
            DeviceMetadata,
            (AnomalyResult.device_id == DeviceMetadata.device_id)
            & (AnomalyResult.tenant_id == DeviceMetadata.tenant_id),
        )
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.id != anomaly.id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.status.in_(["resolved", "false_positive"]))
        .order_by(AnomalyResult.updated_at.desc())
        .limit(limit * 2)  # Get more to filter
        .all()
    )

    cases = []
    for s, device_name in similar[:limit]:
        # Calculate simple similarity based on score proximity
        score_diff = abs(anomaly.anomaly_score - s.anomaly_score)
        similarity = max(0, 1 - score_diff)

        similarity_factors = []
        if anomaly.device_id == s.device_id:
            similarity_factors.append("Same device")
            similarity += 0.2
        if abs(anomaly.anomaly_score - s.anomaly_score) < 0.1:
            similarity_factors.append("Similar anomaly score")

        resolution_hours = None
        if s.updated_at and s.timestamp:
            resolution_hours = (s.updated_at - s.timestamp).total_seconds() / 3600

        cases.append(SimilarCase(
            case_id=str(uuid.uuid4()),
            anomaly_id=s.id,
            device_id=s.device_id,
            device_name=device_name,
            detected_at=s.timestamp,
            resolved_at=s.updated_at if s.status == "resolved" else None,
            similarity_score=min(1.0, similarity),
            similarity_factors=similarity_factors if similarity_factors else ["Similar pattern"],
            anomaly_type="device_anomaly",
            severity=_get_severity(s.anomaly_score),
            resolution_status=s.status,
            resolution_summary=s.notes[:100] if s.notes else None,
            time_to_resolution_hours=resolution_hours,
        ))

    # Sort by similarity
    cases.sort(key=lambda c: c.similarity_score, reverse=True)
    return cases


# ============================================
# API Endpoints
# ============================================


@router.get("/{anomaly_id}/investigation", response_model=InvestigationPanelResponse)
def get_investigation_panel(
    anomaly_id: int,
    include_ai_analysis: bool = Query(True),
    include_similar_cases: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Get complete investigation panel data for an anomaly."""
    tenant_id = get_tenant_id()

    # Get the anomaly
    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    # Calculate baseline statistics
    baseline_stats = _calculate_baseline_stats(db, tenant_id, anomaly.device_id)

    # Build feature contributions
    contributions = _build_feature_contributions(anomaly, baseline_stats)

    # Build explanation
    explanation = _build_explanation(anomaly, contributions)

    # Build baseline comparison
    baseline_comparison = _build_baseline_comparison(anomaly, baseline_stats)

    # Generate evidence events
    evidence_events = _generate_evidence_events(anomaly)

    # Generate remediation suggestions
    remediations = _generate_remediation_suggestions(anomaly, contributions)

    # Find similar cases
    similar_cases = []
    if include_similar_cases:
        similar_cases = _find_similar_cases(db, tenant_id, anomaly)

    # Calculate confidence score (based on baseline data availability)
    confidence = 0.5 + (min(len(baseline_stats), 8) / 16)  # 0.5 to 1.0 based on stats

    return InvestigationPanelResponse(
        anomaly_id=anomaly.id,
        device_id=anomaly.device_id,
        anomaly_score=anomaly.anomaly_score,
        severity=_get_severity(anomaly.anomaly_score),
        confidence_score=round(confidence, 2),
        detected_at=anomaly.timestamp,
        explanation=explanation,
        baseline_comparison=baseline_comparison,
        evidence_events=evidence_events,
        evidence_event_count=len(evidence_events),
        ai_analysis=None,  # Will be populated by separate endpoint
        suggested_remediations=remediations,
        similar_cases=similar_cases,
    )


@router.get("/{anomaly_id}/explanation", response_model=AnomalyExplanation)
def get_anomaly_explanation(
    anomaly_id: int,
    db: Session = Depends(get_db),
):
    """Get feature contribution explanation for an anomaly."""
    tenant_id = get_tenant_id()

    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    baseline_stats = _calculate_baseline_stats(db, tenant_id, anomaly.device_id)
    contributions = _build_feature_contributions(anomaly, baseline_stats)
    return _build_explanation(anomaly, contributions)


@router.get("/{anomaly_id}/baseline", response_model=BaselineComparison)
def get_baseline_comparison(
    anomaly_id: int,
    baseline_days: int = Query(30, ge=7, le=90),
    db: Session = Depends(get_db),
):
    """Get baseline comparison for an anomaly."""
    tenant_id = get_tenant_id()

    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    baseline_stats = _calculate_baseline_stats(db, tenant_id, anomaly.device_id, days=baseline_days)
    comparison = _build_baseline_comparison(anomaly, baseline_stats)

    if not comparison:
        raise HTTPException(status_code=404, detail="Insufficient baseline data")

    return comparison


@router.get("/{anomaly_id}/timeline", response_model=HistoricalTimelineResponse)
def get_historical_timeline(
    anomaly_id: int,
    metric: str = Query(..., description="Metric name to chart"),
    days: int = Query(30, ge=7, le=90),
    db: Session = Depends(get_db),
):
    """Get historical timeline data for a specific metric."""
    tenant_id = get_tenant_id()

    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    # Validate metric name
    valid_metrics = [
        "total_battery_level_drop", "total_free_storage_kb", "download", "upload",
        "offline_time", "disconnect_count", "wifi_signal_strength", "connection_time"
    ]
    if metric not in valid_metrics:
        raise HTTPException(status_code=400, detail=f"Invalid metric. Valid options: {valid_metrics}")

    # Get historical data
    cutoff = datetime.now(UTC) - timedelta(days=days)
    results = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.device_id == anomaly.device_id)
        .filter(AnomalyResult.timestamp >= cutoff)
        .order_by(AnomalyResult.timestamp)
        .all()
    )

    # Build data points
    data_points = []
    values = []
    for r in results:
        value = getattr(r, metric)
        if value is not None:
            values.append(value)
            data_points.append(TimeSeriesDataPoint(
                timestamp=r.timestamp,
                value=value,
                is_anomalous=(r.anomaly_label == -1),
            ))

    # Calculate baseline stats
    if values:
        import statistics
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else mean * 0.1
    else:
        mean = 0
        std = 1

    return HistoricalTimelineResponse(
        metric_name=metric,
        data_points=data_points,
        baseline_mean=mean,
        baseline_std=std,
        baseline_upper=mean + (2 * std),
        baseline_lower=max(0, mean - (2 * std)),
    )


@router.get("/{anomaly_id}/ai-analysis", response_model=AIAnalysisResponse)
def get_ai_analysis(
    anomaly_id: int,
    regenerate: bool = Query(False),
    db: Session = Depends(get_db),
):
    """Get or generate AI root cause analysis for an anomaly."""
    tenant_id = get_tenant_id()

    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    # Check cache first (unless regenerate requested)
    if not regenerate:
        cached = (
            db.query(AnomalyExplanationCache)
            .filter(AnomalyExplanationCache.tenant_id == tenant_id)
            .filter(AnomalyExplanationCache.anomaly_id == anomaly_id)
            .first()
        )
        if cached and cached.ai_analysis_json:
            try:
                analysis_data = json.loads(cached.ai_analysis_json)
                return AIAnalysisResponse(**analysis_data)
            except (json.JSONDecodeError, ValueError):
                pass  # Generate new analysis

    # Build context for LLM
    baseline_stats = _calculate_baseline_stats(db, tenant_id, anomaly.device_id)
    contributions = _build_feature_contributions(anomaly, baseline_stats)

    # Generate analysis using LLM
    llm_client = get_default_llm_client()

    # Features where HIGH values are benign (not problems)
    # Only low values of these metrics are issues
    benign_high_features = {
        "freestorage", "free_storage", "totalfreestoragekb", "total_free_storage_kb",
        "availablestorage", "available_storage", "freememory", "free_memory",
        "availablememory", "available_memory", "batterylevel", "battery_level",
        "batterycapacity", "battery_capacity", "signalstrength", "signal_strength",
        "wifisignalstrength", "wifi_signal_strength",
    }

    def is_benign_high_deviation(fc) -> bool:
        """Check if this is a benign high deviation (e.g., high free storage is good)."""
        feature_lower = fc.feature_name.lower().replace("_", "")
        is_benign_high = any(benign in feature_lower for benign in benign_high_features)
        # If it's a benign-high feature and the deviation is positive (higher than baseline), ignore it
        return is_benign_high and fc.deviation_sigma > 0

    # Build prompt - filter out benign high deviations
    top_features = [fc for fc in contributions if not is_benign_high_deviation(fc)][:5]
    feature_summary = "\n".join([
        f"- {fc.feature_display_name}: {fc.current_value_display} (baseline: {fc.baseline_value_display}, deviation: {fc.deviation_sigma:.1f}σ)"
        for fc in top_features
    ])

    # Determine severity description
    severity_word = get_severity_word(anomaly.anomaly_score)
    severity_emoji = get_severity_emoji(anomaly.anomaly_score)
    common_causes = get_common_root_causes()

    prompt = f"""<role>
You are a device health analyst for an enterprise mobile device management system (SOTI MobiControl). You analyze anomalies detected by an Isolation Forest ML model that monitors daily device telemetry from Android/iOS devices used in warehouses, retail, and field operations.
</role>

<output_format>
{NO_THINKING_INSTRUCTION}

Structure your response EXACTLY as:

PRIMARY HYPOTHESIS
Title: [5-10 word cause title]
Confidence: [High/Medium/Low]
Description: [2-3 sentences explaining what likely happened]

SUPPORTING EVIDENCE
- [Specific metric + why it supports the hypothesis]
- [Another piece of evidence]

ALTERNATIVE HYPOTHESIS
Title: [Alternative cause]
Confidence: [Lower than primary]
Why less likely: [1 sentence]

RECOMMENDED ACTIONS
Urgency: [Immediate/Soon/Monitor]
1. [First action - be specific]
2. [Second action]

BUSINESS IMPACT
[1-2 sentences in plain English about operational impact]
</output_format>

<anomaly_data>
Device ID: {anomaly.device_id}
Detection Time: {anomaly.timestamp}

Anomaly Score: {anomaly.anomaly_score:.4f} {severity_emoji}
- Score interpretation: Isolation Forest scores range from -1 (most anomalous) to 0 (normal).
- This device scored {anomaly.anomaly_score:.4f}, which is {severity_word}ly anomalous.

Top Contributing Factors (metrics most different from normal):
{feature_summary}
</anomaly_data>

<reference>
{common_causes}
</reference>

<instructions>
1. Base your hypothesis ONLY on the metrics provided - do not invent data
2. Consider combinations of metrics (e.g., high battery drain + high app time = heavy usage, not necessarily a problem)
3. Confidence should reflect how clearly the evidence points to the cause
4. Actions should be specific to what an IT admin can actually do
5. Keep total response under 300 words
6. IMPORTANT: Some metrics are GOOD when high - do NOT flag these as problems:
   - High free storage (only LOW storage is a problem)
   - High free memory (only LOW memory is a problem)
   - High battery level (only LOW battery is a problem)
   - Strong signal strength (only WEAK signal is a problem)
   If these metrics show HIGH values, ignore them in your analysis - they are benign.
</instructions>"""

    try:
        raw_llm_response = llm_client.generate(prompt, max_tokens=500, temperature=0.3)
        # Strip <think> tags from models like DeepSeek R1 that include internal reasoning
        llm_response = strip_thinking_tags(raw_llm_response)
        # Filter out benign-high evidence (e.g., high free storage is good, not a problem)
        llm_response = _filter_benign_evidence(llm_response)
        model_used = llm_client.model_name or "unknown"
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}")
        llm_response = None
        model_used = "fallback"

    # Build response (with fallback if LLM fails)
    analysis_id = str(uuid.uuid4())

    if llm_response:
        # Parse LLM response (simplified - in production, use structured output)
        primary_hypothesis = RootCauseHypothesis(
            hypothesis_id=str(uuid.uuid4()),
            title="Device Resource Exhaustion",
            description=llm_response[:500] if len(llm_response) > 500 else llm_response,
            likelihood=0.75,
            evidence_for=[
                EvidenceHypothesis(
                    statement=f"{top_features[0].feature_display_name} shows significant deviation" if top_features else "Multiple metrics deviated from baseline",
                    strength="strong",
                    source="telemetry",
                )
            ],
            evidence_against=[],
            recommended_actions=["Review device resource usage", "Check for recent changes"],
        )
    else:
        # Fallback analysis based on data
        if anomaly.total_free_storage_kb and anomaly.total_free_storage_kb < 500000:
            title = "Storage Exhaustion"
            desc = "Device is running critically low on storage space, which can cause performance degradation and app failures."
        elif anomaly.total_battery_level_drop and anomaly.total_battery_level_drop > 40:
            title = "Excessive Battery Consumption"
            desc = "Device experienced abnormal battery drain, potentially indicating runaway processes or hardware issues."
        elif anomaly.disconnect_count and anomaly.disconnect_count > 10:
            title = "Network Connectivity Issues"
            desc = "Device experienced frequent network disconnections, suggesting connectivity or infrastructure problems."
        else:
            title = "Behavioral Pattern Anomaly"
            desc = "Device metrics deviated significantly from established baseline patterns."

        primary_hypothesis = RootCauseHypothesis(
            hypothesis_id=str(uuid.uuid4()),
            title=title,
            description=desc,
            likelihood=0.70,
            evidence_for=[
                EvidenceHypothesis(
                    statement="Multiple metrics deviated from established baseline",
                    strength="moderate",
                    source="telemetry",
                )
            ],
            evidence_against=[],
            recommended_actions=["Review recent device activity", "Contact device user for context"],
        )

    analysis = AIAnalysisResponse(
        analysis_id=analysis_id,
        generated_at=datetime.now(UTC),
        model_used=model_used,
        primary_hypothesis=primary_hypothesis,
        alternative_hypotheses=[
            RootCauseHypothesis(
                hypothesis_id=str(uuid.uuid4()),
                title="Normal Usage Variation",
                description="The observed pattern may represent legitimate but unusual device usage.",
                likelihood=0.25,
                evidence_for=[
                    EvidenceHypothesis(
                        statement="No critical security events detected",
                        strength="weak",
                        source="inference",
                    )
                ],
                evidence_against=[],
                recommended_actions=["Monitor for pattern recurrence"],
            )
        ],
        confidence_score=0.70 if not llm_response else 0.85,
        confidence_level="medium" if not llm_response else "high",
        confidence_explanation="Analysis based on statistical deviation and pattern matching" if not llm_response else "Analysis generated using AI model with telemetry context",
        similar_cases_analyzed=0,
    )

    # Cache the analysis
    try:
        existing_cache = (
            db.query(AnomalyExplanationCache)
            .filter(AnomalyExplanationCache.tenant_id == tenant_id)
            .filter(AnomalyExplanationCache.anomaly_id == anomaly_id)
            .first()
        )

        if existing_cache:
            existing_cache.ai_analysis_json = json.dumps(analysis.model_dump(), default=str)
            existing_cache.ai_model_used = model_used
            existing_cache.updated_at = datetime.now(UTC)
        else:
            explanation = _build_explanation(anomaly, contributions)
            cache_entry = AnomalyExplanationCache(
                tenant_id=tenant_id,
                anomaly_id=anomaly_id,
                summary_text=explanation.summary_text,
                detailed_explanation=explanation.detailed_explanation,
                feature_contributions_json=json.dumps([fc.model_dump() for fc in contributions]),
                top_contributing_features=json.dumps(explanation.top_contributing_features),
                ai_analysis_json=json.dumps(analysis.model_dump(), default=str),
                ai_model_used=model_used,
            )
            db.add(cache_entry)

        db.commit()
    except Exception as e:
        logger.warning(f"Failed to cache AI analysis: {e}")
        db.rollback()

    return analysis


@router.post("/{anomaly_id}/ai-analysis/feedback")
def submit_ai_feedback(
    anomaly_id: int,
    request: AIAnalysisFeedbackRequest,
    _: None = Depends(require_role(["analyst", "admin"])),
    db: Session = Depends(get_db),
):
    """Submit feedback on AI analysis."""
    tenant_id = get_tenant_id()

    # Update cache with feedback
    cached = (
        db.query(AnomalyExplanationCache)
        .filter(AnomalyExplanationCache.tenant_id == tenant_id)
        .filter(AnomalyExplanationCache.anomaly_id == anomaly_id)
        .first()
    )

    if not cached:
        raise HTTPException(status_code=404, detail="No analysis found for this anomaly")

    cached.feedback_rating = request.rating
    cached.feedback_text = request.feedback_text
    cached.actual_root_cause = request.actual_root_cause
    cached.updated_at = datetime.now(UTC)

    db.commit()

    return {"success": True, "message": "Feedback recorded"}


@router.get("/{anomaly_id}/similar-cases", response_model=list[SimilarCase])
def get_similar_cases(
    anomaly_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get similar historical anomaly cases."""
    tenant_id = get_tenant_id()

    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    return _find_similar_cases(db, tenant_id, anomaly, limit=limit)


@router.get("/{anomaly_id}/remediations", response_model=list[RemediationSuggestion])
def get_remediations(
    anomaly_id: int,
    db: Session = Depends(get_db),
):
    """Get remediation suggestions for an anomaly."""
    tenant_id = get_tenant_id()

    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    baseline_stats = _calculate_baseline_stats(db, tenant_id, anomaly.device_id)
    contributions = _build_feature_contributions(anomaly, baseline_stats)

    return _generate_remediation_suggestions(anomaly, contributions)


@router.post("/{anomaly_id}/remediations/{remediation_id}/outcome")
def record_remediation_outcome(
    anomaly_id: int,
    remediation_id: str,
    request: RemediationOutcomeRequest,
    _: None = Depends(require_role(["analyst", "admin"])),
    db: Session = Depends(get_db),
):
    """Record the outcome of a remediation action."""
    tenant_id = get_tenant_id()
    user = get_current_user()

    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    # Create outcome record
    outcome = RemediationOutcome(
        tenant_id=tenant_id,
        anomaly_id=anomaly_id,
        remediation_title=f"Remediation {remediation_id}",
        remediation_source="user_applied",
        applied_by=user.user_id if user else "unknown",
        outcome=request.outcome,
        outcome_recorded_at=datetime.now(UTC),
        outcome_notes=request.notes,
        anomaly_context_json=json.dumps({
            "anomaly_score": anomaly.anomaly_score,
            "status": anomaly.status,
        }),
    )
    db.add(outcome)
    db.commit()

    return {"success": True, "message": "Outcome recorded", "outcome_id": outcome.id}


@router.post("/{anomaly_id}/learn")
def learn_from_fix(
    anomaly_id: int,
    request: LearnFromFixRequest,
    _: None = Depends(require_role(["analyst", "admin"])),
    db: Session = Depends(get_db),
):
    """Learn from a successful fix to improve future suggestions."""
    tenant_id = get_tenant_id()

    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    # Create pattern hash based on anomaly characteristics
    pattern_key = {
        "severity": _get_severity(anomaly.anomaly_score),
        "has_storage_issue": (anomaly.total_free_storage_kb or 0) < 500000,
        "has_battery_issue": (anomaly.total_battery_level_drop or 0) > 30,
        "has_network_issue": (anomaly.disconnect_count or 0) > 5,
    }
    pattern_hash = hashlib.sha256(json.dumps(pattern_key, sort_keys=True).encode()).hexdigest()[:32]

    # Check if similar pattern already exists
    existing = (
        db.query(LearnedRemediation)
        .filter(LearnedRemediation.tenant_id == tenant_id)
        .filter(LearnedRemediation.pattern_hash == pattern_hash)
        .first()
    )

    if existing:
        # Update existing pattern
        existing.success_count += 1
        existing.times_applied += 1
        existing.current_confidence = min(0.99, existing.current_confidence + 0.05)
        existing.last_successful_case_id = anomaly_id

        # Add to learned cases
        cases = json.loads(existing.learned_from_cases_json or "[]")
        if anomaly_id not in cases:
            cases.append(anomaly_id)
            existing.learned_from_cases_json = json.dumps(cases[-20:])  # Keep last 20

        existing.updated_at = datetime.now(UTC)
        db.commit()

        return {
            "success": True,
            "message": "Existing pattern updated",
            "learned_remediation_id": existing.id,
            "current_confidence": existing.current_confidence,
        }

    # Create new learned remediation
    learned = LearnedRemediation(
        tenant_id=tenant_id,
        pattern_name=f"Learned from case #{anomaly_id}",
        pattern_hash=pattern_hash,
        anomaly_types=json.dumps(["device_anomaly"]),
        severity_range=json.dumps([_get_severity(anomaly.anomaly_score)]),
        remediation_title=request.remediation_description[:255],
        remediation_description=request.remediation_description,
        remediation_steps_json=json.dumps([request.remediation_description]),
        initial_confidence=0.6,
        current_confidence=0.6,
        success_count=1,
        times_applied=1,
        learned_from_cases_json=json.dumps([anomaly_id]),
        last_successful_case_id=anomaly_id,
    )
    db.add(learned)
    db.commit()

    return {
        "success": True,
        "message": "New remediation pattern learned",
        "learned_remediation_id": learned.id,
        "initial_confidence": learned.initial_confidence,
    }
