"""
Root Cause Analysis for Anomaly Detection.

This module identifies probable root causes of anomalies using:
- Feature correlation analysis
- Temporal precedence (what changed before anomaly)
- Cohort comparison (what's different about affected devices)
- Causal graph traversal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RootCause:
    """A potential root cause for an anomaly."""

    cause_type: str  # temporal, cohort, correlation, causal
    feature: str
    evidence: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class RootCauseAnalysisResult:
    """Result of root cause analysis for a device."""

    device_id: Any
    anomaly_type: str
    probable_causes: list[RootCause]
    top_cause: RootCause | None
    analysis_confidence: float
    recommendations: list[str] = field(default_factory=list)


class RootCauseAnalyzer:
    """
    Identifies root causes of anomalies using multiple techniques.

    Techniques:
    1. Feature correlation analysis
    2. Temporal precedence (what changed before anomaly)
    3. Cohort comparison (what's different about affected devices)
    4. Causal graph traversal
    """

    def __init__(self):
        self._causal_graph = self._build_causal_graph()
        self._feature_recommendations = self._build_recommendations()
        self._benign_high_features = self._build_benign_high_features()

    def _build_benign_high_features(self) -> set[str]:
        """
        Features where HIGH values are benign (not problems).

        These features should NOT be flagged as issues when they deviate
        upward from baseline. Only low values would be problematic.
        """
        return {
            # Storage - high free storage is good, low is bad
            "FreeStorage",
            "FreeStorageKb",
            "free_storage",
            "free_storage_kb",
            "total_free_storage_kb",
            "TotalFreeStorageKb",
            "AvailableStorage",
            "available_storage",
            # Memory - high free memory is good
            "FreeMemory",
            "FreeMemoryMb",
            "free_memory",
            "AvailableMemory",
            "available_memory",
            # Signal strength - high is good (less negative dBm)
            "SignalStrength",
            "signal_strength",
            "wifi_signal_strength",
            "WifiSignalStrength",
            # Battery level - high is good
            "BatteryLevel",
            "battery_level",
            "BatteryCapacity",
            "battery_capacity",
        }

    def _build_causal_graph(self) -> dict[str, list[str]]:
        """Build domain knowledge causal graph (cause -> effects)."""
        return {
            # Battery causes
            "ScreenOnTime": ["BatteryDrain", "BatteryDrainPerHour"],
            "AppForegroundTime": ["BatteryDrain", "BatteryDrainPerHour"],
            "BackgroundAppActivity": ["BatteryDrain"],
            "PoorSignal": ["BatteryDrain", "NetworkDrops"],
            "ChargingPattern": ["BatteryHealth", "BatteryCapacity"],
            "BatteryHealth": ["BatteryDrain", "BatteryCapacity"],
            "Temperature": ["BatteryDrain", "CPUThrottle"],
            # Network causes
            "LocationMovement": ["APHopping", "NetworkDrops"],
            "WeakWifiCoverage": ["NetworkDrops", "SignalStrength"],
            "CellCoverage": ["TowerHopping", "NetworkDrops"],
            "APHopping": ["BatteryDrain", "NetworkDrops"],
            "HighDataUsage": ["BatteryDrain", "NetworkCongestion"],
            # App causes
            "AppVersion": ["AppCrash", "ANR"],
            "LowMemory": ["AppCrash", "ANR", "SlowPerformance"],
            "LowStorage": ["AppCrash", "InstallFailure"],
            "HighCPU": ["BatteryDrain", "Temperature", "SlowPerformance"],
            # System causes
            "OsVersion": ["AppCrash", "SecurityVulnerability"],
            "AgentVersion": ["DataCollectionIssues"],
            "Rooted": ["SecurityIssues", "AppBehavior"],
            # Network Traffic causes (NEW - from network_traffic_features.py)
            "high_upload_ratio": ["DataExfiltration", "SecurityRisk"],
            "exfiltration_risk": ["DataExfiltration", "SecurityBreach"],
            "unusual_night_activity": ["SecurityRisk", "MalwareBehavior"],
            "interface_switching_rate": ["BatteryDrain", "NetworkInstability"],
            "traffic_concentration": ["DataExfiltration", "SingleAppAbuse"],
            # Security Posture causes (NEW - from security_features.py)
            "is_rooted_or_jailbroken": ["SecurityBreach", "MalwareBehavior", "AppBehavior"],
            "has_developer_risk": ["SecurityRisk", "DataLeakage"],
            "patch_age_critical": ["SecurityVulnerability", "ExploitRisk"],
            "low_security_score": ["SecurityBreach", "ComplianceViolation"],
            # Location Intelligence causes (NEW - from location_features.py)
            "dead_zone_time_pct": ["NetworkDrops", "BatteryDrain", "DataLoss"],
            "location_entropy": ["APHopping", "BatteryDrain"],
            "ap_hopping_rate": ["BatteryDrain", "NetworkDrops"],
            "signal_variability": ["NetworkDrops", "PoorConnectivity"],
            "problematic_ap_count": ["NetworkDrops", "SlowPerformance"],
            # Temporal Pattern causes (NEW - from temporal_features.py)
            "unusual_peak_hour": ["Malware", "AppAbuse"],
            "night_usage_anomaly": ["SecurityRisk", "MalwareBehavior"],
            "hourly_entropy_low": ["AppAbuse", "AutomatedBehavior"],
            "weekend_usage_anomaly": ["DeviceMisuse", "PolicyViolation"],
        }

    def _build_recommendations(self) -> dict[str, str]:
        """Build recommendations for common root causes."""
        return {
            # Core recommendations
            "ScreenOnTime": "Reduce screen-on time or lower brightness",
            "AppForegroundTime": "Identify power-hungry apps and optimize usage",
            "BackgroundAppActivity": "Review and restrict background app activity",
            "PoorSignal": "Move to area with better signal coverage",
            "ChargingPattern": "Use proper charging practices (avoid overcharging)",
            "Temperature": "Reduce device usage to prevent overheating",
            "LocationMovement": "High mobility causing network handoffs",
            "WeakWifiCoverage": "Improve WiFi coverage or switch to cellular",
            "LowMemory": "Close unused apps or restart device",
            "LowStorage": "Free up storage by removing unused apps/files",
            "HighCPU": "Identify CPU-intensive apps and optimize",
            "AppVersion": "Update apps to latest versions",
            "OsVersion": "Update to latest OS version",
            # Network Traffic recommendations (NEW)
            "high_upload_ratio": "Investigate apps with high upload activity - potential data exfiltration",
            "exfiltration_risk": "Review network traffic for unauthorized data transfers",
            "unusual_night_activity": "Check for malware or unauthorized background activity during off-hours",
            "interface_switching_rate": "Stabilize network connection - frequent switching drains battery",
            "traffic_concentration": "Review app dominating network traffic - may indicate abuse",
            "upload_anomaly_score": "Audit applications sending excessive data",
            # Security Posture recommendations (NEW)
            "is_rooted_or_jailbroken": "Device is compromised - reimage or replace immediately",
            "has_developer_risk": "Disable developer mode and USB debugging in production",
            "patch_age_critical": "Apply security patches urgently - device is vulnerable",
            "low_security_score": "Review security settings and enable encryption",
            "security_score": "Improve security posture: enable passcode, encryption, update patches",
            # Location Intelligence recommendations (NEW)
            "dead_zone_time_pct": "Device frequently in poor coverage areas - consider WiFi extenders",
            "location_entropy": "High mobility pattern - may need better network handoff policies",
            "ap_hopping_rate": "Frequent AP changes causing battery drain - review AP placement",
            "signal_variability": "Inconsistent signal - investigate WiFi infrastructure",
            "problematic_ap_count": "Multiple problematic APs - review WiFi network configuration",
            # Temporal Pattern recommendations (NEW)
            "unusual_peak_hour": "Activity at unusual hours - verify legitimate use",
            "night_usage_anomaly": "Unexpected night activity - check for malware or misuse",
            "hourly_entropy_low": "Unusually consistent usage pattern - may indicate automated behavior",
            "weekend_usage_anomaly": "Unusual weekend activity - verify business need",
        }

    def _is_benign_deviation(self, feature: str, value: float, baseline: float) -> bool:
        """
        Check if a deviation is benign (not a problem).

        For features like free storage, memory, signal strength - higher values
        are better, so we only care about deviations where the value is LOW.
        """
        # Normalize feature name for matching
        feature_lower = feature.lower()

        # Check if this feature is in our benign-high set
        is_benign_high_feature = any(
            benign.lower() in feature_lower or feature_lower in benign.lower()
            for benign in self._benign_high_features
        )

        if is_benign_high_feature:
            # For benign-high features, only flag if value is BELOW baseline
            # If value is above baseline, it's benign (good)
            return value > baseline

        return False

    def analyze(
        self,
        anomaly_features: dict[str, Any],
        historical_features: list[dict[str, Any]] | None = None,
        cohort_baseline: dict[str, Any] | None = None,
        device_id: Any = None,
        anomaly_type: str = "unknown",
    ) -> RootCauseAnalysisResult:
        """
        Identify probable root causes for an anomaly.

        Args:
            anomaly_features: Feature values at time of anomaly
            historical_features: Previous feature values (for temporal analysis)
            cohort_baseline: Baseline stats for device's cohort
            device_id: Device identifier
            anomaly_type: Type of anomaly detected

        Returns:
            RootCauseAnalysisResult with ranked causes
        """
        root_causes: list[RootCause] = []

        # 1. Check temporal precedence
        if historical_features:
            temporal_causes = self._analyze_temporal_precedence(
                anomaly_features, historical_features
            )
            root_causes.extend(temporal_causes)

        # 2. Check cohort deviation
        if cohort_baseline:
            cohort_causes = self._analyze_cohort_deviation(anomaly_features, cohort_baseline)
            root_causes.extend(cohort_causes)

        # 3. Analyze feature correlations
        correlation_causes = self._analyze_feature_correlations(anomaly_features)
        root_causes.extend(correlation_causes)

        # 4. Traverse causal graph
        causal_causes = self._analyze_causal_graph(anomaly_features, anomaly_type)
        root_causes.extend(causal_causes)

        # Rank causes by confidence
        ranked_causes = self._rank_causes(root_causes)

        # Generate recommendations
        recommendations = []
        for cause in ranked_causes[:3]:
            if cause.recommendation:
                recommendations.append(cause.recommendation)
            elif cause.feature in self._feature_recommendations:
                recommendations.append(self._feature_recommendations[cause.feature])

        # Calculate overall confidence
        analysis_confidence = ranked_causes[0].confidence if ranked_causes else 0.0

        return RootCauseAnalysisResult(
            device_id=device_id,
            anomaly_type=anomaly_type,
            probable_causes=ranked_causes,
            top_cause=ranked_causes[0] if ranked_causes else None,
            analysis_confidence=analysis_confidence,
            recommendations=recommendations,
        )

    def _analyze_temporal_precedence(
        self,
        current: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> list[RootCause]:
        """Find what changed before the anomaly."""
        if len(history) < 2:
            return []

        causes = []
        recent = history[-1] if history else {}

        for key, current_val in current.items():
            if key not in recent or not isinstance(current_val, (int, float)):
                continue

            recent_val = recent.get(key)
            if not isinstance(recent_val, (int, float)) or recent_val == 0:
                continue

            # Calculate change
            change = current_val - recent_val
            change_pct = abs(change) / abs(recent_val)

            if change_pct > 0.5:  # >50% change
                # Skip benign deviations (e.g., high free storage is good, not a problem)
                if self._is_benign_deviation(key, current_val, recent_val):
                    continue

                direction = "increased" if change > 0 else "decreased"
                causes.append(
                    RootCause(
                        cause_type="temporal",
                        feature=key,
                        evidence=f"{key} {direction} by {change_pct:.0%}",
                        confidence=min(0.8, change_pct),
                        details={
                            "previous_value": recent_val,
                            "current_value": current_val,
                            "change_pct": change_pct,
                        },
                        recommendation=self._feature_recommendations.get(key, ""),
                    )
                )

        return causes

    def _analyze_cohort_deviation(
        self,
        current: dict[str, Any],
        cohort_baseline: dict[str, Any],
    ) -> list[RootCause]:
        """Find features that deviate significantly from cohort."""
        causes = []

        for key, current_val in current.items():
            if not isinstance(current_val, (int, float)):
                continue

            # Check for cohort stats
            cohort_key = f"{key}_cohort_median"
            if cohort_key not in cohort_baseline:
                cohort_key = key

            if cohort_key not in cohort_baseline:
                continue

            cohort_val = cohort_baseline[cohort_key]
            if not isinstance(cohort_val, (int, float)) or cohort_val == 0:
                continue

            # Calculate deviation
            deviation = abs(current_val - cohort_val) / abs(cohort_val)

            if deviation > 1.0:  # More than 100% deviation
                # Skip benign deviations (e.g., high free storage is good, not a problem)
                if self._is_benign_deviation(key, current_val, cohort_val):
                    continue

                causes.append(
                    RootCause(
                        cause_type="cohort",
                        feature=key,
                        evidence=f"{key} is {deviation:.0%} different from cohort average",
                        confidence=min(0.7, deviation / 2),
                        details={
                            "device_value": current_val,
                            "cohort_value": cohort_val,
                            "deviation": deviation,
                        },
                    )
                )

        return causes

    def _analyze_feature_correlations(
        self,
        features: dict[str, Any],
    ) -> list[RootCause]:
        """Analyze correlations between features."""
        causes = []

        # Known correlated feature pairs
        correlations = [
            ("BatteryDrainPerHour", "ScreenOnTime", "Screen time correlates with battery drain"),
            ("CrashRate", "LowMemory", "Low memory correlates with crashes"),
            ("DropRate", "SignalStrength", "Weak signal correlates with connection drops"),
            ("Temperature", "CPUUsage", "High CPU correlates with temperature"),
        ]

        for effect, cause_feat, explanation in correlations:
            if effect not in features or cause_feat not in features:
                continue

            effect_val = features.get(effect, 0)
            cause_val = features.get(cause_feat, 0)

            # Check if both are elevated/problematic
            if (
                isinstance(effect_val, (int, float))
                and isinstance(cause_val, (int, float))
                and effect_val > 0
                and cause_val > 0
            ):
                causes.append(
                    RootCause(
                        cause_type="correlation",
                        feature=cause_feat,
                        evidence=explanation,
                        confidence=0.6,
                        details={
                            "effect_feature": effect,
                            "effect_value": effect_val,
                            "cause_value": cause_val,
                        },
                    )
                )

        return causes

    def _analyze_causal_graph(
        self,
        features: dict[str, Any],
        anomaly_type: str,
    ) -> list[RootCause]:
        """Traverse causal graph to find upstream causes."""
        causes = []

        # Map anomaly types to effects
        anomaly_effects = {
            "battery": ["BatteryDrain", "BatteryDrainPerHour"],
            "crash": ["AppCrash", "CrashRate"],
            "network": ["NetworkDrops", "DropRate", "APHopping"],
            "performance": ["SlowPerformance", "HighCPU"],
            "storage": ["LowStorage", "StorageUtilization"],
        }

        target_effects = anomaly_effects.get(anomaly_type.lower(), [])

        # Find causes that lead to these effects
        for cause, effects in self._causal_graph.items():
            for effect in effects:
                if effect in target_effects or anomaly_type.lower() in effect.lower():
                    # Check if cause feature is elevated
                    cause_variants = [
                        cause,
                        cause.lower(),
                        f"{cause}_cohort_z",
                        f"{cause}Rate",
                    ]

                    for variant in cause_variants:
                        if variant in features:
                            val = features[variant]
                            if isinstance(val, (int, float)) and val > 0:
                                causes.append(
                                    RootCause(
                                        cause_type="causal",
                                        feature=cause,
                                        evidence=f"{cause} can cause {effect}",
                                        confidence=0.5,
                                        details={
                                            "causal_path": f"{cause} -> {effect}",
                                            "feature_value": val,
                                        },
                                        recommendation=self._feature_recommendations.get(cause, ""),
                                    )
                                )
                            break

        return causes

    def _rank_causes(self, causes: list[RootCause]) -> list[RootCause]:
        """Rank causes by confidence and deduplicate."""
        if not causes:
            return []

        # Aggregate confidence for same feature from different sources
        feature_confidence: dict[str, float] = {}
        feature_causes: dict[str, RootCause] = {}

        for cause in causes:
            feat = cause.feature
            if feat in feature_confidence:
                feature_confidence[feat] += cause.confidence
            else:
                feature_confidence[feat] = cause.confidence
                feature_causes[feat] = cause

        # Update confidence and sort
        ranked = []
        for feat, conf in sorted(feature_confidence.items(), key=lambda x: -x[1]):
            cause = feature_causes[feat]
            cause.confidence = min(0.95, conf)  # Cap at 0.95
            ranked.append(cause)

        return ranked


def analyze_root_cause(
    anomaly_features: dict[str, Any],
    historical_features: list[dict[str, Any]] | None = None,
    cohort_baseline: dict[str, Any] | None = None,
    device_id: Any = None,
    anomaly_type: str = "unknown",
) -> RootCauseAnalysisResult:
    """
    Convenience function to analyze root causes.

    Args:
        anomaly_features: Feature values at time of anomaly
        historical_features: Previous feature values
        cohort_baseline: Baseline stats for device's cohort
        device_id: Device identifier
        anomaly_type: Type of anomaly detected

    Returns:
        RootCauseAnalysisResult with ranked causes
    """
    analyzer = RootCauseAnalyzer()
    return analyzer.analyze(
        anomaly_features=anomaly_features,
        historical_features=historical_features,
        cohort_baseline=cohort_baseline,
        device_id=device_id,
        anomaly_type=anomaly_type,
    )


def analyze_dataframe_root_causes(
    df: pd.DataFrame,
    anomaly_col: str = "anomaly_label",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Analyze root causes for anomalies in a dataframe.

    Args:
        df: DataFrame with anomaly labels and features
        anomaly_col: Column containing anomaly labels
        feature_cols: Feature columns to analyze

    Returns:
        DataFrame with root_cause_analysis column added
    """
    if anomaly_col not in df.columns:
        logger.warning(f"Anomaly column {anomaly_col} not found")
        return df

    df = df.copy()
    analyzer = RootCauseAnalyzer()

    # Get feature columns
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.columns
            if c not in ["DeviceId", "Timestamp", anomaly_col]
            and df[c].dtype in [np.float64, np.int64, float, int]
        ]

    analyses = []
    for _idx, row in df.iterrows():
        if row.get(anomaly_col) != -1:
            analyses.append(None)
            continue

        features = {col: row[col] for col in feature_cols if col in row.index}
        result = analyzer.analyze(
            anomaly_features=features,
            device_id=row.get("DeviceId"),
        )
        analyses.append(
            {
                "top_cause": result.top_cause.feature if result.top_cause else None,
                "confidence": result.analysis_confidence,
                "recommendations": result.recommendations[:2],
            }
        )

    df["root_cause_analysis"] = analyses
    return df
