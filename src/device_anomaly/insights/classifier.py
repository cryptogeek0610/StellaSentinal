"""Insight classifier that maps raw anomalies to customer-facing categories.

Takes raw anomaly detection results and feature values, analyzes the
contributing factors, and classifies them into InsightCategory types
that customers can understand.

The classifier uses:
1. Rule-based matching for clear threshold violations
2. Feature contribution analysis for complex patterns
3. Confidence scoring based on evidence strength
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from device_anomaly.insights.categories import (
    CATEGORY_METADATA,
    EntityType,
    InsightCategory,
    InsightSeverity,
    get_category_severity,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationEvidence:
    """Evidence supporting a classification decision."""

    feature_name: str
    feature_value: float
    threshold: Optional[float] = None
    comparison: Optional[str] = None  # "above", "below", "equals"
    baseline_value: Optional[float] = None
    z_score: Optional[float] = None
    contribution_weight: float = 0.0


@dataclass
class ClassifiedInsight:
    """Result of classifying an anomaly into an insight category."""

    category: InsightCategory
    severity: InsightSeverity
    confidence: float  # 0.0 to 1.0
    evidence: List[ClassificationEvidence] = field(default_factory=list)
    primary_metric: Optional[str] = None
    primary_value: Optional[float] = None
    affected_entity_type: EntityType = EntityType.DEVICE
    affected_entity_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierConfig:
    """Configuration for the insight classifier."""

    # Battery thresholds
    battery_shift_failure_drain_rate: float = 12.5  # %/hour (100% / 8 hours)
    battery_rapid_drain_multiplier: float = 1.5  # vs baseline
    battery_incomplete_charge_threshold: float = 90.0  # %
    battery_health_threshold: float = 70.0  # %

    # Device thresholds
    excessive_drops_daily: int = 5
    excessive_reboots_daily: int = 3

    # Network thresholds
    wifi_ap_hopping_threshold: int = 10  # unique APs per day
    cell_tower_hopping_threshold: int = 20  # unique towers per day
    network_disconnect_threshold: int = 5  # per day
    network_type_degradation_threshold: int = 3  # types used

    # App thresholds
    app_crash_threshold: int = 3  # per day
    app_anr_threshold: int = 2  # per day
    app_power_drain_multiplier: float = 2.0  # vs expected

    # Confidence thresholds
    min_confidence_for_insight: float = 0.5
    high_confidence_threshold: float = 0.8


class InsightClassifier:
    """Classifies raw anomalies into customer-facing insight categories.

    Usage:
        classifier = InsightClassifier()
        insights = classifier.classify(
            anomaly_result=anomaly,
            features=feature_values,
            device_context=device_metadata
        )
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        """Initialize the classifier.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or ClassifierConfig()

    def classify(
        self,
        features: Dict[str, Any],
        device_context: Optional[Dict[str, Any]] = None,
        anomaly_score: Optional[float] = None,
        feature_contributions: Optional[Dict[str, float]] = None,
    ) -> List[ClassifiedInsight]:
        """Analyze features and return all applicable insight categories.

        Args:
            features: Feature values at anomaly time
            device_context: Device metadata (model, OS, location, etc.)
            anomaly_score: Overall anomaly score from detector
            feature_contributions: Feature importance/contribution scores

        Returns:
            List of ClassifiedInsight with category, confidence, and evidence
        """
        device_context = device_context or {}
        feature_contributions = feature_contributions or {}
        insights: List[ClassifiedInsight] = []

        # Battery classifications
        battery_insights = self._classify_battery(features, device_context)
        insights.extend(battery_insights)

        # Device classifications
        device_insights = self._classify_device(features, device_context)
        insights.extend(device_insights)

        # Network classifications
        network_insights = self._classify_network(features, device_context)
        insights.extend(network_insights)

        # App classifications
        app_insights = self._classify_apps(features, device_context)
        insights.extend(app_insights)

        # Filter by minimum confidence
        insights = [
            i for i in insights
            if i.confidence >= self.config.min_confidence_for_insight
        ]

        # Sort by confidence (descending)
        insights.sort(key=lambda x: x.confidence, reverse=True)

        return insights

    def _classify_battery(
        self,
        features: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[ClassifiedInsight]:
        """Classify battery-related insights."""
        insights = []

        # Get battery features
        drain_rate = features.get("BatteryDrainPerHour", 0)
        battery_level = features.get("BatteryLevel", features.get("battery_start", 100))
        total_drain = features.get("TotalBatteryLevelDrop", 0)
        charge_good = features.get("ChargePatternGoodCount", 0)
        charge_bad = features.get("ChargePatternBadCount", 0)
        battery_health = features.get("BatteryHealth", 100)
        shift_hours = context.get("shift_hours", 8)

        # 1. Battery Shift Failure
        if drain_rate > 0:
            projected_end = battery_level - (drain_rate * shift_hours)
            if projected_end < 10:
                confidence = min(1.0, (10 - projected_end) / 20 + 0.5)
                evidence = [
                    ClassificationEvidence(
                        feature_name="BatteryDrainPerHour",
                        feature_value=drain_rate,
                        threshold=self.config.battery_shift_failure_drain_rate,
                        comparison="above",
                        contribution_weight=0.8,
                    ),
                    ClassificationEvidence(
                        feature_name="ProjectedBatteryAtShiftEnd",
                        feature_value=projected_end,
                        threshold=10,
                        comparison="below",
                        contribution_weight=0.7,
                    ),
                ]
                insights.append(ClassifiedInsight(
                    category=InsightCategory.BATTERY_SHIFT_FAILURE,
                    severity=InsightSeverity.CRITICAL,
                    confidence=confidence,
                    evidence=evidence,
                    primary_metric="BatteryDrainPerHour",
                    primary_value=drain_rate,
                    metadata={
                        "projected_battery_at_shift_end": projected_end,
                        "shift_hours": shift_hours,
                        "current_battery": battery_level,
                    },
                ))

        # 2. Battery Rapid Drain
        baseline_drain = context.get("baseline_battery_drain_per_hour", 5)
        if drain_rate > baseline_drain * self.config.battery_rapid_drain_multiplier:
            multiplier = drain_rate / baseline_drain if baseline_drain > 0 else 2.0
            confidence = min(1.0, 0.5 + (multiplier - 1.5) * 0.25)
            evidence = [
                ClassificationEvidence(
                    feature_name="BatteryDrainPerHour",
                    feature_value=drain_rate,
                    baseline_value=baseline_drain,
                    comparison="above",
                    contribution_weight=0.9,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.BATTERY_RAPID_DRAIN,
                severity=InsightSeverity.HIGH,
                confidence=confidence,
                evidence=evidence,
                primary_metric="BatteryDrainPerHour",
                primary_value=drain_rate,
                metadata={
                    "baseline_drain_rate": baseline_drain,
                    "drain_multiplier": multiplier,
                },
            ))

        # 3. Battery Charge Incomplete
        if battery_level < self.config.battery_incomplete_charge_threshold:
            # Check if this is at shift start (morning)
            hour = context.get("hour_of_day", 12)
            if 5 <= hour <= 9:  # Morning shift start
                confidence = 0.6 + (90 - battery_level) / 100
                evidence = [
                    ClassificationEvidence(
                        feature_name="BatteryLevel",
                        feature_value=battery_level,
                        threshold=self.config.battery_incomplete_charge_threshold,
                        comparison="below",
                        contribution_weight=0.8,
                    ),
                ]
                insights.append(ClassifiedInsight(
                    category=InsightCategory.BATTERY_CHARGE_INCOMPLETE,
                    severity=InsightSeverity.MEDIUM,
                    confidence=min(1.0, confidence),
                    evidence=evidence,
                    primary_metric="BatteryLevel",
                    primary_value=battery_level,
                    metadata={"hour_of_day": hour},
                ))

        # 4. Battery Charge Pattern Issues
        if charge_bad > 0 and charge_bad >= charge_good:
            total_charges = charge_good + charge_bad
            bad_ratio = charge_bad / total_charges if total_charges > 0 else 0
            confidence = 0.5 + bad_ratio * 0.4
            evidence = [
                ClassificationEvidence(
                    feature_name="ChargePatternBadCount",
                    feature_value=charge_bad,
                    contribution_weight=0.7,
                ),
                ClassificationEvidence(
                    feature_name="ChargePatternGoodCount",
                    feature_value=charge_good,
                    contribution_weight=0.3,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.BATTERY_CHARGE_PATTERN,
                severity=InsightSeverity.MEDIUM,
                confidence=min(1.0, confidence),
                evidence=evidence,
                primary_metric="ChargePatternBadCount",
                primary_value=charge_bad,
                metadata={"bad_charge_ratio": bad_ratio},
            ))

        # 5. Battery Health Degraded
        if battery_health < self.config.battery_health_threshold:
            confidence = 0.6 + (self.config.battery_health_threshold - battery_health) / 50
            evidence = [
                ClassificationEvidence(
                    feature_name="BatteryHealth",
                    feature_value=battery_health,
                    threshold=self.config.battery_health_threshold,
                    comparison="below",
                    contribution_weight=0.9,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.BATTERY_HEALTH_DEGRADED,
                severity=InsightSeverity.HIGH,
                confidence=min(1.0, confidence),
                evidence=evidence,
                primary_metric="BatteryHealth",
                primary_value=battery_health,
            ))

        return insights

    def _classify_device(
        self,
        features: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[ClassifiedInsight]:
        """Classify device-related insights (drops, reboots)."""
        insights = []

        drop_count = features.get("TotalDropCnt", 0)
        reboot_count = features.get("RebootCount", features.get("ForceStopCount", 0))

        # 1. Excessive Drops
        if drop_count >= self.config.excessive_drops_daily:
            confidence = min(1.0, 0.5 + (drop_count - 5) * 0.1)
            evidence = [
                ClassificationEvidence(
                    feature_name="TotalDropCnt",
                    feature_value=drop_count,
                    threshold=self.config.excessive_drops_daily,
                    comparison="above",
                    contribution_weight=0.9,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.EXCESSIVE_DROPS,
                severity=InsightSeverity.HIGH if drop_count > 10 else InsightSeverity.MEDIUM,
                confidence=confidence,
                evidence=evidence,
                primary_metric="TotalDropCnt",
                primary_value=drop_count,
            ))

        # 2. Excessive Reboots
        if reboot_count >= self.config.excessive_reboots_daily:
            confidence = min(1.0, 0.5 + (reboot_count - 3) * 0.15)
            evidence = [
                ClassificationEvidence(
                    feature_name="RebootCount",
                    feature_value=reboot_count,
                    threshold=self.config.excessive_reboots_daily,
                    comparison="above",
                    contribution_weight=0.9,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.EXCESSIVE_REBOOTS,
                severity=InsightSeverity.HIGH,
                confidence=confidence,
                evidence=evidence,
                primary_metric="RebootCount",
                primary_value=reboot_count,
            ))

        # 3. Device Abuse Pattern (combined drops + reboots)
        if drop_count >= 3 and reboot_count >= 2:
            confidence = min(1.0, 0.6 + (drop_count + reboot_count) * 0.05)
            evidence = [
                ClassificationEvidence(
                    feature_name="TotalDropCnt",
                    feature_value=drop_count,
                    contribution_weight=0.5,
                ),
                ClassificationEvidence(
                    feature_name="RebootCount",
                    feature_value=reboot_count,
                    contribution_weight=0.5,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.DEVICE_ABUSE_PATTERN,
                severity=InsightSeverity.HIGH,
                confidence=confidence,
                evidence=evidence,
                primary_metric="CombinedAbuseScore",
                primary_value=drop_count + reboot_count,
            ))

        return insights

    def _classify_network(
        self,
        features: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[ClassifiedInsight]:
        """Classify network-related insights."""
        insights = []

        unique_aps = features.get("UniqueAPsConnected", 0)
        wifi_disconnects = features.get("WifiDisconnectCount", 0)
        cell_towers = features.get("CellTowerChanges", features.get("UniqueCellIds", 0))
        total_disconnects = features.get("DisconnectCount", features.get("TotalDropCnt", 0))
        time_no_network = features.get("TimeOnNoNetwork", 0)
        network_type_count = features.get("NetworkTypeCount", 1)

        # Network type times
        time_5g = features.get("TimeOn5G", 0)
        time_4g = features.get("TimeOn4G", 0)
        time_3g = features.get("TimeOn3G", 0)
        time_2g = features.get("TimeOn2G", 0)

        # 1. WiFi AP Hopping
        if unique_aps >= self.config.wifi_ap_hopping_threshold:
            confidence = min(1.0, 0.5 + (unique_aps - 10) * 0.05)
            evidence = [
                ClassificationEvidence(
                    feature_name="UniqueAPsConnected",
                    feature_value=unique_aps,
                    threshold=self.config.wifi_ap_hopping_threshold,
                    comparison="above",
                    contribution_weight=0.8,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.WIFI_AP_HOPPING,
                severity=InsightSeverity.MEDIUM,
                confidence=confidence,
                evidence=evidence,
                primary_metric="UniqueAPsConnected",
                primary_value=unique_aps,
            ))

        # 2. Cellular Tower Hopping
        if cell_towers >= self.config.cell_tower_hopping_threshold:
            confidence = min(1.0, 0.5 + (cell_towers - 20) * 0.025)
            evidence = [
                ClassificationEvidence(
                    feature_name="CellTowerChanges",
                    feature_value=cell_towers,
                    threshold=self.config.cell_tower_hopping_threshold,
                    comparison="above",
                    contribution_weight=0.8,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.CELLULAR_TOWER_HOPPING,
                severity=InsightSeverity.MEDIUM,
                confidence=confidence,
                evidence=evidence,
                primary_metric="CellTowerChanges",
                primary_value=cell_towers,
            ))

        # 3. Network Disconnect Pattern
        if total_disconnects >= self.config.network_disconnect_threshold:
            confidence = min(1.0, 0.5 + (total_disconnects - 5) * 0.1)
            evidence = [
                ClassificationEvidence(
                    feature_name="DisconnectCount",
                    feature_value=total_disconnects,
                    threshold=self.config.network_disconnect_threshold,
                    comparison="above",
                    contribution_weight=0.8,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.NETWORK_DISCONNECT_PATTERN,
                severity=InsightSeverity.HIGH,
                confidence=confidence,
                evidence=evidence,
                primary_metric="DisconnectCount",
                primary_value=total_disconnects,
            ))

        # 4. Network Type Degradation (5G -> 4G -> 3G pattern)
        if network_type_count >= self.config.network_type_degradation_threshold:
            # Check if using older tech more than newer
            old_time = time_3g + time_2g
            new_time = time_5g + time_4g
            if old_time > new_time and old_time > 0:
                confidence = 0.5 + (old_time / (old_time + new_time + 1)) * 0.3
                evidence = [
                    ClassificationEvidence(
                        feature_name="NetworkTypeCount",
                        feature_value=network_type_count,
                        contribution_weight=0.4,
                    ),
                    ClassificationEvidence(
                        feature_name="TimeOn3G",
                        feature_value=time_3g,
                        contribution_weight=0.3,
                    ),
                    ClassificationEvidence(
                        feature_name="TimeOn5G",
                        feature_value=time_5g,
                        contribution_weight=0.3,
                    ),
                ]
                insights.append(ClassifiedInsight(
                    category=InsightCategory.CELLULAR_TECH_DEGRADATION,
                    severity=InsightSeverity.LOW,
                    confidence=min(1.0, confidence),
                    evidence=evidence,
                    primary_metric="NetworkTypeCount",
                    primary_value=network_type_count,
                    metadata={
                        "time_on_old_tech": old_time,
                        "time_on_new_tech": new_time,
                    },
                ))

        # 5. Device Hidden Pattern (extended offline)
        # Check for suspicious offline patterns
        if time_no_network > 4 * 3600:  # More than 4 hours offline
            hour = context.get("hour_of_day", 12)
            # Suspicious if offline during expected work hours or evening
            if 9 <= hour <= 22:
                confidence = 0.5 + (time_no_network / 3600 - 4) * 0.05
                evidence = [
                    ClassificationEvidence(
                        feature_name="TimeOnNoNetwork",
                        feature_value=time_no_network / 3600,  # Convert to hours
                        contribution_weight=0.9,
                    ),
                ]
                insights.append(ClassifiedInsight(
                    category=InsightCategory.DEVICE_HIDDEN_PATTERN,
                    severity=InsightSeverity.HIGH,
                    confidence=min(0.9, confidence),  # Cap at 0.9 - needs investigation
                    evidence=evidence,
                    primary_metric="TimeOnNoNetwork",
                    primary_value=time_no_network / 3600,
                    metadata={"hour_of_day": hour},
                ))

        return insights

    def _classify_apps(
        self,
        features: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[ClassifiedInsight]:
        """Classify app-related insights."""
        insights = []

        crash_count = features.get("CrashCount", 0)
        anr_count = features.get("ANRCount", 0)
        app_battery_drain = features.get("TotalBatteryAppDrain", 0)
        app_foreground_time = features.get("AppForegroundTime", 0)
        max_app_drain = features.get("MaxSingleAppDrain", 0)

        # 1. App Crash Pattern
        if crash_count >= self.config.app_crash_threshold:
            confidence = min(1.0, 0.5 + (crash_count - 3) * 0.15)
            evidence = [
                ClassificationEvidence(
                    feature_name="CrashCount",
                    feature_value=crash_count,
                    threshold=self.config.app_crash_threshold,
                    comparison="above",
                    contribution_weight=0.9,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.APP_CRASH_PATTERN,
                severity=InsightSeverity.HIGH,
                confidence=confidence,
                evidence=evidence,
                primary_metric="CrashCount",
                primary_value=crash_count,
            ))

        # 2. App ANR Pattern
        if anr_count >= self.config.app_anr_threshold:
            confidence = min(1.0, 0.5 + (anr_count - 2) * 0.2)
            evidence = [
                ClassificationEvidence(
                    feature_name="ANRCount",
                    feature_value=anr_count,
                    threshold=self.config.app_anr_threshold,
                    comparison="above",
                    contribution_weight=0.9,
                ),
            ]
            insights.append(ClassifiedInsight(
                category=InsightCategory.APP_ANR_PATTERN,
                severity=InsightSeverity.MEDIUM,
                confidence=confidence,
                evidence=evidence,
                primary_metric="ANRCount",
                primary_value=anr_count,
            ))

        # 3. App Power Drain (disproportionate battery usage)
        if app_foreground_time > 0 and app_battery_drain > 0:
            # Calculate drain per hour of foreground time
            drain_per_hour = (app_battery_drain / (app_foreground_time / 3600)) if app_foreground_time > 0 else 0
            baseline_drain_per_hour = context.get("baseline_app_drain_per_hour", 5)

            if drain_per_hour > baseline_drain_per_hour * self.config.app_power_drain_multiplier:
                multiplier = drain_per_hour / baseline_drain_per_hour if baseline_drain_per_hour > 0 else 2.0
                confidence = min(1.0, 0.5 + (multiplier - 2) * 0.2)
                evidence = [
                    ClassificationEvidence(
                        feature_name="TotalBatteryAppDrain",
                        feature_value=app_battery_drain,
                        baseline_value=baseline_drain_per_hour * (app_foreground_time / 3600),
                        comparison="above",
                        contribution_weight=0.7,
                    ),
                    ClassificationEvidence(
                        feature_name="AppForegroundTime",
                        feature_value=app_foreground_time / 3600,  # hours
                        contribution_weight=0.3,
                    ),
                ]
                insights.append(ClassifiedInsight(
                    category=InsightCategory.APP_POWER_DRAIN,
                    severity=InsightSeverity.MEDIUM,
                    confidence=confidence,
                    evidence=evidence,
                    primary_metric="DrainPerForegroundHour",
                    primary_value=drain_per_hour,
                    metadata={
                        "drain_multiplier": multiplier,
                        "baseline_drain_per_hour": baseline_drain_per_hour,
                    },
                ))

        return insights

    def classify_for_location(
        self,
        location_id: str,
        device_insights: List[List[ClassifiedInsight]],
        location_context: Optional[Dict[str, Any]] = None,
    ) -> List[ClassifiedInsight]:
        """Aggregate device insights into location-level insights.

        Args:
            location_id: The location ID
            device_insights: List of insight lists from individual devices
            location_context: Location metadata

        Returns:
            Location-level insights
        """
        location_context = location_context or {}
        location_insights = []

        # Count devices with each insight category
        category_counts: Dict[InsightCategory, int] = {}
        for device_insight_list in device_insights:
            seen_categories = set()
            for insight in device_insight_list:
                if insight.category not in seen_categories:
                    category_counts[insight.category] = category_counts.get(insight.category, 0) + 1
                    seen_categories.add(insight.category)

        total_devices = len(device_insights)

        # Check for anomaly clusters (multiple devices with same issue)
        for category, count in category_counts.items():
            if count >= 3 and count / total_devices >= 0.2:  # At least 3 devices or 20%
                confidence = min(1.0, 0.5 + count / total_devices)
                evidence = [
                    ClassificationEvidence(
                        feature_name="DeviceCountWithIssue",
                        feature_value=count,
                        contribution_weight=0.8,
                    ),
                    ClassificationEvidence(
                        feature_name="PercentageAffected",
                        feature_value=count / total_devices * 100,
                        contribution_weight=0.5,
                    ),
                ]
                location_insights.append(ClassifiedInsight(
                    category=InsightCategory.LOCATION_ANOMALY_CLUSTER,
                    severity=InsightSeverity.HIGH if count >= 5 else InsightSeverity.MEDIUM,
                    confidence=confidence,
                    evidence=evidence,
                    primary_metric="DeviceCountWithIssue",
                    primary_value=count,
                    affected_entity_type=EntityType.LOCATION,
                    affected_entity_id=location_id,
                    metadata={
                        "clustered_category": category.value,
                        "devices_affected": count,
                        "total_devices": total_devices,
                        "percentage_affected": count / total_devices * 100,
                    },
                ))

        return location_insights
