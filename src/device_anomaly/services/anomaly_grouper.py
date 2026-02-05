"""Smart anomaly grouping service.

Groups anomalies based on:
1. Common remediation - anomalies with the same suggested fix
2. Insight category - same type of issue (battery, network, etc.)
3. Similarity - anomalies with similar metric patterns
4. Temporal + Cohort - same device model with issues in same time window
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from device_anomaly.api.models import (
    AnomalyGroup,
    AnomalyGroupMember,
    GroupedAnomaliesResponse,
    RemediationSuggestion,
)
from device_anomaly.database.schema import AnomalyResult, DeviceMetadata
from device_anomaly.insights.categories import (
    InsightCategory,
)

logger = logging.getLogger(__name__)


# Mapping from primary metric to insight category
METRIC_TO_CATEGORY = {
    "total_battery_level_drop": InsightCategory.BATTERY_RAPID_DRAIN,
    "total_free_storage_kb": InsightCategory.DEVICE_PERFORMANCE_DEGRADED,
    "download": InsightCategory.NETWORK_THROUGHPUT_ISSUE,
    "upload": InsightCategory.NETWORK_THROUGHPUT_ISSUE,
    "offline_time": InsightCategory.NETWORK_DISCONNECT_PATTERN,
    "disconnect_count": InsightCategory.NETWORK_DISCONNECT_PATTERN,
    "wifi_signal_strength": InsightCategory.WIFI_DEAD_ZONE,
    "connection_time": InsightCategory.NETWORK_DISCONNECT_PATTERN,
}

# Category display names
CATEGORY_DISPLAY_NAMES = {
    InsightCategory.BATTERY_SHIFT_FAILURE: "Battery Shift Failures",
    InsightCategory.BATTERY_RAPID_DRAIN: "Battery Drain Issues",
    InsightCategory.BATTERY_CHARGE_INCOMPLETE: "Incomplete Charging",
    InsightCategory.BATTERY_CHARGE_PATTERN: "Charging Pattern Issues",
    InsightCategory.BATTERY_PERIODIC_DRAIN: "Periodic Battery Drain",
    InsightCategory.BATTERY_HEALTH_DEGRADED: "Degraded Battery Health",
    InsightCategory.EXCESSIVE_DROPS: "Excessive Device Drops",
    InsightCategory.EXCESSIVE_REBOOTS: "Excessive Reboots",
    InsightCategory.DEVICE_ABUSE_PATTERN: "Device Abuse Patterns",
    InsightCategory.DEVICE_PERFORMANCE_DEGRADED: "Performance Issues",
    InsightCategory.WIFI_AP_HOPPING: "WiFi AP Hopping",
    InsightCategory.WIFI_STICKINESS: "WiFi Stickiness Issues",
    InsightCategory.WIFI_DEAD_ZONE: "WiFi Dead Zones",
    InsightCategory.CELLULAR_TOWER_HOPPING: "Cellular Tower Hopping",
    InsightCategory.CELLULAR_CARRIER_ISSUE: "Carrier Issues",
    InsightCategory.CELLULAR_TECH_DEGRADATION: "Cellular Degradation",
    InsightCategory.NETWORK_DISCONNECT_PATTERN: "Network Disconnections",
    InsightCategory.NETWORK_THROUGHPUT_ISSUE: "Throughput Issues",
    InsightCategory.DEVICE_HIDDEN_PATTERN: "Hidden Device Patterns",
    InsightCategory.APP_CRASH_PATTERN: "App Crashes",
    InsightCategory.APP_POWER_DRAIN: "App Power Drain",
    InsightCategory.APP_ANR_PATTERN: "App Not Responding",
    InsightCategory.APP_PERFORMANCE_ISSUE: "App Performance Issues",
    InsightCategory.COHORT_PERFORMANCE_ISSUE: "Cohort Performance",
    InsightCategory.FIRMWARE_BUG: "Firmware Issues",
    InsightCategory.OS_VERSION_ISSUE: "OS Version Issues",
    InsightCategory.PROBLEM_COMBINATION: "Device Combination Issues",
    InsightCategory.LOCATION_ANOMALY_CLUSTER: "Location Anomaly Clusters",
    InsightCategory.LOCATION_PERFORMANCE_GAP: "Location Performance Gaps",
    InsightCategory.LOCATION_BASELINE_DEVIATION: "Location Baseline Deviations",
    InsightCategory.LOCATION_INFRASTRUCTURE_ISSUE: "Location Infrastructure Issues",
}

# Remediation title to category mapping
REMEDIATION_TITLES = {
    "Clear Device Storage": "storage_clear",
    "Investigate Battery Drain": "battery_investigate",
    "Diagnose Network Connectivity": "network_diagnose",
    "Contact Device User": "user_contact",
    "Restart Device": "device_restart",
    "Clear App Cache": "cache_clear",
    "Update Firmware": "firmware_update",
    "Check Charging Infrastructure": "charging_check",
}


@dataclass
class ClassifiedAnomaly:
    """Anomaly with classification data."""

    anomaly: AnomalyResult
    severity: str
    primary_metric: str | None
    primary_category: InsightCategory | None
    suggested_remediation_title: str | None
    device_name: str | None
    device_model: str | None
    location: str | None
    grouped: bool = False


def _get_severity(score: float) -> str:
    """Convert anomaly score to severity level."""
    if score <= -0.7:
        return "critical"
    if score <= -0.5:
        return "high"
    if score <= -0.3:
        return "medium"
    return "low"


def _get_severity_rank(severity: str) -> int:
    """Get numeric rank for severity (lower = more severe)."""
    ranks = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return ranks.get(severity, 3)


def _find_primary_metric(anomaly: AnomalyResult) -> str | None:
    """Find the primary contributing metric for an anomaly."""
    metrics = {
        "total_battery_level_drop": (
            anomaly.total_battery_level_drop,
            30,
            True,
        ),  # threshold, above_is_bad
        "total_free_storage_kb": (
            anomaly.total_free_storage_kb,
            500000,
            False,
        ),  # below threshold is bad
        "download": (anomaly.download, 500, True),
        "upload": (anomaly.upload, 200, True),
        "offline_time": (anomaly.offline_time, 30, True),
        "disconnect_count": (anomaly.disconnect_count, 5, True),
        "wifi_signal_strength": (
            anomaly.wifi_signal_strength,
            -70,
            False,
        ),  # below threshold is bad
        "connection_time": (anomaly.connection_time, 5, True),
    }

    # Find metric with worst deviation
    worst_metric = None
    worst_score = 0

    for metric, (value, threshold, above_is_bad) in metrics.items():
        if value is None:
            continue

        if above_is_bad:
            if value > threshold:
                score = value / threshold if threshold > 0 else value
                if score > worst_score:
                    worst_score = score
                    worst_metric = metric
        else:
            if value < threshold:
                score = threshold / value if value > 0 else threshold
                if score > worst_score:
                    worst_score = score
                    worst_metric = metric

    return worst_metric


def _get_suggested_remediation_title(anomaly: AnomalyResult) -> str | None:
    """Determine suggested remediation based on anomaly metrics."""
    # Storage issues
    if anomaly.total_free_storage_kb and anomaly.total_free_storage_kb < 500000:
        return "Clear Device Storage"

    # Battery issues
    if anomaly.total_battery_level_drop and anomaly.total_battery_level_drop > 30:
        return "Investigate Battery Drain"

    # Network issues
    if anomaly.disconnect_count and anomaly.disconnect_count > 5:
        return "Diagnose Network Connectivity"

    if anomaly.offline_time and anomaly.offline_time > 30:
        return "Diagnose Network Connectivity"

    return None


class AnomalyGrouper:
    """Smart grouping service for anomalies.

    Grouping priority order:
    1. Remediation Match (highest value - actionable)
    2. Category Match (semantic grouping)
    3. Similarity Cluster (pattern matching)
    4. Temporal + Cohort (device model + time window)
    """

    def __init__(self, db: Session, tenant_id: str):
        self.db = db
        self.tenant_id = tenant_id
        self._device_metadata: dict[int, DeviceMetadata] = {}

    def _load_device_metadata(self, device_ids: set[int]) -> None:
        """Load device metadata for a set of device IDs."""
        if not device_ids:
            return

        try:
            devices = (
                self.db.query(DeviceMetadata)
                .filter(DeviceMetadata.tenant_id == self.tenant_id)
                .filter(DeviceMetadata.device_id.in_(list(device_ids)))
                .all()
            )

            for device in devices:
                self._device_metadata[device.device_id] = device
        except Exception:
            # Database schema might not match - gracefully handle
            # This can happen if migrations haven't been run
            pass

    def _classify_anomaly(self, anomaly: AnomalyResult) -> ClassifiedAnomaly:
        """Classify a single anomaly."""
        severity = _get_severity(anomaly.anomaly_score)
        primary_metric = _find_primary_metric(anomaly)

        # Determine category from primary metric
        primary_category = None
        if primary_metric:
            primary_category = METRIC_TO_CATEGORY.get(primary_metric)

        # Get suggested remediation
        remediation_title = _get_suggested_remediation_title(anomaly)

        # Get device metadata
        device = self._device_metadata.get(anomaly.device_id)
        device_name = device.device_name if device else None
        device_model = device.device_model if device else None
        location = device.location if device else None

        return ClassifiedAnomaly(
            anomaly=anomaly,
            severity=severity,
            primary_metric=primary_metric,
            primary_category=primary_category,
            suggested_remediation_title=remediation_title,
            device_name=device_name,
            device_model=device_model,
            location=location,
        )

    def _create_group_member(self, classified: ClassifiedAnomaly) -> AnomalyGroupMember:
        """Create an AnomalyGroupMember from a classified anomaly."""
        return AnomalyGroupMember(
            anomaly_id=classified.anomaly.id,
            device_id=classified.anomaly.device_id,
            anomaly_score=classified.anomaly.anomaly_score,
            severity=classified.severity,
            status=classified.anomaly.status,
            timestamp=classified.anomaly.timestamp,
            device_name=classified.device_name,
            device_model=classified.device_model,
            location=classified.location,
            primary_metric=classified.primary_metric,
        )

    def _create_remediation_suggestion(self, title: str) -> RemediationSuggestion:
        """Create a RemediationSuggestion for a group."""
        remediation_steps = {
            "Clear Device Storage": [
                "Review and remove unused applications",
                "Clear application caches",
                "Remove old downloads and media files",
                "Consider offloading data to cloud storage",
            ],
            "Investigate Battery Drain": [
                "Check battery usage by app in device settings",
                "Identify background apps consuming power",
                "Review location services and sync settings",
                "Consider disabling power-hungry features",
            ],
            "Diagnose Network Connectivity": [
                "Check WiFi signal strength at device location",
                "Verify network credentials are current",
                "Test device connectivity in different locations",
                "Consider network infrastructure review",
            ],
        }

        steps = remediation_steps.get(title, ["Contact device user for more information"])

        return RemediationSuggestion(
            remediation_id=str(uuid.uuid4()),
            title=title,
            description="Apply this fix to all devices in the group",
            detailed_steps=steps,
            priority=1,
            confidence_score=0.8,
            confidence_level="high",
            source="policy",
            source_details="Common remediation for grouped anomalies",
            is_automated=False,
        )

    def _build_group(
        self,
        group_id: str,
        group_type: str,
        group_category: str,
        group_name: str,
        classified_anomalies: list[ClassifiedAnomaly],
        grouping_factors: list[str],
        suggested_remediation: RemediationSuggestion | None = None,
    ) -> AnomalyGroup:
        """Build an AnomalyGroup from classified anomalies."""
        # Mark anomalies as grouped
        for c in classified_anomalies:
            c.grouped = True

        # Calculate group metrics
        device_ids = {c.anomaly.device_id for c in classified_anomalies}
        open_count = sum(
            1 for c in classified_anomalies if c.anomaly.status in ("open", "investigating")
        )

        # Get worst severity
        severities = [c.severity for c in classified_anomalies]
        worst_severity = min(severities, key=_get_severity_rank)

        # Time range
        timestamps = [c.anomaly.timestamp for c in classified_anomalies]
        time_start = min(timestamps)
        time_end = max(timestamps)

        # Common location
        locations = [c.location for c in classified_anomalies if c.location]
        common_location = (
            locations[0] if locations and all(loc == locations[0] for loc in locations) else None
        )

        # Common device model
        models = [c.device_model for c in classified_anomalies if c.device_model]
        common_model = models[0] if models and all(m == models[0] for m in models) else None

        # Sample anomalies (first 5)
        sample = [self._create_group_member(c) for c in classified_anomalies[:5]]

        return AnomalyGroup(
            group_id=group_id,
            group_name=group_name,
            group_category=group_category,
            group_type=group_type,
            severity=worst_severity,
            total_count=len(classified_anomalies),
            open_count=open_count,
            device_count=len(device_ids),
            suggested_remediation=suggested_remediation,
            common_location=common_location,
            common_device_model=common_model,
            time_range_start=time_start,
            time_range_end=time_end,
            sample_anomalies=sample,
            grouping_factors=grouping_factors,
        )

    def _group_by_remediation(
        self,
        classified: list[ClassifiedAnomaly],
    ) -> list[AnomalyGroup]:
        """Group anomalies by common remediation (highest priority)."""
        groups = []

        # Group by remediation title
        by_remediation: dict[str, list[ClassifiedAnomaly]] = defaultdict(list)
        for c in classified:
            if c.grouped or not c.suggested_remediation_title:
                continue
            by_remediation[c.suggested_remediation_title].append(c)

        for title, anomalies in by_remediation.items():
            if len(anomalies) < 2:
                continue

            # Sort by severity, then by score
            anomalies.sort(key=lambda c: (_get_severity_rank(c.severity), c.anomaly.anomaly_score))

            group_id = hashlib.md5(f"remediation:{title}".encode()).hexdigest()[:12]
            device_count = len({c.anomaly.device_id for c in anomalies})
            group_name = f"{title} ({device_count} devices)"

            groups.append(
                self._build_group(
                    group_id=group_id,
                    group_type="remediation_match",
                    group_category=REMEDIATION_TITLES.get(title, "remediation"),
                    group_name=group_name,
                    classified_anomalies=anomalies,
                    grouping_factors=[f"Same suggested fix: {title}"],
                    suggested_remediation=self._create_remediation_suggestion(title),
                )
            )

        return groups

    def _group_by_category(
        self,
        classified: list[ClassifiedAnomaly],
    ) -> list[AnomalyGroup]:
        """Group remaining anomalies by insight category."""
        groups = []

        # Group by category
        by_category: dict[InsightCategory, list[ClassifiedAnomaly]] = defaultdict(list)
        for c in classified:
            if c.grouped or not c.primary_category:
                continue
            by_category[c.primary_category].append(c)

        for category, anomalies in by_category.items():
            if len(anomalies) < 2:
                continue

            # Sort by severity, then by score
            anomalies.sort(key=lambda c: (_get_severity_rank(c.severity), c.anomaly.anomaly_score))

            category_name = CATEGORY_DISPLAY_NAMES.get(category, category.value)
            group_id = hashlib.md5(f"category:{category.value}".encode()).hexdigest()[:12]
            device_count = len({c.anomaly.device_id for c in anomalies})

            # Check for common location
            locations = [c.location for c in anomalies if c.location]
            if locations and all(loc == locations[0] for loc in locations):
                group_name = f"{category_name} at {locations[0]} ({device_count} devices)"
                grouping_factors = [
                    f"Same category: {category_name}",
                    f"Same location: {locations[0]}",
                ]
            else:
                group_name = f"{category_name} ({device_count} devices)"
                grouping_factors = [f"Same category: {category_name}"]

            groups.append(
                self._build_group(
                    group_id=group_id,
                    group_type="category_match",
                    group_category=category.value,
                    group_name=group_name,
                    classified_anomalies=anomalies,
                    grouping_factors=grouping_factors,
                )
            )

        return groups

    def _group_by_temporal_cohort(
        self,
        classified: list[ClassifiedAnomaly],
        window_hours: int = 24,
    ) -> list[AnomalyGroup]:
        """Group remaining anomalies by device model and time window."""
        groups = []

        # Get ungrouped anomalies with device model
        ungrouped = [c for c in classified if not c.grouped and c.device_model]
        if len(ungrouped) < 2:
            return groups

        # Group by device model
        by_model: dict[str, list[ClassifiedAnomaly]] = defaultdict(list)
        for c in ungrouped:
            by_model[c.device_model].append(c)

        for model, anomalies in by_model.items():
            if len(anomalies) < 2:
                continue

            # Check if anomalies are within the time window
            timestamps = [c.anomaly.timestamp for c in anomalies]
            time_range = max(timestamps) - min(timestamps)

            if time_range <= timedelta(hours=window_hours):
                anomalies.sort(
                    key=lambda c: (_get_severity_rank(c.severity), c.anomaly.anomaly_score)
                )

                group_id = hashlib.md5(f"cohort:{model}".encode()).hexdigest()[:12]
                device_count = len({c.anomaly.device_id for c in anomalies})
                group_name = f"Issues on {model} ({device_count} devices)"

                groups.append(
                    self._build_group(
                        group_id=group_id,
                        group_type="temporal_cluster",
                        group_category="cohort_performance_issue",
                        group_name=group_name,
                        classified_anomalies=anomalies,
                        grouping_factors=[
                            f"Same device model: {model}",
                            f"Within {window_hours}h time window",
                        ],
                    )
                )

        return groups

    def _group_by_location(
        self,
        classified: list[ClassifiedAnomaly],
    ) -> list[AnomalyGroup]:
        """Group remaining anomalies by location."""
        groups = []

        # Get ungrouped anomalies with location
        ungrouped = [c for c in classified if not c.grouped and c.location]
        if len(ungrouped) < 2:
            return groups

        # Group by location
        by_location: dict[str, list[ClassifiedAnomaly]] = defaultdict(list)
        for c in ungrouped:
            by_location[c.location].append(c)

        for location, anomalies in by_location.items():
            if len(anomalies) < 3:  # Require at least 3 for location grouping
                continue

            anomalies.sort(key=lambda c: (_get_severity_rank(c.severity), c.anomaly.anomaly_score))

            group_id = hashlib.md5(f"location:{location}".encode()).hexdigest()[:12]
            device_count = len({c.anomaly.device_id for c in anomalies})
            group_name = f"Multiple Issues at {location} ({device_count} devices)"

            groups.append(
                self._build_group(
                    group_id=group_id,
                    group_type="location_cluster",
                    group_category="location_anomaly_cluster",
                    group_name=group_name,
                    classified_anomalies=anomalies,
                    grouping_factors=[f"Same location: {location}"],
                )
            )

        return groups

    def group_anomalies(
        self,
        anomalies: list[AnomalyResult],
        min_group_size: int = 2,
        temporal_window_hours: int = 24,
    ) -> GroupedAnomaliesResponse:
        """Group anomalies using multi-factor clustering.

        Args:
            anomalies: List of anomalies to group
            min_group_size: Minimum anomalies to form a group
            temporal_window_hours: Time window for temporal grouping

        Returns:
            GroupedAnomaliesResponse with groups and ungrouped anomalies
        """
        if not anomalies:
            return GroupedAnomaliesResponse(
                groups=[],
                total_anomalies=0,
                total_groups=0,
                ungrouped_count=0,
                ungrouped_anomalies=[],
                grouping_method="smart_auto",
                computed_at=datetime.now(UTC),
            )

        # Load device metadata
        device_ids = {a.device_id for a in anomalies}
        self._load_device_metadata(device_ids)

        # Classify all anomalies
        classified = [self._classify_anomaly(a) for a in anomalies]

        all_groups: list[AnomalyGroup] = []

        # Priority 1: Group by remediation (most actionable)
        remediation_groups = self._group_by_remediation(classified)
        all_groups.extend(remediation_groups)

        # Priority 2: Group by category
        category_groups = self._group_by_category(classified)
        all_groups.extend(category_groups)

        # Priority 3: Group by temporal + cohort
        temporal_groups = self._group_by_temporal_cohort(classified, temporal_window_hours)
        all_groups.extend(temporal_groups)

        # Priority 4: Group by location
        location_groups = self._group_by_location(classified)
        all_groups.extend(location_groups)

        # Filter by minimum group size
        all_groups = [g for g in all_groups if g.total_count >= min_group_size]

        # Sort groups by severity, then by count
        all_groups.sort(key=lambda g: (_get_severity_rank(g.severity), -g.total_count))

        # Collect ungrouped anomalies, deduplicated by device_id (keep worst score per device)
        ungrouped_classified = [c for c in classified if not c.grouped]
        seen_devices: dict[int, ClassifiedAnomaly] = {}
        for c in ungrouped_classified:
            device_id = c.anomaly.device_id
            if device_id not in seen_devices:
                seen_devices[device_id] = c
            else:
                # Keep the one with worse score (more negative = more anomalous)
                if c.anomaly.anomaly_score < seen_devices[device_id].anomaly.anomaly_score:
                    seen_devices[device_id] = c

        ungrouped = [self._create_group_member(c) for c in seen_devices.values()]

        # Calculate impact metrics for hero card
        grouped_count = len(anomalies) - len(ungrouped_classified)
        coverage_percent = (grouped_count / len(anomalies) * 100) if anomalies else 0.0

        # Find top impact group (count * severity weight)
        severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        top_impact_group = None
        top_impact_score = 0
        for group in all_groups:
            impact = group.total_count * severity_weights.get(group.severity, 1)
            if impact > top_impact_score:
                top_impact_score = impact
                top_impact_group = group

        return GroupedAnomaliesResponse(
            groups=all_groups,
            total_anomalies=len(anomalies),
            total_groups=len(all_groups),
            ungrouped_count=len(ungrouped),
            ungrouped_anomalies=ungrouped[:20],  # Limit ungrouped to 20 for response size
            grouping_method="smart_auto",
            computed_at=datetime.now(UTC),
            coverage_percent=round(coverage_percent, 1),
            top_impact_group_id=top_impact_group.group_id if top_impact_group else None,
            top_impact_group_name=top_impact_group.group_name if top_impact_group else None,
        )
