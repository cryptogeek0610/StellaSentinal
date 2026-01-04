"""Entity aggregation for insights.

Aggregates device-level anomalies and metrics up the entity hierarchy:
Device -> User -> Location -> Region

Enables Carl's requirement: "Relate any anomalies to location (warehouse 1 vs warehouse 2)"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from device_anomaly.database.schema import AggregatedInsight, LocationMetadata
from device_anomaly.insights.categories import EntityType, InsightCategory, InsightSeverity
from device_anomaly.insights.classifier import ClassifiedInsight
from device_anomaly.insights.location_mapper import LocationMapper

logger = logging.getLogger(__name__)


@dataclass
class EntityMetrics:
    """Aggregated metrics for an entity."""

    entity_type: EntityType
    entity_id: str
    entity_name: str

    # Device counts
    total_devices: int = 0
    devices_with_anomalies: int = 0
    anomaly_rate: float = 0.0

    # Insight counts by category
    insight_counts: Dict[InsightCategory, int] = field(default_factory=dict)

    # Insight counts by severity
    severity_counts: Dict[InsightSeverity, int] = field(default_factory=dict)

    # Key metrics (aggregated from devices)
    avg_battery_drain_per_hour: float = 0.0
    avg_disconnect_rate: float = 0.0
    avg_drop_rate: float = 0.0
    avg_crash_rate: float = 0.0
    total_drops: int = 0
    total_reboots: int = 0
    total_crashes: int = 0

    # Comparison data
    vs_fleet_percentile: Optional[int] = None
    vs_baseline_percent: Optional[float] = None


@dataclass
class LocationInsight:
    """Insight aggregated at the location level."""

    location_id: str
    location_name: str
    parent_region: Optional[str]

    # Summary metrics
    total_devices: int
    devices_with_issues: int
    issue_rate: float

    # Top issues at this location
    top_issues: List[Tuple[InsightCategory, int]]  # (category, count)

    # Comparison to other locations
    rank_among_locations: int
    total_locations: int
    better_than_percent: float

    # Trend
    trend_direction: str  # improving, stable, worsening
    trend_change_percent: float


@dataclass
class UserInsight:
    """Insight aggregated at the user level."""

    user_id: str
    user_name: Optional[str]
    location_id: Optional[str]

    # Devices
    device_count: int
    devices_with_issues: int

    # Key patterns
    total_drops: int
    total_reboots: int
    drop_rate_per_device: float
    reboot_rate_per_device: float

    # Comparison
    vs_peer_percent: float  # How much better/worse than peers


@dataclass
class CohortInsight:
    """Insight aggregated at the device cohort level."""

    cohort_id: str  # e.g., "Samsung_SM-G991B_13_R10"
    manufacturer: str
    model: str
    os_version: str
    firmware_version: Optional[str]

    # Devices
    device_count: int
    devices_with_issues: int
    issue_rate: float

    # Performance metrics
    avg_battery_drain: float
    avg_crash_rate: float
    avg_disconnect_rate: float

    # Comparison to other cohorts
    vs_fleet_percent: float
    is_problem_cohort: bool


class EntityAggregator:
    """Aggregates anomalies and metrics up the entity hierarchy.

    Usage:
        aggregator = EntityAggregator(db_session, tenant_id, location_mapper)
        location_insights = aggregator.aggregate_by_location(anomalies, period)
        user_insights = aggregator.aggregate_by_user(anomalies, device_user_map)
    """

    def __init__(
        self,
        db_session: Session,
        tenant_id: str,
        location_mapper: Optional[LocationMapper] = None,
    ):
        """Initialize the aggregator.

        Args:
            db_session: SQLAlchemy database session
            tenant_id: Tenant ID for multi-tenant filtering
            location_mapper: Optional LocationMapper instance
        """
        self.db = db_session
        self.tenant_id = tenant_id
        self.location_mapper = location_mapper or LocationMapper(db_session, tenant_id)

    def aggregate_by_location(
        self,
        insights: List[ClassifiedInsight],
        devices_df: pd.DataFrame,
        period_days: int = 7,
    ) -> Dict[str, LocationInsight]:
        """Aggregate insights by location.

        Args:
            insights: List of classified insights from devices
            devices_df: DataFrame with device data (must include device_id)
            period_days: Period for metrics calculation

        Returns:
            Dict mapping location_id -> LocationInsight
        """
        # Ensure devices have location mapping
        if "location_id" not in devices_df.columns:
            devices_df = self.location_mapper.bulk_map_devices(devices_df)

        # Get all locations
        locations = self.location_mapper.get_all_locations()
        location_insights: Dict[str, LocationInsight] = {}

        # Group devices by location
        for location in locations:
            loc_devices = devices_df[devices_df["location_id"] == location.location_id]
            if loc_devices.empty:
                continue

            device_ids = set(loc_devices["device_id"].tolist()) if "device_id" in loc_devices.columns else set()

            # Filter insights for this location's devices
            loc_insights = [
                i for i in insights
                if i.affected_entity_id in device_ids or str(i.metadata.get("device_id")) in map(str, device_ids)
            ]

            # Count issues by category
            category_counts: Dict[InsightCategory, int] = {}
            devices_with_issues = set()

            for insight in loc_insights:
                category_counts[insight.category] = category_counts.get(insight.category, 0) + 1
                if insight.affected_entity_id:
                    devices_with_issues.add(insight.affected_entity_id)

            # Top issues
            top_issues = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            total_devices = len(device_ids)
            issue_count = len(devices_with_issues)

            location_insights[location.location_id] = LocationInsight(
                location_id=location.location_id,
                location_name=location.location_name,
                parent_region=location.parent_region,
                total_devices=total_devices,
                devices_with_issues=issue_count,
                issue_rate=issue_count / total_devices if total_devices > 0 else 0,
                top_issues=top_issues,
                rank_among_locations=0,  # Will be calculated below
                total_locations=len(locations),
                better_than_percent=0,  # Will be calculated below
                trend_direction="stable",
                trend_change_percent=0,
            )

        # Calculate rankings
        sorted_locations = sorted(
            location_insights.values(),
            key=lambda x: x.issue_rate,
        )

        for rank, loc_insight in enumerate(sorted_locations, 1):
            loc_insight.rank_among_locations = rank
            loc_insight.better_than_percent = (len(sorted_locations) - rank) / len(sorted_locations) * 100 if sorted_locations else 0

        return location_insights

    def aggregate_by_user(
        self,
        insights: List[ClassifiedInsight],
        device_user_map: Dict[int, str],
        user_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, UserInsight]:
        """Aggregate insights by user.

        Args:
            insights: List of classified insights from devices
            device_user_map: Dict mapping device_id -> user_id
            user_names: Optional dict mapping user_id -> user_name

        Returns:
            Dict mapping user_id -> UserInsight
        """
        user_names = user_names or {}
        user_insights: Dict[str, UserInsight] = {}

        # Group insights by user
        user_device_insights: Dict[str, List[ClassifiedInsight]] = {}
        user_devices: Dict[str, set] = {}

        for insight in insights:
            device_id = insight.metadata.get("device_id") or insight.affected_entity_id
            if device_id is None:
                continue

            try:
                device_id_int = int(device_id)
            except (ValueError, TypeError):
                continue

            user_id = device_user_map.get(device_id_int)
            if user_id is None:
                continue

            if user_id not in user_device_insights:
                user_device_insights[user_id] = []
                user_devices[user_id] = set()

            user_device_insights[user_id].append(insight)
            user_devices[user_id].add(device_id_int)

        # Calculate user-level metrics
        for user_id, user_insight_list in user_device_insights.items():
            device_ids = user_devices[user_id]
            devices_with_issues = set()

            total_drops = 0
            total_reboots = 0

            for insight in user_insight_list:
                devices_with_issues.add(insight.affected_entity_id)

                if insight.category == InsightCategory.EXCESSIVE_DROPS:
                    total_drops += int(insight.primary_value or 0)
                elif insight.category == InsightCategory.EXCESSIVE_REBOOTS:
                    total_reboots += int(insight.primary_value or 0)

            device_count = len(device_ids)
            user_insights[user_id] = UserInsight(
                user_id=user_id,
                user_name=user_names.get(user_id),
                location_id=None,  # Would need device->location mapping
                device_count=device_count,
                devices_with_issues=len(devices_with_issues),
                total_drops=total_drops,
                total_reboots=total_reboots,
                drop_rate_per_device=total_drops / device_count if device_count > 0 else 0,
                reboot_rate_per_device=total_reboots / device_count if device_count > 0 else 0,
                vs_peer_percent=0,  # Will be calculated below
            )

        # Calculate vs peer comparison
        if user_insights:
            avg_drop_rate = sum(u.drop_rate_per_device for u in user_insights.values()) / len(user_insights)
            for user_insight in user_insights.values():
                if avg_drop_rate > 0:
                    user_insight.vs_peer_percent = (
                        (user_insight.drop_rate_per_device - avg_drop_rate) / avg_drop_rate * 100
                    )

        return user_insights

    def aggregate_by_cohort(
        self,
        insights: List[ClassifiedInsight],
        device_metadata: pd.DataFrame,
    ) -> Dict[str, CohortInsight]:
        """Aggregate insights by device cohort (manufacturer+model+OS+firmware).

        Args:
            insights: List of classified insights from devices
            device_metadata: DataFrame with device metadata including:
                - DeviceId, Manufacturer, Model, OSVersion, FirmwareVersion

        Returns:
            Dict mapping cohort_id -> CohortInsight
        """
        cohort_insights: Dict[str, CohortInsight] = {}

        # Build cohort ID for each device
        def get_cohort_id(row: pd.Series) -> str:
            manufacturer = str(row.get("Manufacturer", "Unknown"))
            model = str(row.get("Model", "Unknown"))
            os_version = str(row.get("OSVersion", row.get("OsVersionName", "Unknown")))
            firmware = str(row.get("FirmwareVersion", row.get("OEMVersion", "")))
            return f"{manufacturer}_{model}_{os_version}_{firmware}"

        if device_metadata.empty:
            return cohort_insights

        device_metadata = device_metadata.copy()
        device_metadata["cohort_id"] = device_metadata.apply(get_cohort_id, axis=1)

        # Build device_id -> cohort_id mapping
        device_id_col = "DeviceId" if "DeviceId" in device_metadata.columns else "device_id"
        device_cohort_map = dict(zip(
            device_metadata[device_id_col].astype(str),
            device_metadata["cohort_id"]
        ))

        # Group insights by cohort
        cohort_device_insights: Dict[str, List[ClassifiedInsight]] = {}
        cohort_devices: Dict[str, set] = {}

        for insight in insights:
            device_id = str(insight.metadata.get("device_id") or insight.affected_entity_id or "")
            cohort_id = device_cohort_map.get(device_id)

            if cohort_id is None:
                continue

            if cohort_id not in cohort_device_insights:
                cohort_device_insights[cohort_id] = []
                cohort_devices[cohort_id] = set()

            cohort_device_insights[cohort_id].append(insight)
            cohort_devices[cohort_id].add(device_id)

        # Get cohort metadata
        cohort_metadata = device_metadata.drop_duplicates(subset=["cohort_id"]).set_index("cohort_id")

        # Calculate cohort-level metrics
        for cohort_id, cohort_insight_list in cohort_device_insights.items():
            device_ids = cohort_devices[cohort_id]
            devices_with_issues = set()

            for insight in cohort_insight_list:
                devices_with_issues.add(insight.affected_entity_id)

            device_count = len(device_ids)
            issue_rate = len(devices_with_issues) / device_count if device_count > 0 else 0

            # Get metadata for this cohort
            if cohort_id in cohort_metadata.index:
                row = cohort_metadata.loc[cohort_id]
                manufacturer = str(row.get("Manufacturer", "Unknown"))
                model = str(row.get("Model", "Unknown"))
                os_version = str(row.get("OSVersion", row.get("OsVersionName", "Unknown")))
                firmware = str(row.get("FirmwareVersion", row.get("OEMVersion", "")))
            else:
                parts = cohort_id.split("_")
                manufacturer = parts[0] if len(parts) > 0 else "Unknown"
                model = parts[1] if len(parts) > 1 else "Unknown"
                os_version = parts[2] if len(parts) > 2 else "Unknown"
                firmware = parts[3] if len(parts) > 3 else ""

            cohort_insights[cohort_id] = CohortInsight(
                cohort_id=cohort_id,
                manufacturer=manufacturer,
                model=model,
                os_version=os_version,
                firmware_version=firmware or None,
                device_count=device_count,
                devices_with_issues=len(devices_with_issues),
                issue_rate=issue_rate,
                avg_battery_drain=0,  # Would need to aggregate from metrics
                avg_crash_rate=0,
                avg_disconnect_rate=0,
                vs_fleet_percent=0,  # Will be calculated below
                is_problem_cohort=False,
            )

        # Calculate fleet comparison and identify problem cohorts
        if cohort_insights:
            avg_issue_rate = sum(c.issue_rate for c in cohort_insights.values()) / len(cohort_insights)
            for cohort_insight in cohort_insights.values():
                if avg_issue_rate > 0:
                    cohort_insight.vs_fleet_percent = (
                        (cohort_insight.issue_rate - avg_issue_rate) / avg_issue_rate * 100
                    )
                cohort_insight.is_problem_cohort = cohort_insight.issue_rate > avg_issue_rate * 1.5

        return cohort_insights

    def rank_entities(
        self,
        entity_metrics: Dict[str, EntityMetrics],
        metric: str = "anomaly_rate",
        order: str = "worst",
        limit: int = 10,
    ) -> List[Tuple[str, EntityMetrics]]:
        """Rank entities by a specific metric.

        Args:
            entity_metrics: Dict of entity_id -> EntityMetrics
            metric: Metric to rank by (anomaly_rate, total_drops, avg_battery_drain, etc.)
            order: "worst" for descending, "best" for ascending
            limit: Maximum number of results

        Returns:
            List of (entity_id, EntityMetrics) tuples, sorted by metric
        """
        items = list(entity_metrics.items())

        def get_metric_value(item: Tuple[str, EntityMetrics]) -> float:
            metrics = item[1]
            return getattr(metrics, metric, 0)

        reverse = order == "worst"
        items.sort(key=get_metric_value, reverse=reverse)

        return items[:limit]

    def save_aggregated_insights(
        self,
        location_insights: Dict[str, LocationInsight],
        computed_at: Optional[datetime] = None,
    ) -> int:
        """Save location-level insights to the database.

        Args:
            location_insights: Dict of location_id -> LocationInsight
            computed_at: Timestamp for when insights were computed

        Returns:
            Number of insights saved
        """
        computed_at = computed_at or datetime.utcnow()
        saved_count = 0

        for location_id, insight in location_insights.items():
            if insight.issue_rate == 0:
                continue  # Skip locations with no issues

            # Determine severity based on issue rate
            if insight.issue_rate > 0.3:
                severity = InsightSeverity.CRITICAL
            elif insight.issue_rate > 0.15:
                severity = InsightSeverity.HIGH
            elif insight.issue_rate > 0.05:
                severity = InsightSeverity.MEDIUM
            else:
                severity = InsightSeverity.LOW

            # Build headline
            top_issue_name = insight.top_issues[0][0].value if insight.top_issues else "issues"
            headline = (
                f"{insight.location_name} has {insight.devices_with_issues} devices "
                f"with {top_issue_name} ({insight.issue_rate:.0%} of devices)"
            )

            # Check if insight already exists
            existing = (
                self.db.query(AggregatedInsight)
                .filter(
                    AggregatedInsight.tenant_id == self.tenant_id,
                    AggregatedInsight.entity_type == EntityType.LOCATION.value,
                    AggregatedInsight.entity_id == location_id,
                    AggregatedInsight.is_active == True,  # noqa: E712
                )
                .first()
            )

            import json

            insight_data = {
                "total_devices": insight.total_devices,
                "devices_with_issues": insight.devices_with_issues,
                "issue_rate": insight.issue_rate,
                "top_issues": [(cat.value, count) for cat, count in insight.top_issues],
                "rank": insight.rank_among_locations,
                "total_locations": insight.total_locations,
            }

            if existing:
                # Update existing
                existing.headline = headline
                existing.severity = severity.value
                existing.insight_data_json = json.dumps(insight_data)
                existing.affected_device_count = insight.devices_with_issues
                existing.trend_direction = insight.trend_direction
                existing.current_value = insight.issue_rate
                existing.computed_at = computed_at
            else:
                # Create new
                new_insight = AggregatedInsight(
                    tenant_id=self.tenant_id,
                    entity_type=EntityType.LOCATION.value,
                    entity_id=location_id,
                    entity_name=insight.location_name,
                    insight_category=InsightCategory.LOCATION_ANOMALY_CLUSTER.value,
                    severity=severity.value,
                    headline=headline,
                    insight_data_json=json.dumps(insight_data),
                    affected_device_count=insight.devices_with_issues,
                    trend_direction=insight.trend_direction,
                    current_value=insight.issue_rate,
                    computed_at=computed_at,
                    is_active=True,
                )
                self.db.add(new_insight)

            saved_count += 1

        self.db.commit()
        return saved_count
