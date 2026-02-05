"""Comparison engine for contextualizing anomaly metrics.

Carl's key insight: "Comparisons matter more than absolutes"

Every metric should be presented with context:
- vs fleet average
- vs cohort (same manufacturer/model/OS)
- vs historical baseline
- vs peer locations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from device_anomaly.database.schema import (
    DeviceFeature,
    LocationMetadata,
)

logger = logging.getLogger(__name__)


class ComparisonType(StrEnum):
    """Types of comparisons available."""

    FLEET = "fleet"
    COHORT = "cohort"
    HISTORICAL = "historical"
    LOCATION = "location"
    PEER = "peer"


@dataclass
class FleetComparison:
    """Result of comparing a device metric to the fleet."""

    metric_name: str
    device_value: float
    fleet_mean: float
    fleet_median: float
    fleet_std: float
    fleet_min: float
    fleet_max: float
    percentile: int  # Device's percentile in fleet (0-100)
    percent_from_mean: float  # How far from mean in percentage
    z_score: float  # Standard deviations from mean
    sample_size: int
    comparison_period_days: int

    @property
    def is_outlier(self) -> bool:
        """Check if device is a statistical outlier (>2 std from mean)."""
        return abs(self.z_score) > 2

    @property
    def direction(self) -> str:
        """Return 'above', 'below', or 'average' based on comparison."""
        if self.z_score > 0.5:
            return "above"
        elif self.z_score < -0.5:
            return "below"
        return "average"

    def to_text(self) -> str:
        """Generate human-readable comparison text."""
        if self.direction == "above":
            return f"{self.percent_from_mean:.0f}% higher than fleet average (top {100 - self.percentile}%)"
        elif self.direction == "below":
            return f"{abs(self.percent_from_mean):.0f}% lower than fleet average (bottom {self.percentile}%)"
        return "in line with fleet average"


@dataclass
class CohortComparison:
    """Result of comparing a device metric to its cohort."""

    metric_name: str
    device_value: float
    cohort_id: str
    cohort_name: str  # Human-readable name (e.g., "Samsung Galaxy A52, Android 13")
    cohort_mean: float
    cohort_median: float
    cohort_std: float
    percentile: int
    percent_from_mean: float
    z_score: float
    cohort_size: int
    comparison_period_days: int

    @property
    def is_cohort_outlier(self) -> bool:
        """Check if device is an outlier within its cohort."""
        return abs(self.z_score) > 2

    @property
    def performs_better_than_cohort(self) -> bool:
        """Check if device performs better than cohort average.

        Note: For metrics where lower is better (drain, crashes), "better"
        means below average.
        """
        return self.z_score < -0.5

    def to_text(self) -> str:
        """Generate human-readable comparison text."""
        if self.z_score > 0.5:
            return f"{self.percent_from_mean:.0f}% higher than other {self.cohort_name} devices"
        elif self.z_score < -0.5:
            return f"{abs(self.percent_from_mean):.0f}% lower than other {self.cohort_name} devices"
        return f"typical for {self.cohort_name} devices"


@dataclass
class HistoricalComparison:
    """Result of comparing a device metric to its historical baseline."""

    metric_name: str
    current_value: float
    baseline_value: float
    baseline_period: str  # e.g., "last 30 days", "same period last month"
    percent_change: float
    trend_direction: str  # improving, stable, worsening
    trend_slope: float  # Rate of change
    is_significant_change: bool
    historical_min: float
    historical_max: float
    data_points: int

    def to_text(self) -> str:
        """Generate human-readable comparison text."""
        if not self.is_significant_change:
            return f"stable compared to {self.baseline_period}"

        if self.trend_direction == "worsening":
            return f"{abs(self.percent_change):.0f}% worse than {self.baseline_period}"
        elif self.trend_direction == "improving":
            return f"{abs(self.percent_change):.0f}% better than {self.baseline_period}"
        return f"stable compared to {self.baseline_period}"


@dataclass
class LocationComparison:
    """Result of comparing two locations on multiple metrics."""

    location_a_id: str
    location_a_name: str
    location_b_id: str
    location_b_name: str
    comparison_period_days: int
    device_count_a: int
    device_count_b: int

    # Metric comparisons
    metric_comparisons: dict[str, tuple[float, float, float]]  # metric -> (a_val, b_val, diff_pct)

    # Overall assessment
    overall_winner: str | None  # location_id of better performer, or None if similar
    key_differences: list[str]  # List of notable differences

    def to_text(self) -> str:
        """Generate human-readable comparison summary."""
        lines = [
            f"Comparing {self.location_a_name} ({self.device_count_a} devices) "
            f"vs {self.location_b_name} ({self.device_count_b} devices):"
        ]

        for metric, (val_a, val_b, diff_pct) in self.metric_comparisons.items():
            if abs(diff_pct) < 5:
                lines.append(f"  • {metric}: Similar ({val_a:.1f} vs {val_b:.1f})")
            else:
                better = self.location_a_name if val_a < val_b else self.location_b_name
                lines.append(f"  • {metric}: {better} is {abs(diff_pct):.0f}% better")

        return "\n".join(lines)


class ComparisonEngine:
    """Engine for computing metric comparisons.

    Provides context for anomaly metrics by comparing to:
    - Fleet average (all devices)
    - Cohort average (same manufacturer/model/OS)
    - Historical baseline (device's own history)
    - Peer locations (other locations with similar characteristics)

    Usage:
        engine = ComparisonEngine(db_session, tenant_id)
        fleet_cmp = engine.compare_to_fleet(device_id, "BatteryDrainPerHour", 8.5)
        cohort_cmp = engine.compare_to_cohort(device_id, "BatteryDrainPerHour", 8.5)
    """

    def __init__(
        self,
        db_session: Session,
        tenant_id: str,
        comparison_days: int = 7,
    ):
        """Initialize the comparison engine.

        Args:
            db_session: SQLAlchemy database session
            tenant_id: Tenant ID for multi-tenant filtering
            comparison_days: Default period for comparisons
        """
        self.db = db_session
        self.tenant_id = tenant_id
        self.comparison_days = comparison_days

        # Cache for fleet statistics (refreshed periodically)
        self._fleet_stats_cache: dict[str, dict[str, float]] = {}
        self._fleet_stats_timestamp: datetime | None = None
        self._cache_ttl_minutes = 60

    def compare_to_fleet(
        self,
        device_id: int,
        metric: str,
        value: float,
        period_days: int | None = None,
    ) -> FleetComparison:
        """Compare a device's metric value to the entire fleet.

        Args:
            device_id: The device ID
            metric: Metric name (e.g., "BatteryDrainPerHour")
            value: The device's current metric value
            period_days: Comparison period (defaults to comparison_days)

        Returns:
            FleetComparison with statistical context
        """
        period_days = period_days or self.comparison_days
        stats = self._get_fleet_statistics(metric, period_days)

        if stats["count"] == 0:
            # No data, return neutral comparison
            return FleetComparison(
                metric_name=metric,
                device_value=value,
                fleet_mean=value,
                fleet_median=value,
                fleet_std=0,
                fleet_min=value,
                fleet_max=value,
                percentile=50,
                percent_from_mean=0,
                z_score=0,
                sample_size=0,
                comparison_period_days=period_days,
            )

        # Calculate statistics
        z_score = (value - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0
        percent_from_mean = (
            ((value - stats["mean"]) / stats["mean"] * 100) if stats["mean"] != 0 else 0
        )

        # Estimate percentile (simplified - could use actual distribution)
        # Using normal distribution approximation
        from scipy import stats as scipy_stats

        try:
            percentile = int(scipy_stats.norm.cdf(z_score) * 100)
        except Exception:
            # Fallback if scipy not available
            percentile = 50 + int(z_score * 15)
            percentile = max(0, min(100, percentile))

        return FleetComparison(
            metric_name=metric,
            device_value=value,
            fleet_mean=stats["mean"],
            fleet_median=stats["median"],
            fleet_std=stats["std"],
            fleet_min=stats["min"],
            fleet_max=stats["max"],
            percentile=percentile,
            percent_from_mean=percent_from_mean,
            z_score=z_score,
            sample_size=stats["count"],
            comparison_period_days=period_days,
        )

    def compare_to_cohort(
        self,
        device_id: int,
        metric: str,
        value: float,
        device_metadata: dict[str, Any] | None = None,
        period_days: int | None = None,
    ) -> CohortComparison | None:
        """Compare a device's metric to its cohort (same manufacturer/model/OS).

        Args:
            device_id: The device ID
            metric: Metric name
            value: The device's current metric value
            device_metadata: Optional dict with Manufacturer, Model, OSVersion
            period_days: Comparison period

        Returns:
            CohortComparison if cohort has enough devices, None otherwise
        """
        period_days = period_days or self.comparison_days

        # Get device metadata if not provided
        if device_metadata is None:
            device_metadata = self._get_device_metadata(device_id)
            if device_metadata is None:
                return None

        manufacturer = device_metadata.get("Manufacturer", "Unknown")
        model = device_metadata.get("Model", "Unknown")
        os_version = device_metadata.get(
            "OSVersion", device_metadata.get("OsVersionName", "Unknown")
        )

        cohort_id = f"{manufacturer}_{model}_{os_version}"
        cohort_name = f"{manufacturer} {model}, {os_version}"

        # Get cohort statistics
        stats = self._get_cohort_statistics(manufacturer, model, os_version, metric, period_days)

        if stats["count"] < 3:
            # Not enough devices in cohort for meaningful comparison
            return None

        z_score = (value - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0
        percent_from_mean = (
            ((value - stats["mean"]) / stats["mean"] * 100) if stats["mean"] != 0 else 0
        )

        from scipy import stats as scipy_stats

        try:
            percentile = int(scipy_stats.norm.cdf(z_score) * 100)
        except Exception:
            percentile = 50 + int(z_score * 15)
            percentile = max(0, min(100, percentile))

        return CohortComparison(
            metric_name=metric,
            device_value=value,
            cohort_id=cohort_id,
            cohort_name=cohort_name,
            cohort_mean=stats["mean"],
            cohort_median=stats["median"],
            cohort_std=stats["std"],
            percentile=percentile,
            percent_from_mean=percent_from_mean,
            z_score=z_score,
            cohort_size=stats["count"],
            comparison_period_days=period_days,
        )

    def compare_to_historical(
        self,
        device_id: int,
        metric: str,
        current_value: float,
        baseline_days: int = 30,
    ) -> HistoricalComparison | None:
        """Compare device's current metric to its historical baseline.

        Args:
            device_id: The device ID
            metric: Metric name
            current_value: Current metric value
            baseline_days: Days to look back for baseline

        Returns:
            HistoricalComparison if historical data exists, None otherwise
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=baseline_days)

        # Query historical values from DeviceFeature
        historical_query = (
            self.db.query(DeviceFeature.feature_values_json)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.device_id == device_id,
                DeviceFeature.computed_at >= start_date,
                DeviceFeature.computed_at <= end_date,
            )
            .order_by(DeviceFeature.computed_at)
            .all()
        )

        if not historical_query:
            return None

        import json

        historical_values = []
        for (json_str,) in historical_query:
            try:
                features = json.loads(json_str) if isinstance(json_str, str) else json_str
                if metric in features:
                    historical_values.append(float(features[metric]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        if len(historical_values) < 3:
            return None

        baseline_value = np.mean(historical_values)
        percent_change = (
            ((current_value - baseline_value) / baseline_value * 100) if baseline_value != 0 else 0
        )

        # Determine trend direction
        # For metrics where lower is better (drain, crashes), increase = worsening
        is_lower_better = metric in {
            "BatteryDrainPerHour",
            "TotalDropCnt",
            "CrashCount",
            "ANRCount",
            "RebootCount",
            "WifiDisconnectCount",
        }

        if abs(percent_change) < 10:
            trend_direction = "stable"
            is_significant = False
        elif (is_lower_better and percent_change > 0) or (
            not is_lower_better and percent_change < 0
        ):
            trend_direction = "worsening"
            is_significant = True
        else:
            trend_direction = "improving"
            is_significant = True

        # Calculate trend slope (simple linear regression)
        if len(historical_values) >= 3:
            x = np.arange(len(historical_values))
            slope, _ = np.polyfit(x, historical_values, 1)
        else:
            slope = 0

        return HistoricalComparison(
            metric_name=metric,
            current_value=current_value,
            baseline_value=baseline_value,
            baseline_period=f"last {baseline_days} days",
            percent_change=percent_change,
            trend_direction=trend_direction,
            trend_slope=slope,
            is_significant_change=is_significant,
            historical_min=min(historical_values),
            historical_max=max(historical_values),
            data_points=len(historical_values),
        )

    def compare_locations(
        self,
        location_a_id: str,
        location_b_id: str,
        metrics: list[str] | None = None,
        period_days: int | None = None,
    ) -> LocationComparison | None:
        """Compare two locations across multiple metrics.

        Args:
            location_a_id: First location ID
            location_b_id: Second location ID
            metrics: List of metrics to compare (defaults to key metrics)
            period_days: Comparison period

        Returns:
            LocationComparison with multi-metric comparison
        """
        period_days = period_days or self.comparison_days
        metrics = metrics or [
            "BatteryDrainPerHour",
            "TotalDropCnt",
            "WifiDisconnectCount",
            "CrashCount",
            "RebootCount",
        ]

        # Get location metadata
        loc_a = (
            self.db.query(LocationMetadata)
            .filter(
                LocationMetadata.tenant_id == self.tenant_id,
                LocationMetadata.location_id == location_a_id,
            )
            .first()
        )

        loc_b = (
            self.db.query(LocationMetadata)
            .filter(
                LocationMetadata.tenant_id == self.tenant_id,
                LocationMetadata.location_id == location_b_id,
            )
            .first()
        )

        if not loc_a or not loc_b:
            return None

        # Get device counts and metrics for each location
        stats_a = self._get_location_statistics(location_a_id, metrics, period_days)
        stats_b = self._get_location_statistics(location_b_id, metrics, period_days)

        if stats_a["device_count"] == 0 or stats_b["device_count"] == 0:
            return None

        # Compare each metric
        metric_comparisons: dict[str, tuple[float, float, float]] = {}
        key_differences: list[str] = []

        wins_a = 0
        wins_b = 0

        for metric in metrics:
            val_a = stats_a.get(metric, 0)
            val_b = stats_b.get(metric, 0)

            # Calculate percentage difference
            if val_b != 0:
                diff_pct = ((val_a - val_b) / val_b) * 100
            elif val_a != 0:
                diff_pct = 100  # A has value, B doesn't
            else:
                diff_pct = 0

            metric_comparisons[metric] = (val_a, val_b, diff_pct)

            # For metrics where lower is better
            if abs(diff_pct) > 10:  # Significant difference
                if val_a < val_b:
                    wins_a += 1
                    key_differences.append(
                        f"{loc_a.location_name} has {abs(diff_pct):.0f}% lower {metric}"
                    )
                else:
                    wins_b += 1
                    key_differences.append(
                        f"{loc_b.location_name} has {abs(diff_pct):.0f}% lower {metric}"
                    )

        # Determine overall winner
        if wins_a > wins_b:
            overall_winner = location_a_id
        elif wins_b > wins_a:
            overall_winner = location_b_id
        else:
            overall_winner = None

        return LocationComparison(
            location_a_id=location_a_id,
            location_a_name=loc_a.location_name,
            location_b_id=location_b_id,
            location_b_name=loc_b.location_name,
            comparison_period_days=period_days,
            device_count_a=stats_a["device_count"],
            device_count_b=stats_b["device_count"],
            metric_comparisons=metric_comparisons,
            overall_winner=overall_winner,
            key_differences=key_differences[:5],  # Top 5 differences
        )

    def get_fleet_percentile(
        self,
        metric: str,
        value: float,
        period_days: int | None = None,
    ) -> int:
        """Get the fleet percentile for a metric value.

        Args:
            metric: Metric name
            value: Metric value
            period_days: Comparison period

        Returns:
            Percentile (0-100) where 100 = best
        """
        comparison = self.compare_to_fleet(0, metric, value, period_days)
        return comparison.percentile

    def get_comparison_context(
        self,
        device_id: int,
        metric: str,
        value: float,
        device_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get full comparison context for a metric.

        Returns fleet, cohort, and historical comparisons in a single call.

        Args:
            device_id: Device ID
            metric: Metric name
            value: Current metric value
            device_metadata: Optional device metadata

        Returns:
            Dict with 'fleet', 'cohort', 'historical' comparison results
        """
        fleet_cmp = self.compare_to_fleet(device_id, metric, value)
        cohort_cmp = self.compare_to_cohort(device_id, metric, value, device_metadata)
        historical_cmp = self.compare_to_historical(device_id, metric, value)

        return {
            "fleet": fleet_cmp,
            "cohort": cohort_cmp,
            "historical": historical_cmp,
            "summary": self._generate_comparison_summary(fleet_cmp, cohort_cmp, historical_cmp),
        }

    def _get_fleet_statistics(
        self,
        metric: str,
        period_days: int,
    ) -> dict[str, float]:
        """Get fleet-wide statistics for a metric (with caching).

        Args:
            metric: Metric name
            period_days: Period for calculation

        Returns:
            Dict with mean, median, std, min, max, count
        """
        cache_key = f"{metric}_{period_days}"
        now = datetime.utcnow()

        # Check cache
        if (
            self._fleet_stats_timestamp
            and (now - self._fleet_stats_timestamp).total_seconds() < self._cache_ttl_minutes * 60
            and cache_key in self._fleet_stats_cache
        ):
            return self._fleet_stats_cache[cache_key]

        # Query fleet statistics from DeviceFeature
        end_date = now
        start_date = end_date - timedelta(days=period_days)

        # Get all feature values for this period
        query = (
            self.db.query(DeviceFeature.feature_values_json)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.computed_at >= start_date,
                DeviceFeature.computed_at <= end_date,
            )
            .all()
        )

        import json

        values = []
        for (json_str,) in query:
            try:
                features = json.loads(json_str) if isinstance(json_str, str) else json_str
                if metric in features:
                    val = float(features[metric])
                    if not np.isnan(val):
                        values.append(val)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        if not values:
            stats = {
                "mean": 0,
                "median": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "count": 0,
            }
        else:
            stats = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

        # Update cache
        self._fleet_stats_cache[cache_key] = stats
        self._fleet_stats_timestamp = now

        return stats

    def _get_cohort_statistics(
        self,
        manufacturer: str,
        model: str,
        os_version: str,
        metric: str,
        period_days: int,
    ) -> dict[str, float]:
        """Get statistics for a device cohort.

        Args:
            manufacturer: Device manufacturer
            model: Device model
            os_version: OS version
            metric: Metric name
            period_days: Period for calculation

        Returns:
            Dict with mean, median, std, count
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        # Query features for devices in this cohort
        # This requires joining with device metadata - simplified for now
        # In production, would query devices by cohort first, then get their features

        # For now, query all and filter by cohort info in feature metadata
        query = (
            self.db.query(DeviceFeature.feature_values_json, DeviceFeature.metadata_json)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.computed_at >= start_date,
                DeviceFeature.computed_at <= end_date,
            )
            .all()
        )

        import json

        values = []
        for features_json, metadata_json in query:
            try:
                metadata = (
                    json.loads(metadata_json)
                    if isinstance(metadata_json, str)
                    else (metadata_json or {})
                )

                # Check if device matches cohort
                if (
                    metadata.get("Manufacturer") == manufacturer
                    and metadata.get("Model") == model
                    and (
                        metadata.get("OSVersion") == os_version
                        or metadata.get("OsVersionName") == os_version
                    )
                ):
                    features = (
                        json.loads(features_json)
                        if isinstance(features_json, str)
                        else features_json
                    )
                    if metric in features:
                        val = float(features[metric])
                        if not np.isnan(val):
                            values.append(val)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        if not values:
            return {
                "mean": 0,
                "median": 0,
                "std": 0,
                "count": 0,
            }

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "count": len(values),
        }

    def _get_location_statistics(
        self,
        location_id: str,
        metrics: list[str],
        period_days: int,
    ) -> dict[str, float]:
        """Get statistics for a location.

        Args:
            location_id: Location ID
            metrics: List of metrics to calculate
            period_days: Period for calculation

        Returns:
            Dict with device_count and mean value for each metric
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        # Query features for devices at this location
        query = (
            self.db.query(DeviceFeature.feature_values_json, DeviceFeature.metadata_json)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.computed_at >= start_date,
                DeviceFeature.computed_at <= end_date,
            )
            .all()
        )

        import json

        device_ids = set()
        metric_values: dict[str, list[float]] = {m: [] for m in metrics}

        for features_json, metadata_json in query:
            try:
                metadata = (
                    json.loads(metadata_json)
                    if isinstance(metadata_json, str)
                    else (metadata_json or {})
                )

                # Check if device is at this location
                if metadata.get("location_id") == location_id:
                    device_ids.add(metadata.get("device_id"))
                    features = (
                        json.loads(features_json)
                        if isinstance(features_json, str)
                        else features_json
                    )

                    for metric in metrics:
                        if metric in features:
                            val = float(features[metric])
                            if not np.isnan(val):
                                metric_values[metric].append(val)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        result: dict[str, float] = {"device_count": len(device_ids)}

        for metric, values in metric_values.items():
            result[metric] = float(np.mean(values)) if values else 0

        return result

    def _get_device_metadata(self, device_id: int) -> dict[str, Any] | None:
        """Get device metadata from the most recent feature record.

        Args:
            device_id: Device ID

        Returns:
            Dict with Manufacturer, Model, OSVersion, or None
        """
        feature = (
            self.db.query(DeviceFeature.metadata_json)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.device_id == device_id,
            )
            .order_by(DeviceFeature.computed_at.desc())
            .first()
        )

        if not feature:
            return None

        import json

        try:
            return json.loads(feature[0]) if isinstance(feature[0], str) else feature[0]
        except (json.JSONDecodeError, TypeError):
            return None

    def _generate_comparison_summary(
        self,
        fleet: FleetComparison,
        cohort: CohortComparison | None,
        historical: HistoricalComparison | None,
    ) -> str:
        """Generate a single-sentence summary of all comparisons.

        Args:
            fleet: Fleet comparison result
            cohort: Optional cohort comparison result
            historical: Optional historical comparison result

        Returns:
            Human-readable summary string
        """
        parts = []

        # Fleet context
        parts.append(fleet.to_text())

        # Cohort context if available and different from fleet
        if cohort and abs(cohort.z_score - fleet.z_score) > 0.5:
            parts.append(cohort.to_text())

        # Historical context if available and significant
        if historical and historical.is_significant_change:
            parts.append(historical.to_text())

        return "; ".join(parts) if parts else "No comparison data available"
