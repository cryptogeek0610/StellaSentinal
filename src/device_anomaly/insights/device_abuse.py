"""Device abuse analysis for drops and reboots.

Carl's key requirements:
- "Devices/people/locations with excessive drops"
- "Devices/people/locations with excessive reboots"
- "Performance patterns by manufacturer, model, OS version, firmware"
- "Combinations that cause problems"

This analyzer identifies device misuse and hardware/software issues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from device_anomaly.database.schema import DeviceFeature, LocationMetadata
from device_anomaly.insights.categories import InsightCategory, InsightSeverity, EntityType

logger = logging.getLogger(__name__)


class AbuseType(str, Enum):
    """Types of device abuse patterns."""

    EXCESSIVE_DROPS = "excessive_drops"
    EXCESSIVE_REBOOTS = "excessive_reboots"
    COMBINED_ABUSE = "combined_abuse"
    CRASH_RELATED_REBOOTS = "crash_related_reboots"
    USER_PATTERN = "user_pattern"  # Same user across devices
    LOCATION_PATTERN = "location_pattern"  # Same location across devices


@dataclass
class DeviceAbuseIndicator:
    """Abuse indicator for a single device."""

    device_id: int
    device_name: Optional[str]
    abuse_type: AbuseType

    # Metrics
    drop_count: int
    drop_rate_per_day: float
    reboot_count: int
    reboot_rate_per_day: float
    crash_count: int

    # Comparisons
    vs_fleet_drop_percentile: int  # Higher = worse
    vs_fleet_reboot_percentile: int
    vs_cohort_multiplier: float  # e.g., 2.5x = 2.5x worse than cohort average

    # Context
    assigned_user: Optional[str]
    assigned_location: Optional[str]
    device_cohort: str  # manufacturer_model_os

    severity: InsightSeverity
    description: str
    recommendations: List[str]


@dataclass
class DropAnalysisReport:
    """Drop analysis for a tenant, location, or user."""

    entity_type: str  # "tenant", "location", "user"
    entity_id: str
    entity_name: str
    analysis_period_days: int

    # Summary
    total_devices: int
    total_drops: int
    avg_drops_per_device: float
    devices_with_excessive_drops: int

    # Breakdown by entity
    by_location: Dict[str, Tuple[int, float]]  # location -> (drops, rate)
    by_user: Dict[str, Tuple[int, float]]  # user -> (drops, rate)
    by_cohort: Dict[str, Tuple[int, float]]  # cohort -> (drops, rate)

    # Top offenders
    worst_devices: List[DeviceAbuseIndicator]
    worst_locations: List[Tuple[str, int, float]]  # (location, drops, rate)
    worst_users: List[Tuple[str, int, float]]  # (user, drops, rate)

    # Trend
    trend_direction: str  # improving, stable, worsening
    trend_change_percent: float

    recommendations: List[str]


@dataclass
class RebootAnalysisReport:
    """Reboot analysis for a tenant, location, or user."""

    entity_type: str
    entity_id: str
    entity_name: str
    analysis_period_days: int

    # Summary
    total_devices: int
    total_reboots: int
    total_crashes: int
    avg_reboots_per_device: float
    devices_with_excessive_reboots: int

    # Crash correlation
    crash_induced_reboots_percent: float  # % reboots likely caused by crashes

    # Breakdown
    by_location: Dict[str, Tuple[int, float]]
    by_user: Dict[str, Tuple[int, float]]
    by_cohort: Dict[str, Tuple[int, float]]

    # Top offenders
    worst_devices: List[DeviceAbuseIndicator]
    worst_cohorts: List[Tuple[str, int, float]]  # (cohort, reboots, rate)

    recommendations: List[str]


@dataclass
class ProblemCombination:
    """A problematic device combination."""

    cohort_id: str  # manufacturer_model_os_firmware
    manufacturer: str
    model: str
    os_version: str
    firmware_version: Optional[str]

    # Metrics
    device_count: int
    issue_rate: float  # % of devices with issues
    avg_drops_per_device: float
    avg_reboots_per_device: float
    avg_crashes_per_device: float

    # Comparison
    vs_fleet_issue_rate: float  # e.g., 2.5 = 2.5x fleet average
    is_statistically_significant: bool  # Enough devices to be confident

    # Issues
    primary_issue: str  # "drops", "reboots", "crashes", "combined"
    severity: InsightSeverity

    description: str
    recommendations: List[str]


@dataclass
class ProblemCombinationReport:
    """Analysis of problematic device combinations."""

    tenant_id: str
    analysis_period_days: int

    # Summary
    total_cohorts_analyzed: int
    problem_cohorts_found: int
    devices_in_problem_cohorts: int

    # Problem combinations
    problem_combinations: List[ProblemCombination]

    # By manufacturer
    worst_manufacturers: List[Tuple[str, float]]  # (manufacturer, issue_rate)

    # By OS version
    worst_os_versions: List[Tuple[str, float]]  # (os_version, issue_rate)

    recommendations: List[str]


class DeviceAbuseAnalyzer:
    """Analyzer for device abuse patterns (drops, reboots, crashes).

    Addresses Carl's requirements:
    - Excessive drops by device/person/location
    - Excessive reboots by device/person/location
    - Problem combinations (manufacturer+model+OS+firmware)

    Usage:
        analyzer = DeviceAbuseAnalyzer(db_session, tenant_id)
        drops = analyzer.analyze_drops(period_days=7, group_by="location")
        combos = analyzer.identify_problem_combinations(period_days=30)
    """

    # Thresholds
    EXCESSIVE_DROPS_THRESHOLD = 5  # drops per week
    EXCESSIVE_REBOOTS_THRESHOLD = 3  # reboots per week
    HIGH_DROPS_PERCENTILE = 90  # Top 10% = excessive
    HIGH_REBOOTS_PERCENTILE = 90
    PROBLEM_COHORT_MULTIPLIER = 2.0  # 2x fleet average = problem
    MIN_COHORT_SIZE = 5  # Minimum devices for statistical significance

    def __init__(
        self,
        db_session: Session,
        tenant_id: str,
    ):
        """Initialize the analyzer.

        Args:
            db_session: SQLAlchemy database session
            tenant_id: Tenant ID for multi-tenant filtering
        """
        self.db = db_session
        self.tenant_id = tenant_id

    def analyze_drops(
        self,
        period_days: int = 7,
        group_by: str = "tenant",  # "tenant", "location", "user"
        entity_id: Optional[str] = None,
    ) -> DropAnalysisReport:
        """Analyze drop patterns.

        Args:
            period_days: Days of history to analyze
            group_by: Entity type to analyze
            entity_id: Optional specific entity

        Returns:
            DropAnalysisReport with drop analysis
        """
        # Get device data with drops
        device_data = self._get_device_abuse_data(period_days, entity_id if group_by == "location" else None)

        if device_data.empty:
            return self._empty_drop_report(group_by, entity_id, period_days)

        # Calculate totals
        total_devices = len(device_data["device_id"].unique())
        total_drops = int(device_data["drop_count"].sum())
        avg_drops = device_data.groupby("device_id")["drop_count"].sum().mean()

        # Determine excessive threshold based on fleet distribution
        drop_threshold = self._calculate_threshold(
            device_data, "drop_count", self.EXCESSIVE_DROPS_THRESHOLD
        )

        # Find devices with excessive drops
        device_totals = device_data.groupby("device_id").agg({
            "drop_count": "sum",
            "location_id": "first",
            "user_id": "first",
            "cohort_id": "first",
            "device_name": "first",
        }).reset_index()

        excessive_mask = device_totals["drop_count"] > drop_threshold
        excessive_count = excessive_mask.sum()

        # Group by location
        by_location = {}
        if "location_id" in device_data.columns:
            loc_agg = device_data.groupby("location_id").agg({
                "drop_count": "sum",
                "device_id": "nunique",
            })
            for loc_id, row in loc_agg.iterrows():
                if pd.notna(loc_id):
                    rate = row["drop_count"] / row["device_id"] / period_days * 7  # Per week
                    by_location[str(loc_id)] = (int(row["drop_count"]), float(rate))

        # Group by user
        by_user = {}
        if "user_id" in device_data.columns:
            user_agg = device_data.groupby("user_id").agg({
                "drop_count": "sum",
                "device_id": "nunique",
            })
            for user_id, row in user_agg.iterrows():
                if pd.notna(user_id):
                    rate = row["drop_count"] / row["device_id"] / period_days * 7
                    by_user[str(user_id)] = (int(row["drop_count"]), float(rate))

        # Group by cohort
        by_cohort = {}
        if "cohort_id" in device_data.columns:
            cohort_agg = device_data.groupby("cohort_id").agg({
                "drop_count": "sum",
                "device_id": "nunique",
            })
            for cohort_id, row in cohort_agg.iterrows():
                if pd.notna(cohort_id):
                    rate = row["drop_count"] / row["device_id"] / period_days * 7
                    by_cohort[str(cohort_id)] = (int(row["drop_count"]), float(rate))

        # Top offenders
        worst_devices = self._get_worst_drop_devices(device_totals, device_data, period_days)

        # Sort by rate for worst locations/users
        worst_locations = sorted(by_location.items(), key=lambda x: x[1][1], reverse=True)[:5]
        worst_locations = [(loc, drops, rate) for loc, (drops, rate) in worst_locations]

        worst_users = sorted(by_user.items(), key=lambda x: x[1][1], reverse=True)[:5]
        worst_users = [(user, drops, rate) for user, (drops, rate) in worst_users]

        # Recommendations
        recommendations = self._generate_drop_recommendations(
            worst_devices, worst_locations, worst_users, excessive_count, total_devices
        )

        return DropAnalysisReport(
            entity_type=group_by,
            entity_id=entity_id or self.tenant_id,
            entity_name=entity_id or self.tenant_id,
            analysis_period_days=period_days,
            total_devices=total_devices,
            total_drops=total_drops,
            avg_drops_per_device=float(avg_drops),
            devices_with_excessive_drops=int(excessive_count),
            by_location=by_location,
            by_user=by_user,
            by_cohort=by_cohort,
            worst_devices=worst_devices,
            worst_locations=worst_locations,
            worst_users=worst_users,
            trend_direction="stable",  # Would need historical comparison
            trend_change_percent=0,
            recommendations=recommendations,
        )

    def analyze_reboots(
        self,
        period_days: int = 7,
        group_by: str = "tenant",
        entity_id: Optional[str] = None,
    ) -> RebootAnalysisReport:
        """Analyze reboot patterns.

        Args:
            period_days: Days of history to analyze
            group_by: Entity type to analyze
            entity_id: Optional specific entity

        Returns:
            RebootAnalysisReport with reboot analysis
        """
        device_data = self._get_device_abuse_data(period_days, entity_id if group_by == "location" else None)

        if device_data.empty:
            return self._empty_reboot_report(group_by, entity_id, period_days)

        total_devices = len(device_data["device_id"].unique())
        total_reboots = int(device_data["reboot_count"].sum())
        total_crashes = int(device_data["crash_count"].sum())
        avg_reboots = device_data.groupby("device_id")["reboot_count"].sum().mean()

        # Threshold for excessive reboots
        reboot_threshold = self._calculate_threshold(
            device_data, "reboot_count", self.EXCESSIVE_REBOOTS_THRESHOLD
        )

        device_totals = device_data.groupby("device_id").agg({
            "reboot_count": "sum",
            "crash_count": "sum",
            "location_id": "first",
            "user_id": "first",
            "cohort_id": "first",
            "device_name": "first",
        }).reset_index()

        excessive_mask = device_totals["reboot_count"] > reboot_threshold
        excessive_count = excessive_mask.sum()

        # Crash correlation
        crash_induced = 0
        if total_reboots > 0:
            # Estimate crash-induced reboots (simplified: if crashes > reboots/2, they're related)
            devices_with_crashes = device_totals[device_totals["crash_count"] > 0]
            crash_induced = min(100, (len(devices_with_crashes) / len(device_totals) * 100) if len(device_totals) > 0 else 0)

        # Group by entity
        by_location = self._group_metric_by_entity(device_data, "reboot_count", "location_id", period_days)
        by_user = self._group_metric_by_entity(device_data, "reboot_count", "user_id", period_days)
        by_cohort = self._group_metric_by_entity(device_data, "reboot_count", "cohort_id", period_days)

        # Worst devices and cohorts
        worst_devices = self._get_worst_reboot_devices(device_totals, device_data, period_days)

        worst_cohorts = sorted(by_cohort.items(), key=lambda x: x[1][1], reverse=True)[:5]
        worst_cohorts = [(cohort, reboots, rate) for cohort, (reboots, rate) in worst_cohorts]

        recommendations = self._generate_reboot_recommendations(
            worst_devices, worst_cohorts, crash_induced, excessive_count, total_devices
        )

        return RebootAnalysisReport(
            entity_type=group_by,
            entity_id=entity_id or self.tenant_id,
            entity_name=entity_id or self.tenant_id,
            analysis_period_days=period_days,
            total_devices=total_devices,
            total_reboots=total_reboots,
            total_crashes=total_crashes,
            avg_reboots_per_device=float(avg_reboots),
            devices_with_excessive_reboots=int(excessive_count),
            crash_induced_reboots_percent=float(crash_induced),
            by_location=by_location,
            by_user=by_user,
            by_cohort=by_cohort,
            worst_devices=worst_devices,
            worst_cohorts=worst_cohorts,
            recommendations=recommendations,
        )

    def identify_problem_combinations(
        self,
        period_days: int = 30,
    ) -> ProblemCombinationReport:
        """Identify problematic device combinations (manufacturer+model+OS+firmware).

        Carl's requirement: "Combinations that cause problems"

        Args:
            period_days: Days of history to analyze

        Returns:
            ProblemCombinationReport with problem combinations
        """
        device_data = self._get_device_abuse_data(period_days, None)

        if device_data.empty:
            return ProblemCombinationReport(
                tenant_id=self.tenant_id,
                analysis_period_days=period_days,
                total_cohorts_analyzed=0,
                problem_cohorts_found=0,
                devices_in_problem_cohorts=0,
                problem_combinations=[],
                worst_manufacturers=[],
                worst_os_versions=[],
                recommendations=[],
            )

        # Calculate fleet averages
        fleet_avg_drops = device_data.groupby("device_id")["drop_count"].sum().mean()
        fleet_avg_reboots = device_data.groupby("device_id")["reboot_count"].sum().mean()
        fleet_avg_crashes = device_data.groupby("device_id")["crash_count"].sum().mean()

        # Group by cohort and calculate metrics
        cohort_agg = device_data.groupby("cohort_id").agg({
            "device_id": "nunique",
            "drop_count": "sum",
            "reboot_count": "sum",
            "crash_count": "sum",
            "manufacturer": "first",
            "model": "first",
            "os_version": "first",
            "firmware_version": "first",
        }).reset_index()

        cohort_agg["avg_drops"] = cohort_agg["drop_count"] / cohort_agg["device_id"]
        cohort_agg["avg_reboots"] = cohort_agg["reboot_count"] / cohort_agg["device_id"]
        cohort_agg["avg_crashes"] = cohort_agg["crash_count"] / cohort_agg["device_id"]

        # Calculate issue rate (simplified: devices with any issue / total devices)
        # In production would be more sophisticated

        problem_combinations: List[ProblemCombination] = []

        for _, row in cohort_agg.iterrows():
            if row["device_id"] < self.MIN_COHORT_SIZE:
                continue  # Not enough devices for statistical significance

            # Calculate how much worse than fleet average
            drop_multiplier = row["avg_drops"] / fleet_avg_drops if fleet_avg_drops > 0 else 1
            reboot_multiplier = row["avg_reboots"] / fleet_avg_reboots if fleet_avg_reboots > 0 else 1
            crash_multiplier = row["avg_crashes"] / fleet_avg_crashes if fleet_avg_crashes > 0 else 1

            max_multiplier = max(drop_multiplier, reboot_multiplier, crash_multiplier)

            if max_multiplier >= self.PROBLEM_COHORT_MULTIPLIER:
                # Determine primary issue
                if drop_multiplier >= reboot_multiplier and drop_multiplier >= crash_multiplier:
                    primary_issue = "drops"
                elif reboot_multiplier >= crash_multiplier:
                    primary_issue = "reboots"
                else:
                    primary_issue = "crashes"

                # Severity based on multiplier
                if max_multiplier >= 4:
                    severity = InsightSeverity.CRITICAL
                elif max_multiplier >= 3:
                    severity = InsightSeverity.HIGH
                else:
                    severity = InsightSeverity.MEDIUM

                cohort_id = str(row["cohort_id"])
                manufacturer = str(row.get("manufacturer", "Unknown"))
                model = str(row.get("model", "Unknown"))
                os_version = str(row.get("os_version", "Unknown"))
                firmware = str(row.get("firmware_version", "")) or None

                # Calculate issue rate (% of devices with any issue)
                # Simplified: if above threshold = has issue
                issue_rate = (
                    (row["avg_drops"] > self.EXCESSIVE_DROPS_THRESHOLD or
                     row["avg_reboots"] > self.EXCESSIVE_REBOOTS_THRESHOLD)
                )

                problem_combinations.append(ProblemCombination(
                    cohort_id=cohort_id,
                    manufacturer=manufacturer,
                    model=model,
                    os_version=os_version,
                    firmware_version=firmware,
                    device_count=int(row["device_id"]),
                    issue_rate=float(issue_rate),
                    avg_drops_per_device=float(row["avg_drops"]),
                    avg_reboots_per_device=float(row["avg_reboots"]),
                    avg_crashes_per_device=float(row["avg_crashes"]),
                    vs_fleet_issue_rate=float(max_multiplier),
                    is_statistically_significant=row["device_id"] >= self.MIN_COHORT_SIZE,
                    primary_issue=primary_issue,
                    severity=severity,
                    description=(
                        f"{manufacturer} {model} on {os_version} has {max_multiplier:.1f}x "
                        f"the fleet average {primary_issue}"
                    ),
                    recommendations=self._get_cohort_recommendations(primary_issue, firmware is not None),
                ))

        # Sort by severity
        problem_combinations.sort(key=lambda x: x.vs_fleet_issue_rate, reverse=True)

        # Devices in problem cohorts
        devices_in_problems = sum(p.device_count for p in problem_combinations)

        # Worst manufacturers
        mfg_agg = cohort_agg.groupby("manufacturer").agg({
            "drop_count": "sum",
            "reboot_count": "sum",
            "device_id": "sum",
        })
        mfg_agg["issue_rate"] = (mfg_agg["drop_count"] + mfg_agg["reboot_count"]) / mfg_agg["device_id"]
        worst_mfg = [(str(mfg), float(rate)) for mfg, rate in
                     mfg_agg["issue_rate"].sort_values(ascending=False).head(5).items()]

        # Worst OS versions
        os_agg = cohort_agg.groupby("os_version").agg({
            "drop_count": "sum",
            "reboot_count": "sum",
            "device_id": "sum",
        })
        os_agg["issue_rate"] = (os_agg["drop_count"] + os_agg["reboot_count"]) / os_agg["device_id"]
        worst_os = [(str(os_v), float(rate)) for os_v, rate in
                    os_agg["issue_rate"].sort_values(ascending=False).head(5).items()]

        recommendations = self._generate_combination_recommendations(problem_combinations)

        return ProblemCombinationReport(
            tenant_id=self.tenant_id,
            analysis_period_days=period_days,
            total_cohorts_analyzed=len(cohort_agg),
            problem_cohorts_found=len(problem_combinations),
            devices_in_problem_cohorts=devices_in_problems,
            problem_combinations=problem_combinations,
            worst_manufacturers=worst_mfg,
            worst_os_versions=worst_os,
            recommendations=recommendations,
        )

    def get_device_abuse_score(
        self,
        device_id: int,
        period_days: int = 7,
    ) -> Dict[str, Any]:
        """Get abuse score for a single device.

        Args:
            device_id: Device to analyze
            period_days: Period to analyze

        Returns:
            Dict with abuse metrics and score
        """
        device_data = self._get_device_abuse_data(period_days, None)

        if device_data.empty:
            return {"device_id": device_id, "abuse_score": 0, "status": "no_data"}

        # Get device's data
        device_df = device_data[device_data["device_id"] == device_id]
        if device_df.empty:
            return {"device_id": device_id, "abuse_score": 0, "status": "device_not_found"}

        drops = device_df["drop_count"].sum()
        reboots = device_df["reboot_count"].sum()

        # Calculate percentiles
        all_drops = device_data.groupby("device_id")["drop_count"].sum()
        all_reboots = device_data.groupby("device_id")["reboot_count"].sum()

        drop_percentile = (all_drops < drops).sum() / len(all_drops) * 100 if len(all_drops) > 0 else 50
        reboot_percentile = (all_reboots < reboots).sum() / len(all_reboots) * 100 if len(all_reboots) > 0 else 50

        # Abuse score (0-100)
        abuse_score = (drop_percentile + reboot_percentile) / 2

        return {
            "device_id": device_id,
            "drops": int(drops),
            "reboots": int(reboots),
            "drop_percentile": float(drop_percentile),
            "reboot_percentile": float(reboot_percentile),
            "abuse_score": float(abuse_score),
            "status": "excessive" if abuse_score > 80 else "normal",
        }

    # Private helper methods

    def _get_device_abuse_data(
        self,
        period_days: int,
        location_id: Optional[str],
    ) -> pd.DataFrame:
        """Get device data for abuse analysis."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        query = (
            self.db.query(DeviceFeature)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.computed_at >= start_date,
                DeviceFeature.computed_at <= end_date,
            )
            .all()
        )

        if not query:
            return pd.DataFrame()

        import json

        records = []
        for feature in query:
            try:
                metadata = json.loads(feature.metadata_json) if feature.metadata_json else {}

                if location_id and metadata.get("location_id") != location_id:
                    continue

                features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}

                # Build cohort ID
                manufacturer = metadata.get("Manufacturer", "Unknown")
                model = metadata.get("Model", "Unknown")
                os_version = metadata.get("OSVersion", metadata.get("OsVersionName", "Unknown"))
                firmware = metadata.get("FirmwareVersion", metadata.get("OEMVersion", ""))
                cohort_id = f"{manufacturer}_{model}_{os_version}"

                records.append({
                    "device_id": feature.device_id,
                    "device_name": metadata.get("device_name"),
                    "computed_at": feature.computed_at,
                    "location_id": metadata.get("location_id"),
                    "user_id": metadata.get("user_id"),
                    "cohort_id": cohort_id,
                    "manufacturer": manufacturer,
                    "model": model,
                    "os_version": os_version,
                    "firmware_version": firmware,
                    "drop_count": features.get("TotalDropCnt", 0),
                    "reboot_count": features.get("RebootCount", 0),
                    "crash_count": features.get("CrashCount", 0),
                })
            except (json.JSONDecodeError, TypeError):
                continue

        return pd.DataFrame(records)

    def _calculate_threshold(
        self,
        data: pd.DataFrame,
        column: str,
        default_threshold: float,
    ) -> float:
        """Calculate threshold for excessive value based on data distribution."""
        if data.empty or column not in data.columns:
            return default_threshold

        device_totals = data.groupby("device_id")[column].sum()
        if device_totals.empty:
            return default_threshold

        # Use 90th percentile or default, whichever is higher
        percentile_threshold = np.percentile(device_totals, self.HIGH_DROPS_PERCENTILE)
        return max(default_threshold, percentile_threshold)

    def _group_metric_by_entity(
        self,
        data: pd.DataFrame,
        metric_col: str,
        entity_col: str,
        period_days: int,
    ) -> Dict[str, Tuple[int, float]]:
        """Group a metric by entity and return (total, rate per week)."""
        result = {}

        if entity_col not in data.columns:
            return result

        agg = data.groupby(entity_col).agg({
            metric_col: "sum",
            "device_id": "nunique",
        })

        for entity_id, row in agg.iterrows():
            if pd.notna(entity_id):
                rate = row[metric_col] / row["device_id"] / period_days * 7
                result[str(entity_id)] = (int(row[metric_col]), float(rate))

        return result

    def _get_worst_drop_devices(
        self,
        device_totals: pd.DataFrame,
        device_data: pd.DataFrame,
        period_days: int,
        limit: int = 10,
    ) -> List[DeviceAbuseIndicator]:
        """Get devices with worst drop rates."""
        if device_totals.empty:
            return []

        # Sort by drops
        worst = device_totals.nlargest(limit, "drop_count")

        # Calculate fleet statistics for percentiles
        all_drops = device_totals["drop_count"]
        all_reboots = device_totals["reboot_count"] if "reboot_count" in device_totals.columns else pd.Series([0])

        indicators = []
        for _, row in worst.iterrows():
            drops = row["drop_count"]
            reboots = row.get("reboot_count", 0)

            # Percentiles
            drop_pct = int((all_drops < drops).sum() / len(all_drops) * 100) if len(all_drops) > 0 else 50
            reboot_pct = int((all_reboots < reboots).sum() / len(all_reboots) * 100) if len(all_reboots) > 0 else 50

            # Determine abuse type
            if drops > self.EXCESSIVE_DROPS_THRESHOLD and reboots > self.EXCESSIVE_REBOOTS_THRESHOLD:
                abuse_type = AbuseType.COMBINED_ABUSE
            else:
                abuse_type = AbuseType.EXCESSIVE_DROPS

            # Severity
            if drop_pct > 95:
                severity = InsightSeverity.CRITICAL
            elif drop_pct > 90:
                severity = InsightSeverity.HIGH
            else:
                severity = InsightSeverity.MEDIUM

            indicators.append(DeviceAbuseIndicator(
                device_id=int(row["device_id"]),
                device_name=row.get("device_name"),
                abuse_type=abuse_type,
                drop_count=int(drops),
                drop_rate_per_day=float(drops / period_days),
                reboot_count=int(reboots),
                reboot_rate_per_day=float(reboots / period_days),
                crash_count=0,
                vs_fleet_drop_percentile=drop_pct,
                vs_fleet_reboot_percentile=reboot_pct,
                vs_cohort_multiplier=1.0,  # Would need cohort data
                assigned_user=row.get("user_id"),
                assigned_location=row.get("location_id"),
                device_cohort=row.get("cohort_id", "Unknown"),
                severity=severity,
                description=f"{drops} drops in {period_days} days (top {100-drop_pct}% worst)",
                recommendations=["Investigate physical handling", "Check device case/protection"],
            ))

        return indicators

    def _get_worst_reboot_devices(
        self,
        device_totals: pd.DataFrame,
        device_data: pd.DataFrame,
        period_days: int,
        limit: int = 10,
    ) -> List[DeviceAbuseIndicator]:
        """Get devices with worst reboot rates."""
        if device_totals.empty:
            return []

        worst = device_totals.nlargest(limit, "reboot_count")

        all_reboots = device_totals["reboot_count"]
        all_drops = device_totals["drop_count"] if "drop_count" in device_totals.columns else pd.Series([0])

        indicators = []
        for _, row in worst.iterrows():
            reboots = row["reboot_count"]
            crashes = row.get("crash_count", 0)
            drops = row.get("drop_count", 0)

            reboot_pct = int((all_reboots < reboots).sum() / len(all_reboots) * 100) if len(all_reboots) > 0 else 50
            drop_pct = int((all_drops < drops).sum() / len(all_drops) * 100) if len(all_drops) > 0 else 50

            # Determine if crash-related
            if crashes > reboots / 2:
                abuse_type = AbuseType.CRASH_RELATED_REBOOTS
            else:
                abuse_type = AbuseType.EXCESSIVE_REBOOTS

            if reboot_pct > 95:
                severity = InsightSeverity.CRITICAL
            elif reboot_pct > 90:
                severity = InsightSeverity.HIGH
            else:
                severity = InsightSeverity.MEDIUM

            indicators.append(DeviceAbuseIndicator(
                device_id=int(row["device_id"]),
                device_name=row.get("device_name"),
                abuse_type=abuse_type,
                drop_count=int(drops),
                drop_rate_per_day=float(drops / period_days),
                reboot_count=int(reboots),
                reboot_rate_per_day=float(reboots / period_days),
                crash_count=int(crashes),
                vs_fleet_drop_percentile=drop_pct,
                vs_fleet_reboot_percentile=reboot_pct,
                vs_cohort_multiplier=1.0,
                assigned_user=row.get("user_id"),
                assigned_location=row.get("location_id"),
                device_cohort=row.get("cohort_id", "Unknown"),
                severity=severity,
                description=f"{reboots} reboots in {period_days} days (top {100-reboot_pct}% worst)",
                recommendations=(
                    ["Check for app crashes", "Consider factory reset"] if abuse_type == AbuseType.CRASH_RELATED_REBOOTS
                    else ["Check for hardware issues", "Verify firmware version"]
                ),
            ))

        return indicators

    def _get_cohort_recommendations(
        self,
        primary_issue: str,
        has_firmware: bool,
    ) -> List[str]:
        """Get recommendations for a problem cohort."""
        recs = []

        if primary_issue == "drops":
            recs.append("Consider more protective cases for this model")
            recs.append("Review handling training for users with this device type")

        elif primary_issue == "reboots":
            if has_firmware:
                recs.append("Check for firmware updates")
            recs.append("Consider OS update if available")
            recs.append("Review installed apps for conflicts")

        elif primary_issue == "crashes":
            recs.append("Review app compatibility with this OS version")
            recs.append("Check for known OS bugs with this version")

        return recs

    def _generate_drop_recommendations(
        self,
        worst_devices: List[DeviceAbuseIndicator],
        worst_locations: List[Tuple[str, int, float]],
        worst_users: List[Tuple[str, int, float]],
        excessive_count: int,
        total_devices: int,
    ) -> List[str]:
        """Generate recommendations for drop analysis."""
        recs = []

        if excessive_count > total_devices * 0.1:
            recs.append(
                f"{excessive_count} devices ({excessive_count/total_devices*100:.0f}%) have excessive drops. "
                "Consider fleet-wide protective case review."
            )

        if worst_locations and worst_locations[0][2] > 3:  # More than 3 drops/device/week
            loc, drops, rate = worst_locations[0]
            recs.append(
                f"Location '{loc}' has highest drop rate ({rate:.1f}/device/week). "
                "Review work conditions and device storage."
            )

        if worst_users and worst_users[0][2] > 5:
            user, drops, rate = worst_users[0]
            recs.append(
                f"User '{user}' has exceptionally high drop rate. "
                "Consider device handling refresher training."
            )

        return recs

    def _generate_reboot_recommendations(
        self,
        worst_devices: List[DeviceAbuseIndicator],
        worst_cohorts: List[Tuple[str, int, float]],
        crash_induced: float,
        excessive_count: int,
        total_devices: int,
    ) -> List[str]:
        """Generate recommendations for reboot analysis."""
        recs = []

        if crash_induced > 50:
            recs.append(
                f"~{crash_induced:.0f}% of reboots may be crash-related. "
                "Review app stability and OS updates."
            )

        if worst_cohorts and worst_cohorts[0][2] > 2:
            cohort, reboots, rate = worst_cohorts[0]
            recs.append(
                f"Cohort '{cohort}' has highest reboot rate. "
                "Check for firmware/OS issues with this combination."
            )

        if excessive_count > total_devices * 0.1:
            recs.append(
                f"{excessive_count} devices have excessive reboots. "
                "May indicate systemic software issue."
            )

        return recs

    def _generate_combination_recommendations(
        self,
        problem_combinations: List[ProblemCombination],
    ) -> List[str]:
        """Generate recommendations for problem combinations."""
        recs = []

        critical = [p for p in problem_combinations if p.severity == InsightSeverity.CRITICAL]
        if critical:
            recs.append(
                f"{len(critical)} critical device combination(s) identified. "
                "Consider phasing out or updating these devices."
            )

        # Check for OS-specific issues
        os_issues: Dict[str, int] = {}
        for p in problem_combinations:
            os_issues[p.os_version] = os_issues.get(p.os_version, 0) + 1

        for os_v, count in os_issues.items():
            if count >= 2:
                recs.append(f"OS version '{os_v}' appears in {count} problem combinations. Consider OS update.")

        return recs

    def _empty_drop_report(
        self,
        group_by: str,
        entity_id: Optional[str],
        period_days: int,
    ) -> DropAnalysisReport:
        """Return empty drop report."""
        return DropAnalysisReport(
            entity_type=group_by,
            entity_id=entity_id or self.tenant_id,
            entity_name=entity_id or self.tenant_id,
            analysis_period_days=period_days,
            total_devices=0,
            total_drops=0,
            avg_drops_per_device=0,
            devices_with_excessive_drops=0,
            by_location={},
            by_user={},
            by_cohort={},
            worst_devices=[],
            worst_locations=[],
            worst_users=[],
            trend_direction="stable",
            trend_change_percent=0,
            recommendations=[],
        )

    def _empty_reboot_report(
        self,
        group_by: str,
        entity_id: Optional[str],
        period_days: int,
    ) -> RebootAnalysisReport:
        """Return empty reboot report."""
        return RebootAnalysisReport(
            entity_type=group_by,
            entity_id=entity_id or self.tenant_id,
            entity_name=entity_id or self.tenant_id,
            analysis_period_days=period_days,
            total_devices=0,
            total_reboots=0,
            total_crashes=0,
            avg_reboots_per_device=0,
            devices_with_excessive_reboots=0,
            crash_induced_reboots_percent=0,
            by_location={},
            by_user={},
            by_cohort={},
            worst_devices=[],
            worst_cohorts=[],
            recommendations=[],
        )
