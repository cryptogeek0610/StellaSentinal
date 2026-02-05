"""
Data Reconciliation Service.

Monitors and reports on data quality between XSight and MobiControl:
- Device ID alignment between data sources
- Data freshness and staleness
- Missing data patterns
- Temporal alignment gaps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DataFreshness:
    """Data freshness metrics for a source."""

    source_name: str
    latest_timestamp: datetime | None = None
    staleness_hours: float = 0.0
    device_count: int = 0
    record_count: int = 0
    status: str = "unknown"  # "fresh", "stale", "critical", "unknown"


@dataclass
class ReconciliationReport:
    """Report on data quality and reconciliation status."""

    report_date: datetime

    # Device reconciliation
    xsight_device_count: int
    mobicontrol_device_count: int
    matched_devices: int
    xsight_only: int
    mobicontrol_only: int
    match_rate: float

    # Data freshness
    xsight_latest_data: datetime | None = None
    mobicontrol_latest_data: datetime | None = None
    xsight_staleness_hours: float = 0.0
    mobicontrol_staleness_hours: float = 0.0

    # Missing data
    devices_missing_telemetry: int = 0
    devices_missing_inventory: int = 0

    # Temporal alignment
    avg_time_gap_hours: float = 0.0
    max_time_gap_hours: float = 0.0

    # Quality assessment
    overall_quality_score: float = 0.0  # 0-100
    quality_grade: str = "Unknown"  # A, B, C, D, F

    # Issues and recommendations
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class MissingDevice:
    """A device missing from one data source."""

    device_id: int
    present_in: str  # "xsight" or "mobicontrol"
    missing_from: str
    last_seen: datetime | None = None
    device_name: str | None = None
    device_model: str | None = None


# =============================================================================
# Data Reconciliation Service
# =============================================================================


class DataReconciliationService:
    """
    Service for monitoring data quality between XSight and MobiControl.

    Features:
    - Device ID reconciliation between sources
    - Data freshness monitoring
    - Temporal alignment analysis
    - Quality scoring and recommendations
    """

    def __init__(
        self,
        freshness_warning_hours: float = 24.0,
        freshness_critical_hours: float = 48.0,
        match_rate_warning: float = 0.9,
        match_rate_critical: float = 0.8,
    ):
        """
        Initialize the service.

        Args:
            freshness_warning_hours: Hours before data is considered stale
            freshness_critical_hours: Hours before data is critically stale
            match_rate_warning: Device match rate warning threshold
            match_rate_critical: Device match rate critical threshold
        """
        self.freshness_warning_hours = freshness_warning_hours
        self.freshness_critical_hours = freshness_critical_hours
        self.match_rate_warning = match_rate_warning
        self.match_rate_critical = match_rate_critical

    def generate_reconciliation_report(
        self,
        xsight_df: pd.DataFrame | None = None,
        mobicontrol_df: pd.DataFrame | None = None,
        report_date: datetime | None = None,
    ) -> ReconciliationReport:
        """
        Generate comprehensive reconciliation report.

        Args:
            xsight_df: XSight telemetry data (or loaded automatically)
            mobicontrol_df: MobiControl inventory data (or loaded automatically)
            report_date: Report date (defaults to now)

        Returns:
            ReconciliationReport with quality metrics
        """
        report_date = report_date or datetime.now(UTC)

        # Load data if not provided
        if xsight_df is None:
            xsight_df = self._load_xsight_data()
        if mobicontrol_df is None:
            mobicontrol_df = self._load_mobicontrol_data()

        # Handle empty data
        if xsight_df is None or xsight_df.empty:
            xsight_df = pd.DataFrame(columns=["DeviceId", "Timestamp"])
        if mobicontrol_df is None or mobicontrol_df.empty:
            mobicontrol_df = pd.DataFrame(columns=["DeviceId", "LastCheckInTime"])

        # Get device sets from each source
        xsight_devices = (
            set(xsight_df["DeviceId"].unique()) if "DeviceId" in xsight_df.columns else set()
        )
        mc_devices = (
            set(mobicontrol_df["DeviceId"].unique())
            if "DeviceId" in mobicontrol_df.columns
            else set()
        )

        # Compute set operations
        matched = xsight_devices & mc_devices
        xsight_only = xsight_devices - mc_devices
        mc_only = mc_devices - xsight_devices

        total_devices = len(xsight_devices | mc_devices)
        match_rate = len(matched) / max(1, total_devices)

        # Get data freshness
        xsight_latest = self._get_latest_timestamp(xsight_df, "Timestamp")
        mc_latest = self._get_latest_timestamp(mobicontrol_df, "LastCheckInTime")

        xsight_staleness = self._compute_staleness_hours(xsight_latest, report_date)
        mc_staleness = self._compute_staleness_hours(mc_latest, report_date)

        # Compute temporal alignment for matched devices
        time_gaps = self._compute_temporal_gaps(xsight_df, mobicontrol_df, matched)

        # Generate issues and recommendations
        issues, recommendations = self._analyze_issues(
            match_rate=match_rate,
            xsight_staleness=xsight_staleness,
            mc_staleness=mc_staleness,
            time_gaps=time_gaps,
            xsight_only_count=len(xsight_only),
            mc_only_count=len(mc_only),
        )

        # Compute quality score
        quality_score, quality_grade = self._compute_quality_score(
            match_rate=match_rate,
            xsight_staleness=xsight_staleness,
            mc_staleness=mc_staleness,
            issues_count=len(issues),
        )

        return ReconciliationReport(
            report_date=report_date,
            xsight_device_count=len(xsight_devices),
            mobicontrol_device_count=len(mc_devices),
            matched_devices=len(matched),
            xsight_only=len(xsight_only),
            mobicontrol_only=len(mc_only),
            match_rate=round(match_rate, 3),
            xsight_latest_data=xsight_latest,
            mobicontrol_latest_data=mc_latest,
            xsight_staleness_hours=round(xsight_staleness, 1),
            mobicontrol_staleness_hours=round(mc_staleness, 1),
            devices_missing_telemetry=len(mc_only),
            devices_missing_inventory=len(xsight_only),
            avg_time_gap_hours=round(time_gaps.get("avg", 0), 1),
            max_time_gap_hours=round(time_gaps.get("max", 0), 1),
            overall_quality_score=quality_score,
            quality_grade=quality_grade,
            issues=issues,
            recommendations=recommendations,
        )

    def get_data_freshness(
        self,
        xsight_df: pd.DataFrame | None = None,
        mobicontrol_df: pd.DataFrame | None = None,
    ) -> dict[str, DataFreshness]:
        """
        Get data freshness metrics for all sources.

        Returns:
            Dictionary of source name to freshness metrics
        """
        now = datetime.now(UTC)

        # Load data if not provided
        if xsight_df is None:
            xsight_df = self._load_xsight_data()
        if mobicontrol_df is None:
            mobicontrol_df = self._load_mobicontrol_data()

        freshness = {}

        # XSight freshness
        if xsight_df is not None and not xsight_df.empty:
            latest = self._get_latest_timestamp(xsight_df, "Timestamp")
            staleness = self._compute_staleness_hours(latest, now)
            status = self._get_freshness_status(staleness)

            freshness["xsight"] = DataFreshness(
                source_name="XSight DW",
                latest_timestamp=latest,
                staleness_hours=round(staleness, 1),
                device_count=xsight_df["DeviceId"].nunique()
                if "DeviceId" in xsight_df.columns
                else 0,
                record_count=len(xsight_df),
                status=status,
            )
        else:
            freshness["xsight"] = DataFreshness(
                source_name="XSight DW",
                status="unavailable",
            )

        # MobiControl freshness
        if mobicontrol_df is not None and not mobicontrol_df.empty:
            latest = self._get_latest_timestamp(mobicontrol_df, "LastCheckInTime")
            staleness = self._compute_staleness_hours(latest, now)
            status = self._get_freshness_status(staleness)

            freshness["mobicontrol"] = DataFreshness(
                source_name="MobiControl",
                latest_timestamp=latest,
                staleness_hours=round(staleness, 1),
                device_count=mobicontrol_df["DeviceId"].nunique()
                if "DeviceId" in mobicontrol_df.columns
                else 0,
                record_count=len(mobicontrol_df),
                status=status,
            )
        else:
            freshness["mobicontrol"] = DataFreshness(
                source_name="MobiControl",
                status="unavailable",
            )

        return freshness

    def get_missing_devices(
        self,
        source: str = "xsight",
        xsight_df: pd.DataFrame | None = None,
        mobicontrol_df: pd.DataFrame | None = None,
        limit: int = 100,
    ) -> list[MissingDevice]:
        """
        Get list of devices missing from one source but present in another.

        Args:
            source: Which source to check ("xsight" or "mobicontrol")
            xsight_df: XSight data
            mobicontrol_df: MobiControl data
            limit: Maximum devices to return

        Returns:
            List of missing device records
        """
        # Load data if not provided
        if xsight_df is None:
            xsight_df = self._load_xsight_data()
        if mobicontrol_df is None:
            mobicontrol_df = self._load_mobicontrol_data()

        if xsight_df is None or mobicontrol_df is None:
            return []

        xsight_devices = (
            set(xsight_df["DeviceId"].unique()) if "DeviceId" in xsight_df.columns else set()
        )
        mc_devices = (
            set(mobicontrol_df["DeviceId"].unique())
            if "DeviceId" in mobicontrol_df.columns
            else set()
        )

        missing_devices = []

        if source == "xsight":
            # Devices in MC but not in XSight
            missing_ids = mc_devices - xsight_devices
            for device_id in list(missing_ids)[:limit]:
                device_row = (
                    mobicontrol_df[mobicontrol_df["DeviceId"] == device_id].iloc[0]
                    if device_id in mobicontrol_df["DeviceId"].values
                    else {}
                )

                missing_devices.append(
                    MissingDevice(
                        device_id=int(device_id),
                        present_in="mobicontrol",
                        missing_from="xsight",
                        last_seen=device_row.get("LastCheckInTime")
                        if isinstance(device_row, dict)
                        else getattr(device_row, "LastCheckInTime", None),
                        device_name=device_row.get("DevName")
                        if isinstance(device_row, dict)
                        else getattr(device_row, "DevName", None),
                        device_model=device_row.get("ModelId")
                        if isinstance(device_row, dict)
                        else getattr(device_row, "ModelId", None),
                    )
                )
        else:
            # Devices in XSight but not in MC
            missing_ids = xsight_devices - mc_devices
            for device_id in list(missing_ids)[:limit]:
                device_row = (
                    xsight_df[xsight_df["DeviceId"] == device_id].iloc[0]
                    if device_id in xsight_df["DeviceId"].values
                    else {}
                )

                missing_devices.append(
                    MissingDevice(
                        device_id=int(device_id),
                        present_in="xsight",
                        missing_from="mobicontrol",
                        last_seen=device_row.get("Timestamp")
                        if isinstance(device_row, dict)
                        else getattr(device_row, "Timestamp", None),
                    )
                )

        return missing_devices

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _load_xsight_data(self) -> pd.DataFrame | None:
        """Load XSight telemetry data."""
        try:
            from device_anomaly.data_access.unified_loader import load_unified_device_dataset

            end_date = datetime.now(UTC).date()
            start_date = end_date - timedelta(days=7)

            return load_unified_device_dataset(start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.warning(f"Failed to load XSight data: {e}")
            return None

    def _load_mobicontrol_data(self) -> pd.DataFrame | None:
        """Load MobiControl inventory data."""
        try:
            from device_anomaly.data_access.mc_loader import load_mc_device_inventory_snapshot

            end_date = datetime.now(UTC).date()
            start_date = end_date - timedelta(days=7)

            return load_mc_device_inventory_snapshot(start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.warning(f"Failed to load MobiControl data: {e}")
            return None

    def _get_latest_timestamp(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
    ) -> datetime | None:
        """Get latest timestamp from DataFrame."""
        if df.empty or timestamp_col not in df.columns:
            return None

        try:
            ts_series = pd.to_datetime(df[timestamp_col], errors="coerce")
            latest = ts_series.max()
            if pd.isna(latest):
                return None
            return latest.to_pydatetime().replace(tzinfo=UTC)
        except Exception:
            return None

    def _compute_staleness_hours(
        self,
        latest: datetime | None,
        reference: datetime,
    ) -> float:
        """Compute staleness in hours."""
        if latest is None:
            # Use a large finite value instead of inf for JSON compatibility
            return 9999.0

        # Ensure both are timezone-aware
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=UTC)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)

        delta = reference - latest
        return delta.total_seconds() / 3600

    def _compute_temporal_gaps(
        self,
        xsight_df: pd.DataFrame,
        mc_df: pd.DataFrame,
        matched_devices: set[int],
    ) -> dict[str, float]:
        """Compute temporal alignment gaps between sources."""
        if not matched_devices:
            return {"avg": 0, "max": 0}

        gaps = []

        for device_id in list(matched_devices)[:100]:  # Sample for performance
            xs_times = None
            mc_times = None

            if "DeviceId" in xsight_df.columns and "Timestamp" in xsight_df.columns:
                xs_device = xsight_df[xsight_df["DeviceId"] == device_id]
                if not xs_device.empty:
                    xs_times = pd.to_datetime(xs_device["Timestamp"], errors="coerce").max()

            if "DeviceId" in mc_df.columns and "LastCheckInTime" in mc_df.columns:
                mc_device = mc_df[mc_df["DeviceId"] == device_id]
                if not mc_device.empty:
                    mc_times = pd.to_datetime(mc_device["LastCheckInTime"], errors="coerce").max()

            if xs_times is not None and mc_times is not None:
                gap_hours = abs((xs_times - mc_times).total_seconds()) / 3600
                gaps.append(gap_hours)

        if not gaps:
            return {"avg": 0, "max": 0}

        return {
            "avg": float(sum(gaps) / len(gaps)),
            "max": float(max(gaps)),
        }

    def _get_freshness_status(self, staleness_hours: float) -> str:
        """Get freshness status from staleness hours."""
        if staleness_hours >= 9999.0:
            return "unavailable"
        if staleness_hours <= self.freshness_warning_hours:
            return "fresh"
        if staleness_hours <= self.freshness_critical_hours:
            return "stale"
        return "critical"

    def _analyze_issues(
        self,
        match_rate: float,
        xsight_staleness: float,
        mc_staleness: float,
        time_gaps: dict[str, float],
        xsight_only_count: int,
        mc_only_count: int,
    ) -> tuple[list[str], list[str]]:
        """Analyze data quality issues and generate recommendations."""
        issues = []
        recommendations = []

        if match_rate < self.match_rate_critical:
            issues.append(
                f"Critical: Device match rate is {match_rate:.1%} (below {self.match_rate_critical:.0%})"
            )
            recommendations.append(
                "Urgent: Investigate data collection gaps. Check device enrollment and sync status."
            )
        elif match_rate < self.match_rate_warning:
            issues.append(
                f"Warning: Device match rate is {match_rate:.1%} (below {self.match_rate_warning:.0%})"
            )
            recommendations.append(
                "Review unmatched devices - may indicate enrollment issues or data collection gaps."
            )

        if xsight_staleness > self.freshness_critical_hours:
            issues.append(f"Critical: XSight data is {xsight_staleness:.1f} hours old")
            recommendations.append("Check XSight data collection pipeline urgently.")
        elif xsight_staleness > self.freshness_warning_hours:
            issues.append(f"Warning: XSight data is {xsight_staleness:.1f} hours old")
            recommendations.append("Monitor XSight data pipeline for delays.")

        if mc_staleness > self.freshness_critical_hours:
            issues.append(f"Critical: MobiControl data is {mc_staleness:.1f} hours old")
            recommendations.append("Check MobiControl sync status urgently.")
        elif mc_staleness > self.freshness_warning_hours:
            issues.append(f"Warning: MobiControl data is {mc_staleness:.1f} hours old")
            recommendations.append("Monitor MobiControl sync for delays.")

        avg_gap = time_gaps.get("avg", 0)
        if avg_gap > 24:
            issues.append(f"High temporal misalignment: avg {avg_gap:.1f}h gap between sources")
            recommendations.append(
                "Consider implementing temporal-aware joins to reduce stale data in unified dataset."
            )

        if xsight_only_count > 100:
            issues.append(f"{xsight_only_count} devices have telemetry but no inventory data")
            recommendations.append(
                "Review devices with telemetry but missing inventory - may need re-enrollment."
            )

        if mc_only_count > 100:
            issues.append(f"{mc_only_count} devices have inventory but no telemetry")
            recommendations.append(
                "Review devices with inventory but no telemetry - may have XSight collection issues."
            )

        return issues, recommendations

    def _compute_quality_score(
        self,
        match_rate: float,
        xsight_staleness: float,
        mc_staleness: float,
        issues_count: int,
    ) -> tuple[float, str]:
        """Compute overall quality score and grade."""
        # Base score from match rate (0-40 points)
        match_score = min(40, match_rate * 40)

        # Freshness score (0-30 points each)
        # Cap staleness at 48 hours for score calculation (larger values = 0 points)
        xs_capped = min(xsight_staleness, 48.0)
        mc_capped = min(mc_staleness, 48.0)
        xs_fresh_score = max(0, 30 - (xs_capped / 48) * 30)
        mc_fresh_score = max(0, 30 - (mc_capped / 48) * 30)

        # Issue penalty
        issue_penalty = min(20, issues_count * 5)

        total_score = max(0, match_score + xs_fresh_score + mc_fresh_score - issue_penalty)

        # Convert to grade
        if total_score >= 90:
            grade = "A"
        elif total_score >= 80:
            grade = "B"
        elif total_score >= 70:
            grade = "C"
        elif total_score >= 60:
            grade = "D"
        else:
            grade = "F"

        return round(total_score, 1), grade


# =============================================================================
# Convenience Functions
# =============================================================================


def get_data_quality_summary() -> dict[str, Any]:
    """Get quick data quality summary."""
    service = DataReconciliationService()
    report = service.generate_reconciliation_report()

    return {
        "match_rate": report.match_rate,
        "quality_score": report.overall_quality_score,
        "quality_grade": report.quality_grade,
        "xsight_staleness_hours": report.xsight_staleness_hours,
        "mobicontrol_staleness_hours": report.mobicontrol_staleness_hours,
        "issues_count": len(report.issues),
        "status": "healthy"
        if report.quality_grade in ("A", "B")
        else "warning"
        if report.quality_grade == "C"
        else "critical",
    }
