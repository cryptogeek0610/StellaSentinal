"""Battery shift analysis for work shift readiness.

Carl's key requirements:
- "Batteries that don't last a shift"
- "Batteries not fully charged in the morning"
- "Consistent or periodic rapid drain issues"
- "Charging pattern correlations"

This analyzer provides shift-aware battery insights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, Integer
from sqlalchemy.orm import Session

from device_anomaly.database.schema import (
    DeviceFeature,
    LocationMetadata,
    ShiftPerformance,
)
from device_anomaly.insights.categories import InsightCategory, InsightSeverity
from device_anomaly.insights.location_mapper import LocationMapper, ShiftSchedule

logger = logging.getLogger(__name__)


@dataclass
class BatteryProjection:
    """Projection of battery level at a future time."""

    device_id: int
    current_level: float
    current_drain_rate: float  # % per hour
    projected_level: float
    target_time: datetime
    will_survive: bool
    estimated_dead_time: Optional[datetime]
    confidence: float  # 0-1


@dataclass
class DeviceShiftReadiness:
    """Shift readiness assessment for a single device."""

    device_id: int
    device_name: Optional[str]
    current_battery: float
    drain_rate_per_hour: float
    shift_hours: float
    projected_end_battery: float
    will_complete_shift: bool
    estimated_dead_time: Optional[str]  # Time format like "14:30"
    was_fully_charged: bool  # >= 90% at shift start
    readiness_score: float  # 0-100
    recommendations: List[str]


@dataclass
class ShiftReadinessReport:
    """Shift readiness report for a location."""

    location_id: str
    location_name: str
    shift_name: str
    shift_date: datetime
    shift_start: time
    shift_end: time
    shift_duration_hours: float

    # Summary
    total_devices: int
    devices_ready: int
    devices_at_risk: int
    devices_critical: int
    readiness_percentage: float

    # Device details
    device_readiness: List[DeviceShiftReadiness]

    # Aggregated insights
    avg_battery_at_start: float
    avg_drain_rate: float
    devices_not_fully_charged: int
    devices_will_die_during_shift: int

    # Comparison
    vs_last_week_readiness: Optional[float]  # % change


@dataclass
class ChargingPatternIssue:
    """A charging pattern issue for a device."""

    device_id: int
    device_name: Optional[str]
    issue_type: str  # "incomplete_charge", "short_charges", "usb_instead_ac", "missed_overnight"
    description: str
    frequency: str  # "daily", "3x_per_week", etc.
    severity: InsightSeverity
    recommendation: str


@dataclass
class ChargingPatternReport:
    """Charging pattern analysis for a location."""

    location_id: str
    location_name: str
    analysis_period_days: int

    # Summary
    total_devices: int
    devices_with_issues: int
    issue_rate: float

    # Pattern breakdown
    incomplete_charges: int  # Devices not reaching 90%+
    short_charges: int  # Charges < 30 min
    usb_only_charges: int  # USB instead of AC
    missed_overnight: int  # Devices not charged overnight

    # Issues
    issues: List[ChargingPatternIssue]

    # Top recommendations
    recommendations: List[str]


@dataclass
class PeriodicDrainEvent:
    """A periodic drain event detected in a device."""

    day_of_week: Optional[int]  # 0=Monday, None if daily
    time_of_day: time
    duration_minutes: int
    avg_drain_rate: float
    occurrence_count: int
    confidence: float


@dataclass
class PeriodicDrainReport:
    """Periodic drain analysis for a device."""

    device_id: int
    device_name: Optional[str]
    analysis_period_days: int

    # Detected patterns
    has_periodic_pattern: bool
    patterns: List[PeriodicDrainEvent]

    # Overall drain profile
    avg_daily_drain: float
    peak_drain_time: Optional[time]
    lowest_drain_time: Optional[time]

    # Correlation analysis
    correlated_apps: List[Tuple[str, float]]  # (app_name, correlation)
    correlated_locations: List[Tuple[str, float]]


class BatteryShiftAnalyzer:
    """Analyzer for shift-aware battery insights.

    Addresses Carl's key requirements:
    - Batteries not lasting shifts
    - Morning charge completeness
    - Periodic drain patterns
    - Charging behavior correlations

    Usage:
        analyzer = BatteryShiftAnalyzer(db_session, tenant_id)
        report = analyzer.analyze_shift_readiness("warehouse_1", date.today())
    """

    # Thresholds
    FULLY_CHARGED_THRESHOLD = 90.0  # % to consider "fully charged"
    CRITICAL_BATTERY_THRESHOLD = 20.0  # % at which battery is critical
    HIGH_DRAIN_RATE_THRESHOLD = 10.0  # %/hour considered high
    SHORT_CHARGE_MINUTES = 30  # Charges shorter than this are "short"

    def __init__(
        self,
        db_session: Session,
        tenant_id: str,
        location_mapper: Optional[LocationMapper] = None,
    ):
        """Initialize the analyzer.

        Args:
            db_session: SQLAlchemy database session
            tenant_id: Tenant ID for multi-tenant filtering
            location_mapper: Optional LocationMapper instance
        """
        self.db = db_session
        self.tenant_id = tenant_id
        self.location_mapper = location_mapper or LocationMapper(db_session, tenant_id)

    def analyze_shift_readiness(
        self,
        location_id: str,
        shift_date: datetime,
        shift_name: Optional[str] = None,
    ) -> Optional[ShiftReadinessReport]:
        """Analyze battery readiness for an upcoming shift.

        Args:
            location_id: Location to analyze
            shift_date: Date of the shift
            shift_name: Optional specific shift name (defaults to current/next shift)

        Returns:
            ShiftReadinessReport or None if no shift data
        """
        # Get location and shift info
        location = self.db.query(LocationMetadata).filter(
            LocationMetadata.tenant_id == self.tenant_id,
            LocationMetadata.location_id == location_id,
        ).first()

        if not location:
            logger.warning(f"Location not found: {location_id}")
            return None

        # Get shift schedule
        shift = self._get_shift_for_analysis(location, shift_date, shift_name)
        if not shift:
            logger.warning(f"No shift schedule for location: {location_id}")
            return None

        # Get devices at this location with recent battery data
        devices_df = self._get_location_devices_with_battery(location_id, shift_date)
        if devices_df.empty:
            return None

        shift_duration = self._calculate_shift_duration(shift)

        # Analyze each device
        device_readiness: List[DeviceShiftReadiness] = []

        for _, row in devices_df.iterrows():
            readiness = self._analyze_device_shift_readiness(row, shift, shift_duration)
            device_readiness.append(readiness)

        # Calculate summary metrics
        devices_ready = sum(1 for d in device_readiness if d.will_complete_shift)
        devices_at_risk = sum(1 for d in device_readiness if not d.will_complete_shift and d.projected_end_battery > 0)
        devices_critical = sum(1 for d in device_readiness if d.projected_end_battery <= 0)
        devices_not_charged = sum(1 for d in device_readiness if not d.was_fully_charged)

        # Get last week's readiness for comparison
        last_week_readiness = self._get_historical_readiness(location_id, shift_date - timedelta(days=7))

        total_devices = len(device_readiness)
        current_readiness = (devices_ready / total_devices * 100) if total_devices > 0 else 0

        vs_last_week = None
        if last_week_readiness is not None:
            vs_last_week = current_readiness - last_week_readiness

        return ShiftReadinessReport(
            location_id=location_id,
            location_name=location.location_name,
            shift_name=shift.name,
            shift_date=shift_date,
            shift_start=shift.start_time,
            shift_end=shift.end_time,
            shift_duration_hours=shift_duration,
            total_devices=total_devices,
            devices_ready=devices_ready,
            devices_at_risk=devices_at_risk,
            devices_critical=devices_critical,
            readiness_percentage=current_readiness,
            device_readiness=device_readiness,
            avg_battery_at_start=float(np.mean([d.current_battery for d in device_readiness])) if device_readiness else 0,
            avg_drain_rate=float(np.mean([d.drain_rate_per_hour for d in device_readiness])) if device_readiness else 0,
            devices_not_fully_charged=devices_not_charged,
            devices_will_die_during_shift=devices_critical,
            vs_last_week_readiness=vs_last_week,
        )

    def analyze_charging_patterns(
        self,
        location_id: str,
        period_days: int = 14,
    ) -> Optional[ChargingPatternReport]:
        """Analyze charging patterns at a location.

        Args:
            location_id: Location to analyze
            period_days: Days of history to analyze

        Returns:
            ChargingPatternReport with charging pattern insights
        """
        location = self.db.query(LocationMetadata).filter(
            LocationMetadata.tenant_id == self.tenant_id,
            LocationMetadata.location_id == location_id,
        ).first()

        if not location:
            return None

        # Get charging data for devices at this location
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        charging_data = self._get_location_charging_data(location_id, start_date, end_date)
        if charging_data.empty:
            return None

        issues: List[ChargingPatternIssue] = []

        # Analyze each device's charging patterns
        device_ids = charging_data["device_id"].unique()

        incomplete_charges = 0
        short_charges = 0
        usb_only = 0
        missed_overnight = 0

        for device_id in device_ids:
            device_data = charging_data[charging_data["device_id"] == device_id]
            device_issues = self._analyze_device_charging_patterns(device_id, device_data)
            issues.extend(device_issues)

            # Count issue types
            for issue in device_issues:
                if issue.issue_type == "incomplete_charge":
                    incomplete_charges += 1
                elif issue.issue_type == "short_charges":
                    short_charges += 1
                elif issue.issue_type == "usb_instead_ac":
                    usb_only += 1
                elif issue.issue_type == "missed_overnight":
                    missed_overnight += 1

        total_devices = len(device_ids)
        devices_with_issues = len(set(i.device_id for i in issues))

        # Generate recommendations
        recommendations = self._generate_charging_recommendations(
            incomplete_charges, short_charges, usb_only, missed_overnight, total_devices
        )

        return ChargingPatternReport(
            location_id=location_id,
            location_name=location.location_name,
            analysis_period_days=period_days,
            total_devices=total_devices,
            devices_with_issues=devices_with_issues,
            issue_rate=devices_with_issues / total_devices if total_devices > 0 else 0,
            incomplete_charges=incomplete_charges,
            short_charges=short_charges,
            usb_only_charges=usb_only,
            missed_overnight=missed_overnight,
            issues=issues,
            recommendations=recommendations,
        )

    def identify_periodic_drain_issues(
        self,
        device_id: int,
        lookback_days: int = 14,
    ) -> Optional[PeriodicDrainReport]:
        """Identify periodic/recurring drain patterns for a device.

        Args:
            device_id: Device to analyze
            lookback_days: Days of history to analyze

        Returns:
            PeriodicDrainReport with detected patterns
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        # Get device's battery history
        drain_data = self._get_device_drain_history(device_id, start_date, end_date)
        if drain_data.empty or len(drain_data) < 7:  # Need enough data points
            return None

        # Analyze for periodic patterns
        patterns = self._detect_periodic_patterns(drain_data)

        # Calculate overall drain profile
        avg_daily_drain = drain_data["drain_per_hour"].mean() * 24 if "drain_per_hour" in drain_data.columns else 0

        # Find peak drain times
        if "hour" in drain_data.columns:
            hourly_drain = drain_data.groupby("hour")["drain_per_hour"].mean()
            if not hourly_drain.empty:
                peak_hour = hourly_drain.idxmax()
                lowest_hour = hourly_drain.idxmin()
                peak_drain_time = time(hour=int(peak_hour))
                lowest_drain_time = time(hour=int(lowest_hour))
            else:
                peak_drain_time = None
                lowest_drain_time = None
        else:
            peak_drain_time = None
            lowest_drain_time = None

        # Correlate with apps if data available
        correlated_apps = self._correlate_drain_with_apps(device_id, drain_data)
        correlated_locations = self._correlate_drain_with_locations(device_id, drain_data)

        return PeriodicDrainReport(
            device_id=device_id,
            device_name=None,  # Would need to fetch from metadata
            analysis_period_days=lookback_days,
            has_periodic_pattern=len(patterns) > 0,
            patterns=patterns,
            avg_daily_drain=float(avg_daily_drain),
            peak_drain_time=peak_drain_time,
            lowest_drain_time=lowest_drain_time,
            correlated_apps=correlated_apps,
            correlated_locations=correlated_locations,
        )

    def project_battery_at_time(
        self,
        device_id: int,
        target_time: datetime,
        current_level: Optional[float] = None,
        drain_rate: Optional[float] = None,
    ) -> BatteryProjection:
        """Project battery level at a future time.

        Args:
            device_id: Device ID
            target_time: Future time to project to
            current_level: Current battery level (fetched if not provided)
            drain_rate: Drain rate %/hour (calculated if not provided)

        Returns:
            BatteryProjection with estimated battery at target time
        """
        # Get current battery if not provided
        if current_level is None or drain_rate is None:
            current_data = self._get_current_device_battery(device_id)
            if current_data:
                current_level = current_level or current_data.get("battery_level", 50)
                drain_rate = drain_rate or current_data.get("drain_rate", 5)
            else:
                current_level = current_level or 50
                drain_rate = drain_rate or 5

        # Calculate hours until target
        now = datetime.utcnow()
        hours_until_target = (target_time - now).total_seconds() / 3600

        # Project battery level
        projected_level = current_level - (drain_rate * hours_until_target)
        projected_level = max(0, projected_level)

        # Will it survive?
        will_survive = projected_level > 0

        # Estimate when it will die
        if drain_rate > 0:
            hours_until_dead = current_level / drain_rate
            estimated_dead_time = now + timedelta(hours=hours_until_dead) if hours_until_dead < hours_until_target else None
        else:
            estimated_dead_time = None

        # Confidence based on drain rate variability
        confidence = 0.8  # Default, would be better with historical variance

        return BatteryProjection(
            device_id=device_id,
            current_level=current_level,
            current_drain_rate=drain_rate,
            projected_level=projected_level,
            target_time=target_time,
            will_survive=will_survive,
            estimated_dead_time=estimated_dead_time,
            confidence=confidence,
        )

    def save_shift_performance(
        self,
        report: ShiftReadinessReport,
    ) -> int:
        """Save shift performance data to database.

        Args:
            report: ShiftReadinessReport to save

        Returns:
            Number of records saved
        """
        saved = 0

        for device in report.device_readiness:
            # Calculate shift times as datetime
            shift_start_dt = datetime.combine(report.shift_date.date(), report.shift_start)
            shift_end_dt = datetime.combine(report.shift_date.date(), report.shift_end)

            # Handle overnight shifts
            if report.shift_end < report.shift_start:
                shift_end_dt += timedelta(days=1)

            # Parse estimated dead time
            estimated_dead = None
            if device.estimated_dead_time:
                try:
                    dead_time = datetime.strptime(device.estimated_dead_time, "%H:%M").time()
                    estimated_dead = datetime.combine(report.shift_date.date(), dead_time)
                except ValueError:
                    pass

            record = ShiftPerformance(
                tenant_id=self.tenant_id,
                device_id=device.device_id,
                location_id=report.location_id,
                shift_date=report.shift_date.date(),
                shift_name=report.shift_name,
                shift_start=shift_start_dt,
                shift_end=shift_end_dt,
                shift_duration_hours=report.shift_duration_hours,
                battery_start=device.current_battery,
                battery_end=device.projected_end_battery,
                battery_drain_total=device.current_battery - device.projected_end_battery,
                battery_drain_rate_per_hour=device.drain_rate_per_hour,
                will_complete_shift=device.will_complete_shift,
                estimated_dead_time=estimated_dead,
                was_fully_charged_at_start=device.was_fully_charged,
            )

            self.db.add(record)
            saved += 1

        self.db.commit()
        return saved

    # Private helper methods

    def _get_shift_for_analysis(
        self,
        location: LocationMetadata,
        shift_date: datetime,
        shift_name: Optional[str] = None,
    ) -> Optional[ShiftSchedule]:
        """Get the relevant shift schedule for analysis."""
        shifts = self.location_mapper._load_shift_schedules(location)
        if not shifts:
            # Default shift
            return ShiftSchedule(
                name="Standard",
                start_time=time(6, 0),
                end_time=time(14, 0),
            )

        if shift_name:
            for shift in shifts:
                if shift.name.lower() == shift_name.lower():
                    return shift

        # Find current or next shift based on time
        current_time = shift_date.time() if isinstance(shift_date, datetime) else time(6, 0)
        for shift in shifts:
            if shift.start_time <= current_time <= shift.end_time:
                return shift

        # Return first shift as default
        return shifts[0] if shifts else None

    def _calculate_shift_duration(self, shift: ShiftSchedule) -> float:
        """Calculate shift duration in hours."""
        start = datetime.combine(datetime.today(), shift.start_time)
        end = datetime.combine(datetime.today(), shift.end_time)

        if end < start:  # Overnight shift
            end += timedelta(days=1)

        return (end - start).total_seconds() / 3600

    def _get_location_devices_with_battery(
        self,
        location_id: str,
        as_of_date: datetime,
    ) -> pd.DataFrame:
        """Get devices at a location with their current battery data."""
        # Query recent device features
        query = (
            self.db.query(DeviceFeature)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.computed_at >= as_of_date - timedelta(hours=24),
                DeviceFeature.computed_at <= as_of_date,
            )
            .order_by(DeviceFeature.computed_at.desc())
            .all()
        )

        if not query:
            return pd.DataFrame()

        import json

        records = []
        seen_devices = set()

        for feature in query:
            if feature.device_id in seen_devices:
                continue

            try:
                metadata = json.loads(feature.metadata_json) if feature.metadata_json else {}
                if metadata.get("location_id") != location_id:
                    continue

                features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}

                records.append({
                    "device_id": feature.device_id,
                    "device_name": metadata.get("device_name"),
                    "battery_level": features.get("BatteryLevel", 50),
                    "drain_rate": features.get("BatteryDrainPerHour", 5),
                    "computed_at": feature.computed_at,
                })
                seen_devices.add(feature.device_id)

            except (json.JSONDecodeError, TypeError):
                continue

        return pd.DataFrame(records)

    def _analyze_device_shift_readiness(
        self,
        device_row: pd.Series,
        shift: ShiftSchedule,
        shift_duration: float,
    ) -> DeviceShiftReadiness:
        """Analyze shift readiness for a single device."""
        current_battery = float(device_row.get("battery_level", 50))
        drain_rate = float(device_row.get("drain_rate", 5))

        # Project end battery
        projected_end = current_battery - (drain_rate * shift_duration)
        projected_end = max(0, projected_end)

        # Will complete shift?
        will_complete = projected_end > self.CRITICAL_BATTERY_THRESHOLD

        # Estimate dead time if applicable
        estimated_dead_time = None
        if not will_complete:
            hours_until_dead = current_battery / drain_rate if drain_rate > 0 else 999
            dead_datetime = datetime.combine(datetime.today(), shift.start_time) + timedelta(hours=hours_until_dead)
            estimated_dead_time = dead_datetime.strftime("%H:%M")

        # Was fully charged?
        was_fully_charged = current_battery >= self.FULLY_CHARGED_THRESHOLD

        # Calculate readiness score (0-100)
        readiness_score = min(100, (projected_end / 50) * 100)  # 50% at end = 100 score
        if not was_fully_charged:
            readiness_score *= 0.9  # Penalty for not being fully charged

        # Generate recommendations
        recommendations = []
        if not was_fully_charged:
            recommendations.append("Charge device to 100% before shift")
        if drain_rate > self.HIGH_DRAIN_RATE_THRESHOLD:
            recommendations.append("Investigate high battery drain")
        if not will_complete:
            recommendations.append(f"Battery will die around {estimated_dead_time}")

        return DeviceShiftReadiness(
            device_id=int(device_row.get("device_id", 0)),
            device_name=device_row.get("device_name"),
            current_battery=current_battery,
            drain_rate_per_hour=drain_rate,
            shift_hours=shift_duration,
            projected_end_battery=projected_end,
            will_complete_shift=will_complete,
            estimated_dead_time=estimated_dead_time,
            was_fully_charged=was_fully_charged,
            readiness_score=readiness_score,
            recommendations=recommendations,
        )

    def _get_historical_readiness(
        self,
        location_id: str,
        shift_date: datetime,
    ) -> Optional[float]:
        """Get historical shift readiness percentage."""
        # Query ShiftPerformance for the given date
        query = (
            self.db.query(
                func.count(ShiftPerformance.id).label("total"),
                func.sum(
                    func.cast(ShiftPerformance.will_complete_shift, Integer)
                ).label("ready"),
            )
            .filter(
                ShiftPerformance.tenant_id == self.tenant_id,
                ShiftPerformance.location_id == location_id,
                ShiftPerformance.shift_date == shift_date.date(),
            )
            .first()
        )

        if query and query.total and query.total > 0:
            return (query.ready or 0) / query.total * 100

        return None

    def _get_location_charging_data(
        self,
        location_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Get charging data for devices at a location."""
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
                if metadata.get("location_id") != location_id:
                    continue

                features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}

                records.append({
                    "device_id": feature.device_id,
                    "computed_at": feature.computed_at,
                    "battery_level": features.get("BatteryLevel", 50),
                    "ac_charge_count": features.get("AcChargeCount", 0),
                    "usb_charge_count": features.get("UsbChargeCount", 0),
                    "charge_pattern_good": features.get("ChargePatternGoodCount", 0),
                    "charge_pattern_bad": features.get("ChargePatternBadCount", 0),
                })
            except (json.JSONDecodeError, TypeError):
                continue

        return pd.DataFrame(records)

    def _analyze_device_charging_patterns(
        self,
        device_id: int,
        device_data: pd.DataFrame,
    ) -> List[ChargingPatternIssue]:
        """Analyze charging patterns for a single device."""
        issues: List[ChargingPatternIssue] = []

        if device_data.empty:
            return issues

        # Check for incomplete charges
        max_battery = device_data["battery_level"].max()
        if max_battery < self.FULLY_CHARGED_THRESHOLD:
            issues.append(ChargingPatternIssue(
                device_id=device_id,
                device_name=None,
                issue_type="incomplete_charge",
                description=f"Device rarely reaches full charge (max {max_battery:.0f}%)",
                frequency="daily",
                severity=InsightSeverity.MEDIUM,
                recommendation="Ensure device is charged overnight to 100%",
            ))

        # Check for USB-only charging
        total_usb = device_data["usb_charge_count"].sum()
        total_ac = device_data["ac_charge_count"].sum()
        if total_usb > 0 and total_ac == 0:
            issues.append(ChargingPatternIssue(
                device_id=device_id,
                device_name=None,
                issue_type="usb_instead_ac",
                description="Device only charges via USB (slower than AC)",
                frequency="daily",
                severity=InsightSeverity.LOW,
                recommendation="Use AC charger for faster, more complete charging",
            ))

        # Check for bad charging patterns
        bad_patterns = device_data["charge_pattern_bad"].sum()
        good_patterns = device_data["charge_pattern_good"].sum()
        if bad_patterns > good_patterns and bad_patterns > 0:
            issues.append(ChargingPatternIssue(
                device_id=device_id,
                device_name=None,
                issue_type="short_charges",
                description=f"Frequent short/interrupted charging sessions ({bad_patterns} bad vs {good_patterns} good)",
                frequency="daily",
                severity=InsightSeverity.MEDIUM,
                recommendation="Allow device to charge fully without interruption",
            ))

        return issues

    def _generate_charging_recommendations(
        self,
        incomplete: int,
        short: int,
        usb_only: int,
        missed: int,
        total: int,
    ) -> List[str]:
        """Generate location-wide charging recommendations."""
        recommendations = []

        if incomplete > total * 0.2:
            recommendations.append(
                f"{incomplete} devices ({incomplete/total*100:.0f}%) aren't reaching full charge. "
                "Review overnight charging procedures."
            )

        if short > total * 0.1:
            recommendations.append(
                f"{short} devices have frequent short charges. "
                "Consider designated charging stations."
            )

        if usb_only > total * 0.3:
            recommendations.append(
                f"{usb_only} devices only use USB charging. "
                "Provide AC chargers for faster charging."
            )

        if missed > total * 0.1:
            recommendations.append(
                f"{missed} devices missed overnight charging. "
                "Enforce end-of-shift docking procedures."
            )

        return recommendations

    def _get_device_drain_history(
        self,
        device_id: int,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Get battery drain history for a device."""
        query = (
            self.db.query(DeviceFeature)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.device_id == device_id,
                DeviceFeature.computed_at >= start_date,
                DeviceFeature.computed_at <= end_date,
            )
            .order_by(DeviceFeature.computed_at)
            .all()
        )

        if not query:
            return pd.DataFrame()

        import json

        records = []
        for feature in query:
            try:
                features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}
                records.append({
                    "computed_at": feature.computed_at,
                    "drain_per_hour": features.get("BatteryDrainPerHour", 0),
                    "battery_level": features.get("BatteryLevel", 50),
                    "hour": feature.computed_at.hour,
                    "day_of_week": feature.computed_at.weekday(),
                })
            except (json.JSONDecodeError, TypeError):
                continue

        return pd.DataFrame(records)

    def _detect_periodic_patterns(
        self,
        drain_data: pd.DataFrame,
    ) -> List[PeriodicDrainEvent]:
        """Detect periodic drain patterns in the data."""
        patterns: List[PeriodicDrainEvent] = []

        if drain_data.empty or "hour" not in drain_data.columns:
            return patterns

        # Group by hour and check for consistently high drain
        hourly_stats = drain_data.groupby("hour")["drain_per_hour"].agg(["mean", "std", "count"])

        # Find hours with consistently high drain
        overall_mean = drain_data["drain_per_hour"].mean()
        threshold = overall_mean * 1.5

        for hour, row in hourly_stats.iterrows():
            if row["mean"] > threshold and row["count"] >= 3:
                patterns.append(PeriodicDrainEvent(
                    day_of_week=None,  # Daily pattern
                    time_of_day=time(hour=int(hour)),
                    duration_minutes=60,
                    avg_drain_rate=float(row["mean"]),
                    occurrence_count=int(row["count"]),
                    confidence=min(0.9, row["count"] / 14),  # More occurrences = higher confidence
                ))

        return patterns

    def _correlate_drain_with_apps(
        self,
        device_id: int,
        drain_data: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        """Find apps correlated with high battery drain.

        Note: Would require app usage data not currently available.
        Returns empty list as placeholder.
        """
        return []

    def _correlate_drain_with_locations(
        self,
        device_id: int,
        drain_data: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        """Find locations correlated with high battery drain.

        Note: Would require location history not currently available.
        Returns empty list as placeholder.
        """
        return []

    def _get_current_device_battery(
        self,
        device_id: int,
    ) -> Optional[Dict[str, float]]:
        """Get current battery level and drain rate for a device."""
        feature = (
            self.db.query(DeviceFeature)
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
            features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}
            return {
                "battery_level": features.get("BatteryLevel", 50),
                "drain_rate": features.get("BatteryDrainPerHour", 5),
            }
        except (json.JSONDecodeError, TypeError):
            return None
