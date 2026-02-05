"""App power and crash analysis.

Carl's key requirements:
- "Crashes"
- "Apps consuming too much power (foreground time vs drain)"

This analyzer identifies problematic apps and their impact on battery and stability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from device_anomaly.database.schema import DeviceFeature
from device_anomaly.insights.categories import InsightSeverity

logger = logging.getLogger(__name__)


class AppIssueType(StrEnum):
    """Types of app issues."""

    EXCESSIVE_POWER = "excessive_power"
    FREQUENT_CRASHES = "frequent_crashes"
    ANR_PATTERN = "anr_pattern"
    BACKGROUND_DRAIN = "background_drain"
    INEFFICIENT_USAGE = "inefficient_usage"


@dataclass
class AppPowerProfile:
    """Power profile for a single app."""

    package_name: str
    app_name: str | None

    # Usage metrics
    foreground_time_hours: float
    background_time_hours: float
    total_time_hours: float

    # Power metrics
    battery_drain_percent: float
    drain_per_foreground_hour: float
    drain_per_total_hour: float
    background_drain_percent: float  # % of drain from background

    # Efficiency score (lower = more efficient)
    efficiency_score: float  # drain_per_hour normalized to 0-100

    # Comparison
    vs_app_category_average: float  # Multiplier vs category average
    devices_with_app: int


@dataclass
class AppCrashProfile:
    """Crash profile for a single app."""

    package_name: str
    app_name: str | None

    # Crash metrics
    crash_count: int
    anr_count: int
    total_incidents: int
    crash_rate_per_hour: float

    # Affected devices
    devices_affected: int
    devices_with_app: int
    device_impact_rate: float  # % of devices with app that crashed

    # Trend
    trend_direction: str  # improving, stable, worsening
    vs_last_period: float  # % change

    severity: InsightSeverity


@dataclass
class AppPowerReport:
    """App power consumption analysis."""

    tenant_id: str
    analysis_period_days: int

    # Summary
    total_apps_analyzed: int
    apps_with_power_issues: int
    total_battery_drain_from_apps: float

    # Top power consumers
    top_power_consumers: list[AppPowerProfile]

    # Most inefficient (high drain per usage)
    most_inefficient: list[AppPowerProfile]

    # Background drain offenders
    background_drain_offenders: list[AppPowerProfile]

    # App category breakdown
    drain_by_category: dict[str, float]  # category -> total drain %

    recommendations: list[str]


@dataclass
class AppCrashReport:
    """App crash analysis."""

    tenant_id: str
    analysis_period_days: int

    # Summary
    total_apps_analyzed: int
    apps_with_crashes: int
    total_crashes: int
    total_anrs: int
    avg_crashes_per_device: float

    # Crash ranking
    top_crashers: list[AppCrashProfile]

    # Most impactful (affects most devices)
    most_impactful: list[AppCrashProfile]

    # ANR offenders
    anr_offenders: list[AppCrashProfile]

    # Trend
    overall_trend: str
    trend_change_percent: float

    recommendations: list[str]


@dataclass
class AppBatteryCorrelation:
    """Correlation between app usage and battery drain for a device."""

    device_id: int
    analysis_period_days: int

    # Top apps by drain contribution
    top_drain_apps: list[tuple[str, float]]  # (app_name, drain %)

    # Correlation strength
    app_vs_battery_correlation: float  # 0-1, how much apps explain drain

    # Anomalies
    unexplained_drain_percent: float  # Drain not attributable to apps
    has_rogue_app: bool

    # Recommendations
    recommendations: list[str]


class AppPowerAnalyzer:
    """Analyzer for app power consumption and crash patterns.

    Addresses Carl's requirements:
    - App crashes
    - Apps consuming too much power (foreground time vs drain)

    Usage:
        analyzer = AppPowerAnalyzer(db_session, tenant_id)
        power_report = analyzer.analyze_app_power_efficiency(period_days=7)
        crash_report = analyzer.analyze_app_crashes(period_days=7)
    """

    # Thresholds
    HIGH_DRAIN_PER_HOUR = 5.0  # % per foreground hour = excessive
    HIGH_BACKGROUND_DRAIN = 20.0  # % of total drain from background = concerning
    HIGH_CRASH_RATE = 0.5  # crashes per hour = problematic
    HIGH_DEVICE_IMPACT = 0.1  # 10% of devices affected = significant

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

    def analyze_app_power_efficiency(
        self,
        period_days: int = 7,
    ) -> AppPowerReport:
        """Analyze app power consumption efficiency across the fleet.

        Args:
            period_days: Days of history to analyze

        Returns:
            AppPowerReport with power efficiency analysis
        """
        app_data = self._get_app_power_data(period_days)

        if app_data.empty:
            return self._empty_power_report(period_days)

        # Aggregate by app
        app_profiles = self._build_app_power_profiles(app_data)

        if not app_profiles:
            return self._empty_power_report(period_days)

        # Sort for different views
        top_consumers = sorted(app_profiles, key=lambda x: x.battery_drain_percent, reverse=True)[:10]
        most_inefficient = sorted(app_profiles, key=lambda x: x.drain_per_foreground_hour, reverse=True)[:10]
        background_offenders = sorted(app_profiles, key=lambda x: x.background_drain_percent, reverse=True)[:10]

        # Count apps with issues
        apps_with_issues = sum(
            1 for p in app_profiles
            if p.drain_per_foreground_hour > self.HIGH_DRAIN_PER_HOUR or
            p.background_drain_percent > self.HIGH_BACKGROUND_DRAIN
        )

        # Total drain from apps
        total_drain = sum(p.battery_drain_percent for p in app_profiles)

        # Category breakdown (simplified - would need app categorization)
        drain_by_category = self._calculate_drain_by_category(app_profiles)

        recommendations = self._generate_power_recommendations(
            top_consumers, most_inefficient, background_offenders, apps_with_issues
        )

        return AppPowerReport(
            tenant_id=self.tenant_id,
            analysis_period_days=period_days,
            total_apps_analyzed=len(app_profiles),
            apps_with_power_issues=apps_with_issues,
            total_battery_drain_from_apps=total_drain,
            top_power_consumers=top_consumers,
            most_inefficient=most_inefficient[:5],
            background_drain_offenders=[p for p in background_offenders if p.background_drain_percent > 10][:5],
            drain_by_category=drain_by_category,
            recommendations=recommendations,
        )

    def analyze_app_crashes(
        self,
        period_days: int = 7,
    ) -> AppCrashReport:
        """Analyze app crash patterns across the fleet.

        Args:
            period_days: Days of history to analyze

        Returns:
            AppCrashReport with crash analysis
        """
        crash_data = self._get_app_crash_data(period_days)

        if crash_data.empty:
            return self._empty_crash_report(period_days)

        # Aggregate by app
        crash_profiles = self._build_app_crash_profiles(crash_data, period_days)

        if not crash_profiles:
            return self._empty_crash_report(period_days)

        # Sort for different views
        top_crashers = sorted(crash_profiles, key=lambda x: x.total_incidents, reverse=True)[:10]
        most_impactful = sorted(crash_profiles, key=lambda x: x.device_impact_rate, reverse=True)[:10]
        anr_offenders = sorted(crash_profiles, key=lambda x: x.anr_count, reverse=True)[:10]

        # Totals
        total_crashes = sum(p.crash_count for p in crash_profiles)
        total_anrs = sum(p.anr_count for p in crash_profiles)
        apps_with_crashes = sum(1 for p in crash_profiles if p.total_incidents > 0)

        # Avg per device
        device_count = len(crash_data["device_id"].unique())
        avg_crashes = total_crashes / device_count if device_count > 0 else 0

        recommendations = self._generate_crash_recommendations(
            top_crashers, most_impactful, anr_offenders
        )

        return AppCrashReport(
            tenant_id=self.tenant_id,
            analysis_period_days=period_days,
            total_apps_analyzed=len(crash_profiles),
            apps_with_crashes=apps_with_crashes,
            total_crashes=total_crashes,
            total_anrs=total_anrs,
            avg_crashes_per_device=avg_crashes,
            top_crashers=top_crashers[:5],
            most_impactful=[p for p in most_impactful if p.device_impact_rate > 0.05][:5],
            anr_offenders=[p for p in anr_offenders if p.anr_count > 0][:5],
            overall_trend="stable",  # Would need historical comparison
            trend_change_percent=0,
            recommendations=recommendations,
        )

    def correlate_app_battery_drain(
        self,
        device_id: int,
        period_days: int = 7,
    ) -> AppBatteryCorrelation | None:
        """Correlate app usage with battery drain for a specific device.

        Args:
            device_id: Device to analyze
            period_days: Period to analyze

        Returns:
            AppBatteryCorrelation or None if insufficient data
        """
        device_data = self._get_device_app_data(device_id, period_days)

        if device_data.empty:
            return None

        # Calculate app contributions to drain
        app_drains = self._calculate_device_app_drains(device_data)

        if not app_drains:
            return None

        # Total battery drain
        device_data["total_battery_drain"].sum() if "total_battery_drain" in device_data.columns else 100

        # Top apps by drain
        top_apps = sorted(app_drains.items(), key=lambda x: x[1], reverse=True)[:5]

        # Calculate explained vs unexplained drain
        explained_drain = sum(drain for _, drain in top_apps)
        unexplained_percent = max(0, 100 - explained_drain)

        # Check for rogue app
        has_rogue = any(drain > 30 for _, drain in top_apps)  # Any app using >30%

        recommendations = []
        if has_rogue:
            rogue_app = top_apps[0][0]
            recommendations.append(f"'{rogue_app}' is consuming excessive battery. Consider uninstalling or restricting.")

        if unexplained_percent > 30:
            recommendations.append("Significant unexplained battery drain. Check for system issues or hidden processes.")

        return AppBatteryCorrelation(
            device_id=device_id,
            analysis_period_days=period_days,
            top_drain_apps=top_apps,
            app_vs_battery_correlation=1 - (unexplained_percent / 100),
            unexplained_drain_percent=unexplained_percent,
            has_rogue_app=has_rogue,
            recommendations=recommendations,
        )

    def get_app_impact_score(
        self,
        package_name: str,
        period_days: int = 7,
    ) -> dict[str, Any]:
        """Get overall impact score for an app.

        Args:
            package_name: App package name
            period_days: Period to analyze

        Returns:
            Dict with impact metrics and score
        """
        app_data = self._get_app_power_data(period_days)
        crash_data = self._get_app_crash_data(period_days)

        if app_data.empty and crash_data.empty:
            return {"package_name": package_name, "impact_score": 0, "status": "no_data"}

        # Filter to this app
        power_data = app_data[app_data["package_name"] == package_name] if not app_data.empty else pd.DataFrame()
        crash_app_data = crash_data[crash_data["package_name"] == package_name] if not crash_data.empty else pd.DataFrame()

        total_drain = power_data["battery_drain"].sum() if not power_data.empty else 0
        total_crashes = crash_app_data["crash_count"].sum() if not crash_app_data.empty else 0
        total_anrs = crash_app_data["anr_count"].sum() if not crash_app_data.empty else 0
        devices_affected = len(power_data["device_id"].unique()) if not power_data.empty else 0

        # Calculate impact score (0-100)
        # Weight: 40% power, 40% crashes, 20% ANRs
        power_score = min(100, total_drain / 10 * 100)  # 10% drain = max score
        crash_score = min(100, total_crashes / 50 * 100)  # 50 crashes = max score
        anr_score = min(100, total_anrs / 20 * 100)  # 20 ANRs = max score

        impact_score = power_score * 0.4 + crash_score * 0.4 + anr_score * 0.2

        return {
            "package_name": package_name,
            "total_drain_percent": float(total_drain),
            "total_crashes": int(total_crashes),
            "total_anrs": int(total_anrs),
            "devices_affected": devices_affected,
            "impact_score": float(impact_score),
            "status": "high_impact" if impact_score > 50 else "normal",
        }

    # Private helper methods

    def _get_app_power_data(self, period_days: int) -> pd.DataFrame:
        """Get app power consumption data."""
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
                features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}

                # Get app-level battery data if available
                # This is simplified - actual implementation would parse app-level metrics
                total_app_drain = features.get("TotalBatteryAppDrain", 0)
                app_foreground = features.get("AppForegroundTime", 0)
                total_drain = features.get("TotalBatteryLevelDrop", 0)

                records.append({
                    "device_id": feature.device_id,
                    "computed_at": feature.computed_at,
                    "package_name": "system_apps",  # Simplified - would have per-app data
                    "battery_drain": total_app_drain,
                    "foreground_time": app_foreground,
                    "background_time": 0,  # Would need per-app data
                    "total_battery_drain": total_drain,
                })
            except (json.JSONDecodeError, TypeError):
                continue

        return pd.DataFrame(records)

    def _get_app_crash_data(self, period_days: int) -> pd.DataFrame:
        """Get app crash data."""
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
                features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}

                # Simplified - would have per-app crash data
                crash_count = features.get("CrashCount", 0)
                anr_count = features.get("ANRCount", 0)

                if crash_count > 0 or anr_count > 0:
                    records.append({
                        "device_id": feature.device_id,
                        "computed_at": feature.computed_at,
                        "package_name": "apps",  # Simplified
                        "crash_count": crash_count,
                        "anr_count": anr_count,
                    })
            except (json.JSONDecodeError, TypeError):
                continue

        return pd.DataFrame(records)

    def _get_device_app_data(self, device_id: int, period_days: int) -> pd.DataFrame:
        """Get app data for a specific device."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        query = (
            self.db.query(DeviceFeature)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.device_id == device_id,
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
                features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}

                records.append({
                    "device_id": feature.device_id,
                    "computed_at": feature.computed_at,
                    "total_battery_drain": features.get("TotalBatteryLevelDrop", 0),
                    "app_drain": features.get("TotalBatteryAppDrain", 0),
                    "foreground_time": features.get("AppForegroundTime", 0),
                })
            except (json.JSONDecodeError, TypeError):
                continue

        return pd.DataFrame(records)

    def _build_app_power_profiles(self, app_data: pd.DataFrame) -> list[AppPowerProfile]:
        """Build power profiles from app data."""
        if app_data.empty:
            return []

        profiles = []

        # Aggregate by package name
        agg = app_data.groupby("package_name").agg({
            "battery_drain": "sum",
            "foreground_time": "sum",
            "background_time": "sum",
            "device_id": "nunique",
        }).reset_index()

        for _, row in agg.iterrows():
            foreground_hours = row["foreground_time"] / 60 if row["foreground_time"] > 0 else 0.01
            background_hours = row["background_time"] / 60 if row["background_time"] > 0 else 0
            total_hours = foreground_hours + background_hours

            drain_per_fg = row["battery_drain"] / foreground_hours if foreground_hours > 0 else 0
            drain_per_total = row["battery_drain"] / total_hours if total_hours > 0 else 0

            # Background drain percent
            bg_drain_pct = (background_hours / total_hours * 100) if total_hours > 0 else 0

            # Efficiency score (higher = worse)
            efficiency = min(100, drain_per_fg * 10)

            profiles.append(AppPowerProfile(
                package_name=row["package_name"],
                app_name=row["package_name"],  # Would need mapping
                foreground_time_hours=foreground_hours,
                background_time_hours=background_hours,
                total_time_hours=total_hours,
                battery_drain_percent=row["battery_drain"],
                drain_per_foreground_hour=drain_per_fg,
                drain_per_total_hour=drain_per_total,
                background_drain_percent=bg_drain_pct,
                efficiency_score=efficiency,
                vs_app_category_average=1.0,  # Would need category data
                devices_with_app=int(row["device_id"]),
            ))

        return profiles

    def _build_app_crash_profiles(
        self,
        crash_data: pd.DataFrame,
        period_days: int,
    ) -> list[AppCrashProfile]:
        """Build crash profiles from crash data."""
        if crash_data.empty:
            return []

        profiles = []

        agg = crash_data.groupby("package_name").agg({
            "crash_count": "sum",
            "anr_count": "sum",
            "device_id": "nunique",
        }).reset_index()

        total_devices = len(crash_data["device_id"].unique())
        total_hours = period_days * 24

        for _, row in agg.iterrows():
            crash_count = int(row["crash_count"])
            anr_count = int(row["anr_count"])
            total_incidents = crash_count + anr_count
            devices_affected = int(row["device_id"])

            crash_rate = crash_count / total_hours if total_hours > 0 else 0
            impact_rate = devices_affected / total_devices if total_devices > 0 else 0

            # Severity based on impact
            if impact_rate > 0.2:
                severity = InsightSeverity.CRITICAL
            elif impact_rate > 0.1:
                severity = InsightSeverity.HIGH
            elif total_incidents > 10:
                severity = InsightSeverity.MEDIUM
            else:
                severity = InsightSeverity.LOW

            profiles.append(AppCrashProfile(
                package_name=row["package_name"],
                app_name=row["package_name"],
                crash_count=crash_count,
                anr_count=anr_count,
                total_incidents=total_incidents,
                crash_rate_per_hour=crash_rate,
                devices_affected=devices_affected,
                devices_with_app=devices_affected,  # Simplified
                device_impact_rate=impact_rate,
                trend_direction="stable",
                vs_last_period=0,
                severity=severity,
            ))

        return profiles

    def _calculate_drain_by_category(
        self,
        profiles: list[AppPowerProfile],
    ) -> dict[str, float]:
        """Calculate drain by app category."""
        # Simplified categorization
        categories = {
            "System": 0,
            "Communication": 0,
            "Productivity": 0,
            "Entertainment": 0,
            "Other": 0,
        }

        for profile in profiles:
            # Simple heuristic - would use actual categorization
            if "system" in profile.package_name.lower():
                categories["System"] += profile.battery_drain_percent
            elif any(x in profile.package_name.lower() for x in ["mail", "chat", "message"]):
                categories["Communication"] += profile.battery_drain_percent
            elif any(x in profile.package_name.lower() for x in ["office", "docs", "sheets"]):
                categories["Productivity"] += profile.battery_drain_percent
            elif any(x in profile.package_name.lower() for x in ["game", "video", "music"]):
                categories["Entertainment"] += profile.battery_drain_percent
            else:
                categories["Other"] += profile.battery_drain_percent

        return categories

    def _calculate_device_app_drains(
        self,
        device_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate app drain contributions for a device."""
        if device_data.empty:
            return {}

        # Simplified - would have per-app breakdown
        total_app_drain = device_data["app_drain"].sum()
        total_drain = device_data["total_battery_drain"].sum()

        if total_drain == 0:
            return {}

        # Return simplified breakdown
        app_contribution = (total_app_drain / total_drain * 100) if total_drain > 0 else 0
        return {
            "apps": app_contribution,
            "system": 100 - app_contribution,
        }

    def _generate_power_recommendations(
        self,
        top_consumers: list[AppPowerProfile],
        inefficient: list[AppPowerProfile],
        background: list[AppPowerProfile],
        apps_with_issues: int,
    ) -> list[str]:
        """Generate power consumption recommendations."""
        recommendations = []

        if top_consumers and top_consumers[0].battery_drain_percent > 20:
            app = top_consumers[0]
            recommendations.append(
                f"'{app.app_name}' consumes {app.battery_drain_percent:.0f}% of battery. "
                "Review if this app is necessary or can be optimized."
            )

        if inefficient:
            worst = inefficient[0]
            if worst.drain_per_foreground_hour > self.HIGH_DRAIN_PER_HOUR:
                recommendations.append(
                    f"'{worst.app_name}' uses {worst.drain_per_foreground_hour:.1f}%/hour of foreground use. "
                    "Check for app updates or alternatives."
                )

        if background:
            bg_offenders = [p for p in background if p.background_drain_percent > self.HIGH_BACKGROUND_DRAIN]
            if bg_offenders:
                recommendations.append(
                    f"{len(bg_offenders)} app(s) have high background drain. "
                    "Consider restricting background activity."
                )

        return recommendations

    def _generate_crash_recommendations(
        self,
        top_crashers: list[AppCrashProfile],
        impactful: list[AppCrashProfile],
        anr_offenders: list[AppCrashProfile],
    ) -> list[str]:
        """Generate crash recommendations."""
        recommendations = []

        if top_crashers and top_crashers[0].crash_count > 10:
            app = top_crashers[0]
            recommendations.append(
                f"'{app.app_name}' crashed {app.crash_count} times. "
                "Check for app updates or report to developer."
            )

        if impactful:
            critical = [p for p in impactful if p.severity == InsightSeverity.CRITICAL]
            if critical:
                recommendations.append(
                    f"{len(critical)} app(s) are affecting >20% of devices. "
                    "Priority fix required."
                )

        if anr_offenders:
            high_anr = [p for p in anr_offenders if p.anr_count > 5]
            if high_anr:
                recommendations.append(
                    f"{len(high_anr)} app(s) show frequent ANR (Not Responding) events. "
                    "May indicate performance issues."
                )

        return recommendations

    def _empty_power_report(self, period_days: int) -> AppPowerReport:
        """Return empty power report."""
        return AppPowerReport(
            tenant_id=self.tenant_id,
            analysis_period_days=period_days,
            total_apps_analyzed=0,
            apps_with_power_issues=0,
            total_battery_drain_from_apps=0,
            top_power_consumers=[],
            most_inefficient=[],
            background_drain_offenders=[],
            drain_by_category={},
            recommendations=[],
        )

    def _empty_crash_report(self, period_days: int) -> AppCrashReport:
        """Return empty crash report."""
        return AppCrashReport(
            tenant_id=self.tenant_id,
            analysis_period_days=period_days,
            total_apps_analyzed=0,
            apps_with_crashes=0,
            total_crashes=0,
            total_anrs=0,
            avg_crashes_per_device=0,
            top_crashers=[],
            most_impactful=[],
            anr_offenders=[],
            overall_trend="stable",
            trend_change_percent=0,
            recommendations=[],
        )
