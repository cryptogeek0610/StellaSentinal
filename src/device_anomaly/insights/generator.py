"""Insight generator orchestrating all insight components.

This is the main entry point for generating customer-facing insights.
It coordinates all specialized analyzers and produces prioritized,
actionable insights aligned with Carl's vision.

Carl's direction:
- "XSight has the data. XSight needs the story."
- Pre-interpreted, contextualized, and actionable insights
- Comparisons matter more than absolutes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from device_anomaly.database.schema import AggregatedInsight, DeviceFeature
from device_anomaly.insights.app_power import AppPowerAnalyzer
from device_anomaly.insights.battery_shift import BatteryShiftAnalyzer
from device_anomaly.insights.categories import InsightCategory, InsightSeverity, EntityType
from device_anomaly.insights.classifier import ClassifiedInsight, InsightClassifier
from device_anomaly.insights.comparisons import ComparisonEngine
from device_anomaly.insights.device_abuse import DeviceAbuseAnalyzer
from device_anomaly.insights.entities import EntityAggregator, LocationInsight
from device_anomaly.insights.location_mapper import LocationMapper
from device_anomaly.insights.network_patterns import NetworkPatternAnalyzer
from device_anomaly.insights.templates import get_template, render_headline, render_impact

logger = logging.getLogger(__name__)


@dataclass
class CustomerInsight:
    """A customer-facing insight ready for display.

    This is the final output format - fully rendered with business language,
    contextual comparisons, and actionable recommendations.
    """

    insight_id: str
    category: InsightCategory
    severity: InsightSeverity

    # Customer-facing content
    headline: str
    impact_statement: str
    comparison_context: str
    recommended_actions: List[str]

    # Metadata
    entity_type: EntityType
    entity_id: str
    entity_name: str
    affected_device_count: int

    # Technical details (for drill-down)
    primary_metric: str
    primary_value: float
    threshold_exceeded: Optional[float]
    confidence_score: float

    # Trend
    trend_direction: str  # improving, stable, worsening
    trend_change_percent: Optional[float]

    # Timestamps
    detected_at: datetime
    last_updated: datetime

    # For ticket creation
    ticket_summary: Optional[str] = None
    ticket_description: Optional[str] = None


@dataclass
class DailyInsightDigest:
    """Daily digest of prioritized insights."""

    tenant_id: str
    digest_date: date
    generated_at: datetime

    # Summary counts
    total_insights: int
    critical_count: int
    high_count: int
    medium_count: int

    # Top priority insights
    top_insights: List[CustomerInsight]

    # By domain
    battery_insights: List[CustomerInsight]
    network_insights: List[CustomerInsight]
    device_insights: List[CustomerInsight]
    app_insights: List[CustomerInsight]
    location_insights: List[CustomerInsight]

    # Trending issues (getting worse)
    trending_issues: List[CustomerInsight]

    # New issues (first seen today)
    new_issues: List[CustomerInsight]

    # Executive summary
    executive_summary: str

    # All insights combined (required for save_insights_to_db)
    all_insights: List[CustomerInsight] = field(default_factory=list)


@dataclass
class TrendingInsight:
    """An insight that's getting worse over time."""

    insight: CustomerInsight
    trend_period_days: int
    change_percent: float
    predicted_severity_change: Optional[str]  # e.g., "may become critical in 3 days"


@dataclass
class LocationInsightReport:
    """Comprehensive insight report for a location."""

    location_id: str
    location_name: str
    report_date: date

    # Summary
    total_devices: int
    devices_with_issues: int
    issue_rate: float

    # Shift readiness (if applicable)
    shift_readiness: Optional[Dict[str, Any]]

    # All insights for this location
    insights: List[CustomerInsight]

    # Top issues
    top_issues: List[Tuple[InsightCategory, int]]

    # Comparison to other locations
    rank_among_locations: int
    better_than_percent: float

    # Recommendations
    recommendations: List[str]


class InsightGenerator:
    """Main orchestrator for generating customer-facing insights.

    Coordinates all specialized analyzers to produce prioritized,
    actionable insights for customers.

    Usage:
        generator = InsightGenerator(db_session, tenant_id)
        digest = generator.generate_daily_insights(date.today())
        location_report = generator.generate_location_insights("warehouse_1", 7)
    """

    def __init__(
        self,
        db_session: Session,
        tenant_id: str,
    ):
        """Initialize the insight generator.

        Args:
            db_session: SQLAlchemy database session
            tenant_id: Tenant ID for multi-tenant filtering
        """
        self.db = db_session
        self.tenant_id = tenant_id

        # Initialize all components
        self.location_mapper = LocationMapper(db_session, tenant_id)
        self.classifier = InsightClassifier()
        self.comparison_engine = ComparisonEngine(db_session, tenant_id)
        self.entity_aggregator = EntityAggregator(db_session, tenant_id, self.location_mapper)

        # Specialized analyzers
        self.battery_analyzer = BatteryShiftAnalyzer(db_session, tenant_id, self.location_mapper)
        self.network_analyzer = NetworkPatternAnalyzer(db_session, tenant_id)
        self.abuse_analyzer = DeviceAbuseAnalyzer(db_session, tenant_id)
        self.app_analyzer = AppPowerAnalyzer(db_session, tenant_id)

    def generate_daily_insights(
        self,
        insight_date: date,
        period_days: int = 7,
    ) -> DailyInsightDigest:
        """Generate the daily insight digest.

        Args:
            insight_date: Date for the digest
            period_days: Days of data to analyze

        Returns:
            DailyInsightDigest with prioritized insights
        """
        logger.info(f"Generating daily insights for {insight_date}")

        # Collect insights from all analyzers
        all_insights: List[CustomerInsight] = []

        # Battery insights
        battery_insights = self._generate_battery_insights(insight_date, period_days)
        all_insights.extend(battery_insights)

        # Network insights
        network_insights = self._generate_network_insights(period_days)
        all_insights.extend(network_insights)

        # Device abuse insights
        device_insights = self._generate_device_insights(period_days)
        all_insights.extend(device_insights)

        # App insights
        app_insights = self._generate_app_insights(period_days)
        all_insights.extend(app_insights)

        # Location insights
        location_insights = self._generate_location_aggregated_insights(period_days)
        all_insights.extend(location_insights)

        # Sort by severity and priority
        all_insights.sort(key=lambda x: self._get_priority_score(x), reverse=True)

        # Count by severity
        critical = [i for i in all_insights if i.severity == InsightSeverity.CRITICAL]
        high = [i for i in all_insights if i.severity == InsightSeverity.HIGH]
        medium = [i for i in all_insights if i.severity == InsightSeverity.MEDIUM]

        # Identify trending issues
        trending = [i for i in all_insights if i.trend_direction == "worsening"]

        # Identify new issues (detected today)
        today = datetime.combine(insight_date, datetime.min.time())
        new_issues = [i for i in all_insights if i.detected_at >= today]

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            all_insights, critical, high, trending, new_issues
        )

        return DailyInsightDigest(
            tenant_id=self.tenant_id,
            digest_date=insight_date,
            generated_at=datetime.utcnow(),
            total_insights=len(all_insights),
            critical_count=len(critical),
            high_count=len(high),
            medium_count=len(medium),
            top_insights=all_insights[:10],
            battery_insights=battery_insights[:5],
            network_insights=network_insights[:5],
            device_insights=device_insights[:5],
            app_insights=app_insights[:5],
            location_insights=location_insights[:5],
            trending_issues=[i for i in trending[:5]],
            new_issues=new_issues[:5],
            executive_summary=executive_summary,
            all_insights=all_insights,
        )

    def generate_device_insight(
        self,
        device_id: int,
        anomaly_score: float,
        features: Dict[str, float],
        feature_contributions: Dict[str, float],
        device_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CustomerInsight]:
        """Generate a customer insight from a device anomaly.

        Args:
            device_id: Device ID
            anomaly_score: Anomaly detection score
            features: Device feature values
            feature_contributions: Feature contribution to anomaly
            device_context: Optional device metadata

        Returns:
            CustomerInsight or None if not significant
        """
        device_context = device_context or {}

        # Classify the anomaly
        classified = self.classifier.classify(
            features, device_context, anomaly_score, feature_contributions
        )

        if not classified:
            return None

        # Take the highest confidence classification
        top_classification = classified[0]

        # Get comparison context
        primary_metric = top_classification.primary_metric or "AnomalyScore"
        primary_value = top_classification.primary_value or anomaly_score

        comparison = self.comparison_engine.get_comparison_context(
            device_id, primary_metric, primary_value, device_context
        )

        # Render template
        template = get_template(top_classification.category)

        # Build template context
        template_context = {
            "device_id": device_id,
            "device_name": device_context.get("device_name", f"Device {device_id}"),
            "location": device_context.get("location_name", "Unknown"),
            "value": primary_value,
            "metric": primary_metric,
            "manufacturer": device_context.get("Manufacturer", "Unknown"),
            "model": device_context.get("Model", "Unknown"),
            **{k: v for k, v in features.items()},
        }

        headline = render_headline(top_classification.category, template_context)
        impact = render_impact(top_classification.category, template_context)

        return CustomerInsight(
            insight_id=f"{device_id}_{top_classification.category.value}_{datetime.utcnow().strftime('%Y%m%d')}",
            category=top_classification.category,
            severity=top_classification.severity,
            headline=headline,
            impact_statement=impact,
            comparison_context=comparison.get("summary", ""),
            recommended_actions=template.actions if template else [],
            entity_type=EntityType.DEVICE,
            entity_id=str(device_id),
            entity_name=device_context.get("device_name", f"Device {device_id}"),
            affected_device_count=1,
            primary_metric=primary_metric,
            primary_value=primary_value,
            threshold_exceeded=top_classification.threshold,
            confidence_score=top_classification.confidence,
            trend_direction="stable",
            trend_change_percent=None,
            detected_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
        )

    def generate_location_insights(
        self,
        location_id: str,
        period_days: int = 7,
    ) -> LocationInsightReport:
        """Generate comprehensive insights for a location.

        Args:
            location_id: Location to analyze
            period_days: Days of data to analyze

        Returns:
            LocationInsightReport with location-specific insights
        """
        # Get location metadata
        location = self.location_mapper.get_location(location_id)
        location_name = location.location_name if location else location_id

        # Collect all insights for this location
        insights: List[CustomerInsight] = []

        # Battery/shift insights
        shift_readiness = None
        shift_report = self.battery_analyzer.analyze_shift_readiness(
            location_id, datetime.utcnow()
        )
        if shift_report:
            shift_readiness = {
                "readiness_percentage": shift_report.readiness_percentage,
                "devices_ready": shift_report.devices_ready,
                "devices_at_risk": shift_report.devices_at_risk,
                "avg_battery": shift_report.avg_battery_at_start,
            }

            # Convert to insights
            for device in shift_report.device_readiness:
                if not device.will_complete_shift:
                    insights.append(self._device_readiness_to_insight(device, location_name))

        # Network insights for this location
        wifi_report = self.network_analyzer.analyze_wifi_roaming(location_id, period_days)
        if wifi_report:
            for issue in wifi_report.issues[:5]:
                insights.append(self._wifi_issue_to_insight(issue, location_name))

        # Device abuse for this location
        drop_report = self.abuse_analyzer.analyze_drops(period_days, "location", location_id)
        if drop_report:
            for device in drop_report.worst_devices[:5]:
                insights.append(self._drop_device_to_insight(device, location_name))

        # Sort by priority
        insights.sort(key=lambda x: self._get_priority_score(x), reverse=True)

        # Count issues by category
        category_counts: Dict[InsightCategory, int] = {}
        for insight in insights:
            category_counts[insight.category] = category_counts.get(insight.category, 0) + 1

        top_issues = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Get devices at location
        devices_df = self._get_location_devices(location_id, period_days)
        total_devices = len(devices_df["device_id"].unique()) if not devices_df.empty else 0
        devices_with_issues = len(set(i.entity_id for i in insights))

        # Location ranking
        all_locations = self.location_mapper.get_all_locations()
        rank = 1  # Would calculate actual rank

        # Generate recommendations
        recommendations = self._generate_location_recommendations(
            insights, shift_readiness, top_issues
        )

        return LocationInsightReport(
            location_id=location_id,
            location_name=location_name,
            report_date=date.today(),
            total_devices=total_devices,
            devices_with_issues=devices_with_issues,
            issue_rate=devices_with_issues / total_devices if total_devices > 0 else 0,
            shift_readiness=shift_readiness,
            insights=insights[:20],
            top_issues=top_issues,
            rank_among_locations=rank,
            better_than_percent=50,  # Would calculate
            recommendations=recommendations,
        )

    def get_trending_issues(
        self,
        lookback_days: int = 14,
        limit: int = 10,
    ) -> List[TrendingInsight]:
        """Get issues that are trending worse.

        Args:
            lookback_days: Days to analyze for trends
            limit: Maximum trending issues to return

        Returns:
            List of trending insights
        """
        # Query historical aggregated insights
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        historical = (
            self.db.query(AggregatedInsight)
            .filter(
                AggregatedInsight.tenant_id == self.tenant_id,
                AggregatedInsight.computed_at >= cutoff,
                AggregatedInsight.trend_direction == "worsening",
            )
            .order_by(AggregatedInsight.confidence_score.desc())
            .limit(limit)
            .all()
        )

        trending: List[TrendingInsight] = []

        for record in historical:
            insight = self._aggregated_to_customer_insight(record)
            if insight:
                change_pct = 0
                if record.baseline_value and record.current_value:
                    change_pct = ((record.current_value - record.baseline_value) /
                                  record.baseline_value * 100)

                trending.append(TrendingInsight(
                    insight=insight,
                    trend_period_days=lookback_days,
                    change_percent=change_pct,
                    predicted_severity_change=None,
                ))

        return trending

    def save_insights_to_db(
        self,
        insights: List[CustomerInsight],
    ) -> int:
        """Save generated insights to the database.

        Args:
            insights: List of insights to save

        Returns:
            Number of insights saved
        """
        saved = 0

        for insight in insights:
            existing = (
                self.db.query(AggregatedInsight)
                .filter(
                    AggregatedInsight.tenant_id == self.tenant_id,
                    AggregatedInsight.entity_type == insight.entity_type.value,
                    AggregatedInsight.entity_id == insight.entity_id,
                    AggregatedInsight.insight_category == insight.category.value,
                )
                .first()
            )

            if existing:
                # Update existing
                existing.headline = insight.headline
                existing.severity = insight.severity.value
                existing.affected_device_count = insight.affected_device_count
                existing.trend_direction = insight.trend_direction
                existing.confidence_score = insight.confidence_score
                existing.current_value = insight.primary_value
                existing.computed_at = datetime.utcnow()
            else:
                # Create new
                new_insight = AggregatedInsight(
                    tenant_id=self.tenant_id,
                    entity_type=insight.entity_type.value,
                    entity_id=insight.entity_id,
                    entity_name=insight.entity_name,
                    insight_category=insight.category.value,
                    severity=insight.severity.value,
                    headline=insight.headline,
                    impact_statement=insight.impact_statement,
                    comparison_context=insight.comparison_context,
                    affected_device_count=insight.affected_device_count,
                    trend_direction=insight.trend_direction,
                    confidence_score=insight.confidence_score,
                    current_value=insight.primary_value,
                    computed_at=datetime.utcnow(),
                    is_active=True,
                )
                self.db.add(new_insight)

            saved += 1

        self.db.commit()
        return saved

    # Private methods for generating insights from each analyzer

    def _generate_battery_insights(
        self,
        insight_date: date,
        period_days: int,
    ) -> List[CustomerInsight]:
        """Generate battery-related insights."""
        insights: List[CustomerInsight] = []

        # Get all locations
        locations = self.location_mapper.get_all_locations()

        for location in locations[:10]:  # Limit for performance
            # Shift readiness
            report = self.battery_analyzer.analyze_shift_readiness(
                location.location_id,
                datetime.combine(insight_date, datetime.min.time())
            )

            if report and report.devices_at_risk > 0:
                insights.append(CustomerInsight(
                    insight_id=f"battery_shift_{location.location_id}_{insight_date}",
                    category=InsightCategory.BATTERY_SHIFT_FAILURE,
                    severity=InsightSeverity.HIGH if report.devices_at_risk > 5 else InsightSeverity.MEDIUM,
                    headline=f"{report.devices_at_risk} devices at {location.location_name} won't last shift",
                    impact_statement=f"At current drain rates, these devices will die before shift end",
                    comparison_context=f"{report.readiness_percentage:.0f}% shift readiness",
                    recommended_actions=["Charge devices before shift", "Check drain rates"],
                    entity_type=EntityType.LOCATION,
                    entity_id=location.location_id,
                    entity_name=location.location_name,
                    affected_device_count=report.devices_at_risk,
                    primary_metric="ShiftReadiness",
                    primary_value=report.readiness_percentage,
                    threshold_exceeded=80,
                    confidence_score=0.8,
                    trend_direction="stable",
                    trend_change_percent=report.vs_last_week_readiness,
                    detected_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                ))

            # Charging patterns
            charging = self.battery_analyzer.analyze_charging_patterns(
                location.location_id, period_days
            )

            if charging and charging.devices_with_issues > 0:
                insights.append(CustomerInsight(
                    insight_id=f"battery_charging_{location.location_id}_{insight_date}",
                    category=InsightCategory.BATTERY_CHARGE_PATTERN,
                    severity=InsightSeverity.MEDIUM,
                    headline=f"{charging.devices_with_issues} devices at {location.location_name} have charging issues",
                    impact_statement="Poor charging patterns lead to devices not ready for shift",
                    comparison_context=f"{charging.issue_rate*100:.0f}% of devices affected",
                    recommended_actions=charging.recommendations,
                    entity_type=EntityType.LOCATION,
                    entity_id=location.location_id,
                    entity_name=location.location_name,
                    affected_device_count=charging.devices_with_issues,
                    primary_metric="ChargingIssueRate",
                    primary_value=charging.issue_rate * 100,
                    threshold_exceeded=10,
                    confidence_score=0.7,
                    trend_direction="stable",
                    trend_change_percent=None,
                    detected_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                ))

        return insights

    def _generate_network_insights(self, period_days: int) -> List[CustomerInsight]:
        """Generate network-related insights."""
        insights: List[CustomerInsight] = []

        # WiFi roaming analysis
        wifi_report = self.network_analyzer.analyze_wifi_roaming(None, period_days)

        if wifi_report and wifi_report.devices_with_roaming_issues > 0:
            insights.append(CustomerInsight(
                insight_id=f"wifi_roaming_{self.tenant_id}_{date.today()}",
                category=InsightCategory.WIFI_AP_HOPPING,
                severity=InsightSeverity.MEDIUM,
                headline=f"{wifi_report.devices_with_roaming_issues} devices have WiFi roaming issues",
                impact_statement="Excessive AP switching causes connectivity interruptions",
                comparison_context=f"Fleet average: {wifi_report.avg_aps_per_device:.1f} APs per device",
                recommended_actions=wifi_report.recommendations,
                entity_type=EntityType.MANUFACTURER,  # Fleet-wide
                entity_id=self.tenant_id,
                entity_name="Fleet",
                affected_device_count=wifi_report.devices_with_roaming_issues,
                primary_metric="APHoppingRate",
                primary_value=wifi_report.avg_aps_per_device,
                threshold_exceeded=5,
                confidence_score=0.7,
                trend_direction="stable",
                trend_change_percent=None,
                detected_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            ))

        # Hidden devices
        hidden_report = self.network_analyzer.detect_hidden_devices(period_days)

        if hidden_report and hidden_report.devices_highly_suspicious > 0:
            insights.append(CustomerInsight(
                insight_id=f"hidden_devices_{self.tenant_id}_{date.today()}",
                category=InsightCategory.DEVICE_HIDDEN_PATTERN,
                severity=InsightSeverity.HIGH,
                headline=f"{hidden_report.devices_highly_suspicious} devices may be hidden or off-site",
                impact_statement="These devices show suspicious offline patterns",
                comparison_context=f"{hidden_report.devices_flagged} total flagged for review",
                recommended_actions=hidden_report.recommendations,
                entity_type=EntityType.DEVICE,
                entity_id=self.tenant_id,
                entity_name="Fleet",
                affected_device_count=hidden_report.devices_highly_suspicious,
                primary_metric="HiddenScore",
                primary_value=100,
                threshold_exceeded=80,
                confidence_score=0.6,
                trend_direction="stable",
                trend_change_percent=None,
                detected_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            ))

        return insights

    def _generate_device_insights(self, period_days: int) -> List[CustomerInsight]:
        """Generate device abuse insights."""
        insights: List[CustomerInsight] = []

        # Drop analysis
        drop_report = self.abuse_analyzer.analyze_drops(period_days)

        if drop_report and drop_report.devices_with_excessive_drops > 0:
            insights.append(CustomerInsight(
                insight_id=f"excessive_drops_{self.tenant_id}_{date.today()}",
                category=InsightCategory.EXCESSIVE_DROPS,
                severity=InsightSeverity.HIGH,
                headline=f"{drop_report.devices_with_excessive_drops} devices have excessive drops",
                impact_statement=f"Total {drop_report.total_drops} drops in {period_days} days",
                comparison_context=f"Average {drop_report.avg_drops_per_device:.1f} drops per device",
                recommended_actions=drop_report.recommendations,
                entity_type=EntityType.DEVICE,
                entity_id=self.tenant_id,
                entity_name="Fleet",
                affected_device_count=drop_report.devices_with_excessive_drops,
                primary_metric="DropCount",
                primary_value=drop_report.total_drops,
                threshold_exceeded=5,
                confidence_score=0.8,
                trend_direction=drop_report.trend_direction,
                trend_change_percent=drop_report.trend_change_percent,
                detected_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            ))

        # Reboot analysis
        reboot_report = self.abuse_analyzer.analyze_reboots(period_days)

        if reboot_report and reboot_report.devices_with_excessive_reboots > 0:
            insights.append(CustomerInsight(
                insight_id=f"excessive_reboots_{self.tenant_id}_{date.today()}",
                category=InsightCategory.EXCESSIVE_REBOOTS,
                severity=InsightSeverity.HIGH,
                headline=f"{reboot_report.devices_with_excessive_reboots} devices have excessive reboots",
                impact_statement=f"Total {reboot_report.total_reboots} reboots ({reboot_report.crash_induced_reboots_percent:.0f}% crash-related)",
                comparison_context=f"Average {reboot_report.avg_reboots_per_device:.1f} reboots per device",
                recommended_actions=reboot_report.recommendations,
                entity_type=EntityType.DEVICE,
                entity_id=self.tenant_id,
                entity_name="Fleet",
                affected_device_count=reboot_report.devices_with_excessive_reboots,
                primary_metric="RebootCount",
                primary_value=reboot_report.total_reboots,
                threshold_exceeded=3,
                confidence_score=0.8,
                trend_direction="stable",
                trend_change_percent=None,
                detected_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            ))

        # Problem combinations
        combo_report = self.abuse_analyzer.identify_problem_combinations(period_days)

        for combo in combo_report.problem_combinations[:3]:
            insights.append(CustomerInsight(
                insight_id=f"problem_combo_{combo.cohort_id}_{date.today()}",
                category=InsightCategory.PROBLEM_COMBINATION,
                severity=combo.severity,
                headline=combo.description,
                impact_statement=f"{combo.device_count} devices affected",
                comparison_context=f"{combo.vs_fleet_issue_rate:.1f}x fleet average issues",
                recommended_actions=combo.recommendations,
                entity_type=EntityType.COHORT,
                entity_id=combo.cohort_id,
                entity_name=f"{combo.manufacturer} {combo.model}",
                affected_device_count=combo.device_count,
                primary_metric="IssueMultiplier",
                primary_value=combo.vs_fleet_issue_rate,
                threshold_exceeded=2.0,
                confidence_score=0.7 if combo.is_statistically_significant else 0.4,
                trend_direction="stable",
                trend_change_percent=None,
                detected_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            ))

        return insights

    def _generate_app_insights(self, period_days: int) -> List[CustomerInsight]:
        """Generate app-related insights."""
        insights: List[CustomerInsight] = []

        # App crashes
        crash_report = self.app_analyzer.analyze_app_crashes(period_days)

        if crash_report and crash_report.apps_with_crashes > 0:
            insights.append(CustomerInsight(
                insight_id=f"app_crashes_{self.tenant_id}_{date.today()}",
                category=InsightCategory.APP_CRASH_PATTERN,
                severity=InsightSeverity.MEDIUM if crash_report.total_crashes < 50 else InsightSeverity.HIGH,
                headline=f"{crash_report.total_crashes} app crashes in {period_days} days",
                impact_statement=f"Average {crash_report.avg_crashes_per_device:.1f} crashes per device",
                comparison_context=f"{crash_report.apps_with_crashes} apps affected",
                recommended_actions=crash_report.recommendations,
                entity_type=EntityType.APP,
                entity_id=self.tenant_id,
                entity_name="Fleet Apps",
                affected_device_count=int(crash_report.total_crashes),
                primary_metric="CrashCount",
                primary_value=crash_report.total_crashes,
                threshold_exceeded=10,
                confidence_score=0.8,
                trend_direction=crash_report.overall_trend,
                trend_change_percent=crash_report.trend_change_percent,
                detected_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            ))

        # App power
        power_report = self.app_analyzer.analyze_app_power_efficiency(period_days)

        if power_report and power_report.apps_with_power_issues > 0:
            insights.append(CustomerInsight(
                insight_id=f"app_power_{self.tenant_id}_{date.today()}",
                category=InsightCategory.APP_POWER_DRAIN,
                severity=InsightSeverity.MEDIUM,
                headline=f"{power_report.apps_with_power_issues} apps have power efficiency issues",
                impact_statement=f"Apps consuming {power_report.total_battery_drain_from_apps:.0f}% of battery",
                comparison_context=f"{power_report.total_apps_analyzed} apps analyzed",
                recommended_actions=power_report.recommendations,
                entity_type=EntityType.APP,
                entity_id=self.tenant_id,
                entity_name="Fleet Apps",
                affected_device_count=power_report.apps_with_power_issues,
                primary_metric="AppDrainPercent",
                primary_value=power_report.total_battery_drain_from_apps,
                threshold_exceeded=50,
                confidence_score=0.7,
                trend_direction="stable",
                trend_change_percent=None,
                detected_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            ))

        return insights

    def _generate_location_aggregated_insights(
        self,
        period_days: int,
    ) -> List[CustomerInsight]:
        """Generate location-level aggregated insights."""
        insights: List[CustomerInsight] = []

        # Get all classified insights and aggregate by location
        # This would use the entity aggregator to roll up device insights

        locations = self.location_mapper.get_all_locations()

        for location in locations[:5]:  # Limit for performance
            # Get drop analysis for location
            drops = self.abuse_analyzer.analyze_drops(period_days, "location", location.location_id)

            if drops and drops.avg_drops_per_device > 2:
                insights.append(CustomerInsight(
                    insight_id=f"location_drops_{location.location_id}_{date.today()}",
                    category=InsightCategory.LOCATION_ANOMALY_CLUSTER,
                    severity=InsightSeverity.MEDIUM,
                    headline=f"{location.location_name} has elevated device issues",
                    impact_statement=f"{drops.devices_with_excessive_drops} devices with problems",
                    comparison_context=f"Average {drops.avg_drops_per_device:.1f} drops per device",
                    recommended_actions=drops.recommendations,
                    entity_type=EntityType.LOCATION,
                    entity_id=location.location_id,
                    entity_name=location.location_name,
                    affected_device_count=drops.devices_with_excessive_drops,
                    primary_metric="LocationIssueRate",
                    primary_value=drops.avg_drops_per_device,
                    threshold_exceeded=2,
                    confidence_score=0.7,
                    trend_direction=drops.trend_direction,
                    trend_change_percent=drops.trend_change_percent,
                    detected_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                ))

        return insights

    def _get_priority_score(self, insight: CustomerInsight) -> float:
        """Calculate priority score for sorting insights."""
        severity_scores = {
            InsightSeverity.CRITICAL: 100,
            InsightSeverity.HIGH: 75,
            InsightSeverity.MEDIUM: 50,
            InsightSeverity.LOW: 25,
            InsightSeverity.INFO: 10,
        }

        score = severity_scores.get(insight.severity, 50)

        # Boost for trending worse
        if insight.trend_direction == "worsening":
            score += 20

        # Boost for many affected devices
        if insight.affected_device_count > 10:
            score += 10

        return score

    def _generate_executive_summary(
        self,
        all_insights: List[CustomerInsight],
        critical: List[CustomerInsight],
        high: List[CustomerInsight],
        trending: List[CustomerInsight],
        new_issues: List[CustomerInsight],
    ) -> str:
        """Generate executive summary text."""
        lines = []

        if critical:
            lines.append(f"**{len(critical)} Critical Issues** require immediate attention.")

        if trending:
            lines.append(f"**{len(trending)} Issues are Trending Worse** - monitor closely.")

        if new_issues:
            lines.append(f"**{len(new_issues)} New Issues** detected today.")

        if not critical and not trending:
            lines.append("Fleet health is stable with no critical issues detected.")

        # Top insight
        if all_insights:
            top = all_insights[0]
            lines.append(f"\nTop priority: {top.headline}")

        return " ".join(lines)

    def _get_location_devices(
        self,
        location_id: str,
        period_days: int,
    ) -> pd.DataFrame:
        """Get devices at a location."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        query = (
            self.db.query(DeviceFeature)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.computed_at >= start_date,
            )
            .all()
        )

        import json

        records = []
        for feature in query:
            try:
                metadata = json.loads(feature.metadata_json) if feature.metadata_json else {}
                if metadata.get("location_id") == location_id:
                    records.append({"device_id": feature.device_id})
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Failed to parse metadata for device {feature.device_id} "
                    f"(tenant={self.tenant_id}): {e}"
                )
                continue

        return pd.DataFrame(records)

    def _aggregated_to_customer_insight(
        self,
        record: AggregatedInsight,
    ) -> Optional[CustomerInsight]:
        """Convert database record to CustomerInsight."""
        try:
            return CustomerInsight(
                insight_id=str(record.id),
                category=InsightCategory(record.insight_category),
                severity=InsightSeverity(record.severity),
                headline=record.headline or "",
                impact_statement=record.impact_statement or "",
                comparison_context=record.comparison_context or "",
                recommended_actions=[],
                entity_type=EntityType(record.entity_type),
                entity_id=record.entity_id,
                entity_name=record.entity_name or "",
                affected_device_count=record.affected_device_count or 0,
                primary_metric="",
                primary_value=record.current_value or 0,
                threshold_exceeded=None,
                confidence_score=record.confidence_score or 0.5,
                trend_direction=record.trend_direction or "stable",
                trend_change_percent=None,
                detected_at=record.computed_at or datetime.utcnow(),
                last_updated=record.computed_at or datetime.utcnow(),
            )
        except (ValueError, TypeError):
            return None

    def _device_readiness_to_insight(
        self,
        device: Any,
        location_name: str,
    ) -> CustomerInsight:
        """Convert device readiness to insight."""
        return CustomerInsight(
            insight_id=f"battery_{device.device_id}_{date.today()}",
            category=InsightCategory.BATTERY_SHIFT_FAILURE,
            severity=InsightSeverity.HIGH,
            headline=f"Device {device.device_id} won't last shift at {location_name}",
            impact_statement=f"Battery will die at {device.estimated_dead_time}",
            comparison_context=f"Current battery: {device.current_battery:.0f}%",
            recommended_actions=device.recommendations,
            entity_type=EntityType.DEVICE,
            entity_id=str(device.device_id),
            entity_name=device.device_name or f"Device {device.device_id}",
            affected_device_count=1,
            primary_metric="BatteryLevel",
            primary_value=device.current_battery,
            threshold_exceeded=20,
            confidence_score=0.8,
            trend_direction="stable",
            trend_change_percent=None,
            detected_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
        )

    def _wifi_issue_to_insight(
        self,
        issue: Any,
        location_name: str,
    ) -> CustomerInsight:
        """Convert WiFi issue to insight."""
        return CustomerInsight(
            insight_id=f"wifi_{issue.device_id}_{date.today()}",
            category=InsightCategory.WIFI_AP_HOPPING,
            severity=issue.severity,
            headline=issue.description,
            impact_statement=f"At {location_name}",
            comparison_context=f"Connected to {issue.unique_aps_connected} APs",
            recommended_actions=[issue.recommendation],
            entity_type=EntityType.DEVICE,
            entity_id=str(issue.device_id),
            entity_name=issue.device_name or f"Device {issue.device_id}",
            affected_device_count=1,
            primary_metric="UniqueAPs",
            primary_value=issue.unique_aps_connected,
            threshold_exceeded=5,
            confidence_score=0.7,
            trend_direction="stable",
            trend_change_percent=None,
            detected_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
        )

    def _drop_device_to_insight(
        self,
        device: Any,
        location_name: str,
    ) -> CustomerInsight:
        """Convert drop abuse indicator to insight."""
        return CustomerInsight(
            insight_id=f"drops_{device.device_id}_{date.today()}",
            category=InsightCategory.EXCESSIVE_DROPS,
            severity=device.severity,
            headline=device.description,
            impact_statement=f"At {location_name}",
            comparison_context=f"Top {100-device.vs_fleet_drop_percentile}% worst in fleet",
            recommended_actions=device.recommendations,
            entity_type=EntityType.DEVICE,
            entity_id=str(device.device_id),
            entity_name=device.device_name or f"Device {device.device_id}",
            affected_device_count=1,
            primary_metric="DropCount",
            primary_value=device.drop_count,
            threshold_exceeded=5,
            confidence_score=0.8,
            trend_direction="stable",
            trend_change_percent=None,
            detected_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
        )

    def _generate_location_recommendations(
        self,
        insights: List[CustomerInsight],
        shift_readiness: Optional[Dict],
        top_issues: List[Tuple[InsightCategory, int]],
    ) -> List[str]:
        """Generate recommendations for a location."""
        recs = []

        if shift_readiness and shift_readiness.get("readiness_percentage", 100) < 80:
            recs.append("Improve shift readiness by ensuring devices are fully charged before shift start")

        if top_issues:
            top_category = top_issues[0][0]
            if top_category == InsightCategory.EXCESSIVE_DROPS:
                recs.append("Review device handling procedures and consider protective cases")
            elif top_category == InsightCategory.WIFI_AP_HOPPING:
                recs.append("Evaluate WiFi infrastructure and roaming settings")
            elif top_category == InsightCategory.BATTERY_SHIFT_FAILURE:
                recs.append("Review charging infrastructure and overnight procedures")

        return recs
