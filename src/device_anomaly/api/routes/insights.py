"""API routes for customer-facing insights.

Aligned with Carl's (CEO) vision:
- "XSight has the data. XSight needs the story."
- Pre-interpreted, contextualized, and actionable insights
- Comparisons matter more than absolutes
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_db, get_tenant_id
from device_anomaly.database.schema import AggregatedInsight
from device_anomaly.insights import (
    AppPowerAnalyzer,
    BatteryShiftAnalyzer,
    CustomerInsight,
    DailyInsightDigest,
    DeviceAbuseAnalyzer,
    InsightCategory,
    InsightGenerator,
    InsightSeverity,
    LocationMapper,
    NetworkPatternAnalyzer,
)

router = APIRouter(prefix="/insights", tags=["insights"])


# Pydantic models for API responses


class InsightResponse(BaseModel):
    """Customer insight response model."""

    insight_id: str
    category: str
    severity: str
    headline: str
    impact_statement: str
    comparison_context: str
    recommended_actions: List[str]
    entity_type: str
    entity_id: str
    entity_name: str
    affected_device_count: int
    primary_metric: str
    primary_value: float
    trend_direction: str
    trend_change_percent: Optional[float]
    detected_at: datetime
    confidence_score: float

    class Config:
        from_attributes = True


class DailyDigestResponse(BaseModel):
    """Daily insight digest response."""

    tenant_id: str
    digest_date: date
    generated_at: datetime
    total_insights: int
    critical_count: int
    high_count: int
    medium_count: int
    top_insights: List[InsightResponse]
    executive_summary: str
    trending_issues: List[InsightResponse]
    new_issues: List[InsightResponse]


class LocationInsightResponse(BaseModel):
    """Location insight report response."""

    location_id: str
    location_name: str
    report_date: date
    total_devices: int
    devices_with_issues: int
    issue_rate: float
    shift_readiness: Optional[Dict[str, Any]]
    insights: List[InsightResponse]
    top_issues: List[Dict[str, Any]]
    rank_among_locations: int
    better_than_percent: float
    recommendations: List[str]


class ShiftReadinessResponse(BaseModel):
    """Shift readiness report response."""

    location_id: str
    location_name: str
    shift_name: str
    shift_date: date
    readiness_percentage: float
    total_devices: int
    devices_ready: int
    devices_at_risk: int
    devices_critical: int
    avg_battery_at_start: float
    avg_drain_rate: float
    devices_not_fully_charged: int
    vs_last_week_readiness: Optional[float]
    device_details: List[Dict[str, Any]]
    recommendations: List[str]


class NetworkAnalysisResponse(BaseModel):
    """Network pattern analysis response."""

    tenant_id: str
    analysis_period_days: int
    wifi_summary: Dict[str, Any]
    cellular_summary: Optional[Dict[str, Any]]
    disconnect_summary: Dict[str, Any]
    hidden_devices_count: int
    recommendations: List[str]


class UserAbuseItem(BaseModel):
    """User abuse item for by-user analysis."""

    user_id: str
    user_name: Optional[str]
    user_email: Optional[str]
    total_drops: int
    total_reboots: int
    device_count: int
    drops_per_device: float
    drops_per_day: float
    vs_fleet_multiplier: float
    is_excessive: bool


class UserAbuseResponse(BaseModel):
    """User abuse analysis response (Carl's 'People with excessive drops')."""

    tenant_id: str
    period_days: int
    total_users: int
    total_drops: int
    fleet_avg_drops_per_device: float
    users_with_excessive_drops: int
    worst_users: List[UserAbuseItem]
    recommendations: List[str]


class DeviceAbuseResponse(BaseModel):
    """Device abuse analysis response."""

    tenant_id: str
    analysis_period_days: int
    total_devices: int
    total_drops: int
    total_reboots: int
    devices_with_excessive_drops: int
    devices_with_excessive_reboots: int
    worst_locations: List[Dict[str, Any]]
    worst_cohorts: List[Dict[str, Any]]
    worst_users: Optional[List[UserAbuseItem]] = None
    problem_combinations: List[Dict[str, Any]]
    recommendations: List[str]
    financial_impact: Optional[Dict[str, Any]] = None


class AppAnalysisResponse(BaseModel):
    """App power and crash analysis response."""

    tenant_id: str
    analysis_period_days: int
    total_apps_analyzed: int
    apps_with_issues: int
    total_crashes: int
    total_anrs: int
    top_power_consumers: List[Dict[str, Any]]
    top_crashers: List[Dict[str, Any]]
    recommendations: List[str]


class LocationCompareRequest(BaseModel):
    """Request model for location comparison."""

    location_a_id: str
    location_b_id: str
    metrics: Optional[List[str]] = None


class LocationCompareResponse(BaseModel):
    """Response model for location comparison."""

    location_a_id: str
    location_a_name: str
    location_b_id: str
    location_b_name: str
    device_count_a: int
    device_count_b: int
    metric_comparisons: Dict[str, Dict[str, float]]
    overall_winner: Optional[str]
    key_differences: List[str]


# Helper functions


def _customer_insight_to_response(insight: CustomerInsight) -> InsightResponse:
    """Convert CustomerInsight to API response."""
    return InsightResponse(
        insight_id=insight.insight_id,
        category=insight.category.value,
        severity=insight.severity.value,
        headline=insight.headline,
        impact_statement=insight.impact_statement,
        comparison_context=insight.comparison_context,
        recommended_actions=insight.recommended_actions,
        entity_type=insight.entity_type.value,
        entity_id=insight.entity_id,
        entity_name=insight.entity_name,
        affected_device_count=insight.affected_device_count,
        primary_metric=insight.primary_metric,
        primary_value=insight.primary_value,
        trend_direction=insight.trend_direction,
        trend_change_percent=insight.trend_change_percent,
        detected_at=insight.detected_at,
        confidence_score=insight.confidence_score,
    )


# Routes


@router.get("/daily-digest", response_model=DailyDigestResponse)
def get_daily_digest(
    digest_date: Optional[date] = Query(None, description="Date for digest (defaults to today)"),
    period_days: int = Query(7, ge=1, le=30, description="Days of data to analyze"),
    db: Session = Depends(get_db),
):
    """Get the daily insight digest with prioritized issues.

    Returns a summary of all insights for the day, organized by priority
    and category, with executive summary and trending issues.
    """
    tenant_id = get_tenant_id()
    target_date = digest_date or date.today()

    generator = InsightGenerator(db, tenant_id)
    digest = generator.generate_daily_insights(target_date, period_days)

    return DailyDigestResponse(
        tenant_id=digest.tenant_id,
        digest_date=digest.digest_date,
        generated_at=digest.generated_at,
        total_insights=digest.total_insights,
        critical_count=digest.critical_count,
        high_count=digest.high_count,
        medium_count=digest.medium_count,
        top_insights=[_customer_insight_to_response(i) for i in digest.top_insights],
        executive_summary=digest.executive_summary,
        trending_issues=[_customer_insight_to_response(i) for i in digest.trending_issues],
        new_issues=[_customer_insight_to_response(i) for i in digest.new_issues],
    )


@router.get("/location/{location_id}", response_model=LocationInsightResponse)
def get_location_insights(
    location_id: str,
    period_days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Get comprehensive insights for a specific location.

    Includes shift readiness, device issues, network problems,
    and comparison to other locations.
    """
    tenant_id = get_tenant_id()

    generator = InsightGenerator(db, tenant_id)
    report = generator.generate_location_insights(location_id, period_days)

    return LocationInsightResponse(
        location_id=report.location_id,
        location_name=report.location_name,
        report_date=report.report_date,
        total_devices=report.total_devices,
        devices_with_issues=report.devices_with_issues,
        issue_rate=report.issue_rate,
        shift_readiness=report.shift_readiness,
        insights=[_customer_insight_to_response(i) for i in report.insights],
        top_issues=[
            {"category": cat.value, "count": count}
            for cat, count in report.top_issues
        ],
        rank_among_locations=report.rank_among_locations,
        better_than_percent=report.better_than_percent,
        recommendations=report.recommendations,
    )


@router.get("/location/{location_id}/shift-readiness", response_model=ShiftReadinessResponse)
def get_shift_readiness(
    location_id: str,
    shift_date: Optional[date] = Query(None, description="Date for shift analysis"),
    shift_name: Optional[str] = Query(None, description="Specific shift name"),
    db: Session = Depends(get_db),
):
    """Get battery shift readiness analysis for a location.

    Carl's key requirement: "Batteries that don't last a shift"

    Returns which devices will/won't make it through the shift,
    with recommendations for each.
    """
    tenant_id = get_tenant_id()
    target_date = datetime.combine(shift_date or date.today(), datetime.min.time())

    mapper = LocationMapper(db, tenant_id)
    analyzer = BatteryShiftAnalyzer(db, tenant_id, mapper)

    report = analyzer.analyze_shift_readiness(location_id, target_date, shift_name)

    if not report:
        raise HTTPException(status_code=404, detail="Location not found or no shift data")

    # Convert device readiness to dict
    device_details = []
    for device in report.device_readiness:
        device_details.append({
            "device_id": device.device_id,
            "device_name": device.device_name,
            "current_battery": device.current_battery,
            "drain_rate_per_hour": device.drain_rate_per_hour,
            "projected_end_battery": device.projected_end_battery,
            "will_complete_shift": device.will_complete_shift,
            "estimated_dead_time": device.estimated_dead_time,
            "was_fully_charged": device.was_fully_charged,
            "readiness_score": device.readiness_score,
            "recommendations": device.recommendations,
        })

    return ShiftReadinessResponse(
        location_id=report.location_id,
        location_name=report.location_name,
        shift_name=report.shift_name,
        shift_date=report.shift_date.date(),
        readiness_percentage=report.readiness_percentage,
        total_devices=report.total_devices,
        devices_ready=report.devices_ready,
        devices_at_risk=report.devices_at_risk,
        devices_critical=report.devices_critical,
        avg_battery_at_start=report.avg_battery_at_start,
        avg_drain_rate=report.avg_drain_rate,
        devices_not_fully_charged=report.devices_not_fully_charged,
        vs_last_week_readiness=report.vs_last_week_readiness,
        device_details=device_details,
        recommendations=[
            "Ensure all devices are fully charged before shift",
            "Investigate devices with high drain rates",
            "Review charging infrastructure at this location",
        ] if report.devices_at_risk > 0 else [],
    )


@router.get("/trending", response_model=List[InsightResponse])
def get_trending_issues(
    lookback_days: int = Query(14, ge=1, le=30),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get issues that are trending worse.

    Returns insights that have gotten worse over the lookback period,
    sorted by rate of decline.
    """
    tenant_id = get_tenant_id()

    generator = InsightGenerator(db, tenant_id)
    trending = generator.get_trending_issues(lookback_days, limit)

    return [_customer_insight_to_response(t.insight) for t in trending]


@router.post("/compare/locations", response_model=LocationCompareResponse)
def compare_locations(
    request: LocationCompareRequest,
    period_days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Compare two locations across multiple metrics.

    Carl's requirement: "Relate any anomalies to location (warehouse 1 vs warehouse 2)"

    Returns side-by-side comparison with percentage differences.
    """
    tenant_id = get_tenant_id()

    from device_anomaly.insights.comparisons import ComparisonEngine

    engine = ComparisonEngine(db, tenant_id)
    comparison = engine.compare_locations(
        request.location_a_id,
        request.location_b_id,
        request.metrics,
        period_days,
    )

    if not comparison:
        raise HTTPException(status_code=404, detail="One or both locations not found")

    return LocationCompareResponse(
        location_a_id=comparison.location_a_id,
        location_a_name=comparison.location_a_name,
        location_b_id=comparison.location_b_id,
        location_b_name=comparison.location_b_name,
        device_count_a=comparison.device_count_a,
        device_count_b=comparison.device_count_b,
        metric_comparisons={
            metric: {
                "location_a_value": val_a,
                "location_b_value": val_b,
                "difference_percent": diff_pct,
            }
            for metric, (val_a, val_b, diff_pct) in comparison.metric_comparisons.items()
        },
        overall_winner=comparison.overall_winner,
        key_differences=comparison.key_differences,
    )


@router.get("/cohort/{cohort_id}/issues", response_model=List[InsightResponse])
def get_cohort_issues(
    cohort_id: str,
    period_days: int = Query(30, ge=1, le=90),
    db: Session = Depends(get_db),
):
    """Get issues specific to a device cohort.

    Carl's requirement: "Performance patterns by manufacturer, model, OS version, firmware"

    Returns insights for devices matching the cohort (e.g., "Samsung_SM-G991B_13").
    """
    tenant_id = get_tenant_id()

    # Query aggregated insights for this cohort
    cutoff = datetime.utcnow() - timedelta(days=period_days)
    insights = (
        db.query(AggregatedInsight)
        .filter(
            AggregatedInsight.tenant_id == tenant_id,
            AggregatedInsight.entity_type == "cohort",
            AggregatedInsight.entity_id == cohort_id,
            AggregatedInsight.computed_at >= cutoff,
            AggregatedInsight.is_active == True,  # noqa: E712
        )
        .order_by(AggregatedInsight.severity.desc())
        .all()
    )

    # Convert to response
    responses = []
    for insight in insights:
        responses.append(InsightResponse(
            insight_id=str(insight.id),
            category=insight.insight_category,
            severity=insight.severity,
            headline=insight.headline or "",
            impact_statement=insight.impact_statement or "",
            comparison_context=insight.comparison_context or "",
            recommended_actions=[],
            entity_type=insight.entity_type,
            entity_id=insight.entity_id,
            entity_name=insight.entity_name or "",
            affected_device_count=insight.affected_device_count or 0,
            primary_metric="",
            primary_value=insight.current_value or 0,
            trend_direction=insight.trend_direction or "stable",
            trend_change_percent=None,
            detected_at=insight.computed_at or datetime.utcnow(),
            confidence_score=insight.confidence_score or 0.5,
        ))

    return responses


@router.get("/apps/power-hungry", response_model=AppAnalysisResponse)
def get_power_hungry_apps(
    period_days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Get apps with power efficiency issues.

    Carl's requirement: "Apps consuming too much power (foreground time vs drain)"

    Returns apps ranked by power consumption relative to usage time.
    """
    tenant_id = get_tenant_id()

    analyzer = AppPowerAnalyzer(db, tenant_id)
    power_report = analyzer.analyze_app_power_efficiency(period_days)
    crash_report = analyzer.analyze_app_crashes(period_days)

    return AppAnalysisResponse(
        tenant_id=tenant_id,
        analysis_period_days=period_days,
        total_apps_analyzed=power_report.total_apps_analyzed,
        apps_with_issues=power_report.apps_with_power_issues,
        total_crashes=crash_report.total_crashes,
        total_anrs=crash_report.total_anrs,
        top_power_consumers=[
            {
                "package_name": p.package_name,
                "app_name": p.app_name,
                "battery_drain_percent": p.battery_drain_percent,
                "drain_per_hour": p.drain_per_foreground_hour,
                "foreground_hours": p.foreground_time_hours,
                "efficiency_score": p.efficiency_score,
            }
            for p in power_report.top_power_consumers[:10]
        ],
        top_crashers=[
            {
                "package_name": c.package_name,
                "app_name": c.app_name,
                "crash_count": c.crash_count,
                "anr_count": c.anr_count,
                "devices_affected": c.devices_affected,
                "severity": c.severity.value,
            }
            for c in crash_report.top_crashers[:10]
        ],
        recommendations=power_report.recommendations + crash_report.recommendations,
    )


@router.get("/network/analysis", response_model=NetworkAnalysisResponse)
def get_network_analysis(
    location_id: Optional[str] = Query(None),
    period_days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Get network pattern analysis.

    Carl's requirements:
    - "AP hopping/stickiness"
    - "Tower hopping/stickiness"
    - "Server disconnection patterns"

    Returns WiFi roaming issues, cellular patterns, and disconnect analysis.
    """
    tenant_id = get_tenant_id()

    analyzer = NetworkPatternAnalyzer(db, tenant_id)

    wifi_report = analyzer.analyze_wifi_roaming(location_id, period_days)
    cellular_report = analyzer.analyze_cellular_patterns(period_days)
    disconnect_report = analyzer.analyze_disconnect_patterns("tenant", None, period_days)
    hidden_report = analyzer.detect_hidden_devices(period_days)

    return NetworkAnalysisResponse(
        tenant_id=tenant_id,
        analysis_period_days=period_days,
        wifi_summary={
            "total_devices": wifi_report.total_devices,
            "devices_with_roaming_issues": wifi_report.devices_with_roaming_issues,
            "devices_with_stickiness": wifi_report.devices_with_stickiness,
            "avg_aps_per_device": wifi_report.avg_aps_per_device,
            "potential_dead_zones": wifi_report.potential_dead_zones,
        },
        cellular_summary={
            "total_devices": cellular_report.total_devices,
            "devices_with_tower_hopping": cellular_report.devices_with_tower_hopping,
            "devices_with_tech_fallback": cellular_report.devices_with_tech_fallback,
            "best_carrier": cellular_report.best_carrier,
            "worst_carrier": cellular_report.worst_carrier,
            "network_type_distribution": cellular_report.network_type_distribution,
        } if cellular_report else None,
        disconnect_summary={
            "total_disconnects": disconnect_report.total_disconnects,
            "avg_disconnects_per_device": disconnect_report.avg_disconnects_per_device,
            "total_offline_hours": disconnect_report.total_offline_hours,
            "has_predictable_pattern": disconnect_report.has_predictable_pattern,
            "pattern_description": disconnect_report.pattern_description,
        },
        hidden_devices_count=hidden_report.devices_highly_suspicious if hidden_report else 0,
        recommendations=(
            wifi_report.recommendations +
            cellular_report.recommendations +
            disconnect_report.recommendations
        ),
    )


@router.get("/device-abuse", response_model=DeviceAbuseResponse)
def get_device_abuse_analysis(
    period_days: int = Query(7, ge=1, le=30),
    include_user_analysis: bool = Query(True, description="Include by-user analysis"),
    db: Session = Depends(get_db),
):
    """Get device abuse analysis (drops, reboots, problem combinations).

    Carl's requirements:
    - "Devices/people/locations with excessive drops"
    - "Devices/people/locations with excessive reboots"
    - "Combinations that cause problems"
    """
    tenant_id = get_tenant_id()

    analyzer = DeviceAbuseAnalyzer(db, tenant_id)

    drop_report = analyzer.analyze_drops(period_days)
    reboot_report = analyzer.analyze_reboots(period_days)
    combo_report = analyzer.identify_problem_combinations(period_days)

    # By-user analysis (Carl's "People with excessive drops")
    worst_users = None
    if include_user_analysis:
        user_analysis = analyzer.analyze_drops_by_user(period_days, top_n=10)
        if user_analysis.get("worst_users"):
            worst_users = [
                UserAbuseItem(
                    user_id=u["user_id"],
                    user_name=u.get("user_name"),
                    user_email=u.get("user_email"),
                    total_drops=u["total_drops"],
                    total_reboots=u.get("total_reboots", 0),
                    device_count=u["device_count"],
                    drops_per_device=u["drops_per_device"],
                    drops_per_day=u["drops_per_day"],
                    vs_fleet_multiplier=u["vs_fleet_multiplier"],
                    is_excessive=u["is_excessive"],
                )
                for u in user_analysis["worst_users"]
            ]

    # Financial impact estimation (simplified)
    financial_impact = None
    if drop_report.total_drops > 0 or reboot_report.total_reboots > 0:
        # Rough cost estimation
        drop_cost = drop_report.total_drops * 150  # $150 per drop (repair/replacement)
        reboot_downtime_cost = reboot_report.total_reboots * 25  # $25 per reboot (productivity)
        total_cost = drop_cost + reboot_downtime_cost
        financial_impact = {
            "total_estimated_cost": total_cost,
            "cost_breakdown": [
                {
                    "category": "Device Repairs",
                    "amount": drop_cost,
                    "description": f"{drop_report.total_drops} drops x $150 avg repair cost",
                },
                {
                    "category": "Downtime from Reboots",
                    "amount": reboot_downtime_cost,
                    "description": f"{reboot_report.total_reboots} reboots x $25 productivity loss",
                },
            ],
            "potential_savings": int(total_cost * 0.6),  # 60% reduction if addressed
        }

    return DeviceAbuseResponse(
        tenant_id=tenant_id,
        analysis_period_days=period_days,
        total_devices=drop_report.total_devices,
        total_drops=drop_report.total_drops,
        total_reboots=reboot_report.total_reboots,
        devices_with_excessive_drops=drop_report.devices_with_excessive_drops,
        devices_with_excessive_reboots=reboot_report.devices_with_excessive_reboots,
        worst_locations=[
            {"location_id": loc, "drops": drops, "rate_per_device": rate}
            for loc, drops, rate in drop_report.worst_locations
        ],
        worst_cohorts=[
            {"cohort_id": cohort, "reboots": reboots, "rate_per_device": rate}
            for cohort, reboots, rate in reboot_report.worst_cohorts
        ],
        worst_users=worst_users,
        problem_combinations=[
            {
                "cohort_id": combo.cohort_id,
                "manufacturer": combo.manufacturer,
                "model": combo.model,
                "os_version": combo.os_version,
                "device_count": combo.device_count,
                "vs_fleet_multiplier": combo.vs_fleet_issue_rate,
                "primary_issue": combo.primary_issue,
                "severity": combo.severity.value,
            }
            for combo in combo_report.problem_combinations[:10]
        ],
        recommendations=(
            drop_report.recommendations +
            reboot_report.recommendations +
            combo_report.recommendations
        ),
        financial_impact=financial_impact,
    )


@router.get("/device-abuse/by-user", response_model=UserAbuseResponse)
def get_drops_by_user_analysis(
    period_days: int = Query(7, ge=1, le=30),
    top_n: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get drops analysis grouped by assigned user.

    Carl's requirement: "People with excessive drops"

    Returns users ranked by drop count with comparison to fleet average.
    """
    tenant_id = get_tenant_id()

    analyzer = DeviceAbuseAnalyzer(db, tenant_id)
    result = analyzer.analyze_drops_by_user(period_days, top_n)

    worst_users = [
        UserAbuseItem(
            user_id=u["user_id"],
            user_name=u.get("user_name"),
            user_email=u.get("user_email"),
            total_drops=u["total_drops"],
            total_reboots=u.get("total_reboots", 0),
            device_count=u["device_count"],
            drops_per_device=u["drops_per_device"],
            drops_per_day=u["drops_per_day"],
            vs_fleet_multiplier=u["vs_fleet_multiplier"],
            is_excessive=u["is_excessive"],
        )
        for u in result.get("worst_users", [])
    ]

    return UserAbuseResponse(
        tenant_id=result.get("tenant_id", tenant_id),
        period_days=result.get("period_days", period_days),
        total_users=result.get("total_users", 0),
        total_drops=result.get("total_drops", 0),
        fleet_avg_drops_per_device=result.get("fleet_avg_drops_per_device", 0.0),
        users_with_excessive_drops=result.get("users_with_excessive_drops", 0),
        worst_users=worst_users,
        recommendations=result.get("recommendations", []),
    )


@router.get("/by-category/{category}", response_model=List[InsightResponse])
def get_insights_by_category(
    category: str,
    period_days: int = Query(7, ge=1, le=30),
    severity: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get insights filtered by category.

    Useful for drilling into specific issue types like "battery_shift_failure"
    or "excessive_drops".
    """
    tenant_id = get_tenant_id()

    # Validate category
    try:
        InsightCategory(category)
    except ValueError:
        valid = [c.value for c in InsightCategory]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Valid categories: {valid}",
        )

    cutoff = datetime.utcnow() - timedelta(days=period_days)
    query = (
        db.query(AggregatedInsight)
        .filter(
            AggregatedInsight.tenant_id == tenant_id,
            AggregatedInsight.insight_category == category,
            AggregatedInsight.computed_at >= cutoff,
            AggregatedInsight.is_active == True,  # noqa: E712
        )
    )

    if severity:
        try:
            InsightSeverity(severity)
            query = query.filter(AggregatedInsight.severity == severity)
        except ValueError:
            pass

    insights = query.order_by(AggregatedInsight.severity.desc()).limit(limit).all()

    return [
        InsightResponse(
            insight_id=str(i.id),
            category=i.insight_category,
            severity=i.severity,
            headline=i.headline or "",
            impact_statement=i.impact_statement or "",
            comparison_context=i.comparison_context or "",
            recommended_actions=[],
            entity_type=i.entity_type,
            entity_id=i.entity_id,
            entity_name=i.entity_name or "",
            affected_device_count=i.affected_device_count or 0,
            primary_metric="",
            primary_value=i.current_value or 0,
            trend_direction=i.trend_direction or "stable",
            trend_change_percent=None,
            detected_at=i.computed_at or datetime.utcnow(),
            confidence_score=i.confidence_score or 0.5,
        )
        for i in insights
    ]


# ============================================
# Insight Devices Endpoint
# ============================================


@router.get("/insight/{insight_id}/devices")
def get_insight_devices(
    insight_id: str,
    include_ai_grouping: bool = Query(False, description="Include AI pattern similarity grouping"),
    db: Session = Depends(get_db),
):
    """Get all devices affected by a specific insight with grouping options.

    Returns the full list of devices impacted by an insight, grouped by:
    - Location
    - Device model
    - AI-detected patterns (if include_ai_grouping=True)

    This enables drill-down from insight cards to see exactly which devices
    are affected and take targeted action.
    """
    import json
    from device_anomaly.api.models import (
        DeviceGroupingResponse,
        ImpactedDeviceResponse,
        InsightDevicesResponse,
    )
    from device_anomaly.database.schema import DeviceMetadata
    from device_anomaly.services.device_grouper import DeviceGrouper

    tenant_id = get_tenant_id()

    # Find the insight
    insight = (
        db.query(AggregatedInsight)
        .filter(
            AggregatedInsight.tenant_id == tenant_id,
            AggregatedInsight.id == int(insight_id),
        )
        .first()
    )

    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")

    # Parse affected device IDs from JSON
    device_ids: List[int] = []
    if insight.affected_devices_json:
        try:
            device_ids = json.loads(insight.affected_devices_json)
        except json.JSONDecodeError:
            pass

    # If no device list stored, return empty response
    if not device_ids:
        return InsightDevicesResponse(
            insight_id=insight_id,
            insight_headline=insight.headline or "",
            insight_category=insight.insight_category,
            insight_severity=insight.severity,
            total_devices=0,
            devices=[],
            groupings={
                "by_location": [],
                "by_model": [],
                "by_pattern": [],
            },
            ai_pattern_analysis=None,
            generated_at=datetime.utcnow(),
        )

    # Fetch device metadata
    device_records = (
        db.query(DeviceMetadata)
        .filter(
            DeviceMetadata.tenant_id == tenant_id,
            DeviceMetadata.device_id.in_(device_ids),
        )
        .all()
    )

    # Build device response list
    devices: List[ImpactedDeviceResponse] = []
    for d in device_records:
        devices.append(
            ImpactedDeviceResponse(
                device_id=d.device_id,
                device_name=d.device_name,
                device_model=d.device_model,
                location=d.location,
                status=d.status or "unknown",
                last_seen=d.last_seen,
                os_version=d.os_version,
                anomaly_count=0,  # Could be enriched with anomaly data
                severity=insight.severity,
                primary_metric=None,
            )
        )

    # Include devices that were in the list but not found in metadata
    found_ids = {d.device_id for d in device_records}
    for device_id in device_ids:
        if device_id not in found_ids:
            devices.append(
                ImpactedDeviceResponse(
                    device_id=device_id,
                    device_name=f"Device-{device_id}",
                    device_model=None,
                    location=None,
                    status="unknown",
                    last_seen=None,
                    os_version=None,
                    anomaly_count=0,
                    severity=insight.severity,
                    primary_metric=None,
                )
            )

    # Build groupings
    grouper = DeviceGrouper(enable_ai_grouping=include_ai_grouping)

    by_location = grouper.group_by_location(devices)
    by_model = grouper.group_by_model(devices)

    # AI pattern grouping (only if requested)
    by_pattern: List[DeviceGroupingResponse] = []
    ai_analysis: Optional[str] = None

    if include_ai_grouping and len(devices) >= 3:
        by_pattern, ai_analysis = grouper.group_by_pattern_similarity(
            devices,
            insight_category=insight.insight_category,
            insight_headline=insight.headline,
        )

    return InsightDevicesResponse(
        insight_id=insight_id,
        insight_headline=insight.headline or "",
        insight_category=insight.insight_category,
        insight_severity=insight.severity,
        total_devices=len(devices),
        devices=devices,
        groupings={
            "by_location": by_location,
            "by_model": by_model,
            "by_pattern": by_pattern,
        },
        ai_pattern_analysis=ai_analysis,
        generated_at=datetime.utcnow(),
    )
