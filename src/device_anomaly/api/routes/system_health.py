"""API routes for system health monitoring."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/system-health", tags=["system-health"])


# ============================================================================
# Response Models
# ============================================================================

class SystemHealthMetricsResponse(BaseModel):
    """Aggregated system health metrics."""
    avg_cpu_usage: float = Field(description="Average CPU usage percentage")
    avg_memory_usage: float = Field(description="Average memory usage percentage")
    avg_storage_available_pct: float = Field(description="Average storage available percentage")
    avg_device_temp: float = Field(description="Average device temperature (Celsius)")
    avg_battery_temp: float = Field(description="Average battery temperature (Celsius)")
    devices_high_cpu: int = Field(description="Count of devices with high CPU usage")
    devices_high_memory: int = Field(description="Count of devices with high memory usage")
    devices_low_storage: int = Field(description="Count of devices with low storage")
    devices_high_temp: int = Field(description="Count of devices with high temperature")
    total_devices: int = Field(description="Total devices with health data")


class SystemHealthSummaryResponse(BaseModel):
    """Complete system health summary with fleet score."""
    tenant_id: str
    fleet_health_score: float = Field(description="Overall fleet health score 0-100")
    health_trend: str = Field(description="Trend: improving, stable, or degrading")
    total_devices: int
    healthy_count: int
    warning_count: int
    critical_count: int
    metrics: SystemHealthMetricsResponse
    cohort_breakdown: List["CohortHealthResponse"] = []
    recommendations: List[str] = []
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CohortHealthResponse(BaseModel):
    """Health breakdown for a device cohort."""
    cohort_id: str
    cohort_name: str
    device_count: int
    health_score: float
    avg_cpu: float
    avg_memory: float
    avg_storage_pct: float
    devices_at_risk: int


class HealthTrendResponse(BaseModel):
    """Time-series health trend point."""
    timestamp: datetime
    value: float
    device_count: int


class StorageForecastDeviceResponse(BaseModel):
    """Storage forecast for a single device."""
    device_id: int
    device_name: str
    current_storage_pct: float
    storage_trend_gb_per_day: float
    projected_full_date: Optional[datetime]
    days_until_full: Optional[int]
    confidence: float


class StorageForecastResponse(BaseModel):
    """Storage exhaustion forecast summary."""
    tenant_id: str
    devices_at_risk: List[StorageForecastDeviceResponse]
    total_at_risk_count: int
    avg_days_until_full: Optional[float]
    recommendations: List[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Mock Data Functions
# ============================================================================

def get_mock_system_health_summary(period_days: int = 7) -> SystemHealthSummaryResponse:
    """Generate mock system health summary."""
    import random
    random.seed(42)

    metrics = SystemHealthMetricsResponse(
        avg_cpu_usage=35.2 + random.uniform(-5, 5),
        avg_memory_usage=62.4 + random.uniform(-5, 5),
        avg_storage_available_pct=45.8 + random.uniform(-5, 5),
        avg_device_temp=32.1 + random.uniform(-2, 2),
        avg_battery_temp=29.5 + random.uniform(-2, 2),
        devices_high_cpu=random.randint(3, 8),
        devices_high_memory=random.randint(5, 12),
        devices_low_storage=random.randint(8, 15),
        devices_high_temp=random.randint(1, 4),
        total_devices=248,
    )

    cohorts = [
        CohortHealthResponse(
            cohort_id="samsung_android_13",
            cohort_name="Samsung Galaxy A8 - Android 13",
            device_count=85,
            health_score=92.5,
            avg_cpu=28.4,
            avg_memory=58.2,
            avg_storage_pct=52.3,
            devices_at_risk=2,
        ),
        CohortHealthResponse(
            cohort_id="zebra_android_11",
            cohort_name="Zebra TC52 - Android 11",
            device_count=62,
            health_score=78.3,
            avg_cpu=42.1,
            avg_memory=71.5,
            avg_storage_pct=38.2,
            devices_at_risk=8,
        ),
        CohortHealthResponse(
            cohort_id="honeywell_android_12",
            cohort_name="Honeywell CT60 - Android 12",
            device_count=45,
            health_score=85.1,
            avg_cpu=35.7,
            avg_memory=64.3,
            avg_storage_pct=48.9,
            devices_at_risk=3,
        ),
        CohortHealthResponse(
            cohort_id="ipad_ios_17",
            cohort_name="iPad Pro 11 - iOS 17",
            device_count=36,
            health_score=96.2,
            avg_cpu=22.3,
            avg_memory=45.8,
            avg_storage_pct=61.2,
            devices_at_risk=0,
        ),
    ]

    recommendations = []
    if metrics.devices_low_storage > 5:
        recommendations.append(
            f"{metrics.devices_low_storage} devices have low storage - consider clearing cache or archiving data"
        )
    if metrics.devices_high_memory > 10:
        recommendations.append(
            f"{metrics.devices_high_memory} devices have high memory usage - review running apps"
        )
    if metrics.devices_high_temp > 0:
        recommendations.append(
            f"{metrics.devices_high_temp} devices running hot - check for intensive apps or charging issues"
        )

    total = metrics.total_devices
    critical = metrics.devices_high_temp + (metrics.devices_low_storage if metrics.avg_storage_available_pct < 10 else 0)
    warning = metrics.devices_high_cpu + metrics.devices_high_memory + metrics.devices_low_storage
    healthy = max(0, total - warning - critical)

    return SystemHealthSummaryResponse(
        tenant_id="default",
        fleet_health_score=85.7,
        health_trend="stable",
        total_devices=total,
        healthy_count=healthy,
        warning_count=warning,
        critical_count=critical,
        metrics=metrics,
        cohort_breakdown=cohorts,
        recommendations=recommendations,
    )


def get_mock_health_trends(metric: str, period_days: int, granularity: str) -> List[HealthTrendResponse]:
    """Generate mock health trend data."""
    import random
    from datetime import timedelta
    random.seed(hash(metric) % 100)

    base_values = {
        "cpu_usage": 35.0,
        "memory_usage": 62.0,
        "storage_available_pct": 45.0,
        "device_temperature": 32.0,
        "battery_temperature": 29.0,
    }

    base = base_values.get(metric, 50.0)
    now = datetime.now(timezone.utc)
    trends = []

    hours = period_days * 24 if granularity == "hourly" else period_days
    step = timedelta(hours=1) if granularity == "hourly" else timedelta(days=1)

    for i in range(hours):
        ts = now - step * (hours - i - 1)
        # Add some variation with daily pattern
        hour_of_day = ts.hour
        daily_factor = 1 + 0.2 * (hour_of_day >= 9 and hour_of_day <= 17)  # Higher during work hours
        value = base * daily_factor + random.uniform(-5, 5)

        trends.append(HealthTrendResponse(
            timestamp=ts,
            value=max(0, value),
            device_count=random.randint(240, 250),
        ))

    return trends


def get_mock_storage_forecast() -> StorageForecastResponse:
    """Generate mock storage forecast data."""
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    devices = [
        StorageForecastDeviceResponse(
            device_id=1042,
            device_name="Warehouse-Scanner-042",
            current_storage_pct=8.5,
            storage_trend_gb_per_day=-0.15,
            projected_full_date=now + timedelta(days=12),
            days_until_full=12,
            confidence=0.89,
        ),
        StorageForecastDeviceResponse(
            device_id=1087,
            device_name="Floor-Tablet-087",
            current_storage_pct=12.3,
            storage_trend_gb_per_day=-0.22,
            projected_full_date=now + timedelta(days=18),
            days_until_full=18,
            confidence=0.85,
        ),
        StorageForecastDeviceResponse(
            device_id=1156,
            device_name="Shipping-Device-156",
            current_storage_pct=15.7,
            storage_trend_gb_per_day=-0.08,
            projected_full_date=now + timedelta(days=45),
            days_until_full=45,
            confidence=0.72,
        ),
    ]

    return StorageForecastResponse(
        tenant_id="default",
        devices_at_risk=devices,
        total_at_risk_count=len(devices),
        avg_days_until_full=sum(d.days_until_full or 0 for d in devices) / len(devices),
        recommendations=[
            "Clear app caches on devices with <15% storage remaining",
            "Review large media files on Warehouse-Scanner-042",
            "Consider storage expansion for frequently affected devices",
        ],
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/summary", response_model=SystemHealthSummaryResponse)
def get_system_health_summary(
    period_days: int = Query(7, ge=1, le=90, description="Analysis period in days"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get comprehensive system health summary for the fleet.

    Returns fleet health score, aggregated metrics, cohort breakdown,
    and actionable recommendations.
    """
    if mock_mode:
        return get_mock_system_health_summary(period_days)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.system_health_loader import (
            load_system_health_metrics,
            load_cohort_health_breakdown,
            calculate_fleet_health_score,
        )

        # Load metrics
        metrics_data = load_system_health_metrics(period_days=period_days)
        cohorts_data = load_cohort_health_breakdown(period_days=period_days)
        fleet_score, trend = calculate_fleet_health_score(period_days=period_days)

        # Convert to response models
        metrics = SystemHealthMetricsResponse(
            avg_cpu_usage=metrics_data.avg_cpu_usage,
            avg_memory_usage=metrics_data.avg_memory_usage,
            avg_storage_available_pct=metrics_data.avg_storage_available_pct,
            avg_device_temp=metrics_data.avg_device_temp,
            avg_battery_temp=metrics_data.avg_battery_temp,
            devices_high_cpu=metrics_data.devices_high_cpu,
            devices_high_memory=metrics_data.devices_high_memory,
            devices_low_storage=metrics_data.devices_low_storage,
            devices_high_temp=metrics_data.devices_high_temp,
            total_devices=metrics_data.total_devices,
        )

        cohorts = [
            CohortHealthResponse(
                cohort_id=c.cohort_id,
                cohort_name=c.cohort_name,
                device_count=c.device_count,
                health_score=c.health_score,
                avg_cpu=c.avg_cpu,
                avg_memory=c.avg_memory,
                avg_storage_pct=c.avg_storage_pct,
                devices_at_risk=c.devices_at_risk,
            )
            for c in cohorts_data
        ]

        # Generate recommendations
        recommendations = []
        if metrics.devices_low_storage > 5:
            recommendations.append(
                f"{metrics.devices_low_storage} devices have low storage - consider clearing cache or archiving data"
            )
        if metrics.devices_high_memory > 10:
            recommendations.append(
                f"{metrics.devices_high_memory} devices have high memory usage - review running apps"
            )
        if metrics.devices_high_temp > 0:
            recommendations.append(
                f"{metrics.devices_high_temp} devices running hot - check for intensive apps or charging issues"
            )
        if metrics.devices_high_cpu > 5:
            recommendations.append(
                f"{metrics.devices_high_cpu} devices have high CPU usage - investigate background processes"
            )

        # Calculate device status counts
        total = metrics.total_devices
        critical = metrics.devices_high_temp
        warning = metrics.devices_high_cpu + metrics.devices_high_memory + metrics.devices_low_storage
        healthy = max(0, total - warning - critical)

        return SystemHealthSummaryResponse(
            tenant_id=tenant_id,
            fleet_health_score=fleet_score,
            health_trend=trend,
            total_devices=total,
            healthy_count=healthy,
            warning_count=min(warning, total - critical),
            critical_count=min(critical, total),
            metrics=metrics,
            cohort_breakdown=cohorts,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Failed to get system health summary: {e}")
        # Return empty response on error
        return SystemHealthSummaryResponse(
            tenant_id=tenant_id,
            fleet_health_score=0,
            health_trend="unknown",
            total_devices=0,
            healthy_count=0,
            warning_count=0,
            critical_count=0,
            metrics=SystemHealthMetricsResponse(
                avg_cpu_usage=0, avg_memory_usage=0, avg_storage_available_pct=0,
                avg_device_temp=0, avg_battery_temp=0, devices_high_cpu=0,
                devices_high_memory=0, devices_low_storage=0, devices_high_temp=0,
                total_devices=0,
            ),
            cohort_breakdown=[],
            recommendations=["Unable to load system health data - check database connections"],
        )


@router.get("/trends", response_model=List[HealthTrendResponse])
def get_health_trends(
    metric: str = Query(..., description="Metric: cpu_usage, memory_usage, storage_available_pct, device_temperature, battery_temperature"),
    period_days: int = Query(7, ge=1, le=30, description="Period in days"),
    granularity: str = Query("hourly", description="Granularity: hourly or daily"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get time-series health trends for a specific metric.

    Useful for visualizing how system health changes over time.
    """
    if mock_mode:
        return get_mock_health_trends(metric, period_days, granularity)

    try:
        from device_anomaly.data_access.system_health_loader import load_health_trends

        trends_data = load_health_trends(
            metric=metric,
            period_days=period_days,
            granularity=granularity,
        )

        return [
            HealthTrendResponse(
                timestamp=t.timestamp,
                value=t.value,
                device_count=t.device_count,
            )
            for t in trends_data
        ]

    except Exception as e:
        logger.error(f"Failed to get health trends: {e}")
        return []


@router.get("/storage-forecast", response_model=StorageForecastResponse)
def get_storage_forecast(
    lookback_days: int = Query(30, ge=7, le=90, description="Days of history to analyze"),
    forecast_days: int = Query(90, ge=30, le=180, description="Days to forecast"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Predict which devices will run out of storage.

    Uses linear regression on storage history to forecast when devices
    will reach critical storage levels.
    """
    if mock_mode:
        return get_mock_storage_forecast()

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.system_health_loader import calculate_storage_forecast

        forecasts = calculate_storage_forecast(
            lookback_days=lookback_days,
            forecast_days=forecast_days,
        )

        devices = [
            StorageForecastDeviceResponse(
                device_id=f.device_id,
                device_name=f.device_name or f"Device-{f.device_id}",
                current_storage_pct=f.current_storage_pct,
                storage_trend_gb_per_day=f.storage_trend_gb_per_day,
                projected_full_date=f.projected_full_date,
                days_until_full=f.days_until_full,
                confidence=f.confidence,
            )
            for f in forecasts
        ]

        # Generate recommendations
        recommendations = []
        urgent_count = sum(1 for d in devices if d.days_until_full and d.days_until_full < 14)
        if urgent_count > 0:
            recommendations.append(
                f"{urgent_count} devices need immediate attention - storage will be exhausted within 2 weeks"
            )
        if devices:
            recommendations.append("Clear app caches on devices with <15% storage remaining")
            recommendations.append("Review and archive large media files")
            recommendations.append("Consider automated cleanup policies for affected device groups")

        avg_days = None
        if devices:
            days_list = [d.days_until_full for d in devices if d.days_until_full is not None]
            if days_list:
                avg_days = sum(days_list) / len(days_list)

        return StorageForecastResponse(
            tenant_id=tenant_id,
            devices_at_risk=devices,
            total_at_risk_count=len(devices),
            avg_days_until_full=avg_days,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Failed to get storage forecast: {e}")
        return StorageForecastResponse(
            tenant_id=tenant_id,
            devices_at_risk=[],
            total_at_risk_count=0,
            avg_days_until_full=None,
            recommendations=["Unable to calculate storage forecast - check database connections"],
        )


@router.get("/cohort-breakdown", response_model=List[CohortHealthResponse])
def get_cohort_breakdown(
    period_days: int = Query(7, ge=1, le=90, description="Analysis period in days"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get health breakdown by device cohort (model/OS).

    Helps identify if certain device types are experiencing more issues.
    """
    if mock_mode:
        summary = get_mock_system_health_summary(period_days)
        return summary.cohort_breakdown

    try:
        from device_anomaly.data_access.system_health_loader import load_cohort_health_breakdown

        cohorts_data = load_cohort_health_breakdown(period_days=period_days)

        return [
            CohortHealthResponse(
                cohort_id=c.cohort_id,
                cohort_name=c.cohort_name,
                device_count=c.device_count,
                health_score=c.health_score,
                avg_cpu=c.avg_cpu,
                avg_memory=c.avg_memory,
                avg_storage_pct=c.avg_storage_pct,
                devices_at_risk=c.devices_at_risk,
            )
            for c in cohorts_data
        ]

    except Exception as e:
        logger.error(f"Failed to get cohort breakdown: {e}")
        return []
