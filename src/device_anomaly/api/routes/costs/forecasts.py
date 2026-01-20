"""
Battery forecast and NFF (No Fault Found) summary endpoints.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_backend_db, get_db, get_tenant_id
from device_anomaly.api.models_cost import (
    BatteryForecastEntry,
    BatteryForecastResponse,
    DeviceImpactResponse,
    DeviceImpactSummary,
    NFFByDeviceModel,
    NFFByResolution,
    NFFSummaryResponse,
)
from device_anomaly.database.schema import AnomalyResult, DeviceFeature, DeviceMetadata
from device_anomaly.db.models_cost import CostCalculationCache, DeviceTypeCost

from .utils import cents_to_dollars

router = APIRouter()


@router.get("/battery-forecast", response_model=BatteryForecastResponse)
def get_battery_forecast(
    device_id: Optional[int] = Query(None, ge=1),
    horizon_days: int = Query(90, ge=30, le=365),
    db: Session = Depends(get_backend_db),
    results_db: Session = Depends(get_db),
) -> BatteryForecastResponse:
    """Forecast battery replacement costs using real battery health data when available."""
    tenant_id = get_tenant_id()

    # Get device counts by model
    device_query = results_db.query(
        DeviceMetadata.device_model,
        DeviceMetadata.device_id,
    ).filter(DeviceMetadata.tenant_id == tenant_id)
    if device_id is not None:
        device_query = device_query.filter(DeviceMetadata.device_id == device_id)

    devices = device_query.all()
    device_ids_by_model: Dict[str, List[int]] = {}
    for model, dev_id in devices:
        if model:
            device_ids_by_model.setdefault(model, []).append(dev_id)

    # Try to get real battery health data from DeviceFeature
    battery_health_by_device: Dict[int, Optional[float]] = {}
    devices_with_health = 0

    # Query latest DeviceFeature per device
    all_device_ids = [d for ids in device_ids_by_model.values() for d in ids]
    if all_device_ids:
        # Get latest feature record for each device
        subq = (
            results_db.query(DeviceFeature.device_id, func.max(DeviceFeature.computed_at).label("latest"))
            .filter(DeviceFeature.tenant_id == tenant_id, DeviceFeature.device_id.in_(all_device_ids))
            .group_by(DeviceFeature.device_id)
            .subquery()
        )

        latest_features = (
            results_db.query(DeviceFeature)
            .join(subq, (DeviceFeature.device_id == subq.c.device_id) & (DeviceFeature.computed_at == subq.c.latest))
            .filter(DeviceFeature.tenant_id == tenant_id)
            .all()
        )

        for feature in latest_features:
            if feature.feature_values_json:
                try:
                    features = json.loads(feature.feature_values_json)
                    # Look for BatteryHealth in the feature data
                    health = features.get("BatteryHealth") or features.get("battery_health")
                    if health is not None and isinstance(health, (int, float)) and 0 <= health <= 100:
                        battery_health_by_device[feature.device_id] = float(health)
                        devices_with_health += 1
                except (json.JSONDecodeError, TypeError):
                    pass

    # Get cost entries
    cost_entries = (
        db.query(DeviceTypeCost)
        .filter(
            DeviceTypeCost.tenant_id == tenant_id,
            DeviceTypeCost.valid_to.is_(None),
        )
        .all()
    )
    cost_map: Dict[str, DeviceTypeCost] = {entry.device_model: entry for entry in cost_entries}

    forecasts: List[BatteryForecastEntry] = []
    total_cost_30 = Decimal(0)
    total_cost_90 = Decimal(0)
    total_due_30 = 0
    total_due_90 = 0
    total_devices = 0
    overall_data_quality = "estimated"
    any_real_data = False
    all_real_data = True

    for model, dev_ids in device_ids_by_model.items():
        count = len(dev_ids)
        cost_entry = cost_map.get(model)

        if cost_entry:
            base_cost = (
                cents_to_dollars(cost_entry.replacement_cost)
                if cost_entry.replacement_cost
                else cents_to_dollars(cost_entry.purchase_cost)
            )
            lifespan = cost_entry.battery_lifespan_months or cost_entry.warranty_months or cost_entry.depreciation_months or 24
            battery_cost = cost_entry.battery_replacement_cost
            if battery_cost:
                battery_replacement_cost = cents_to_dollars(battery_cost)
            else:
                battery_replacement_cost = (base_cost * Decimal("0.2")).quantize(Decimal("0.01"))
        else:
            base_cost = Decimal(200)
            lifespan = 24
            battery_replacement_cost = Decimal("40.00")

        # Check if we have real battery health data for this model
        health_values = [battery_health_by_device.get(d) for d in dev_ids if battery_health_by_device.get(d) is not None]

        if health_values:
            # Use real battery health data
            any_real_data = True
            avg_health = sum(health_values) / len(health_values)
            model_data_quality = "real" if len(health_values) == count else "mixed"

            # Devices needing replacement based on health
            due_now = sum(1 for h in health_values if h < 60)
            due_soon = sum(1 for h in health_values if 60 <= h < 70)
            due_90 = sum(1 for h in health_values if h < 80)

            # For devices without health data in this model, estimate
            devices_without_health = count - len(health_values)
            if devices_without_health > 0:
                due_now += int(devices_without_health * 0.05)
                due_soon += int(devices_without_health * 0.03)
                due_90 += int(devices_without_health * 0.15)

            avg_age = round(lifespan * (1 - avg_health / 100))
        else:
            all_real_data = False
            model_data_quality = "estimated"
            avg_health = None

            avg_age = round(lifespan * 0.7)
            age_ratio = min(avg_age / lifespan, 1.0) if lifespan else 0.0

            due_now = int(count * 0.1 * age_ratio)
            due_soon = int(count * 0.08 * age_ratio)
            due_90 = max(due_now, int(count * 0.25 * age_ratio))

        oldest_age = round(min(lifespan * 1.2, lifespan + 6))

        cost_30 = battery_replacement_cost * due_now
        cost_90 = battery_replacement_cost * due_90

        forecasts.append(
            BatteryForecastEntry(
                device_model=model,
                device_count=count,
                battery_replacement_cost=battery_replacement_cost,
                battery_lifespan_months=lifespan,
                devices_due_this_month=due_now,
                devices_due_next_month=due_soon,
                devices_due_in_90_days=due_90,
                estimated_cost_30_days=cost_30,
                estimated_cost_90_days=cost_90,
                avg_battery_age_months=avg_age,
                oldest_battery_months=oldest_age,
                avg_battery_health_percent=round(avg_health, 1) if avg_health is not None else None,
                data_quality=model_data_quality,
            )
        )

        total_devices += count
        total_cost_30 += cost_30
        total_cost_90 += cost_90
        total_due_30 += due_now
        total_due_90 += due_90

    # Determine overall data quality
    if any_real_data and all_real_data:
        overall_data_quality = "real"
    elif any_real_data:
        overall_data_quality = "mixed"
    else:
        overall_data_quality = "estimated"

    return BatteryForecastResponse(
        forecasts=forecasts,
        total_devices_with_battery_data=total_devices,
        total_estimated_cost_30_days=total_cost_30,
        total_estimated_cost_90_days=total_cost_90,
        total_replacements_due_30_days=total_due_30,
        total_replacements_due_90_days=total_due_90,
        forecast_generated_at=datetime.now(timezone.utc),
        data_quality=overall_data_quality,
        devices_with_health_data=devices_with_health,
    )


@router.get("/nff/summary", response_model=NFFSummaryResponse)
def get_nff_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    results_db: Session = Depends(get_db),
) -> NFFSummaryResponse:
    """Return summary of No Fault Found (false positive) investigations."""
    tenant_id = get_tenant_id()
    end_dt = end_date or datetime.now(timezone.utc)
    start_dt = start_date or (end_dt - timedelta(days=30))
    prev_start = start_dt - timedelta(days=30)

    base_query = results_db.query(AnomalyResult).filter(
        AnomalyResult.tenant_id == tenant_id,
        AnomalyResult.timestamp >= start_dt,
        AnomalyResult.timestamp <= end_dt,
    )
    total_anomalies = base_query.count()
    nff_query = base_query.filter(AnomalyResult.status == "false_positive")
    total_nff = nff_query.count()

    cost_per_nff = Decimal("75")
    total_nff_cost = cost_per_nff * total_nff
    avg_cost = (total_nff_cost / total_nff) if total_nff else Decimal(0)
    nff_rate = (total_nff / total_anomalies * 100.0) if total_anomalies else 0.0

    device_model_rows = (
        results_db.query(DeviceMetadata.device_model, func.count(AnomalyResult.id))
        .join(DeviceMetadata, DeviceMetadata.device_id == AnomalyResult.device_id)
        .filter(
            AnomalyResult.tenant_id == tenant_id,
            AnomalyResult.status == "false_positive",
            AnomalyResult.timestamp >= start_dt,
            AnomalyResult.timestamp <= end_dt,
        )
        .group_by(DeviceMetadata.device_model)
        .all()
    )
    by_device_model = [
        NFFByDeviceModel(
            device_model=model or "unknown",
            count=count,
            total_cost=cost_per_nff * count,
        )
        for model, count in device_model_rows
    ]

    by_resolution = [
        NFFByResolution(
            resolution="no_fault_found",
            count=total_nff,
            total_cost=total_nff_cost,
        )
    ]

    prev_count = (
        results_db.query(func.count(AnomalyResult.id))
        .filter(
            AnomalyResult.tenant_id == tenant_id,
            AnomalyResult.status == "false_positive",
            AnomalyResult.timestamp >= prev_start,
            AnomalyResult.timestamp < start_dt,
        )
        .scalar()
    ) or 0
    if prev_count == 0:
        trend = 0.0 if total_nff == 0 else 100.0
    else:
        trend = ((total_nff - prev_count) / prev_count) * 100.0

    return NFFSummaryResponse(
        total_nff_count=total_nff,
        total_nff_cost=total_nff_cost,
        avg_cost_per_nff=avg_cost,
        nff_rate_percent=nff_rate,
        by_device_model=by_device_model,
        by_resolution=by_resolution,
        trend_30_days=trend,
    )


@router.get("/device/{device_id}/impact", response_model=DeviceImpactResponse)
def get_device_cost_impact(
    device_id: int,
    db: Session = Depends(get_backend_db),
    results_db: Session = Depends(get_db),
) -> DeviceImpactResponse:
    """Return cost impact summary for a specific device."""
    tenant_id = get_tenant_id()

    device = (
        results_db.query(DeviceMetadata)
        .filter(
            DeviceMetadata.tenant_id == tenant_id,
            DeviceMetadata.device_id == device_id,
        )
        .first()
    )

    device_model = device.device_model if device else None
    device_name = device.device_name if device else None

    anomaly_query = results_db.query(AnomalyResult).filter(
        AnomalyResult.tenant_id == tenant_id,
        AnomalyResult.device_id == device_id,
    )
    total_anomalies = anomaly_query.count()
    open_anomalies = anomaly_query.filter(AnomalyResult.status == "open").count()
    resolved_anomalies = anomaly_query.filter(AnomalyResult.status == "resolved").count()

    per_anomaly_cost = Decimal("100")
    anomaly_cost = per_anomaly_cost * total_anomalies

    unit_cost = None
    current_value = None
    if device_model:
        cost_entry = (
            db.query(DeviceTypeCost)
            .filter(
                DeviceTypeCost.tenant_id == tenant_id,
                DeviceTypeCost.device_model == device_model,
                DeviceTypeCost.valid_to.is_(None),
            )
            .first()
        )
        if cost_entry:
            unit_cost = cents_to_dollars(cost_entry.purchase_cost)
            residual = Decimal(cost_entry.residual_value_percent or 0) / Decimal(100)
            current_value = unit_cost * (Decimal(1) - residual)

    cache = (
        db.query(CostCalculationCache)
        .filter(
            CostCalculationCache.tenant_id == tenant_id,
            CostCalculationCache.entity_type == "device",
            CostCalculationCache.entity_id == str(device_id),
        )
        .order_by(CostCalculationCache.calculated_at.desc())
        .first()
    )
    total_cost = cache.total_cost_decimal if cache else anomaly_cost

    risk_score = min(1.0, total_anomalies / 10.0) if total_anomalies else 0.0
    if risk_score >= 0.7:
        risk_level = "critical"
    elif risk_score >= 0.4:
        risk_level = "high"
    elif risk_score >= 0.2:
        risk_level = "medium"
    else:
        risk_level = "low"

    summary = DeviceImpactSummary(
        device_id=device_id,
        device_model=device_model,
        device_name=device_name,
        location=device.location if device else None,
        unit_cost=unit_cost,
        current_value=current_value,
        total_anomalies=total_anomalies,
        open_anomalies=open_anomalies,
        resolved_anomalies=resolved_anomalies,
        total_estimated_impact=total_cost,
        impact_mtd=total_cost,
        impact_ytd=total_cost,
        risk_score=risk_score,
        risk_level=risk_level,
    )

    monthly_trend = {}
    now = datetime.now(timezone.utc)
    for idx in range(6):
        month = (now - timedelta(days=30 * idx)).strftime("%Y-%m")
        monthly_trend[month] = Decimal(0)

    recommendations = []
    if risk_score >= 0.4:
        recommendations.append("Review device usage patterns and retrain models for this cohort.")
    if open_anomalies > 0:
        recommendations.append("Resolve open anomalies to reduce recurring costs.")
    if not recommendations:
        recommendations.append("Continue monitoring device performance and costs.")

    return DeviceImpactResponse(
        device_id=device_id,
        device_model=device_model,
        device_name=device_name,
        summary=summary,
        recent_anomalies=[],
        monthly_impact_trend=monthly_trend,
        cost_saving_recommendations=recommendations,
    )
