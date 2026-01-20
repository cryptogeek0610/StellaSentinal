"""
Anomaly financial impact calculation endpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_backend_db, get_db, get_tenant_id, require_role
from device_anomaly.api.models_cost import (
    AnomalyImpactResponse,
    ImpactComponent,
    ImpactLevel,
)
from device_anomaly.database.schema import AnomalyResult, DeviceMetadata
from device_anomaly.db.models_cost import DeviceTypeCost, OperationalCost

from .utils import cents_to_dollars, get_severity

router = APIRouter()


@router.get("/impact/{anomaly_id}", response_model=AnomalyImpactResponse)
def get_anomaly_impact(
    anomaly_id: int,
    _: None = Depends(require_role(["analyst", "admin"])),
    db: Session = Depends(get_db),
):
    """
    Calculate financial impact for a specific anomaly.
    """
    tenant_id = get_tenant_id()

    # Get the anomaly
    anomaly = (
        db.query(AnomalyResult)
        .filter(
            AnomalyResult.id == anomaly_id,
            AnomalyResult.tenant_id == tenant_id,
        )
        .first()
    )

    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    # Get device info
    device = db.query(DeviceMetadata).filter(DeviceMetadata.device_id == anomaly.device_id).first()

    device_model = device.device_model if device else None

    # Get cost data from backend db
    backend_db = next(get_backend_db())
    try:
        device_cost = None
        if device_model:
            device_cost = (
                backend_db.query(DeviceTypeCost)
                .filter(
                    DeviceTypeCost.tenant_id == tenant_id,
                    DeviceTypeCost.device_model == device_model,
                    DeviceTypeCost.valid_to.is_(None),
                )
                .first()
            )

        # Check if user has configured any labor/support operational costs
        labor_cost_entry = (
            backend_db.query(OperationalCost)
            .filter(
                OperationalCost.tenant_id == tenant_id,
                OperationalCost.category.in_(["labor", "support"]),
                OperationalCost.is_active == True,
                OperationalCost.valid_to.is_(None),
            )
            .first()
        )

        # Determine if we're using defaults (no custom costs configured)
        using_defaults = device_cost is None and labor_cost_entry is None

        # Calculate impact components
        components = []
        total_impact = Decimal(0)

        # Hardware impact (based on severity)
        severity = get_severity(anomaly.anomaly_score)
        if device_cost:
            unit_cost = cents_to_dollars(device_cost.purchase_cost)
            severity_multipliers = {"critical": 0.15, "high": 0.10, "medium": 0.05, "low": 0.02}
            hardware_risk = unit_cost * Decimal(str(severity_multipliers.get(severity, 0.02)))

            components.append(
                ImpactComponent(
                    type="direct",
                    description="Potential hardware damage based on anomaly severity",
                    amount=hardware_risk,
                    calculation_method=f"{severity_multipliers.get(severity, 0.02)*100:.0f}% of device cost based on {severity} severity",
                    confidence=0.6,
                )
            )
            total_impact += hardware_risk

        # Productivity impact (estimate)
        hourly_rate = Decimal("25")  # Default hourly rate
        downtime_hours = Decimal("1.5")  # Estimated investigation time
        productivity_loss = hourly_rate * downtime_hours

        components.append(
            ImpactComponent(
                type="productivity",
                description="Estimated worker downtime for issue resolution",
                amount=productivity_loss,
                calculation_method=f"{downtime_hours} hours at ${hourly_rate}/hour",
                confidence=0.7,
            )
        )
        total_impact += productivity_loss

        # Support cost
        it_hourly = Decimal("50")
        support_hours = Decimal("0.5")
        support_cost = it_hourly * support_hours

        components.append(
            ImpactComponent(
                type="support",
                description="IT investigation and resolution time",
                amount=support_cost,
                calculation_method=f"{support_hours} hours at ${it_hourly}/hour IT rate",
                confidence=0.8,
            )
        )
        total_impact += support_cost

        # Determine impact level
        if total_impact > 500:
            impact_level = ImpactLevel.HIGH
        elif total_impact > 100:
            impact_level = ImpactLevel.MEDIUM
        else:
            impact_level = ImpactLevel.LOW

    finally:
        backend_db.close()

    return AnomalyImpactResponse(
        anomaly_id=anomaly_id,
        device_id=anomaly.device_id,
        device_model=device_model,
        anomaly_severity=severity,
        total_estimated_impact=total_impact,
        impact_components=components,
        device_unit_cost=cents_to_dollars(device_cost.purchase_cost) if device_cost else None,
        device_replacement_cost=cents_to_dollars(device_cost.replacement_cost)
        if device_cost and device_cost.replacement_cost
        else None,
        estimated_downtime_hours=float(downtime_hours),
        productivity_cost_per_hour=hourly_rate,
        productivity_impact=productivity_loss,
        support_cost_per_hour=it_hourly,
        estimated_support_cost=support_cost,
        similar_cases_count=0,
        overall_confidence=0.7,
        confidence_explanation="Estimate based on device cost data and default labor rates"
        if using_defaults
        else "Estimate based on your configured cost data",
        impact_level=impact_level,
        using_defaults=using_defaults,
        calculated_at=datetime.now(timezone.utc),
    )
