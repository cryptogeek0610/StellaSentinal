"""
Cost summary endpoints.
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_backend_db, get_db, get_tenant_id
from device_anomaly.api.models_cost import (
    CategoryCostSummary,
    CostSummaryResponse,
    DeviceModelCostSummary,
)
from device_anomaly.db.models_cost import DeviceTypeCost, OperationalCost

from .utils import calculate_monthly_equivalent, cents_to_dollars, get_device_count_for_model

router = APIRouter()


@router.get("/summary", response_model=CostSummaryResponse)
def get_cost_summary(
    period: str = Query("current_month", pattern="^(current_month|ytd|last_30d|all_time)$"),
    db: Session = Depends(get_backend_db),
):
    """
    Get total costs aggregated by category.
    """
    tenant_id = get_tenant_id()

    # Get hardware costs
    hardware_costs = (
        db.query(DeviceTypeCost)
        .filter(
            DeviceTypeCost.tenant_id == tenant_id,
            DeviceTypeCost.valid_to.is_(None),
        )
        .all()
    )

    # Get operational costs
    operational_costs = (
        db.query(OperationalCost)
        .filter(
            OperationalCost.tenant_id == tenant_id,
            OperationalCost.valid_to.is_(None),
            OperationalCost.is_active,
        )
        .all()
    )

    # Calculate totals
    results_db = next(get_db())
    try:
        total_hardware_value = Decimal(0)
        device_count = 0
        by_device_model = []

        for cost in hardware_costs:
            count = get_device_count_for_model(results_db, tenant_id, cost.device_model)
            unit_cost = cents_to_dollars(cost.purchase_cost)
            total_value = unit_cost * count

            total_hardware_value += total_value
            device_count += count

            by_device_model.append(
                DeviceModelCostSummary(
                    device_model=cost.device_model,
                    device_count=count,
                    unit_cost=unit_cost,
                    total_value=total_value,
                )
            )

        # Calculate operational totals
        total_operational_monthly = Decimal(0)
        category_totals = {}

        for cost in operational_costs:
            amount = cents_to_dollars(cost.amount)
            monthly = calculate_monthly_equivalent(amount, cost.cost_type)
            total_operational_monthly += monthly

            cat = cost.category
            if cat not in category_totals:
                category_totals[cat] = {"total": Decimal(0), "count": 0}
            category_totals[cat]["total"] += monthly
            category_totals[cat]["count"] += 1

        # Build category breakdown
        total_all = total_hardware_value + (total_operational_monthly * 12)
        by_category = []

        # Add hardware as a category
        if total_hardware_value > 0:
            by_category.append(
                CategoryCostSummary(
                    category="hardware",
                    total_cost=total_hardware_value,
                    item_count=len(hardware_costs),
                    percentage_of_total=float(total_hardware_value / total_all * 100) if total_all > 0 else 0,
                )
            )

        for cat, data in category_totals.items():
            annual = data["total"] * 12
            by_category.append(
                CategoryCostSummary(
                    category=cat,
                    total_cost=annual,
                    item_count=data["count"],
                    percentage_of_total=float(annual / total_all * 100) if total_all > 0 else 0,
                )
            )

    finally:
        results_db.close()

    return CostSummaryResponse(
        tenant_id=tenant_id,
        summary_period=period,
        total_hardware_value=total_hardware_value,
        total_operational_monthly=total_operational_monthly,
        total_operational_annual=total_operational_monthly * 12,
        total_anomaly_impact_mtd=Decimal(0),  # TODO: Calculate from cache
        total_anomaly_impact_ytd=Decimal(0),  # TODO: Calculate from cache
        by_category=by_category,
        by_device_model=by_device_model,
        cost_trend_30d=None,
        anomaly_cost_trend_30d=None,
        calculated_at=datetime.now(UTC),
        device_count=device_count,
        anomaly_count_period=0,
    )
