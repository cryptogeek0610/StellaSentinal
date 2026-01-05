"""API routes for Cost Intelligence Module."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import (
    get_backend_db,
    get_current_user,
    get_db,
    get_tenant_id,
    require_role,
)
from device_anomaly.api.models_cost import (
    AnomalyImpactResponse,
    AuditAction,
    CategoryCostSummary,
    CostBreakdownItem,
    CostCategory,
    CostChangeEntry,
    CostHistoryResponse,
    CostSummaryResponse,
    CostType,
    DeviceImpactResponse,
    DeviceImpactSummary,
    DeviceModelCostSummary,
    DeviceModelInfo,
    DeviceModelsResponse,
    FinancialImpactSummary,
    HardwareCostCreate,
    HardwareCostListResponse,
    HardwareCostResponse,
    HardwareCostUpdate,
    ImpactComponent,
    ImpactLevel,
    OperationalCostCreate,
    OperationalCostListResponse,
    OperationalCostResponse,
    OperationalCostUpdate,
    ScopeType,
)
from device_anomaly.database.schema import AnomalyResult, DeviceMetadata
from device_anomaly.db.models_cost import (
    CostAuditLog,
    CostCalculationCache,
    DeviceTypeCost,
    OperationalCost,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/costs", tags=["costs"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cents_to_dollars(cents: int) -> Decimal:
    """Convert cents to dollars."""
    return Decimal(cents) / 100 if cents else Decimal(0)


def dollars_to_cents(dollars: Decimal) -> int:
    """Convert dollars to cents."""
    return int(dollars * 100) if dollars else 0


def calculate_monthly_equivalent(amount: Decimal, cost_type: str) -> Decimal:
    """Calculate monthly equivalent of a cost based on its type."""
    multipliers = {
        "hourly": Decimal("160"),  # ~40 hours/week * 4 weeks
        "daily": Decimal("22"),    # ~22 working days/month
        "per_incident": Decimal("10"),  # Assume 10 incidents/month
        "fixed_monthly": Decimal("1"),
        "per_device": Decimal("1"),  # Per device is already monthly-ish
    }
    return amount * multipliers.get(cost_type, Decimal("1"))


def create_audit_log(
    db: Session,
    tenant_id: str,
    entity_type: str,
    entity_id: int,
    action: str,
    old_values: Optional[dict] = None,
    new_values: Optional[dict] = None,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
) -> CostAuditLog:
    """Create an audit log entry for cost changes."""
    changed_fields = []
    if old_values and new_values:
        for key in set(old_values.keys()) | set(new_values.keys()):
            if old_values.get(key) != new_values.get(key):
                changed_fields.append(key)

    log = CostAuditLog(
        tenant_id=tenant_id,
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        old_values_json=json.dumps(old_values) if old_values else None,
        new_values_json=json.dumps(new_values) if new_values else None,
        changed_fields_json=json.dumps(changed_fields) if changed_fields else None,
        user_id=user_id,
        user_email=user_email,
        source="api",
    )
    db.add(log)
    return log


# =============================================================================
# HARDWARE COSTS ENDPOINTS
# =============================================================================

@router.get("/hardware", response_model=HardwareCostListResponse)
def list_hardware_costs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None, max_length=255),
    db: Session = Depends(get_backend_db),
):
    """
    List all hardware cost entries for the tenant.

    Returns paginated list with device counts and fleet values.
    """
    tenant_id = get_tenant_id()

    # Base query for active costs
    query = db.query(DeviceTypeCost).filter(
        DeviceTypeCost.tenant_id == tenant_id,
        DeviceTypeCost.valid_to.is_(None),  # Active only
    )

    # Apply search filter
    if search:
        query = query.filter(DeviceTypeCost.device_model.ilike(f"%{search}%"))

    # Get total count
    total = query.count()
    total_pages = max(1, math.ceil(total / page_size))

    # Get paginated results
    offset = (page - 1) * page_size
    costs = query.order_by(DeviceTypeCost.device_model).offset(offset).limit(page_size).all()

    # Build response
    response_costs = []
    total_fleet_value = Decimal(0)

    for cost in costs:
        # Get device count for this model
        device_count = _get_device_count_for_model(db, tenant_id, cost.device_model)
        fleet_value = cents_to_dollars(cost.purchase_cost) * device_count

        response_costs.append(HardwareCostResponse(
            id=cost.id,
            tenant_id=cost.tenant_id,
            device_model=cost.device_model,
            purchase_cost=cents_to_dollars(cost.purchase_cost),
            replacement_cost=cents_to_dollars(cost.replacement_cost) if cost.replacement_cost else None,
            repair_cost_avg=cents_to_dollars(cost.repair_cost_avg) if cost.repair_cost_avg else None,
            depreciation_months=cost.depreciation_months,
            residual_value_percent=cost.residual_value_percent,
            warranty_months=cost.warranty_months,
            currency_code=cost.currency_code,
            notes=cost.notes,
            device_count=device_count,
            total_fleet_value=fleet_value,
            valid_from=cost.valid_from,
            valid_to=cost.valid_to,
            created_at=cost.created_at,
            updated_at=cost.updated_at,
            created_by=cost.created_by,
        ))
        total_fleet_value += fleet_value

    return HardwareCostListResponse(
        costs=response_costs,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        total_fleet_value=total_fleet_value,
    )


@router.get("/hardware/types", response_model=DeviceModelsResponse)
def get_device_models(
    db: Session = Depends(get_db),
):
    """
    Get unique device models from the database.

    Returns all distinct device_model values from DeviceMetadata,
    with counts and whether a cost entry exists.
    """
    tenant_id = get_tenant_id()

    # Get device models from DeviceMetadata
    models_query = (
        db.query(
            DeviceMetadata.device_model,
            func.count(DeviceMetadata.device_id).label("device_count"),
        )
        .filter(
            DeviceMetadata.tenant_id == tenant_id,
            DeviceMetadata.device_model.isnot(None),
        )
        .group_by(DeviceMetadata.device_model)
        .order_by(func.count(DeviceMetadata.device_id).desc())
        .all()
    )

    # Get backend db session for cost entries
    backend_db = next(get_backend_db())
    try:
        # Get existing cost entries
        existing_costs = (
            backend_db.query(DeviceTypeCost.device_model)
            .filter(
                DeviceTypeCost.tenant_id == tenant_id,
                DeviceTypeCost.valid_to.is_(None),
            )
            .all()
        )
        existing_models = {c.device_model for c in existing_costs}

        models = [
            DeviceModelInfo(
                device_model=model,
                device_count=count,
                has_cost_entry=model in existing_models,
            )
            for model, count in models_query
            if model  # Filter out None values
        ]
    finally:
        backend_db.close()

    return DeviceModelsResponse(
        models=models,
        total=len(models),
    )


@router.post("/hardware", response_model=HardwareCostResponse, status_code=201)
def create_hardware_cost(
    request: HardwareCostCreate,
    _: None = Depends(require_role(["admin"])),
    db: Session = Depends(get_backend_db),
):
    """
    Create a hardware cost entry.

    If an entry already exists for the device_model, it will be expired
    and a new version created.
    """
    tenant_id = get_tenant_id()
    user = get_current_user()

    # Check if active entry exists
    existing = (
        db.query(DeviceTypeCost)
        .filter(
            DeviceTypeCost.tenant_id == tenant_id,
            DeviceTypeCost.device_model == request.device_model,
            DeviceTypeCost.valid_to.is_(None),
        )
        .first()
    )

    if existing:
        # Expire the existing entry
        existing.valid_to = datetime.now(timezone.utc)

    # Create new entry
    new_cost = DeviceTypeCost(
        tenant_id=tenant_id,
        device_model=request.device_model,
        currency_code=request.currency_code,
        purchase_cost=dollars_to_cents(request.purchase_cost),
        replacement_cost=dollars_to_cents(request.replacement_cost) if request.replacement_cost else None,
        repair_cost_avg=dollars_to_cents(request.repair_cost_avg) if request.repair_cost_avg else None,
        depreciation_months=request.depreciation_months,
        residual_value_percent=request.residual_value_percent,
        warranty_months=request.warranty_months,
        notes=request.notes,
        created_by=user.user_id,
    )
    db.add(new_cost)
    db.flush()  # Get the ID

    # Create audit log
    create_audit_log(
        db=db,
        tenant_id=tenant_id,
        entity_type="device_type_cost",
        entity_id=new_cost.id,
        action="create",
        new_values=request.model_dump(exclude_none=True),
        user_id=user.user_id,
        user_email=user.email,
    )

    db.commit()

    # Get device count
    results_db = next(get_db())
    try:
        device_count = _get_device_count_for_model(results_db, tenant_id, new_cost.device_model)
    finally:
        results_db.close()

    return HardwareCostResponse(
        id=new_cost.id,
        tenant_id=new_cost.tenant_id,
        device_model=new_cost.device_model,
        purchase_cost=request.purchase_cost,
        replacement_cost=request.replacement_cost,
        repair_cost_avg=request.repair_cost_avg,
        depreciation_months=new_cost.depreciation_months,
        residual_value_percent=new_cost.residual_value_percent,
        warranty_months=new_cost.warranty_months,
        currency_code=new_cost.currency_code,
        notes=new_cost.notes,
        device_count=device_count,
        total_fleet_value=request.purchase_cost * device_count,
        valid_from=new_cost.valid_from,
        valid_to=new_cost.valid_to,
        created_at=new_cost.created_at,
        updated_at=new_cost.updated_at,
        created_by=new_cost.created_by,
    )


@router.delete("/hardware/{cost_id}", status_code=204)
def delete_hardware_cost(
    cost_id: int,
    _: None = Depends(require_role(["admin"])),
    db: Session = Depends(get_backend_db),
):
    """
    Delete a hardware cost entry by expiring it.
    """
    tenant_id = get_tenant_id()
    user = get_current_user()

    cost = (
        db.query(DeviceTypeCost)
        .filter(
            DeviceTypeCost.id == cost_id,
            DeviceTypeCost.tenant_id == tenant_id,
        )
        .first()
    )

    if not cost:
        raise HTTPException(status_code=404, detail="Hardware cost entry not found")

    # Expire the entry
    old_values = {
        "device_model": cost.device_model,
        "purchase_cost": float(cents_to_dollars(cost.purchase_cost)),
    }
    cost.valid_to = datetime.now(timezone.utc)

    # Create audit log
    create_audit_log(
        db=db,
        tenant_id=tenant_id,
        entity_type="device_type_cost",
        entity_id=cost_id,
        action="delete",
        old_values=old_values,
        user_id=user.user_id,
        user_email=user.email,
    )

    db.commit()


# =============================================================================
# OPERATIONAL COSTS ENDPOINTS
# =============================================================================

@router.get("/operational", response_model=OperationalCostListResponse)
def list_operational_costs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    category: Optional[CostCategory] = Query(None),
    is_active: Optional[bool] = Query(None),
    search: Optional[str] = Query(None, max_length=255),
    db: Session = Depends(get_backend_db),
):
    """
    List custom operational costs with filtering.
    """
    tenant_id = get_tenant_id()

    # Base query
    query = db.query(OperationalCost).filter(
        OperationalCost.tenant_id == tenant_id,
        OperationalCost.valid_to.is_(None),  # Active versions only
    )

    # Apply filters
    if category:
        query = query.filter(OperationalCost.category == category.value)
    if is_active is not None:
        query = query.filter(OperationalCost.is_active == is_active)
    if search:
        query = query.filter(OperationalCost.name.ilike(f"%{search}%"))

    # Get total
    total = query.count()
    total_pages = max(1, math.ceil(total / page_size))

    # Paginate
    offset = (page - 1) * page_size
    costs = query.order_by(OperationalCost.name).offset(offset).limit(page_size).all()

    # Build response
    response_costs = []
    total_monthly = Decimal(0)

    for cost in costs:
        amount_decimal = cents_to_dollars(cost.amount)
        monthly_eq = calculate_monthly_equivalent(amount_decimal, cost.cost_type)

        response_costs.append(OperationalCostResponse(
            id=cost.id,
            tenant_id=cost.tenant_id,
            name=cost.name,
            description=cost.description,
            category=CostCategory(cost.category),
            amount=amount_decimal,
            cost_type=CostType(cost.cost_type),
            unit=cost.unit,
            scope_type=ScopeType(cost.scope_type) if cost.scope_type else ScopeType.TENANT,
            scope_id=cost.scope_id,
            currency_code=cost.currency_code,
            is_active=cost.is_active,
            notes=cost.notes,
            valid_from=cost.valid_from,
            valid_to=cost.valid_to,
            monthly_equivalent=monthly_eq,
            annual_equivalent=monthly_eq * 12,
            created_at=cost.created_at,
            updated_at=cost.updated_at,
            created_by=cost.created_by,
        ))

        if cost.is_active:
            total_monthly += monthly_eq

    return OperationalCostListResponse(
        costs=response_costs,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        total_monthly_cost=total_monthly,
        total_annual_cost=total_monthly * 12,
    )


@router.post("/operational", response_model=OperationalCostResponse, status_code=201)
def create_operational_cost(
    request: OperationalCostCreate,
    _: None = Depends(require_role(["admin"])),
    db: Session = Depends(get_backend_db),
):
    """
    Create a new operational cost entry.
    """
    tenant_id = get_tenant_id()
    user = get_current_user()

    new_cost = OperationalCost(
        tenant_id=tenant_id,
        name=request.name,
        description=request.description,
        category=request.category.value,
        amount=dollars_to_cents(request.amount),
        cost_type=request.cost_type.value,
        unit=request.unit,
        scope_type=request.scope_type.value if request.scope_type else "tenant",
        scope_id=request.scope_id,
        currency_code=request.currency_code,
        is_active=request.is_active,
        notes=request.notes,
        created_by=user.user_id,
    )
    db.add(new_cost)
    db.flush()

    # Create audit log
    create_audit_log(
        db=db,
        tenant_id=tenant_id,
        entity_type="operational_cost",
        entity_id=new_cost.id,
        action="create",
        new_values=request.model_dump(exclude_none=True, mode="json"),
        user_id=user.user_id,
        user_email=user.email,
    )

    db.commit()

    monthly_eq = calculate_monthly_equivalent(request.amount, request.cost_type.value)

    return OperationalCostResponse(
        id=new_cost.id,
        tenant_id=new_cost.tenant_id,
        name=new_cost.name,
        description=new_cost.description,
        category=request.category,
        amount=request.amount,
        cost_type=request.cost_type,
        unit=new_cost.unit,
        scope_type=request.scope_type,
        scope_id=new_cost.scope_id,
        currency_code=new_cost.currency_code,
        is_active=new_cost.is_active,
        notes=new_cost.notes,
        valid_from=new_cost.valid_from,
        valid_to=new_cost.valid_to,
        monthly_equivalent=monthly_eq,
        annual_equivalent=monthly_eq * 12,
        created_at=new_cost.created_at,
        updated_at=new_cost.updated_at,
        created_by=new_cost.created_by,
    )


@router.put("/operational/{cost_id}", response_model=OperationalCostResponse)
def update_operational_cost(
    cost_id: int,
    request: OperationalCostUpdate,
    _: None = Depends(require_role(["admin"])),
    db: Session = Depends(get_backend_db),
):
    """
    Update an operational cost entry.
    """
    tenant_id = get_tenant_id()
    user = get_current_user()

    cost = (
        db.query(OperationalCost)
        .filter(
            OperationalCost.id == cost_id,
            OperationalCost.tenant_id == tenant_id,
            OperationalCost.valid_to.is_(None),
        )
        .first()
    )

    if not cost:
        raise HTTPException(status_code=404, detail="Operational cost entry not found")

    # Track old values for audit
    old_values = {
        "name": cost.name,
        "amount": float(cents_to_dollars(cost.amount)),
        "category": cost.category,
    }

    # Update fields
    update_data = request.model_dump(exclude_unset=True)
    new_values = {}

    for field, value in update_data.items():
        if value is not None:
            if field == "amount":
                setattr(cost, field, dollars_to_cents(value))
                new_values[field] = float(value)
            elif field in ("category", "cost_type", "scope_type"):
                setattr(cost, field, value.value if hasattr(value, "value") else value)
                new_values[field] = value.value if hasattr(value, "value") else value
            else:
                setattr(cost, field, value)
                new_values[field] = value

    cost.updated_at = datetime.now(timezone.utc)

    # Create audit log
    create_audit_log(
        db=db,
        tenant_id=tenant_id,
        entity_type="operational_cost",
        entity_id=cost_id,
        action="update",
        old_values=old_values,
        new_values=new_values,
        user_id=user.user_id,
        user_email=user.email,
    )

    db.commit()

    amount_decimal = cents_to_dollars(cost.amount)
    monthly_eq = calculate_monthly_equivalent(amount_decimal, cost.cost_type)

    return OperationalCostResponse(
        id=cost.id,
        tenant_id=cost.tenant_id,
        name=cost.name,
        description=cost.description,
        category=CostCategory(cost.category),
        amount=amount_decimal,
        cost_type=CostType(cost.cost_type),
        unit=cost.unit,
        scope_type=ScopeType(cost.scope_type) if cost.scope_type else ScopeType.TENANT,
        scope_id=cost.scope_id,
        currency_code=cost.currency_code,
        is_active=cost.is_active,
        notes=cost.notes,
        valid_from=cost.valid_from,
        valid_to=cost.valid_to,
        monthly_equivalent=monthly_eq,
        annual_equivalent=monthly_eq * 12,
        created_at=cost.created_at,
        updated_at=cost.updated_at,
        created_by=cost.created_by,
    )


@router.delete("/operational/{cost_id}", status_code=204)
def delete_operational_cost(
    cost_id: int,
    _: None = Depends(require_role(["admin"])),
    db: Session = Depends(get_backend_db),
):
    """
    Delete an operational cost entry.
    """
    tenant_id = get_tenant_id()
    user = get_current_user()

    cost = (
        db.query(OperationalCost)
        .filter(
            OperationalCost.id == cost_id,
            OperationalCost.tenant_id == tenant_id,
        )
        .first()
    )

    if not cost:
        raise HTTPException(status_code=404, detail="Operational cost entry not found")

    old_values = {"name": cost.name, "amount": float(cents_to_dollars(cost.amount))}
    cost.valid_to = datetime.now(timezone.utc)
    cost.is_active = False

    # Create audit log
    create_audit_log(
        db=db,
        tenant_id=tenant_id,
        entity_type="operational_cost",
        entity_id=cost_id,
        action="delete",
        old_values=old_values,
        user_id=user.user_id,
        user_email=user.email,
    )

    db.commit()


# =============================================================================
# COST SUMMARY ENDPOINTS
# =============================================================================

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
            OperationalCost.is_active == True,
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
            count = _get_device_count_for_model(results_db, tenant_id, cost.device_model)
            unit_cost = cents_to_dollars(cost.purchase_cost)
            total_value = unit_cost * count

            total_hardware_value += total_value
            device_count += count

            by_device_model.append(DeviceModelCostSummary(
                device_model=cost.device_model,
                device_count=count,
                unit_cost=unit_cost,
                total_value=total_value,
            ))

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
            by_category.append(CategoryCostSummary(
                category="hardware",
                total_cost=total_hardware_value,
                item_count=len(hardware_costs),
                percentage_of_total=float(total_hardware_value / total_all * 100) if total_all > 0 else 0,
            ))

        for cat, data in category_totals.items():
            annual = data["total"] * 12
            by_category.append(CategoryCostSummary(
                category=cat,
                total_cost=annual,
                item_count=data["count"],
                percentage_of_total=float(annual / total_all * 100) if total_all > 0 else 0,
            ))

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
        calculated_at=datetime.now(timezone.utc),
        device_count=device_count,
        anomaly_count_period=0,
    )


# =============================================================================
# IMPACT CALCULATION ENDPOINTS
# =============================================================================

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
    device = (
        db.query(DeviceMetadata)
        .filter(DeviceMetadata.device_id == anomaly.device_id)
        .first()
    )

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

        # Calculate impact components
        components = []
        total_impact = Decimal(0)

        # Hardware impact (based on severity)
        severity = _get_severity(anomaly.anomaly_score)
        if device_cost:
            unit_cost = cents_to_dollars(device_cost.purchase_cost)
            severity_multipliers = {"critical": 0.15, "high": 0.10, "medium": 0.05, "low": 0.02}
            hardware_risk = unit_cost * Decimal(str(severity_multipliers.get(severity, 0.02)))

            components.append(ImpactComponent(
                type="direct",
                description="Potential hardware damage based on anomaly severity",
                amount=hardware_risk,
                calculation_method=f"{severity_multipliers.get(severity, 0.02)*100:.0f}% of device cost based on {severity} severity",
                confidence=0.6,
            ))
            total_impact += hardware_risk

        # Productivity impact (estimate)
        hourly_rate = Decimal("25")  # Default hourly rate
        downtime_hours = Decimal("1.5")  # Estimated investigation time
        productivity_loss = hourly_rate * downtime_hours

        components.append(ImpactComponent(
            type="productivity",
            description="Estimated worker downtime for issue resolution",
            amount=productivity_loss,
            calculation_method=f"{downtime_hours} hours at ${hourly_rate}/hour",
            confidence=0.7,
        ))
        total_impact += productivity_loss

        # Support cost
        it_hourly = Decimal("50")
        support_hours = Decimal("0.5")
        support_cost = it_hourly * support_hours

        components.append(ImpactComponent(
            type="support",
            description="IT investigation and resolution time",
            amount=support_cost,
            calculation_method=f"{support_hours} hours at ${it_hourly}/hour IT rate",
            confidence=0.8,
        ))
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
        device_replacement_cost=cents_to_dollars(device_cost.replacement_cost) if device_cost and device_cost.replacement_cost else None,
        estimated_downtime_hours=float(downtime_hours),
        productivity_cost_per_hour=hourly_rate,
        productivity_impact=productivity_loss,
        support_cost_per_hour=it_hourly,
        estimated_support_cost=support_cost,
        similar_cases_count=0,
        overall_confidence=0.7,
        confidence_explanation="Estimate based on device cost data and default labor rates",
        impact_level=impact_level,
        calculated_at=datetime.now(timezone.utc),
    )


# =============================================================================
# COST HISTORY ENDPOINTS
# =============================================================================

@router.get("/history", response_model=CostHistoryResponse)
def get_cost_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    entity_type: Optional[str] = Query(None, pattern="^(device_type_cost|operational_cost)$"),
    action: Optional[AuditAction] = Query(None),
    _: None = Depends(require_role(["admin"])),
    db: Session = Depends(get_backend_db),
):
    """
    Get audit trail of cost changes.
    """
    tenant_id = get_tenant_id()

    # Base query
    query = db.query(CostAuditLog).filter(
        CostAuditLog.tenant_id == tenant_id,
    )

    # Apply filters
    if entity_type:
        query = query.filter(CostAuditLog.entity_type == entity_type)
    if action:
        query = query.filter(CostAuditLog.action == action.value)

    # Get counts
    total = query.count()
    total_pages = max(1, math.ceil(total / page_size))

    # Count by action
    action_counts = (
        db.query(CostAuditLog.action, func.count(CostAuditLog.id))
        .filter(CostAuditLog.tenant_id == tenant_id)
        .group_by(CostAuditLog.action)
        .all()
    )
    action_count_map = {a: c for a, c in action_counts}

    # Get unique users
    unique_users = (
        db.query(func.count(func.distinct(CostAuditLog.user_id)))
        .filter(
            CostAuditLog.tenant_id == tenant_id,
            CostAuditLog.user_id.isnot(None),
        )
        .scalar()
    )

    # Paginate
    offset = (page - 1) * page_size
    logs = (
        query
        .order_by(CostAuditLog.timestamp.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    # Build response
    changes = []
    for log in logs:
        # Parse JSON fields
        old_values = json.loads(log.old_values_json) if log.old_values_json else None
        new_values = json.loads(log.new_values_json) if log.new_values_json else None
        changed_fields = json.loads(log.changed_fields_json) if log.changed_fields_json else []

        # Get entity name
        entity_name = ""
        if new_values:
            entity_name = new_values.get("name") or new_values.get("device_model") or ""
        elif old_values:
            entity_name = old_values.get("name") or old_values.get("device_model") or ""

        changes.append(CostChangeEntry(
            id=log.id,
            timestamp=log.timestamp,
            action=AuditAction(log.action),
            entity_type=log.entity_type,
            entity_id=log.entity_id,
            entity_name=entity_name,
            changed_by=log.user_id,
            field_changed=changed_fields[0] if changed_fields else None,
            old_value=str(old_values.get(changed_fields[0])) if old_values and changed_fields else None,
            new_value=str(new_values.get(changed_fields[0])) if new_values and changed_fields else None,
            before_snapshot=old_values,
            after_snapshot=new_values,
        ))

    return CostHistoryResponse(
        changes=changes,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        total_creates=action_count_map.get("create", 0),
        total_updates=action_count_map.get("update", 0),
        total_deletes=action_count_map.get("delete", 0),
        unique_users=unique_users or 0,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_device_count_for_model(db: Session, tenant_id: str, device_model: str) -> int:
    """Get count of devices matching a device model."""
    return (
        db.query(func.count(DeviceMetadata.device_id))
        .filter(
            DeviceMetadata.tenant_id == tenant_id,
            DeviceMetadata.device_model == device_model,
        )
        .scalar()
    ) or 0


def _get_severity(score: float) -> str:
    """Convert anomaly score to severity level."""
    if score <= -0.7:
        return "critical"
    if score <= -0.5:
        return "high"
    if score <= -0.3:
        return "medium"
    return "low"
