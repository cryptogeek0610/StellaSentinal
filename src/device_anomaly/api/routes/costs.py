"""API routes for Cost Intelligence Module."""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

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
    BatteryForecastEntry,
    BatteryForecastResponse,
    CategoryCostSummary,
    CostBreakdownItem,
    CostCategory,
    CostChangeEntry,
    CostHistoryResponse,
    CostSummaryResponse,
    CostType,
    CostAlert,
    CostAlertCreate,
    CostAlertListResponse,
    CostAlertUpdate,
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
    NFFByDeviceModel,
    NFFByResolution,
    NFFSummaryResponse,
    ScopeType,
)
from device_anomaly.database.schema import AnomalyResult, DeviceFeature, DeviceMetadata
from device_anomaly.db.models_cost import (
    CostAuditLog,
    CostCalculationCache,
    DeviceTypeCost,
    OperationalCost,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/costs", tags=["costs"])

_COST_ALERTS_PATH = Path(os.getenv("COST_ALERTS_PATH", "data/cost_alerts.json"))


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_cost_alert_store() -> list[dict]:
    if not _COST_ALERTS_PATH.exists():
        return []
    try:
        payload = json.loads(_COST_ALERTS_PATH.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        alerts = payload.get("alerts", [])
        return alerts if isinstance(alerts, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _save_cost_alert_store(alerts: list[dict]) -> None:
    _COST_ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"alerts": alerts}
    _COST_ALERTS_PATH.write_text(json.dumps(payload, indent=2, default=str))


def _serialize_cost_alert(alert: dict) -> CostAlert:
    data = dict(alert)
    data.pop("tenant_id", None)
    data["created_at"] = _parse_datetime(data.get("created_at")) or datetime.now(timezone.utc)
    data["updated_at"] = _parse_datetime(data.get("updated_at")) or datetime.now(timezone.utc)
    data["last_triggered"] = _parse_datetime(data.get("last_triggered"))
    return CostAlert(**data)


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
        old_values_json=json.dumps(old_values, default=str) if old_values else None,
        new_values_json=json.dumps(new_values, default=str) if new_values else None,
        changed_fields_json=json.dumps(changed_fields) if changed_fields else None,
        user_id=user_id,
        user_email=user_email,
        source="api",
    )
    if db.bind and db.bind.dialect.name == "sqlite":
        max_id = db.query(func.max(CostAuditLog.id)).scalar() or 0
        log.id = max_id + 1
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
        user_email=getattr(user, "email", None),
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


@router.put("/hardware/{cost_id}", response_model=HardwareCostResponse)
def update_hardware_cost(
    cost_id: int,
    request: HardwareCostUpdate,
    _: None = Depends(require_role(["admin"])),
    db: Session = Depends(get_backend_db),
):
    """Update an existing hardware cost entry."""
    tenant_id = get_tenant_id()
    user = get_current_user()

    cost = (
        db.query(DeviceTypeCost)
        .filter(
            DeviceTypeCost.id == cost_id,
            DeviceTypeCost.tenant_id == tenant_id,
            DeviceTypeCost.valid_to.is_(None),
        )
        .first()
    )

    if not cost:
        raise HTTPException(status_code=404, detail="Hardware cost entry not found")

    update_data = request.model_dump(exclude_unset=True)
    if not update_data:
        update_data = {}

    old_values = {
        "device_model": cost.device_model,
        "purchase_cost": float(cents_to_dollars(cost.purchase_cost)),
        "replacement_cost": float(cents_to_dollars(cost.replacement_cost)) if cost.replacement_cost else None,
        "repair_cost_avg": float(cents_to_dollars(cost.repair_cost_avg)) if cost.repair_cost_avg else None,
        "depreciation_months": cost.depreciation_months,
        "residual_value_percent": cost.residual_value_percent,
        "warranty_months": cost.warranty_months,
        "notes": cost.notes,
    }

    if "device_model" in update_data:
        cost.device_model = update_data["device_model"]
    if "purchase_cost" in update_data:
        cost.purchase_cost = dollars_to_cents(update_data["purchase_cost"])
    if "replacement_cost" in update_data:
        cost.replacement_cost = (
            dollars_to_cents(update_data["replacement_cost"])
            if update_data["replacement_cost"] is not None
            else None
        )
    if "repair_cost_avg" in update_data:
        cost.repair_cost_avg = (
            dollars_to_cents(update_data["repair_cost_avg"])
            if update_data["repair_cost_avg"] is not None
            else None
        )
    if "depreciation_months" in update_data:
        cost.depreciation_months = update_data["depreciation_months"]
    if "residual_value_percent" in update_data:
        cost.residual_value_percent = update_data["residual_value_percent"]
    if "warranty_months" in update_data:
        cost.warranty_months = update_data["warranty_months"]
    if "notes" in update_data:
        cost.notes = update_data["notes"]

    create_audit_log(
        db=db,
        tenant_id=tenant_id,
        entity_type="device_type_cost",
        entity_id=cost.id,
        action="update",
        old_values=old_values,
        new_values=request.model_dump(exclude_none=True),
        user_id=user.user_id,
        user_email=getattr(user, "email", None),
    )

    db.commit()

    results_db = next(get_db())
    try:
        device_count = _get_device_count_for_model(results_db, tenant_id, cost.device_model)
    finally:
        results_db.close()

    return HardwareCostResponse(
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
        total_fleet_value=cents_to_dollars(cost.purchase_cost) * device_count,
        valid_from=cost.valid_from,
        valid_to=cost.valid_to,
        created_at=cost.created_at,
        updated_at=cost.updated_at,
        created_by=cost.created_by,
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
        user_email=getattr(user, "email", None),
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
        user_email=getattr(user, "email", None),
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
        user_email=getattr(user, "email", None),
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
        user_email=getattr(user, "email", None),
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
        confidence_explanation="Estimate based on device cost data and default labor rates" if using_defaults else "Estimate based on your configured cost data",
        impact_level=impact_level,
        using_defaults=using_defaults,
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
# COST ALERTS / BATTERY FORECAST / NFF ENDPOINTS
# =============================================================================

@router.get("/alerts", response_model=CostAlertListResponse)
def get_cost_alerts() -> CostAlertListResponse:
    """Return configured cost alerts for the tenant."""
    tenant_id = get_tenant_id()
    alerts_raw = [
        alert for alert in _load_cost_alert_store()
        if alert.get("tenant_id") == tenant_id
    ]
    alerts = [_serialize_cost_alert(alert) for alert in alerts_raw]
    return CostAlertListResponse(alerts=alerts, total=len(alerts))


@router.post("/alerts", response_model=CostAlert, status_code=201)
def create_cost_alert(
    request: CostAlertCreate,
    _: None = Depends(require_role(["admin"])),
) -> CostAlert:
    """Create a cost alert configuration."""
    tenant_id = get_tenant_id()
    alerts = _load_cost_alert_store()
    tenant_alerts = [a for a in alerts if a.get("tenant_id") == tenant_id]
    next_id = max((a.get("id", 0) for a in tenant_alerts), default=0) + 1
    now = datetime.now(timezone.utc)

    new_alert = {
        "id": next_id,
        "tenant_id": tenant_id,
        "name": request.name,
        "threshold_type": request.threshold_type.value,
        "threshold_value": str(request.threshold_value),
        "is_active": request.is_active,
        "notify_email": request.notify_email,
        "notify_webhook": request.notify_webhook,
        "last_triggered": None,
        "trigger_count": 0,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    alerts.append(new_alert)
    _save_cost_alert_store(alerts)
    return _serialize_cost_alert(new_alert)


@router.put("/alerts/{alert_id}", response_model=CostAlert)
def update_cost_alert(
    alert_id: int,
    request: CostAlertUpdate,
    _: None = Depends(require_role(["admin"])),
) -> CostAlert:
    """Update a cost alert configuration."""
    tenant_id = get_tenant_id()
    alerts = _load_cost_alert_store()
    updated = None

    for alert in alerts:
        if alert.get("tenant_id") != tenant_id:
            continue
        if alert.get("id") != alert_id:
            continue
        update_data = request.model_dump(exclude_unset=True)
        if "threshold_type" in update_data and update_data["threshold_type"] is not None:
            alert["threshold_type"] = update_data["threshold_type"].value
        if "threshold_value" in update_data and update_data["threshold_value"] is not None:
            alert["threshold_value"] = str(update_data["threshold_value"])
        if "name" in update_data:
            alert["name"] = update_data["name"]
        if "is_active" in update_data:
            alert["is_active"] = update_data["is_active"]
        if "notify_email" in update_data:
            alert["notify_email"] = update_data["notify_email"]
        if "notify_webhook" in update_data:
            alert["notify_webhook"] = update_data["notify_webhook"]
        alert["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated = alert
        break

    if updated is None:
        raise HTTPException(status_code=404, detail="Cost alert not found")

    _save_cost_alert_store(alerts)
    return _serialize_cost_alert(updated)


@router.delete("/alerts/{alert_id}", status_code=204)
def delete_cost_alert(
    alert_id: int,
    _: None = Depends(require_role(["admin"])),
) -> None:
    """Delete a cost alert configuration."""
    tenant_id = get_tenant_id()
    alerts = _load_cost_alert_store()
    remaining = [
        alert for alert in alerts
        if not (alert.get("tenant_id") == tenant_id and alert.get("id") == alert_id)
    ]
    if len(remaining) == len(alerts):
        raise HTTPException(status_code=404, detail="Cost alert not found")
    _save_cost_alert_store(remaining)


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
    device_query = (
        results_db.query(
            DeviceMetadata.device_model,
            DeviceMetadata.device_id,
        )
        .filter(DeviceMetadata.tenant_id == tenant_id)
    )
    if device_id is not None:
        device_query = device_query.filter(DeviceMetadata.device_id == device_id)

    devices = device_query.all()
    device_ids_by_model: Dict[str, List[int]] = {}
    for model, dev_id in devices:
        if model:
            device_ids_by_model.setdefault(model, []).append(dev_id)

    # Try to get real battery health data from DeviceFeature
    # Get the most recent feature snapshot for each device
    from sqlalchemy import distinct
    from sqlalchemy.orm import aliased

    battery_health_by_device: Dict[int, Optional[float]] = {}
    devices_with_health = 0

    # Query latest DeviceFeature per device
    all_device_ids = [d for ids in device_ids_by_model.values() for d in ids]
    if all_device_ids:
        # Get latest feature record for each device
        subq = (
            results_db.query(
                DeviceFeature.device_id,
                func.max(DeviceFeature.computed_at).label("latest")
            )
            .filter(
                DeviceFeature.tenant_id == tenant_id,
                DeviceFeature.device_id.in_(all_device_ids)
            )
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

            # Devices needing replacement based on health:
            # - Health < 60% = needs replacement now (due this month)
            # - Health 60-80% = due soon (next 1-2 months)
            # - Health < 80% = within 90 days
            due_now = sum(1 for h in health_values if h < 60)
            due_soon = sum(1 for h in health_values if 60 <= h < 70)
            due_90 = sum(1 for h in health_values if h < 80)

            # For devices without health data in this model, estimate
            devices_without_health = count - len(health_values)
            if devices_without_health > 0:
                # Estimate based on average fleet aging
                est_ratio = 0.7  # Assume 70% through lifespan
                due_now += int(devices_without_health * 0.05)
                due_soon += int(devices_without_health * 0.03)
                due_90 += int(devices_without_health * 0.15)

            # Convert health to "age equivalent" for display
            # 100% health = 0 months, 60% health = lifespan months
            avg_age = round(lifespan * (1 - avg_health / 100))
        else:
            # Fall back to estimated ages
            all_real_data = False
            model_data_quality = "estimated"
            avg_health = None

            # Estimate based on lifespan
            avg_age = round(lifespan * 0.7)
            age_ratio = min(avg_age / lifespan, 1.0) if lifespan else 0.0

            due_now = int(count * 0.1 * age_ratio)
            due_soon = int(count * 0.08 * age_ratio)
            due_90 = max(due_now, int(count * 0.25 * age_ratio))

        # Calculate oldest battery age
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
