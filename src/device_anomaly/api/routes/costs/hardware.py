"""
Hardware costs endpoints.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

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
    DeviceModelInfo,
    DeviceModelsResponse,
    HardwareCostCreate,
    HardwareCostListResponse,
    HardwareCostResponse,
    HardwareCostUpdate,
)
from device_anomaly.database.schema import DeviceMetadata
from device_anomaly.db.models_cost import DeviceTypeCost

from .utils import cents_to_dollars, create_audit_log, dollars_to_cents, get_device_count_for_model

router = APIRouter()


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

    # Get results db for device counts
    results_db = next(get_db())
    try:
        for cost in costs:
            # Get device count for this model
            device_count = get_device_count_for_model(results_db, tenant_id, cost.device_model)
            fleet_value = cents_to_dollars(cost.purchase_cost) * device_count

            response_costs.append(
                HardwareCostResponse(
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
                )
            )
            total_fleet_value += fleet_value
    finally:
        results_db.close()

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
        device_count = get_device_count_for_model(results_db, tenant_id, new_cost.device_model)
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
            dollars_to_cents(update_data["replacement_cost"]) if update_data["replacement_cost"] is not None else None
        )
    if "repair_cost_avg" in update_data:
        cost.repair_cost_avg = (
            dollars_to_cents(update_data["repair_cost_avg"]) if update_data["repair_cost_avg"] is not None else None
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
        device_count = get_device_count_for_model(results_db, tenant_id, cost.device_model)
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
