"""
Operational costs endpoints.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import (
    get_backend_db,
    get_current_user,
    get_tenant_id,
    require_role,
)
from device_anomaly.api.models_cost import (
    CostCategory,
    CostType,
    OperationalCostCreate,
    OperationalCostListResponse,
    OperationalCostResponse,
    OperationalCostUpdate,
    ScopeType,
)
from device_anomaly.db.models_cost import OperationalCost

from .utils import (
    calculate_monthly_equivalent,
    cents_to_dollars,
    create_audit_log,
    dollars_to_cents,
)

router = APIRouter()


@router.get("/operational", response_model=OperationalCostListResponse)
def list_operational_costs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    category: CostCategory | None = Query(None),
    is_active: bool | None = Query(None),
    search: str | None = Query(None, max_length=255),
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

        response_costs.append(
            OperationalCostResponse(
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
        )

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

    cost.updated_at = datetime.now(UTC)

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
    cost.valid_to = datetime.now(UTC)
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
