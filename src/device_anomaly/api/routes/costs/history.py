"""
Cost history and audit trail endpoints.
"""
from __future__ import annotations

import json
import math
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_backend_db, get_tenant_id, require_role
from device_anomaly.api.models_cost import (
    AuditAction,
    CostChangeEntry,
    CostHistoryResponse,
)
from device_anomaly.db.models_cost import CostAuditLog

router = APIRouter()


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
    logs = query.order_by(CostAuditLog.timestamp.desc()).offset(offset).limit(page_size).all()

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

        changes.append(
            CostChangeEntry(
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
            )
        )

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
