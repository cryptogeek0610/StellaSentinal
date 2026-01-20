"""
Utility functions for cost calculations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from device_anomaly.database.schema import DeviceMetadata
from device_anomaly.db.models_cost import CostAuditLog


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


def get_device_count_for_model(db: Session, tenant_id: str, device_model: str) -> int:
    """Get count of devices matching a device model."""
    return (
        db.query(func.count(DeviceMetadata.device_id))
        .filter(
            DeviceMetadata.tenant_id == tenant_id,
            DeviceMetadata.device_model == device_model,
        )
        .scalar()
    ) or 0


def get_severity(score: float) -> str:
    """Convert anomaly score to severity level."""
    if score <= -0.7:
        return "critical"
    if score <= -0.5:
        return "high"
    if score <= -0.3:
        return "medium"
    return "low"
