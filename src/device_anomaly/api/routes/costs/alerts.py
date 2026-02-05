"""
Cost alerts endpoints.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from device_anomaly.api.dependencies import get_tenant_id, require_role
from device_anomaly.api.models_cost import (
    CostAlert,
    CostAlertCreate,
    CostAlertListResponse,
    CostAlertUpdate,
)

router = APIRouter()

_COST_ALERTS_PATH = Path(os.getenv("COST_ALERTS_PATH", "data/cost_alerts.json"))


def _parse_datetime(value: str | None) -> datetime | None:
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
    data["created_at"] = _parse_datetime(data.get("created_at")) or datetime.now(UTC)
    data["updated_at"] = _parse_datetime(data.get("updated_at")) or datetime.now(UTC)
    data["last_triggered"] = _parse_datetime(data.get("last_triggered"))
    return CostAlert(**data)


@router.get("/alerts", response_model=CostAlertListResponse)
def get_cost_alerts() -> CostAlertListResponse:
    """Return configured cost alerts for the tenant."""
    tenant_id = get_tenant_id()
    alerts_raw = [alert for alert in _load_cost_alert_store() if alert.get("tenant_id") == tenant_id]
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
    now = datetime.now(UTC)

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
        alert["updated_at"] = datetime.now(UTC).isoformat()
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
    remaining = [alert for alert in alerts if not (alert.get("tenant_id") == tenant_id and alert.get("id") == alert_id)]
    if len(remaining) == len(alerts):
        raise HTTPException(status_code=404, detail="Cost alert not found")
    _save_cost_alert_store(remaining)
