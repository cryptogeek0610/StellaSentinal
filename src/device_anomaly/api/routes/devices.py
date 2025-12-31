"""API routes for device endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import String, cast, func, or_
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_db, get_mock_mode, get_tenant_id
from device_anomaly.api.models import AnomalyResponse, DeviceDetailResponse, DeviceResponse
from device_anomaly.api.mock_mode import get_mock_devices
from device_anomaly.config.settings import get_settings
from device_anomaly.database.schema import AnomalyResult, DeviceMetadata


def escape_like_pattern(value: str) -> str:
    """Escape special characters in LIKE patterns to prevent pattern injection."""
    # Escape backslash first, then % and _
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

router = APIRouter(prefix="/devices", tags=["devices"])


@router.get("")
def list_devices(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    search: Optional[str] = Query(None),
    group_by: Optional[str] = Query(None),
    group_value: Optional[str] = Query(None),
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """List devices with filtering, pagination, and grouping."""
    # Return mock data if Mock Mode is enabled
    if mock_mode:
        mock_devices = get_mock_devices()
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            mock_devices = [
                d for d in mock_devices
                if search_lower in d.get("device_name", "").lower()
                or search_lower in str(d.get("device_id", "")).lower()
            ]
        
        # Apply group filter
        if group_by and group_value:
            mock_devices = [
                d for d in mock_devices
                if d.get("custom_attributes", {}).get(group_by) == group_value
                or d.get("store_id") == group_value
            ]
        
        # Apply pagination
        total = len(mock_devices)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_devices = mock_devices[start:end]
        
        return {
            "devices": paginated_devices,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        }
    
    # Real implementation: Get devices from DeviceMetadata
    tenant_id = get_tenant_id()
    query = db.query(DeviceMetadata).filter(DeviceMetadata.tenant_id == tenant_id)
    
    # Apply search filter (with LIKE pattern escaping to prevent injection)
    if search:
        search_escaped = escape_like_pattern(search)
        query = query.filter(
            or_(
                DeviceMetadata.device_name.ilike(f"%{search_escaped}%", escape="\\"),
                cast(DeviceMetadata.device_id, String).ilike(f"%{search_escaped}%", escape="\\")
            )
        )
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    devices = query.offset(offset).limit(page_size).all()

    device_ids = [device.device_id for device in devices]
    anomaly_counts = {}
    if device_ids:
        counts = (
            db.query(AnomalyResult.device_id, func.count(AnomalyResult.id))
            .filter(AnomalyResult.device_id.in_(device_ids))
            .filter(AnomalyResult.tenant_id == tenant_id)
            .filter(AnomalyResult.anomaly_label == -1)
            .group_by(AnomalyResult.device_id)
            .all()
        )
        anomaly_counts = {device_id: count for device_id, count in counts}
    
    # Convert to response format
    device_list = []
    for device in devices:
        anomaly_count = anomaly_counts.get(device.device_id, 0)
        
        device_dict = DeviceResponse.model_validate(device).model_dump()
        device_dict["anomaly_count"] = anomaly_count
        device_list.append(device_dict)
    
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "devices": device_list,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@router.get("/{device_id}", response_model=DeviceDetailResponse)
def get_device(device_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific device."""
    tenant_id = get_tenant_id()
    device = (
        db.query(DeviceMetadata)
        .filter(DeviceMetadata.device_id == device_id)
        .filter(DeviceMetadata.tenant_id == tenant_id)
        .first()
    )

    # If device metadata doesn't exist, create a basic one
    if not device:
        if get_settings().env in {"local", "development"}:
            device = DeviceMetadata(
                device_id=device_id,
                tenant_id=tenant_id,
                status="unknown",
            )
            db.add(device)
            db.commit()
            db.refresh(device)
        else:
            raise HTTPException(status_code=404, detail="Device not found")

    # Get anomaly count
    anomaly_count = (
        db.query(func.count(AnomalyResult.id))
        .filter(AnomalyResult.device_id == device_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .scalar()
    ) or 0

    # Get recent anomalies (last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    recent_anomalies = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.device_id == device_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.timestamp >= thirty_days_ago)
        .order_by(AnomalyResult.timestamp.desc())
        .limit(20)
        .all()
    )

    response = DeviceDetailResponse.model_validate(device)
    response.anomaly_count = anomaly_count
    response.recent_anomalies = [AnomalyResponse.model_validate(a) for a in recent_anomalies]

    return response
