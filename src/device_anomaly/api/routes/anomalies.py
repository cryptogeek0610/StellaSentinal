"""API routes for anomaly endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_current_user, get_db, get_tenant_id, require_role
from device_anomaly.api.models import (
    AddNoteRequest,
    AnomalyDetailResponse,
    AnomalyFilters,
    AnomalyListResponse,
    AnomalyResponse,
    BulkActionRequest,
    BulkActionResponse,
    GroupedAnomaliesResponse,
    ResolveAnomalyRequest,
)
from device_anomaly.database.schema import AnomalyResult, AnomalyStatus, InvestigationNote
from device_anomaly.services.anomaly_grouper import AnomalyGrouper

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("", response_model=AnomalyListResponse)
def list_anomalies(
    device_id: Optional[int] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    status: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None),
    max_score: Optional[float] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List anomalies with filtering and pagination."""
    tenant_id = get_tenant_id()
    query = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
    )

    if device_id is not None:
        query = query.filter(AnomalyResult.device_id == device_id)
    if start_date:
        query = query.filter(AnomalyResult.timestamp >= start_date)
    if end_date:
        query = query.filter(AnomalyResult.timestamp <= end_date)
    if status:
        query = query.filter(AnomalyResult.status == status)
    if min_score is not None:
        query = query.filter(AnomalyResult.anomaly_score >= min_score)
    if max_score is not None:
        query = query.filter(AnomalyResult.anomaly_score <= max_score)

    # Get total count
    total = query.count()

    # Apply pagination
    offset = (page - 1) * page_size
    anomalies = query.order_by(AnomalyResult.anomaly_score.asc()).offset(offset).limit(page_size).all()

    total_pages = (total + page_size - 1) // page_size

    return AnomalyListResponse(
        anomalies=[AnomalyResponse.model_validate(a) for a in anomalies],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/grouped", response_model=GroupedAnomaliesResponse)
def list_grouped_anomalies(
    status: Optional[str] = Query(None, description="Filter by status: open, investigating, resolved"),
    min_severity: Optional[str] = Query(None, description="Minimum severity: critical, high, medium, low"),
    min_group_size: int = Query(2, ge=1, description="Minimum anomalies to form a group"),
    temporal_window_hours: int = Query(24, ge=1, description="Time window for temporal grouping"),
    db: Session = Depends(get_db),
):
    """Get anomalies organized into smart groups.

    Groups anomalies by:
    1. Common remediation (same suggested fix)
    2. Category (battery issues, network issues, etc.)
    3. Temporal + cohort (same device model within time window)
    4. Location (multiple issues at same location)
    """
    tenant_id = get_tenant_id()

    # Build query
    query = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
    )

    # Filter by status
    if status:
        statuses = [s.strip() for s in status.split(",")]
        query = query.filter(AnomalyResult.status.in_(statuses))
    else:
        # Default to open and investigating
        query = query.filter(AnomalyResult.status.in_(["open", "investigating"]))

    # Filter by severity
    if min_severity:
        severity_thresholds = {
            "critical": -0.7,
            "high": -0.5,
            "medium": -0.3,
            "low": 0.0,
        }
        threshold = severity_thresholds.get(min_severity, 0.0)
        query = query.filter(AnomalyResult.anomaly_score <= threshold)

    # Get all matching anomalies
    anomalies = query.order_by(AnomalyResult.anomaly_score.asc()).all()

    # Group using the service
    grouper = AnomalyGrouper(db, tenant_id)
    return grouper.group_anomalies(
        anomalies,
        min_group_size=min_group_size,
        temporal_window_hours=temporal_window_hours,
    )


@router.post("/bulk-action", response_model=BulkActionResponse)
def bulk_action(
    request: BulkActionRequest,
    _: None = Depends(require_role(["analyst", "admin"])),
    db: Session = Depends(get_db),
):
    """Apply bulk status change to multiple anomalies.

    Supported actions:
    - resolve: Mark anomalies as resolved
    - investigating: Mark anomalies as under investigation
    - dismiss: Mark anomalies as false positive
    """
    tenant_id = get_tenant_id()
    user = get_current_user()

    # Validate action
    valid_actions = {
        "resolve": "resolved",
        "investigating": "investigating",
        "dismiss": "false_positive",
        "false_positive": "false_positive",
        "open": "open",
    }

    if request.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {request.action}. Must be one of: {list(valid_actions.keys())}",
        )

    new_status = valid_actions[request.action]

    # Get anomalies to update
    anomalies_to_update = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.id.in_(request.anomaly_ids))
        .all()
    )

    if not anomalies_to_update:
        return BulkActionResponse(
            success=False,
            affected_count=0,
            failed_ids=request.anomaly_ids,
            message="No matching anomalies found",
        )

    # Track results
    updated_count = 0
    failed_ids = []
    found_ids = set(a.id for a in anomalies_to_update)

    # Update each anomaly
    for anomaly in anomalies_to_update:
        try:
            anomaly.status = new_status
            anomaly.updated_at = datetime.now(timezone.utc)

            # Add investigation note
            note_text = request.notes or f"Bulk action: Status changed to {new_status}"
            note = InvestigationNote(
                tenant_id=tenant_id,
                anomaly_id=anomaly.id,
                user=user.user_id or "system",
                note=note_text,
                action_type="bulk_status_change",
            )
            db.add(note)
            updated_count += 1
        except Exception:
            failed_ids.append(anomaly.id)

    # Add IDs that weren't found
    for aid in request.anomaly_ids:
        if aid not in found_ids:
            failed_ids.append(aid)

    db.commit()

    return BulkActionResponse(
        success=updated_count > 0,
        affected_count=updated_count,
        failed_ids=failed_ids,
        message=f"Successfully updated {updated_count} anomalies to {new_status}",
    )


@router.get("/{anomaly_id}", response_model=AnomalyDetailResponse)
def get_anomaly(anomaly_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific anomaly."""
    tenant_id = get_tenant_id()
    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    # Get investigation notes
    notes = (
        db.query(InvestigationNote)
        .filter(InvestigationNote.anomaly_id == anomaly_id)
        .filter(InvestigationNote.tenant_id == tenant_id)
        .order_by(InvestigationNote.created_at.desc())
        .all()
    )

    response = AnomalyDetailResponse.model_validate(anomaly)
    response.investigation_notes = [
        {
            "id": note.id,
            "user": note.user,
            "note": note.note,
            "action_type": note.action_type,
            "created_at": note.created_at.isoformat(),
        }
        for note in notes
    ]

    return response


@router.post("/{anomaly_id}/resolve", response_model=AnomalyResponse)
def resolve_anomaly(
    anomaly_id: int,
    request: ResolveAnomalyRequest,
    _: None = Depends(require_role(["analyst", "admin"])),
    db: Session = Depends(get_db),
):
    """Mark an anomaly as resolved or update its status."""
    tenant_id = get_tenant_id()
    user = get_current_user()
    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    anomaly.status = request.status
    if request.notes:
        if anomaly.notes:
            anomaly.notes += f"\n\n{request.notes}"
        else:
            anomaly.notes = request.notes
    anomaly.updated_at = datetime.now(timezone.utc)

    # Add investigation note
    note = InvestigationNote(
        tenant_id=tenant_id,
        anomaly_id=anomaly_id,
        user=user.user_id or "system",
        note=request.notes or f"Status changed to {request.status}",
        action_type="status_change",
    )
    db.add(note)

    db.commit()
    db.refresh(anomaly)

    return AnomalyResponse.model_validate(anomaly)


@router.post("/{anomaly_id}/notes", response_model=dict)
def add_note(
    anomaly_id: int,
    request: AddNoteRequest,
    _: None = Depends(require_role(["analyst", "admin"])),
    db: Session = Depends(get_db),
):
    """Add an investigation note to an anomaly."""
    tenant_id = get_tenant_id()
    user = get_current_user()
    anomaly = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.id == anomaly_id)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .first()
    )
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")

    note = InvestigationNote(
        tenant_id=tenant_id,
        anomaly_id=anomaly_id,
        user=user.user_id or "system",
        note=request.note,
        action_type=request.action_type or "note",
    )
    db.add(note)
    db.commit()

    return {"id": note.id, "message": "Note added successfully"}
