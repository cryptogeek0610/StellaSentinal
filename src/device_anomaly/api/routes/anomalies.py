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
    ResolveAnomalyRequest,
)
from device_anomaly.database.schema import AnomalyResult, AnomalyStatus, InvestigationNote

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
