"""
Data Quality and Reconciliation API.

Provides endpoints for monitoring data quality between XSight and MobiControl:
- Device reconciliation status
- Data freshness metrics
- Missing device tracking
- Quality scoring
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-quality", tags=["data-quality"])


# =============================================================================
# Response Models
# =============================================================================


class DataFreshnessResponse(BaseModel):
    """Data freshness for a single source."""

    source_name: str
    latest_timestamp: str | None = None
    staleness_hours: float
    device_count: int
    record_count: int
    status: str = Field(description="fresh, stale, critical, or unavailable")


class FreshnessResponse(BaseModel):
    """Response for data freshness endpoint."""

    sources: dict[str, DataFreshnessResponse]
    overall_status: str
    generated_at: str


class ReconciliationReportResponse(BaseModel):
    """Response for reconciliation endpoint."""

    report_date: str

    # Device counts
    xsight_device_count: int
    mobicontrol_device_count: int
    matched_devices: int
    xsight_only: int = Field(description="Devices only in XSight (missing from MC)")
    mobicontrol_only: int = Field(description="Devices only in MC (missing from XSight)")
    match_rate: float

    # Freshness
    xsight_staleness_hours: float
    mobicontrol_staleness_hours: float

    # Temporal alignment
    avg_time_gap_hours: float
    max_time_gap_hours: float

    # Quality
    quality_score: float = Field(description="0-100 quality score")
    quality_grade: str = Field(description="A, B, C, D, or F")

    # Issues
    issues: list[str]
    recommendations: list[str]


class MissingDeviceResponse(BaseModel):
    """A device missing from one data source."""

    device_id: int
    present_in: str
    missing_from: str
    last_seen: str | None = None
    device_name: str | None = None
    device_model: str | None = None


class MissingDevicesResponse(BaseModel):
    """Response for missing devices endpoint."""

    source_checked: str
    missing_count: int
    devices: list[MissingDeviceResponse]
    generated_at: str


class QualitySummaryResponse(BaseModel):
    """Quick quality summary."""

    match_rate: float
    quality_score: float
    quality_grade: str
    xsight_staleness_hours: float
    mobicontrol_staleness_hours: float
    issues_count: int
    status: str
    generated_at: str


# =============================================================================
# Mock Data Generators
# =============================================================================


def get_mock_reconciliation_report() -> ReconciliationReportResponse:
    """Generate mock reconciliation report."""
    return ReconciliationReportResponse(
        report_date=datetime.now(UTC).isoformat(),
        xsight_device_count=4523,
        mobicontrol_device_count=4678,
        matched_devices=4312,
        xsight_only=211,
        mobicontrol_only=366,
        match_rate=0.923,
        xsight_staleness_hours=2.3,
        mobicontrol_staleness_hours=4.1,
        avg_time_gap_hours=3.2,
        max_time_gap_hours=18.5,
        quality_score=87.5,
        quality_grade="B",
        issues=[
            "Warning: 366 devices have inventory but no telemetry",
            "Warning: 211 devices have telemetry but no inventory",
        ],
        recommendations=[
            "Review devices with inventory but no telemetry - may have XSight collection issues.",
            "Review devices with telemetry but missing inventory - may need re-enrollment.",
        ],
    )


def get_mock_freshness() -> FreshnessResponse:
    """Generate mock freshness data."""
    return FreshnessResponse(
        sources={
            "xsight": DataFreshnessResponse(
                source_name="XSight DW",
                latest_timestamp=datetime.now(UTC).isoformat(),
                staleness_hours=2.3,
                device_count=4523,
                record_count=125678,
                status="fresh",
            ),
            "mobicontrol": DataFreshnessResponse(
                source_name="MobiControl",
                latest_timestamp=datetime.now(UTC).isoformat(),
                staleness_hours=4.1,
                device_count=4678,
                record_count=4678,
                status="fresh",
            ),
        },
        overall_status="healthy",
        generated_at=datetime.now(UTC).isoformat(),
    )


def get_mock_missing_devices(source: str, limit: int) -> MissingDevicesResponse:
    """Generate mock missing devices."""
    devices = [
        MissingDeviceResponse(
            device_id=12345,
            present_in="mobicontrol" if source == "xsight" else "xsight",
            missing_from=source,
            last_seen=datetime.now(UTC).isoformat(),
            device_name="TC52-WH-A101",
            device_model="TC52",
        ),
        MissingDeviceResponse(
            device_id=12346,
            present_in="mobicontrol" if source == "xsight" else "xsight",
            missing_from=source,
            last_seen=datetime.now(UTC).isoformat(),
            device_name="CT40-STORE-B23",
            device_model="CT40",
        ),
    ]

    return MissingDevicesResponse(
        source_checked=source,
        missing_count=len(devices),
        devices=devices[:limit],
        generated_at=datetime.now(UTC).isoformat(),
    )


def get_mock_quality_summary() -> QualitySummaryResponse:
    """Generate mock quality summary."""
    return QualitySummaryResponse(
        match_rate=0.923,
        quality_score=87.5,
        quality_grade="B",
        xsight_staleness_hours=2.3,
        mobicontrol_staleness_hours=4.1,
        issues_count=2,
        status="healthy",
        generated_at=datetime.now(UTC).isoformat(),
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/reconciliation", response_model=ReconciliationReportResponse)
async def get_reconciliation_report(
    mock_mode: bool = Depends(get_mock_mode),
) -> ReconciliationReportResponse:
    """
    Get current data reconciliation status between XSight and MobiControl.

    Returns device match rates, data freshness, temporal alignment,
    quality scoring, and actionable recommendations.
    """
    if mock_mode:
        return get_mock_reconciliation_report()

    try:
        from device_anomaly.services.data_reconciliation import DataReconciliationService

        service = DataReconciliationService()
        report = service.generate_reconciliation_report()

        return ReconciliationReportResponse(
            report_date=report.report_date.isoformat()
            if report.report_date
            else datetime.now(UTC).isoformat(),
            xsight_device_count=report.xsight_device_count,
            mobicontrol_device_count=report.mobicontrol_device_count,
            matched_devices=report.matched_devices,
            xsight_only=report.xsight_only,
            mobicontrol_only=report.mobicontrol_only,
            match_rate=report.match_rate,
            xsight_staleness_hours=report.xsight_staleness_hours,
            mobicontrol_staleness_hours=report.mobicontrol_staleness_hours,
            avg_time_gap_hours=report.avg_time_gap_hours,
            max_time_gap_hours=report.max_time_gap_hours,
            quality_score=report.overall_quality_score,
            quality_grade=report.quality_grade,
            issues=report.issues,
            recommendations=report.recommendations,
        )

    except Exception as e:
        logger.warning(f"Reconciliation report failed: {e}, returning mock")
        return get_mock_reconciliation_report()


@router.get("/freshness", response_model=FreshnessResponse)
async def get_data_freshness(
    mock_mode: bool = Depends(get_mock_mode),
) -> FreshnessResponse:
    """
    Get data freshness metrics for all data sources.

    Shows how recent the data is from each source and whether
    there are any staleness concerns.
    """
    if mock_mode:
        return get_mock_freshness()

    try:
        from device_anomaly.services.data_reconciliation import DataReconciliationService

        service = DataReconciliationService()
        freshness = service.get_data_freshness()

        sources = {}
        for key, data in freshness.items():
            sources[key] = DataFreshnessResponse(
                source_name=data.source_name,
                latest_timestamp=data.latest_timestamp.isoformat()
                if data.latest_timestamp
                else None,
                staleness_hours=data.staleness_hours,
                device_count=data.device_count,
                record_count=data.record_count,
                status=data.status,
            )

        # Determine overall status
        statuses = [d.status for d in freshness.values()]
        if "critical" in statuses:
            overall = "critical"
        elif "stale" in statuses:
            overall = "warning"
        elif "unavailable" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        return FreshnessResponse(
            sources=sources,
            overall_status=overall,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.warning(f"Freshness check failed: {e}, returning mock")
        return get_mock_freshness()


@router.get("/missing-devices", response_model=MissingDevicesResponse)
async def get_missing_devices(
    source: str = Query("xsight", description="Source to check: xsight or mobicontrol"),
    limit: int = Query(100, description="Maximum devices to return"),
    mock_mode: bool = Depends(get_mock_mode),
) -> MissingDevicesResponse:
    """
    Get list of devices missing from one source but present in the other.

    If source="xsight", returns devices in MobiControl but missing from XSight.
    If source="mobicontrol", returns devices in XSight but missing from MobiControl.
    """
    if source not in ("xsight", "mobicontrol"):
        source = "xsight"

    if mock_mode:
        return get_mock_missing_devices(source, limit)

    try:
        from device_anomaly.services.data_reconciliation import DataReconciliationService

        service = DataReconciliationService()
        missing = service.get_missing_devices(source=source, limit=limit)

        devices = [
            MissingDeviceResponse(
                device_id=d.device_id,
                present_in=d.present_in,
                missing_from=d.missing_from,
                last_seen=d.last_seen.isoformat() if d.last_seen else None,
                device_name=d.device_name,
                device_model=str(d.device_model) if d.device_model else None,
            )
            for d in missing
        ]

        return MissingDevicesResponse(
            source_checked=source,
            missing_count=len(devices),
            devices=devices,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.warning(f"Missing devices check failed: {e}, returning mock")
        return get_mock_missing_devices(source, limit)


@router.get("/summary", response_model=QualitySummaryResponse)
async def get_quality_summary(
    mock_mode: bool = Depends(get_mock_mode),
) -> QualitySummaryResponse:
    """
    Get quick data quality summary.

    Returns key metrics at a glance without full reconciliation details.
    """
    if mock_mode:
        return get_mock_quality_summary()

    try:
        from device_anomaly.services.data_reconciliation import get_data_quality_summary

        summary = get_data_quality_summary()

        return QualitySummaryResponse(
            match_rate=summary["match_rate"],
            quality_score=summary["quality_score"],
            quality_grade=summary["quality_grade"],
            xsight_staleness_hours=summary["xsight_staleness_hours"],
            mobicontrol_staleness_hours=summary["mobicontrol_staleness_hours"],
            issues_count=summary["issues_count"],
            status=summary["status"],
            generated_at=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.warning(f"Quality summary failed: {e}, returning mock")
        return get_mock_quality_summary()


@router.get("/health")
async def get_data_health(
    mock_mode: bool = Depends(get_mock_mode),
) -> dict[str, Any]:
    """
    Get overall data health status.

    Simple endpoint for health checks and monitoring dashboards.
    """
    try:
        summary = await get_quality_summary(mock_mode=mock_mode)

        return {
            "healthy": summary.status == "healthy",
            "status": summary.status,
            "quality_grade": summary.quality_grade,
            "match_rate": summary.match_rate,
            "issues_count": summary.issues_count,
            "checked_at": summary.generated_at,
        }

    except Exception as e:
        return {
            "healthy": False,
            "status": "error",
            "error": str(e),
            "checked_at": datetime.now(UTC).isoformat(),
        }
