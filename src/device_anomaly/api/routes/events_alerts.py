"""API routes for events and alerts monitoring."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/events", tags=["events-alerts"])


# ============================================================================
# Response Models
# ============================================================================


class EventEntryResponse(BaseModel):
    """Single event from MainLog."""

    log_id: int
    timestamp: datetime
    event_id: int
    severity: str
    event_class: str
    message: str
    device_id: int | None = None
    login_id: str | None = None


class EventTimelineResponse(BaseModel):
    """Paginated event timeline."""

    tenant_id: str
    events: list[EventEntryResponse]
    total: int
    page: int
    page_size: int
    severity_distribution: dict[str, int]
    event_class_distribution: dict[str, int]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AlertEntryResponse(BaseModel):
    """Single alert."""

    alert_id: int
    alert_key: str
    alert_name: str
    severity: str
    device_id: str | None = None
    status: str
    set_datetime: datetime | None = None
    ack_datetime: datetime | None = None


class AlertNameCount(BaseModel):
    """Alert count by name."""

    name: str
    count: int


class AlertSummaryResponse(BaseModel):
    """Alert summary statistics."""

    tenant_id: str
    total_active: int
    total_acknowledged: int
    total_resolved: int
    by_severity: dict[str, int]
    by_alert_name: list[AlertNameCount]
    recent_alerts: list[AlertEntryResponse]
    avg_acknowledge_time_minutes: float
    avg_resolution_time_minutes: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AlertTrendPointResponse(BaseModel):
    """Single point in alert trend."""

    timestamp: datetime
    count: int
    severity: str


class AlertTrendsResponse(BaseModel):
    """Alert trends over time."""

    tenant_id: str
    trends: list[AlertTrendPointResponse]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CorrelatedEventResponse(BaseModel):
    """Event that occurred before an anomaly."""

    event: EventEntryResponse
    time_before_minutes: float
    frequency_score: float


class EventCorrelationResponse(BaseModel):
    """Event correlation analysis."""

    tenant_id: str
    anomaly_timestamp: datetime
    device_id: int
    correlated_events: list[CorrelatedEventResponse]
    total_events_found: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EventStatisticsResponse(BaseModel):
    """Overall event statistics."""

    tenant_id: str
    total_events: int
    events_per_day: int
    unique_devices: int
    top_event_classes: list[dict[str, Any]]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============================================================================
# Mock Data Functions
# ============================================================================


def get_mock_event_timeline(page: int, page_size: int) -> EventTimelineResponse:
    """Generate mock event timeline."""
    import random

    random.seed(42 + page)

    now = datetime.now(UTC)
    severities = ["Info", "Warning", "Error", "Critical"]
    event_classes = [
        "DeviceConnect",
        "AppCrash",
        "BatteryLow",
        "NetworkDrop",
        "PolicyApply",
        "UserLogin",
    ]

    events = []
    for i in range(page_size):
        ts = now - timedelta(minutes=i * 5 + (page - 1) * page_size * 5)
        severity = random.choices(severities, weights=[50, 30, 15, 5])[0]
        event_class = random.choice(event_classes)

        events.append(
            EventEntryResponse(
                log_id=10000 + i + (page - 1) * page_size,
                timestamp=ts,
                event_id=random.randint(1000, 9999),
                severity=severity,
                event_class=event_class,
                message=f"Sample {event_class} event for device",
                device_id=random.randint(1000, 1250),
                login_id=f"user_{random.randint(1, 10)}",
            )
        )

    return EventTimelineResponse(
        tenant_id="default",
        events=events,
        total=1000,
        page=page,
        page_size=page_size,
        severity_distribution={"Info": 500, "Warning": 300, "Error": 150, "Critical": 50},
        event_class_distribution={
            "DeviceConnect": 250,
            "AppCrash": 180,
            "BatteryLow": 150,
            "NetworkDrop": 200,
            "PolicyApply": 120,
            "UserLogin": 100,
        },
    )


def get_mock_alert_summary() -> AlertSummaryResponse:
    """Generate mock alert summary."""
    now = datetime.now(UTC)

    recent_alerts = [
        AlertEntryResponse(
            alert_id=1,
            alert_key="ALT_001",
            alert_name="High CPU Usage",
            severity="Warning",
            device_id="DEV_1042",
            status="Active",
            set_datetime=now - timedelta(hours=2),
            ack_datetime=None,
        ),
        AlertEntryResponse(
            alert_id=2,
            alert_key="ALT_002",
            alert_name="Low Storage",
            severity="Critical",
            device_id="DEV_1087",
            status="Acknowledged",
            set_datetime=now - timedelta(hours=5),
            ack_datetime=now - timedelta(hours=4),
        ),
        AlertEntryResponse(
            alert_id=3,
            alert_key="ALT_003",
            alert_name="Battery Critical",
            severity="Critical",
            device_id="DEV_1156",
            status="Active",
            set_datetime=now - timedelta(hours=1),
            ack_datetime=None,
        ),
        AlertEntryResponse(
            alert_id=4,
            alert_key="ALT_004",
            alert_name="Connection Lost",
            severity="Warning",
            device_id="DEV_1023",
            status="Resolved",
            set_datetime=now - timedelta(days=1),
            ack_datetime=now - timedelta(hours=20),
        ),
    ]

    return AlertSummaryResponse(
        tenant_id="default",
        total_active=15,
        total_acknowledged=8,
        total_resolved=42,
        by_severity={"Critical": 5, "Warning": 12, "Info": 8, "Low": 40},
        by_alert_name=[
            AlertNameCount(name="Low Storage", count=18),
            AlertNameCount(name="Battery Critical", count=12),
            AlertNameCount(name="High CPU Usage", count=10),
            AlertNameCount(name="Connection Lost", count=8),
            AlertNameCount(name="Policy Violation", count=5),
        ],
        recent_alerts=recent_alerts,
        avg_acknowledge_time_minutes=45.5,
        avg_resolution_time_minutes=180.2,
    )


def get_mock_alert_trends(period_days: int, granularity: str) -> AlertTrendsResponse:
    """Generate mock alert trends."""
    import random

    random.seed(42)

    now = datetime.now(UTC)
    trends = []

    hours = period_days * 24 if granularity == "hourly" else period_days
    step = timedelta(hours=1) if granularity == "hourly" else timedelta(days=1)

    severities = ["Critical", "Warning", "Info"]
    for severity in severities:
        for i in range(hours):
            ts = now - step * (hours - i - 1)
            base = 5 if severity == "Info" else (3 if severity == "Warning" else 1)
            count = base + random.randint(-1, 2)

            trends.append(
                AlertTrendPointResponse(
                    timestamp=ts,
                    count=max(0, count),
                    severity=severity,
                )
            )

    return AlertTrendsResponse(
        tenant_id="default",
        trends=trends,
    )


def get_mock_event_statistics() -> EventStatisticsResponse:
    """Generate mock event statistics."""
    return EventStatisticsResponse(
        tenant_id="default",
        total_events=15680,
        events_per_day=2240,
        unique_devices=248,
        top_event_classes=[
            {"class": "DeviceConnect", "count": 4200},
            {"class": "NetworkDrop", "count": 3100},
            {"class": "AppCrash", "count": 2800},
            {"class": "BatteryLow", "count": 2400},
            {"class": "PolicyApply", "count": 1800},
        ],
    )


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/timeline", response_model=EventTimelineResponse)
def get_event_timeline(
    device_id: int | None = Query(None, description="Filter by device ID"),
    severity: str | None = Query(None, description="Filter by severity"),
    event_class: str | None = Query(None, description="Filter by event class"),
    start_time: datetime | None = Query(None, description="Start of time window"),
    end_time: datetime | None = Query(None, description="End of time window"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=10, le=200, description="Events per page"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get paginated event timeline from MainLog.

    Filter by device, severity, event class, and time range.
    """
    if mock_mode:
        return get_mock_event_timeline(page, page_size)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.events_alerts_loader import load_event_timeline

        timeline_data = load_event_timeline(
            device_id=device_id,
            severity=severity,
            event_class=event_class,
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size,
        )

        events = [
            EventEntryResponse(
                log_id=e.log_id,
                timestamp=e.timestamp,
                event_id=e.event_id,
                severity=e.severity,
                event_class=e.event_class,
                message=e.message,
                device_id=e.device_id,
                login_id=e.login_id,
            )
            for e in timeline_data.events
        ]

        return EventTimelineResponse(
            tenant_id=tenant_id,
            events=events,
            total=timeline_data.total,
            page=timeline_data.page,
            page_size=timeline_data.page_size,
            severity_distribution=timeline_data.severity_distribution,
            event_class_distribution=timeline_data.event_class_distribution,
        )

    except Exception as e:
        logger.error(f"Failed to get event timeline: {e}")
        return EventTimelineResponse(
            tenant_id=tenant_id,
            events=[],
            total=0,
            page=page,
            page_size=page_size,
            severity_distribution={},
            event_class_distribution={},
        )


@router.get("/alerts/summary", response_model=AlertSummaryResponse)
def get_alert_summary(
    period_days: int = Query(30, ge=1, le=90, description="Analysis period in days"),
    status_filter: str | None = Query(None, description="Filter by status"),
    severity_filter: str | None = Query(None, description="Filter by severity"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get alert summary statistics.

    Returns counts by status, severity, and recent alerts.
    """
    if mock_mode:
        return get_mock_alert_summary()

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.events_alerts_loader import load_alert_summary

        start_time = datetime.now(UTC) - timedelta(days=period_days)
        summary_data = load_alert_summary(
            start_time=start_time,
            status_filter=status_filter,
            severity_filter=severity_filter,
        )

        recent = [
            AlertEntryResponse(
                alert_id=a.alert_id,
                alert_key=a.alert_key,
                alert_name=a.alert_name,
                severity=a.severity,
                device_id=a.device_id,
                status=a.status,
                set_datetime=a.set_datetime,
                ack_datetime=a.ack_datetime,
            )
            for a in summary_data.recent_alerts
        ]

        by_name = [
            AlertNameCount(name=n["name"], count=n["count"]) for n in summary_data.by_alert_name
        ]

        return AlertSummaryResponse(
            tenant_id=tenant_id,
            total_active=summary_data.total_active,
            total_acknowledged=summary_data.total_acknowledged,
            total_resolved=summary_data.total_resolved,
            by_severity=summary_data.by_severity,
            by_alert_name=by_name,
            recent_alerts=recent,
            avg_acknowledge_time_minutes=summary_data.avg_acknowledge_time_minutes,
            avg_resolution_time_minutes=summary_data.avg_resolution_time_minutes,
        )

    except Exception as e:
        logger.error(f"Failed to get alert summary: {e}")
        return AlertSummaryResponse(
            tenant_id=tenant_id,
            total_active=0,
            total_acknowledged=0,
            total_resolved=0,
            by_severity={},
            by_alert_name=[],
            recent_alerts=[],
            avg_acknowledge_time_minutes=0,
            avg_resolution_time_minutes=0,
        )


@router.get("/alerts/trends", response_model=AlertTrendsResponse)
def get_alert_trends(
    period_days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    granularity: str = Query("hourly", description="Granularity: hourly or daily"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get alert count trends over time.

    Useful for visualizing alert patterns and identifying spikes.
    """
    if mock_mode:
        return get_mock_alert_trends(period_days, granularity)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.events_alerts_loader import load_alert_trends

        trends_data = load_alert_trends(
            period_days=period_days,
            granularity=granularity,
        )

        trends = [
            AlertTrendPointResponse(
                timestamp=t.timestamp,
                count=t.count,
                severity=t.severity,
            )
            for t in trends_data
        ]

        return AlertTrendsResponse(
            tenant_id=tenant_id,
            trends=trends,
        )

    except Exception as e:
        logger.error(f"Failed to get alert trends: {e}")
        return AlertTrendsResponse(
            tenant_id=tenant_id,
            trends=[],
        )


@router.get("/correlation", response_model=EventCorrelationResponse)
def get_event_correlation(
    anomaly_timestamp: datetime = Query(..., description="When the anomaly was detected"),
    device_id: int = Query(..., description="Device ID of the anomaly"),
    window_minutes: int = Query(60, ge=5, le=240, description="Minutes before anomaly to search"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Find events that occurred before an anomaly.

    Useful for root cause analysis by identifying events that
    commonly precede anomalies.
    """
    if mock_mode:
        # Generate mock correlated events
        now = anomaly_timestamp
        events = [
            CorrelatedEventResponse(
                event=EventEntryResponse(
                    log_id=10001,
                    timestamp=now - timedelta(minutes=5),
                    event_id=2001,
                    severity="Warning",
                    event_class="BatteryLow",
                    message="Battery level dropped below 20%",
                    device_id=device_id,
                    login_id=None,
                ),
                time_before_minutes=5.0,
                frequency_score=0.92,
            ),
            CorrelatedEventResponse(
                event=EventEntryResponse(
                    log_id=10002,
                    timestamp=now - timedelta(minutes=15),
                    event_id=2002,
                    severity="Info",
                    event_class="AppCrash",
                    message="App com.example.scanner crashed",
                    device_id=device_id,
                    login_id="user_5",
                ),
                time_before_minutes=15.0,
                frequency_score=0.75,
            ),
        ]
        return EventCorrelationResponse(
            tenant_id="default",
            anomaly_timestamp=anomaly_timestamp,
            device_id=device_id,
            correlated_events=events,
            total_events_found=len(events),
        )

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.events_alerts_loader import find_correlated_events

        correlated = find_correlated_events(
            anomaly_timestamp=anomaly_timestamp,
            device_id=device_id,
            window_minutes=window_minutes,
        )

        events = [
            CorrelatedEventResponse(
                event=EventEntryResponse(
                    log_id=c.event.log_id,
                    timestamp=c.event.timestamp,
                    event_id=c.event.event_id,
                    severity=c.event.severity,
                    event_class=c.event.event_class,
                    message=c.event.message,
                    device_id=c.event.device_id,
                    login_id=c.event.login_id,
                ),
                time_before_minutes=c.time_before_minutes,
                frequency_score=c.frequency_score,
            )
            for c in correlated
        ]

        return EventCorrelationResponse(
            tenant_id=tenant_id,
            anomaly_timestamp=anomaly_timestamp,
            device_id=device_id,
            correlated_events=events,
            total_events_found=len(events),
        )

    except Exception as e:
        logger.error(f"Failed to find correlated events: {e}")
        return EventCorrelationResponse(
            tenant_id=tenant_id,
            anomaly_timestamp=anomaly_timestamp,
            device_id=device_id,
            correlated_events=[],
            total_events_found=0,
        )


@router.get("/statistics", response_model=EventStatisticsResponse)
def get_event_statistics(
    period_days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get overall event statistics.

    Returns total counts, daily averages, and top event types.
    """
    if mock_mode:
        return get_mock_event_statistics()

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.events_alerts_loader import (
            get_event_statistics as get_stats,
        )

        stats = get_stats(period_days=period_days)

        return EventStatisticsResponse(
            tenant_id=tenant_id,
            total_events=stats.get("total_events", 0),
            events_per_day=stats.get("events_per_day", 0),
            unique_devices=stats.get("unique_devices", 0),
            top_event_classes=stats.get("top_event_classes", []),
        )

    except Exception as e:
        logger.error(f"Failed to get event statistics: {e}")
        return EventStatisticsResponse(
            tenant_id=tenant_id,
            total_events=0,
            events_per_day=0,
            unique_devices=0,
            top_event_classes=[],
        )
