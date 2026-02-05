"""
Events and Alerts Data Loader.

Loads event timeline and alert data from MobiControl for
audit trail visualization and event correlation analysis.

Data Sources:
- MainLog (MobiControl): ~1M rows of device events
- Alert (MobiControl): ~1.3K system alerts
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from device_anomaly.data_access.db_connection import create_mc_engine
from device_anomaly.data_access.db_utils import table_exists

logger = logging.getLogger(__name__)


@dataclass
class EventEntry:
    """Single event from MainLog."""
    log_id: int
    timestamp: datetime
    event_id: int
    severity: str
    event_class: str
    message: str
    device_id: int | None = None
    login_id: str | None = None


@dataclass
class AlertEntry:
    """Single alert from Alert table."""
    alert_id: int
    alert_key: str
    alert_name: str
    severity: str
    device_id: str | None = None
    status: str = "Unknown"
    set_datetime: datetime | None = None
    ack_datetime: datetime | None = None


@dataclass
class EventTimelineData:
    """Event timeline response data."""
    events: list[EventEntry] = field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50
    severity_distribution: dict[str, int] = field(default_factory=dict)
    event_class_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class AlertSummaryData:
    """Alert summary data."""
    total_active: int = 0
    total_acknowledged: int = 0
    total_resolved: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    by_alert_name: list[dict[str, Any]] = field(default_factory=list)
    recent_alerts: list[AlertEntry] = field(default_factory=list)
    avg_acknowledge_time_minutes: float = 0.0
    avg_resolution_time_minutes: float = 0.0


@dataclass
class AlertTrendPoint:
    """Single point in alert trend."""
    timestamp: datetime
    count: int
    severity: str


@dataclass
class CorrelatedEvent:
    """Event that occurred before an anomaly."""
    event: EventEntry
    time_before_minutes: float
    frequency_score: float


def _parse_severity(severity_val: Any) -> str:
    """Parse severity value to string."""
    if severity_val is None:
        return "Unknown"
    if isinstance(severity_val, str):
        return severity_val
    # Map numeric severity to string
    severity_map = {
        0: "Debug",
        1: "Info",
        2: "Warning",
        3: "Error",
        4: "Critical",
    }
    return severity_map.get(int(severity_val), f"Level_{severity_val}")


def load_event_timeline(
    device_id: int | None = None,
    severity: str | None = None,
    event_class: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    page: int = 1,
    page_size: int = 50,
    engine: Engine | None = None,
) -> EventTimelineData:
    """
    Load paginated event timeline from MainLog.

    Args:
        device_id: Filter by device ID
        severity: Filter by severity level
        event_class: Filter by event class
        start_time: Start of time window
        end_time: End of time window
        page: Page number (1-indexed)
        page_size: Number of events per page
        engine: SQLAlchemy engine

    Returns:
        EventTimelineData with paginated events
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "MainLog"):
        logger.warning("MainLog table not found in MobiControlDB")
        return EventTimelineData()

    # Default time window
    if end_time is None:
        end_time = datetime.now(UTC)
    if start_time is None:
        start_time = end_time - timedelta(days=7)

    # Build filters
    filters = ["DateTime >= :start_time", "DateTime <= :end_time"]
    params: dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
    }

    if device_id:
        filters.append("DeviceId = :device_id")
        params["device_id"] = device_id

    if severity:
        filters.append("Severity = :severity")
        params["severity"] = severity

    if event_class:
        filters.append("EventClass = :event_class")
        params["event_class"] = event_class

    where_clause = " AND ".join(filters)

    # Get total count
    count_query = text(f"""
        SELECT COUNT(*) FROM dbo.MainLog WHERE {where_clause}
    """)

    # Get paginated events
    offset = (page - 1) * page_size
    events_query = text(f"""
        SELECT
            ILogId,
            DateTime,
            EventId,
            Severity,
            EventClass,
            ResTxt,
            DeviceId,
            LoginId
        FROM dbo.MainLog
        WHERE {where_clause}
        ORDER BY DateTime DESC
        OFFSET :offset ROWS
        FETCH NEXT :page_size ROWS ONLY
    """)

    # Get distributions
    severity_query = text(f"""
        SELECT Severity, COUNT(*) as cnt
        FROM dbo.MainLog
        WHERE {where_clause}
        GROUP BY Severity
    """)

    class_query = text(f"""
        SELECT EventClass, COUNT(*) as cnt
        FROM dbo.MainLog
        WHERE {where_clause}
        GROUP BY EventClass
        ORDER BY cnt DESC
    """)

    try:
        with engine.connect() as conn:
            total = conn.execute(count_query, params).scalar() or 0

            params["offset"] = offset
            params["page_size"] = page_size
            events_df = pd.read_sql(events_query, conn, params=params)

            # Reset params for distribution queries
            del params["offset"]
            del params["page_size"]
            severity_df = pd.read_sql(severity_query, conn, params=params)
            class_df = pd.read_sql(class_query, conn, params=params)

    except Exception as e:
        logger.error(f"Failed to load event timeline: {e}")
        return EventTimelineData()

    # Build events list
    events = []
    for _, row in events_df.iterrows():
        ts = row["DateTime"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        events.append(EventEntry(
            log_id=int(row["ILogId"]),
            timestamp=ts,
            event_id=int(row["EventId"]) if pd.notna(row["EventId"]) else 0,
            severity=_parse_severity(row["Severity"]),
            event_class=str(row["EventClass"]) if pd.notna(row["EventClass"]) else "Unknown",
            message=str(row["ResTxt"]) if pd.notna(row["ResTxt"]) else "",
            device_id=int(row["DeviceId"]) if pd.notna(row["DeviceId"]) else None,
            login_id=str(row["LoginId"]) if pd.notna(row["LoginId"]) else None,
        ))

    # Build distributions
    severity_dist = {}
    for _, row in severity_df.iterrows():
        sev = _parse_severity(row["Severity"])
        severity_dist[sev] = int(row["cnt"])

    class_dist = {}
    for _, row in class_df.iterrows():
        ec = str(row["EventClass"]) if pd.notna(row["EventClass"]) else "Unknown"
        class_dist[ec] = int(row["cnt"])

    return EventTimelineData(
        events=events,
        total=int(total),
        page=page,
        page_size=page_size,
        severity_distribution=severity_dist,
        event_class_distribution=class_dist,
    )


def load_alert_summary(
    start_time: datetime | None = None,
    status_filter: str | None = None,
    severity_filter: str | None = None,
    engine: Engine | None = None,
) -> AlertSummaryData:
    """
    Load alert summary statistics.

    Args:
        start_time: Start of time window
        status_filter: Filter by status (Active, Acknowledged, Resolved)
        severity_filter: Filter by severity level
        engine: SQLAlchemy engine

    Returns:
        AlertSummaryData with summary statistics
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "Alert"):
        logger.warning("Alert table not found in MobiControlDB")
        return AlertSummaryData()

    if start_time is None:
        start_time = datetime.now(UTC) - timedelta(days=30)

    # Build filters
    filters = ["SetDateTime >= :start_time"]
    params: dict[str, Any] = {"start_time": start_time}

    if status_filter:
        filters.append("Status = :status")
        params["status"] = status_filter

    if severity_filter:
        filters.append("AlertSeverity = :severity")
        params["severity"] = severity_filter

    where_clause = " AND ".join(filters)

    # Get summary counts
    summary_query = text(f"""
        SELECT
            Status,
            COUNT(*) as cnt
        FROM dbo.Alert
        WHERE {where_clause}
        GROUP BY Status
    """)

    # Get severity breakdown
    severity_query = text(f"""
        SELECT
            AlertSeverity,
            COUNT(*) as cnt
        FROM dbo.Alert
        WHERE {where_clause}
        GROUP BY AlertSeverity
    """)

    # Get top alert names
    names_query = text(f"""
        SELECT TOP 10
            AlertName,
            COUNT(*) as cnt
        FROM dbo.Alert
        WHERE {where_clause}
        GROUP BY AlertName
        ORDER BY cnt DESC
    """)

    # Get recent alerts
    recent_query = text(f"""
        SELECT TOP 20
            AlertId,
            AlertKey,
            AlertName,
            AlertSeverity,
            DevId,
            Status,
            SetDateTime,
            AckDateTime
        FROM dbo.Alert
        WHERE {where_clause}
        ORDER BY SetDateTime DESC
    """)

    # Get acknowledgment time stats
    ack_time_query = text(f"""
        SELECT
            AVG(DATEDIFF(MINUTE, SetDateTime, AckDateTime)) as avg_ack_time
        FROM dbo.Alert
        WHERE {where_clause}
            AND AckDateTime IS NOT NULL
    """)

    try:
        with engine.connect() as conn:
            summary_df = pd.read_sql(summary_query, conn, params=params)
            severity_df = pd.read_sql(severity_query, conn, params=params)
            names_df = pd.read_sql(names_query, conn, params=params)
            recent_df = pd.read_sql(recent_query, conn, params=params)
            ack_time = conn.execute(ack_time_query, params).scalar() or 0

    except Exception as e:
        logger.error(f"Failed to load alert summary: {e}")
        return AlertSummaryData()

    # Parse summary counts
    total_active = 0
    total_acknowledged = 0
    total_resolved = 0
    for _, row in summary_df.iterrows():
        status = str(row["Status"]).lower() if pd.notna(row["Status"]) else ""
        cnt = int(row["cnt"])
        if "active" in status or "open" in status:
            total_active += cnt
        elif "ack" in status:
            total_acknowledged += cnt
        elif "resolved" in status or "closed" in status:
            total_resolved += cnt

    # Parse severity breakdown
    by_severity = {}
    for _, row in severity_df.iterrows():
        sev = str(row["AlertSeverity"]) if pd.notna(row["AlertSeverity"]) else "Unknown"
        by_severity[sev] = int(row["cnt"])

    # Parse alert names
    by_name = []
    for _, row in names_df.iterrows():
        by_name.append({
            "name": str(row["AlertName"]) if pd.notna(row["AlertName"]) else "Unknown",
            "count": int(row["cnt"]),
        })

    # Parse recent alerts
    recent_alerts = []
    for _, row in recent_df.iterrows():
        set_dt = row["SetDateTime"]
        if isinstance(set_dt, str):
            set_dt = datetime.fromisoformat(set_dt)
        elif hasattr(set_dt, 'to_pydatetime'):
            set_dt = set_dt.to_pydatetime()
        if set_dt and set_dt.tzinfo is None:
            set_dt = set_dt.replace(tzinfo=UTC)

        ack_dt = row["AckDateTime"]
        if isinstance(ack_dt, str):
            ack_dt = datetime.fromisoformat(ack_dt)
        elif hasattr(ack_dt, 'to_pydatetime'):
            ack_dt = ack_dt.to_pydatetime()
        if ack_dt and ack_dt.tzinfo is None:
            ack_dt = ack_dt.replace(tzinfo=UTC)

        recent_alerts.append(AlertEntry(
            alert_id=int(row["AlertId"]),
            alert_key=str(row["AlertKey"]) if pd.notna(row["AlertKey"]) else "",
            alert_name=str(row["AlertName"]) if pd.notna(row["AlertName"]) else "",
            severity=str(row["AlertSeverity"]) if pd.notna(row["AlertSeverity"]) else "Unknown",
            device_id=str(row["DevId"]) if pd.notna(row["DevId"]) else None,
            status=str(row["Status"]) if pd.notna(row["Status"]) else "Unknown",
            set_datetime=set_dt,
            ack_datetime=ack_dt,
        ))

    return AlertSummaryData(
        total_active=total_active,
        total_acknowledged=total_acknowledged,
        total_resolved=total_resolved,
        by_severity=by_severity,
        by_alert_name=by_name,
        recent_alerts=recent_alerts,
        avg_acknowledge_time_minutes=float(ack_time) if ack_time else 0.0,
    )


def load_alert_trends(
    period_days: int = 7,
    granularity: str = "hourly",
    engine: Engine | None = None,
) -> list[AlertTrendPoint]:
    """
    Load alert count trends over time.

    Args:
        period_days: Number of days to analyze
        granularity: 'hourly' or 'daily'
        engine: SQLAlchemy engine

    Returns:
        List of AlertTrendPoint objects
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "Alert"):
        return []

    start_time = datetime.now(UTC) - timedelta(days=period_days)

    if granularity == "daily":
        time_group = "CAST(SetDateTime AS DATE)"
    else:
        time_group = "DATEADD(HOUR, DATEDIFF(HOUR, 0, SetDateTime), 0)"

    query = text(f"""
        SELECT
            {time_group} as time_bucket,
            AlertSeverity,
            COUNT(*) as cnt
        FROM dbo.Alert
        WHERE SetDateTime >= :start_time
        GROUP BY {time_group}, AlertSeverity
        ORDER BY time_bucket
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_time": start_time})
    except Exception as e:
        logger.error(f"Failed to load alert trends: {e}")
        return []

    trends = []
    for _, row in df.iterrows():
        ts = row["time_bucket"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        trends.append(AlertTrendPoint(
            timestamp=ts,
            count=int(row["cnt"]),
            severity=str(row["AlertSeverity"]) if pd.notna(row["AlertSeverity"]) else "Unknown",
        ))

    return trends


def find_correlated_events(
    anomaly_timestamp: datetime,
    device_id: int,
    window_minutes: int = 60,
    engine: Engine | None = None,
) -> list[CorrelatedEvent]:
    """
    Find events that occurred before an anomaly.

    Useful for root cause analysis by identifying events
    that commonly precede anomalies.

    Args:
        anomaly_timestamp: When the anomaly was detected
        device_id: Device ID of the anomaly
        window_minutes: Minutes before anomaly to search
        engine: SQLAlchemy engine

    Returns:
        List of CorrelatedEvent objects
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "MainLog"):
        return []

    start_time = anomaly_timestamp - timedelta(minutes=window_minutes)

    query = text("""
        SELECT
            ILogId,
            DateTime,
            EventId,
            Severity,
            EventClass,
            ResTxt,
            DeviceId,
            LoginId
        FROM dbo.MainLog
        WHERE DeviceId = :device_id
            AND DateTime >= :start_time
            AND DateTime <= :anomaly_time
        ORDER BY DateTime DESC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "device_id": device_id,
                "start_time": start_time,
                "anomaly_time": anomaly_timestamp,
            })
    except Exception as e:
        logger.error(f"Failed to find correlated events: {e}")
        return []

    correlated = []
    for _, row in df.iterrows():
        ts = row["DateTime"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        time_before = (anomaly_timestamp - ts).total_seconds() / 60

        event = EventEntry(
            log_id=int(row["ILogId"]),
            timestamp=ts,
            event_id=int(row["EventId"]) if pd.notna(row["EventId"]) else 0,
            severity=_parse_severity(row["Severity"]),
            event_class=str(row["EventClass"]) if pd.notna(row["EventClass"]) else "Unknown",
            message=str(row["ResTxt"]) if pd.notna(row["ResTxt"]) else "",
            device_id=int(row["DeviceId"]) if pd.notna(row["DeviceId"]) else None,
            login_id=str(row["LoginId"]) if pd.notna(row["LoginId"]) else None,
        )

        # Higher score for events closer to the anomaly
        frequency_score = 1.0 - (time_before / window_minutes)

        correlated.append(CorrelatedEvent(
            event=event,
            time_before_minutes=time_before,
            frequency_score=frequency_score,
        ))

    return correlated


def get_event_statistics(
    period_days: int = 7,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """
    Get overall event statistics.

    Returns:
        Dictionary with event statistics
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "MainLog"):
        return {
            "total_events": 0,
            "events_per_day": 0,
            "unique_devices": 0,
            "top_event_classes": [],
        }

    start_time = datetime.now(UTC) - timedelta(days=period_days)

    query = text("""
        SELECT
            COUNT(*) as total_events,
            COUNT(DISTINCT DeviceId) as unique_devices
        FROM dbo.MainLog
        WHERE DateTime >= :start_time
    """)

    top_classes_query = text("""
        SELECT TOP 5
            EventClass,
            COUNT(*) as cnt
        FROM dbo.MainLog
        WHERE DateTime >= :start_time
        GROUP BY EventClass
        ORDER BY cnt DESC
    """)

    try:
        with engine.connect() as conn:
            stats = conn.execute(query, {"start_time": start_time}).fetchone()
            classes_df = pd.read_sql(top_classes_query, conn, params={"start_time": start_time})
    except Exception as e:
        logger.error(f"Failed to get event statistics: {e}")
        return {"total_events": 0, "events_per_day": 0, "unique_devices": 0, "top_event_classes": []}

    total = stats[0] if stats else 0
    unique_devices = stats[1] if stats else 0

    top_classes = []
    for _, row in classes_df.iterrows():
        top_classes.append({
            "class": str(row["EventClass"]) if pd.notna(row["EventClass"]) else "Unknown",
            "count": int(row["cnt"]),
        })

    return {
        "total_events": int(total),
        "events_per_day": int(total / period_days) if period_days > 0 else 0,
        "unique_devices": int(unique_devices),
        "top_event_classes": top_classes,
    }
