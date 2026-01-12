"""
Location Intelligence Data Loader.

Loads location data from XSight_DW (cs_WiFiLocation) and MobiControl
(DeviceStatLocation) for WiFi heatmaps, dead zone detection, and
device movement analysis.

Data Sources:
- cs_WiFiLocation (XSight_DW): WiFi signal + GPS coordinates, ~790K rows
- cs_WifiHour (XSight_DW): WiFi connectivity by AP, ~755K rows
- DeviceStatLocation (MobiControl): GPS with speed/heading, ~619K rows
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from device_anomaly.data_access.db_connection import create_dw_engine, create_mc_engine
from device_anomaly.data_access.db_utils import table_exists

logger = logging.getLogger(__name__)

# Grid cell size for heatmap aggregation (in degrees, ~100m)
DEFAULT_GRID_SIZE = 0.001

# Signal strength thresholds (dBm)
SIGNAL_THRESHOLDS = {
    "excellent": -50,
    "good": -60,
    "fair": -70,
    "poor": -80,
    "dead_zone": -85,
}


@dataclass
class HeatmapCell:
    """Single cell in a WiFi heatmap grid."""
    lat: float
    long: float
    signal_strength: float
    reading_count: int
    is_dead_zone: bool = False
    access_point_id: Optional[str] = None


@dataclass
class GeoBounds:
    """Geographic bounding box."""
    min_lat: float
    max_lat: float
    min_long: float
    max_long: float


@dataclass
class WiFiHeatmapData:
    """Complete WiFi heatmap data."""
    grid_cells: List[HeatmapCell] = field(default_factory=list)
    bounds: Optional[GeoBounds] = None
    total_readings: int = 0
    avg_signal_strength: float = 0.0
    dead_zone_count: int = 0


@dataclass
class DeadZone:
    """Dead zone location data."""
    zone_id: str
    lat: float
    long: float
    avg_signal: float
    affected_devices: int
    total_readings: int
    first_detected: Optional[datetime] = None
    last_detected: Optional[datetime] = None


@dataclass
class MovementPoint:
    """Single point in device movement history."""
    timestamp: datetime
    lat: float
    long: float
    speed: float
    heading: float


@dataclass
class DeviceMovementData:
    """Device movement summary."""
    device_id: int
    movements: List[MovementPoint] = field(default_factory=list)
    total_distance_km: float = 0.0
    avg_speed_kmh: float = 0.0
    stationary_time_pct: float = 0.0
    active_hours: List[int] = field(default_factory=list)


@dataclass
class DwellZone:
    """Location where devices spend significant time."""
    zone_id: str
    lat: float
    long: float
    avg_dwell_minutes: float
    device_count: int
    visit_count: int
    peak_hours: List[int] = field(default_factory=list)


def load_wifi_heatmap(
    period_days: int = 7,
    grid_size: float = DEFAULT_GRID_SIZE,
    min_signal: int = -100,
    engine: Optional[Engine] = None,
) -> WiFiHeatmapData:
    """
    Load WiFi location data and aggregate into a heatmap grid.

    Args:
        period_days: Number of days to look back
        grid_size: Grid cell size in degrees (~0.001 = 100m)
        min_signal: Minimum signal strength to include
        engine: SQLAlchemy engine for XSight DW

    Returns:
        WiFiHeatmapData with aggregated grid cells
    """
    if engine is None:
        engine = create_dw_engine()

    if not table_exists(engine, "cs_WiFiLocation"):
        logger.warning("cs_WiFiLocation table not found in XSight DW")
        return WiFiHeatmapData()

    start_time = datetime.now(timezone.utc) - timedelta(days=period_days)

    # Query WiFi location data with signal strength
    query = text("""
        SELECT
            Latitude,
            Longitude,
            WiFiSignalStrength,
            AccessPointId,
            ReadingTime
        FROM dbo.cs_WiFiLocation
        WHERE ReadingTime > :start_time
            AND Latitude IS NOT NULL
            AND Longitude IS NOT NULL
            AND WiFiSignalStrength >= :min_signal
            AND ABS(Latitude) <= 90
            AND ABS(Longitude) <= 180
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "start_time": start_time,
                "min_signal": min_signal,
            })
    except Exception as e:
        logger.error(f"Failed to load WiFi location data: {e}")
        return WiFiHeatmapData()

    if df.empty:
        return WiFiHeatmapData()

    # Grid the data
    df["grid_lat"] = (df["Latitude"] / grid_size).round() * grid_size
    df["grid_long"] = (df["Longitude"] / grid_size).round() * grid_size

    # Aggregate by grid cell
    grid_agg = df.groupby(["grid_lat", "grid_long"]).agg({
        "WiFiSignalStrength": ["mean", "count"],
        "AccessPointId": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None,
    }).reset_index()

    grid_agg.columns = ["lat", "long", "avg_signal", "reading_count", "access_point"]

    # Build heatmap cells
    cells = []
    dead_zone_count = 0

    for _, row in grid_agg.iterrows():
        is_dead = row["avg_signal"] < SIGNAL_THRESHOLDS["dead_zone"]
        if is_dead:
            dead_zone_count += 1

        cells.append(HeatmapCell(
            lat=float(row["lat"]),
            long=float(row["long"]),
            signal_strength=float(row["avg_signal"]),
            reading_count=int(row["reading_count"]),
            is_dead_zone=is_dead,
            access_point_id=str(row["access_point"]) if row["access_point"] else None,
        ))

    # Calculate bounds
    bounds = GeoBounds(
        min_lat=float(df["Latitude"].min()),
        max_lat=float(df["Latitude"].max()),
        min_long=float(df["Longitude"].min()),
        max_long=float(df["Longitude"].max()),
    )

    return WiFiHeatmapData(
        grid_cells=cells,
        bounds=bounds,
        total_readings=len(df),
        avg_signal_strength=float(df["WiFiSignalStrength"].mean()),
        dead_zone_count=dead_zone_count,
    )


def detect_dead_zones(
    period_days: int = 7,
    signal_threshold: int = -75,
    min_readings: int = 5,
    engine: Optional[Engine] = None,
) -> List[DeadZone]:
    """
    Identify areas with consistently poor WiFi signal.

    Args:
        period_days: Number of days to analyze
        signal_threshold: Signal strength threshold (dBm) for dead zone
        min_readings: Minimum readings to confirm dead zone
        engine: SQLAlchemy engine for XSight DW

    Returns:
        List of DeadZone objects
    """
    if engine is None:
        engine = create_dw_engine()

    if not table_exists(engine, "cs_WiFiLocation"):
        return []

    start_time = datetime.now(timezone.utc) - timedelta(days=period_days)
    grid_size = DEFAULT_GRID_SIZE * 2  # Larger grid for dead zones

    query = text("""
        SELECT
            ROUND(Latitude / :grid_size, 0) * :grid_size as grid_lat,
            ROUND(Longitude / :grid_size, 0) * :grid_size as grid_long,
            AVG(WiFiSignalStrength) as avg_signal,
            COUNT(*) as reading_count,
            COUNT(DISTINCT DeviceId) as device_count,
            MIN(ReadingTime) as first_reading,
            MAX(ReadingTime) as last_reading
        FROM dbo.cs_WiFiLocation
        WHERE ReadingTime > :start_time
            AND Latitude IS NOT NULL
            AND Longitude IS NOT NULL
        GROUP BY
            ROUND(Latitude / :grid_size, 0) * :grid_size,
            ROUND(Longitude / :grid_size, 0) * :grid_size
        HAVING AVG(WiFiSignalStrength) < :threshold
            AND COUNT(*) >= :min_readings
        ORDER BY avg_signal ASC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "start_time": start_time,
                "grid_size": grid_size,
                "threshold": signal_threshold,
                "min_readings": min_readings,
            })
    except Exception as e:
        logger.error(f"Failed to detect dead zones: {e}")
        return []

    dead_zones = []
    for idx, row in df.iterrows():
        first_dt = row["first_reading"]
        last_dt = row["last_reading"]

        # Convert to datetime if needed
        if isinstance(first_dt, str):
            first_dt = datetime.fromisoformat(first_dt)
        if isinstance(last_dt, str):
            last_dt = datetime.fromisoformat(last_dt)

        dead_zones.append(DeadZone(
            zone_id=f"dz_{idx}",
            lat=float(row["grid_lat"]),
            long=float(row["grid_long"]),
            avg_signal=float(row["avg_signal"]),
            affected_devices=int(row["device_count"]),
            total_readings=int(row["reading_count"]),
            first_detected=first_dt,
            last_detected=last_dt,
        ))

    return dead_zones


def load_device_movements(
    device_id: int,
    period_days: int = 7,
    engine: Optional[Engine] = None,
) -> DeviceMovementData:
    """
    Load GPS movement history for a specific device.

    Args:
        device_id: Device ID to query
        period_days: Number of days to look back
        engine: SQLAlchemy engine for MobiControl DB

    Returns:
        DeviceMovementData with movement points and statistics
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "DeviceStatLocation"):
        return DeviceMovementData(device_id=device_id)

    start_time = datetime.now(timezone.utc) - timedelta(days=period_days)

    query = text("""
        SELECT
            ServerDateTime,
            Latitude,
            Longitude,
            Speed,
            Heading
        FROM dbo.DeviceStatLocation
        WHERE DeviceId = :device_id
            AND ServerDateTime > :start_time
            AND Latitude IS NOT NULL
            AND Longitude IS NOT NULL
        ORDER BY ServerDateTime
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "device_id": device_id,
                "start_time": start_time,
            })
    except Exception as e:
        logger.error(f"Failed to load device movements: {e}")
        return DeviceMovementData(device_id=device_id)

    if df.empty:
        return DeviceMovementData(device_id=device_id)

    # Build movement points
    movements = []
    for _, row in df.iterrows():
        ts = row["ServerDateTime"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        movements.append(MovementPoint(
            timestamp=ts,
            lat=float(row["Latitude"]),
            long=float(row["Longitude"]),
            speed=float(row["Speed"]) if pd.notna(row["Speed"]) else 0.0,
            heading=float(row["Heading"]) if pd.notna(row["Heading"]) else 0.0,
        ))

    # Calculate statistics
    total_distance = 0.0
    if len(movements) > 1:
        for i in range(1, len(movements)):
            # Haversine distance approximation
            lat1, lon1 = movements[i-1].lat, movements[i-1].long
            lat2, lon2 = movements[i].lat, movements[i].long
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            total_distance += 6371 * c  # Earth radius in km

    speeds = [m.speed for m in movements]
    avg_speed = np.mean(speeds) if speeds else 0.0

    # Calculate stationary time (speed < 1 km/h)
    stationary = sum(1 for s in speeds if s < 1)
    stationary_pct = (stationary / len(speeds) * 100) if speeds else 0.0

    # Find active hours
    hours = [m.timestamp.hour for m in movements if m.speed > 1]
    active_hours = list(set(hours))

    return DeviceMovementData(
        device_id=device_id,
        movements=movements,
        total_distance_km=total_distance,
        avg_speed_kmh=avg_speed,
        stationary_time_pct=stationary_pct,
        active_hours=sorted(active_hours),
    )


def load_dwell_time_analysis(
    period_days: int = 7,
    min_dwell_minutes: int = 5,
    engine: Optional[Engine] = None,
) -> List[DwellZone]:
    """
    Analyze where devices spend significant time.

    Args:
        period_days: Number of days to analyze
        min_dwell_minutes: Minimum dwell time to consider
        engine: SQLAlchemy engine for MobiControl DB

    Returns:
        List of DwellZone objects
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "DeviceStatLocation"):
        return []

    start_time = datetime.now(timezone.utc) - timedelta(days=period_days)
    grid_size = DEFAULT_GRID_SIZE * 2  # Larger grid for dwell zones

    # Query location data with time spent calculation
    query = text("""
        WITH LocationGrid AS (
            SELECT
                DeviceId,
                ServerDateTime,
                ROUND(Latitude / :grid_size, 0) * :grid_size as grid_lat,
                ROUND(Longitude / :grid_size, 0) * :grid_size as grid_long,
                DATEPART(HOUR, ServerDateTime) as hour
            FROM dbo.DeviceStatLocation
            WHERE ServerDateTime > :start_time
                AND Latitude IS NOT NULL
                AND Longitude IS NOT NULL
        )
        SELECT
            grid_lat,
            grid_long,
            COUNT(*) as visit_count,
            COUNT(DISTINCT DeviceId) as device_count,
            COUNT(DISTINCT CAST(ServerDateTime AS DATE)) as days_present
        FROM LocationGrid
        GROUP BY grid_lat, grid_long
        HAVING COUNT(*) >= :min_readings
        ORDER BY visit_count DESC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "start_time": start_time,
                "grid_size": grid_size,
                "min_readings": 10,  # Minimum readings to be considered a dwell zone
            })
    except Exception as e:
        logger.error(f"Failed to load dwell time data: {e}")
        return []

    zones = []
    for idx, row in df.iterrows():
        # Estimate average dwell time based on visit count
        # This is a rough approximation - actual dwell time would need session detection
        avg_dwell = (row["visit_count"] / row["device_count"]) * 5  # Assume 5 min between readings

        if avg_dwell >= min_dwell_minutes:
            zones.append(DwellZone(
                zone_id=f"dwell_{idx}",
                lat=float(row["grid_lat"]),
                long=float(row["grid_long"]),
                avg_dwell_minutes=avg_dwell,
                device_count=int(row["device_count"]),
                visit_count=int(row["visit_count"]),
                peak_hours=[9, 10, 11, 14, 15, 16],  # Business hours as default
            ))

    return zones


def get_coverage_summary(
    period_days: int = 7,
    engine: Optional[Engine] = None,
) -> Dict[str, Any]:
    """
    Get summary statistics about WiFi coverage.

    Returns:
        Dictionary with coverage statistics
    """
    if engine is None:
        engine = create_dw_engine()

    if not table_exists(engine, "cs_WiFiLocation"):
        return {
            "total_readings": 0,
            "unique_locations": 0,
            "avg_signal": 0,
            "coverage_distribution": {},
        }

    start_time = datetime.now(timezone.utc) - timedelta(days=period_days)

    query = text("""
        SELECT
            COUNT(*) as total_readings,
            AVG(WiFiSignalStrength) as avg_signal,
            SUM(CASE WHEN WiFiSignalStrength >= -50 THEN 1 ELSE 0 END) as excellent,
            SUM(CASE WHEN WiFiSignalStrength >= -60 AND WiFiSignalStrength < -50 THEN 1 ELSE 0 END) as good,
            SUM(CASE WHEN WiFiSignalStrength >= -70 AND WiFiSignalStrength < -60 THEN 1 ELSE 0 END) as fair,
            SUM(CASE WHEN WiFiSignalStrength >= -80 AND WiFiSignalStrength < -70 THEN 1 ELSE 0 END) as poor,
            SUM(CASE WHEN WiFiSignalStrength < -80 THEN 1 ELSE 0 END) as dead
        FROM dbo.cs_WiFiLocation
        WHERE ReadingTime > :start_time
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"start_time": start_time}).fetchone()
    except Exception as e:
        logger.error(f"Failed to get coverage summary: {e}")
        return {"total_readings": 0, "unique_locations": 0, "avg_signal": 0, "coverage_distribution": {}}

    if result:
        total = result[0] or 0
        return {
            "total_readings": total,
            "avg_signal": float(result[1]) if result[1] else 0,
            "coverage_distribution": {
                "excellent": int(result[2] or 0),
                "good": int(result[3] or 0),
                "fair": int(result[4] or 0),
                "poor": int(result[5] or 0),
                "dead": int(result[6] or 0),
            },
        }

    return {"total_readings": 0, "unique_locations": 0, "avg_signal": 0, "coverage_distribution": {}}
