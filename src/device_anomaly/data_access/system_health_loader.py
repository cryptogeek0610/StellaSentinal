"""
System Health Data Loader.

Loads system health metrics (CPU, RAM, storage, temperature) from
MobiControl's DeviceStatInt table for fleet-wide health monitoring.

StatType codes used:
- 1: battery_level
- 2: total_storage
- 3: available_storage
- 4: total_ram
- 5: available_ram
- 6: cpu_usage
- 7: memory_usage
- 10: device_temperature
- 11: battery_temperature
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from device_anomaly.data_access.db_connection import create_mc_engine
from device_anomaly.data_access.db_utils import table_exists

logger = logging.getLogger(__name__)

# StatType codes for system health metrics
HEALTH_STAT_TYPES = {
    1: "battery_level",
    2: "total_storage",
    3: "available_storage",
    4: "total_ram",
    5: "available_ram",
    6: "cpu_usage",
    7: "memory_usage",
    10: "device_temperature",
    11: "battery_temperature",
}

# Thresholds for health status
HEALTH_THRESHOLDS = {
    "cpu_usage": {"warning": 70, "critical": 90},
    "memory_usage": {"warning": 70, "critical": 90},
    "storage_available_pct": {"warning": 20, "critical": 10},
    "device_temperature": {"warning": 35, "critical": 45},
    "battery_temperature": {"warning": 40, "critical": 50},
}


@dataclass
class SystemHealthMetrics:
    """Aggregated system health metrics for the fleet."""
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    avg_storage_available_pct: float = 0.0
    avg_device_temp: float = 0.0
    avg_battery_temp: float = 0.0
    devices_high_cpu: int = 0
    devices_high_memory: int = 0
    devices_low_storage: int = 0
    devices_high_temp: int = 0
    total_devices: int = 0


@dataclass
class StorageForecast:
    """Storage exhaustion forecast for a device."""
    device_id: int
    device_name: str = ""
    current_storage_pct: float = 0.0
    storage_trend_gb_per_day: float = 0.0
    projected_full_date: datetime | None = None
    days_until_full: int | None = None
    confidence: float = 0.0


@dataclass
class CohortHealthBreakdown:
    """Health breakdown by device cohort (model/OS)."""
    cohort_id: str
    cohort_name: str
    device_count: int = 0
    health_score: float = 100.0
    avg_cpu: float = 0.0
    avg_memory: float = 0.0
    avg_storage_pct: float = 0.0
    devices_at_risk: int = 0


@dataclass
class HealthTrendPoint:
    """Single point in a health trend time-series."""
    timestamp: datetime
    value: float
    device_count: int = 0


def load_system_health_metrics(
    period_days: int = 7,
    device_ids: list[int] | None = None,
    engine: Engine | None = None,
) -> SystemHealthMetrics:
    """
    Load aggregated system health metrics from DeviceStatInt.

    Args:
        period_days: Number of days to look back
        device_ids: Optional list of specific device IDs
        engine: SQLAlchemy engine (creates one if not provided)

    Returns:
        SystemHealthMetrics with aggregated values
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "DeviceStatInt"):
        logger.warning("DeviceStatInt table not found")
        return SystemHealthMetrics()

    start_time = datetime.now(UTC) - timedelta(days=period_days)
    stat_types = list(HEALTH_STAT_TYPES.keys())

    # Build device filter
    device_filter = ""
    params: dict[str, Any] = {
        "start_time": start_time,
        "stat_types": stat_types,
    }

    if device_ids:
        device_filter = "AND DeviceId IN :device_ids"
        params["device_ids"] = device_ids

    # Query for latest value per device per stat type
    query = text(f"""
        WITH LatestStats AS (
            SELECT
                DeviceId,
                StatType,
                IntValue,
                ROW_NUMBER() OVER (PARTITION BY DeviceId, StatType ORDER BY ServerDateTime DESC) as rn
            FROM dbo.DeviceStatInt
            WHERE ServerDateTime > :start_time
                AND StatType IN :stat_types
                {device_filter}
        )
        SELECT DeviceId, StatType, IntValue
        FROM LatestStats
        WHERE rn = 1
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
    except Exception as e:
        logger.error(f"Failed to load system health metrics: {e}")
        return SystemHealthMetrics()

    if df.empty:
        return SystemHealthMetrics()

    # Pivot to get one row per device
    pivot_df = df.pivot(index="DeviceId", columns="StatType", values="IntValue")

    metrics = SystemHealthMetrics()
    metrics.total_devices = len(pivot_df)

    # Calculate CPU usage stats
    if 6 in pivot_df.columns:
        cpu_values = pivot_df[6].dropna()
        metrics.avg_cpu_usage = float(cpu_values.mean()) if len(cpu_values) > 0 else 0
        metrics.devices_high_cpu = int((cpu_values > HEALTH_THRESHOLDS["cpu_usage"]["warning"]).sum())

    # Calculate memory usage stats
    if 7 in pivot_df.columns:
        mem_values = pivot_df[7].dropna()
        metrics.avg_memory_usage = float(mem_values.mean()) if len(mem_values) > 0 else 0
        metrics.devices_high_memory = int((mem_values > HEALTH_THRESHOLDS["memory_usage"]["warning"]).sum())

    # Calculate storage stats (available_storage / total_storage * 100)
    if 2 in pivot_df.columns and 3 in pivot_df.columns:
        total_storage = pivot_df[2].fillna(0)
        avail_storage = pivot_df[3].fillna(0)
        # Avoid division by zero
        storage_pct = np.where(total_storage > 0, (avail_storage / total_storage) * 100, 0)
        metrics.avg_storage_available_pct = float(np.mean(storage_pct))
        metrics.devices_low_storage = int((storage_pct < HEALTH_THRESHOLDS["storage_available_pct"]["warning"]).sum())

    # Calculate temperature stats
    if 10 in pivot_df.columns:
        temp_values = pivot_df[10].dropna()
        metrics.avg_device_temp = float(temp_values.mean()) if len(temp_values) > 0 else 0
        metrics.devices_high_temp = int((temp_values > HEALTH_THRESHOLDS["device_temperature"]["warning"]).sum())

    if 11 in pivot_df.columns:
        batt_temp = pivot_df[11].dropna()
        metrics.avg_battery_temp = float(batt_temp.mean()) if len(batt_temp) > 0 else 0

    return metrics


def load_health_trends(
    metric: str,
    period_days: int = 7,
    granularity: str = "hourly",
    device_ids: list[int] | None = None,
    engine: Engine | None = None,
) -> list[HealthTrendPoint]:
    """
    Load time-series health trends for a specific metric.

    Args:
        metric: Metric name (cpu_usage, memory_usage, storage_available_pct, device_temperature)
        period_days: Number of days to look back
        granularity: 'hourly' or 'daily'
        device_ids: Optional device filter
        engine: SQLAlchemy engine

    Returns:
        List of HealthTrendPoint objects
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "DeviceStatInt"):
        return []

    # Map metric name to StatType
    metric_to_stat_type = {
        "cpu_usage": 6,
        "memory_usage": 7,
        "device_temperature": 10,
        "battery_temperature": 11,
        "battery_level": 1,
    }

    stat_type = metric_to_stat_type.get(metric)
    if stat_type is None:
        logger.warning(f"Unknown metric: {metric}")
        return []

    start_time = datetime.now(UTC) - timedelta(days=period_days)

    # Build time grouping
    if granularity == "daily":
        time_group = "CAST(ServerDateTime AS DATE)"
    else:
        time_group = "DATEADD(HOUR, DATEDIFF(HOUR, 0, ServerDateTime), 0)"

    device_filter = ""
    params: dict[str, Any] = {
        "start_time": start_time,
        "stat_type": stat_type,
    }

    if device_ids:
        device_filter = "AND DeviceId IN :device_ids"
        params["device_ids"] = device_ids

    query = text(f"""
        SELECT
            {time_group} as time_bucket,
            AVG(CAST(IntValue AS FLOAT)) as avg_value,
            COUNT(DISTINCT DeviceId) as device_count
        FROM dbo.DeviceStatInt
        WHERE ServerDateTime > :start_time
            AND StatType = :stat_type
            {device_filter}
        GROUP BY {time_group}
        ORDER BY time_bucket
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
    except Exception as e:
        logger.error(f"Failed to load health trends: {e}")
        return []

    if df.empty:
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

        trends.append(HealthTrendPoint(
            timestamp=ts,
            value=float(row["avg_value"]),
            device_count=int(row["device_count"]),
        ))

    return trends


def calculate_storage_forecast(
    lookback_days: int = 30,
    forecast_days: int = 90,
    min_data_points: int = 5,
    engine: Engine | None = None,
) -> list[StorageForecast]:
    """
    Predict when devices will run out of storage using linear regression.

    Args:
        lookback_days: Days of historical data to analyze
        forecast_days: Days into the future to predict
        min_data_points: Minimum data points required for reliable prediction
        engine: SQLAlchemy engine

    Returns:
        List of StorageForecast for devices at risk
    """
    if engine is None:
        engine = create_mc_engine()

    if not table_exists(engine, "DeviceStatInt"):
        return []

    start_time = datetime.now(UTC) - timedelta(days=lookback_days)

    # Get storage history for all devices
    query = text("""
        SELECT
            DeviceId,
            ServerDateTime,
            IntValue
        FROM dbo.DeviceStatInt
        WHERE ServerDateTime > :start_time
            AND StatType = 3  -- available_storage
        ORDER BY DeviceId, ServerDateTime
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_time": start_time})
    except Exception as e:
        logger.error(f"Failed to load storage history: {e}")
        return []

    if df.empty:
        return []

    # Also get total storage for percentage calculation
    total_query = text("""
        SELECT DeviceId, MAX(IntValue) as total_storage
        FROM dbo.DeviceStatInt
        WHERE StatType = 2  -- total_storage
        GROUP BY DeviceId
    """)

    try:
        with engine.connect() as conn:
            total_df = pd.read_sql(total_query, conn)
    except Exception:
        total_df = pd.DataFrame()

    forecasts = []

    for device_id, group in df.groupby("DeviceId"):
        if len(group) < min_data_points:
            continue

        # Convert timestamps to days since first observation
        group = group.sort_values("ServerDateTime")
        first_ts = group["ServerDateTime"].min()
        group["days"] = (group["ServerDateTime"] - first_ts).dt.total_seconds() / 86400

        # Linear regression
        x = group["days"].values
        y = group["IntValue"].values

        if len(x) < 2 or np.std(x) == 0:
            continue

        # Calculate slope (storage change per day) using least squares
        slope, intercept = np.polyfit(x, y, 1)

        # If storage is decreasing (negative slope), predict when it hits zero
        if slope < 0:
            # Current storage
            current_storage = y[-1]
            x[-1]

            # Get total storage for this device
            total_storage = 0
            if not total_df.empty:
                device_total = total_df[total_df["DeviceId"] == device_id]
                if not device_total.empty:
                    total_storage = float(device_total["total_storage"].iloc[0])

            # Calculate current percentage
            current_pct = (current_storage / total_storage * 100) if total_storage > 0 else 0

            # Days until storage hits zero
            days_until_zero = -current_storage / slope if slope < 0 else None

            # Only include if predicted to run out within forecast window
            if days_until_zero and days_until_zero <= forecast_days:
                projected_date = datetime.now(UTC) + timedelta(days=days_until_zero)

                # Calculate confidence based on R-squared
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                forecasts.append(StorageForecast(
                    device_id=int(device_id),
                    current_storage_pct=current_pct,
                    storage_trend_gb_per_day=slope / (1024 * 1024 * 1024),  # Convert to GB
                    projected_full_date=projected_date,
                    days_until_full=int(days_until_zero),
                    confidence=max(0, min(1, r_squared)),
                ))

    # Sort by days until full (most urgent first)
    forecasts.sort(key=lambda f: f.days_until_full or float('inf'))

    return forecasts


def load_cohort_health_breakdown(
    period_days: int = 7,
    engine: Engine | None = None,
) -> list[CohortHealthBreakdown]:
    """
    Break down health metrics by device cohort (model/OS).

    Note: This requires device metadata. Returns simplified breakdown
    based on DeviceId ranges if metadata not available.

    Args:
        period_days: Number of days to look back
        engine: SQLAlchemy engine

    Returns:
        List of CohortHealthBreakdown objects
    """
    if engine is None:
        engine = create_mc_engine()

    # First get the metrics per device
    metrics_df = _load_device_health_dataframe(period_days, engine)
    if metrics_df.empty:
        return []

    # Try to join with device info for cohort grouping
    device_query = text("""
        SELECT TOP 1000
            DeviceId,
            COALESCE(DeviceName, CAST(DeviceId AS VARCHAR)) as DeviceName,
            COALESCE(Model, 'Unknown') as Model,
            COALESCE(OSType, 'Unknown') as OSType
        FROM dbo.Device
    """)

    try:
        with engine.connect() as conn:
            devices_df = pd.read_sql(device_query, conn)

        # Merge with metrics
        merged = metrics_df.merge(devices_df, on="DeviceId", how="left")

        # Create cohort identifier
        merged["cohort_id"] = merged["Model"].fillna("Unknown") + "_" + merged["OSType"].fillna("Unknown")
        merged["cohort_name"] = merged["Model"].fillna("Unknown Model")

    except Exception as e:
        logger.debug(f"Could not load device info for cohort breakdown: {e}")
        # Fallback: no cohort grouping, just return overall
        return [CohortHealthBreakdown(
            cohort_id="all_devices",
            cohort_name="All Devices",
            device_count=len(metrics_df),
            health_score=_calculate_health_score(metrics_df),
            avg_cpu=float(metrics_df["cpu_usage"].mean()) if "cpu_usage" in metrics_df else 0,
            avg_memory=float(metrics_df["memory_usage"].mean()) if "memory_usage" in metrics_df else 0,
            avg_storage_pct=float(metrics_df["storage_pct"].mean()) if "storage_pct" in metrics_df else 0,
        )]

    # Group by cohort and calculate stats
    breakdowns = []
    for cohort_id, group in merged.groupby("cohort_id"):
        cohort_name = group["cohort_name"].iloc[0] if "cohort_name" in group else str(cohort_id)

        breakdown = CohortHealthBreakdown(
            cohort_id=str(cohort_id),
            cohort_name=str(cohort_name),
            device_count=len(group),
            health_score=_calculate_health_score(group),
            avg_cpu=float(group["cpu_usage"].mean()) if "cpu_usage" in group else 0,
            avg_memory=float(group["memory_usage"].mean()) if "memory_usage" in group else 0,
            avg_storage_pct=float(group["storage_pct"].mean()) if "storage_pct" in group else 0,
        )

        # Count at-risk devices
        at_risk = 0
        if "cpu_usage" in group.columns:
            at_risk += (group["cpu_usage"] > HEALTH_THRESHOLDS["cpu_usage"]["warning"]).sum()
        if "storage_pct" in group.columns:
            at_risk += (group["storage_pct"] < HEALTH_THRESHOLDS["storage_available_pct"]["warning"]).sum()
        breakdown.devices_at_risk = int(at_risk)

        breakdowns.append(breakdown)

    # Sort by health score (worst first)
    breakdowns.sort(key=lambda b: b.health_score)

    return breakdowns


def _load_device_health_dataframe(
    period_days: int,
    engine: Engine,
) -> pd.DataFrame:
    """Load health metrics as a DataFrame with one row per device."""
    if not table_exists(engine, "DeviceStatInt"):
        return pd.DataFrame()

    start_time = datetime.now(UTC) - timedelta(days=period_days)
    stat_types = [1, 2, 3, 4, 5, 6, 7, 10, 11]

    query = text("""
        WITH LatestStats AS (
            SELECT
                DeviceId,
                StatType,
                IntValue,
                ROW_NUMBER() OVER (PARTITION BY DeviceId, StatType ORDER BY ServerDateTime DESC) as rn
            FROM dbo.DeviceStatInt
            WHERE ServerDateTime > :start_time
                AND StatType IN :stat_types
        )
        SELECT DeviceId, StatType, IntValue
        FROM LatestStats
        WHERE rn = 1
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "start_time": start_time,
                "stat_types": stat_types,
            })
    except Exception as e:
        logger.error(f"Failed to load device health dataframe: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Pivot to get one row per device
    pivot_df = df.pivot(index="DeviceId", columns="StatType", values="IntValue")
    pivot_df = pivot_df.reset_index()

    # Rename columns to meaningful names
    col_renames = {
        1: "battery_level",
        2: "total_storage",
        3: "available_storage",
        4: "total_ram",
        5: "available_ram",
        6: "cpu_usage",
        7: "memory_usage",
        10: "device_temperature",
        11: "battery_temperature",
    }
    pivot_df = pivot_df.rename(columns=col_renames)

    # Calculate storage percentage
    if "total_storage" in pivot_df.columns and "available_storage" in pivot_df.columns:
        total = pivot_df["total_storage"].fillna(0)
        avail = pivot_df["available_storage"].fillna(0)
        pivot_df["storage_pct"] = np.where(total > 0, (avail / total) * 100, 0)

    return pivot_df


def _calculate_health_score(df: pd.DataFrame) -> float:
    """
    Calculate overall health score (0-100) based on metrics.

    Higher score = healthier.
    """
    score = 100.0
    penalties = 0

    if "cpu_usage" in df.columns:
        avg_cpu = df["cpu_usage"].mean()
        if avg_cpu > HEALTH_THRESHOLDS["cpu_usage"]["critical"]:
            penalties += 30
        elif avg_cpu > HEALTH_THRESHOLDS["cpu_usage"]["warning"]:
            penalties += 15

    if "memory_usage" in df.columns:
        avg_mem = df["memory_usage"].mean()
        if avg_mem > HEALTH_THRESHOLDS["memory_usage"]["critical"]:
            penalties += 25
        elif avg_mem > HEALTH_THRESHOLDS["memory_usage"]["warning"]:
            penalties += 10

    if "storage_pct" in df.columns:
        avg_storage = df["storage_pct"].mean()
        if avg_storage < HEALTH_THRESHOLDS["storage_available_pct"]["critical"]:
            penalties += 25
        elif avg_storage < HEALTH_THRESHOLDS["storage_available_pct"]["warning"]:
            penalties += 10

    if "device_temperature" in df.columns:
        avg_temp = df["device_temperature"].mean()
        if avg_temp > HEALTH_THRESHOLDS["device_temperature"]["critical"]:
            penalties += 20
        elif avg_temp > HEALTH_THRESHOLDS["device_temperature"]["warning"]:
            penalties += 10

    return max(0, score - penalties)


def calculate_fleet_health_score(
    period_days: int = 7,
    engine: Engine | None = None,
) -> tuple[float, str]:
    """
    Calculate overall fleet health score and trend.

    Returns:
        Tuple of (score, trend) where trend is 'improving', 'stable', or 'degrading'
    """
    if engine is None:
        engine = create_mc_engine()

    # Get current period metrics
    current_df = _load_device_health_dataframe(period_days, engine)
    current_score = _calculate_health_score(current_df) if not current_df.empty else 100.0

    # Get previous period for comparison
    # This would require querying with different time windows
    # For now, return 'stable' as default trend
    trend = "stable"

    return current_score, trend
