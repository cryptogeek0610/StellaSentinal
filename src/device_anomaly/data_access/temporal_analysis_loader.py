"""
Temporal Analysis Data Loader.

Loads hourly data from XSight_DW for temporal drill-downs,
peak detection, and period comparisons.

Data Sources:
- cs_DataUsageByHour (XSight_DW): ~104M rows of hourly network data
- cs_BatteryLevelDrop (XSight_DW): ~14.8M rows of hourly battery drain
- cs_AppUsageListed (XSight_DW): ~8.5M rows of hourly app usage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text
from sqlalchemy.engine import Engine

from device_anomaly.data_access.db_connection import create_dw_engine
from device_anomaly.data_access.db_utils import table_exists

logger = logging.getLogger(__name__)


@dataclass
class HourlyDataPoint:
    """Single hour's aggregated data."""

    hour: int
    avg_value: float
    min_value: float
    max_value: float
    std_value: float
    sample_count: int


@dataclass
class HourlyBreakdownData:
    """Hourly breakdown analysis."""

    metric: str
    hourly_data: list[HourlyDataPoint] = field(default_factory=list)
    peak_hours: list[int] = field(default_factory=list)
    low_hours: list[int] = field(default_factory=list)
    day_night_ratio: float = 1.0


@dataclass
class PeakDetection:
    """Detected usage peak."""

    timestamp: datetime
    value: float
    z_score: float
    is_significant: bool


@dataclass
class PeriodStats:
    """Statistics for a time period."""

    start: datetime
    end: datetime
    avg: float
    median: float
    std: float
    sample_count: int


@dataclass
class TemporalComparisonData:
    """Comparison between two time periods."""

    metric: str
    period_a: PeriodStats
    period_b: PeriodStats
    change_percent: float
    is_significant: bool
    p_value: float


def load_hourly_breakdown(
    metric: str,
    period_days: int = 7,
    device_ids: list[int] | None = None,
    engine: Engine | None = None,
) -> HourlyBreakdownData:
    """
    Load hour-of-day aggregated metrics.

    Args:
        metric: Metric type (data_usage, battery_drain, app_usage)
        period_days: Number of days to analyze
        device_ids: Optional device filter
        engine: SQLAlchemy engine for XSight DW

    Returns:
        HourlyBreakdownData with hour-by-hour statistics
    """
    if engine is None:
        engine = create_dw_engine()

    # Map metric to table and column
    metric_config = {
        "data_usage": {
            "table": "cs_DataUsageByHour",
            "value_col": "Download + Upload",
            "hour_col": "Hour",
        },
        "battery_drain": {
            "table": "cs_BatteryLevelDrop",
            "value_col": "BatteryLevel",
            "hour_col": "Hour",
        },
        "app_usage": {
            "table": "cs_AppUsageListed",
            "value_col": "TotalForegroundTime",
            "hour_col": "Hour",
        },
    }

    config = metric_config.get(metric)
    if not config:
        logger.warning(f"Unknown metric: {metric}")
        return HourlyBreakdownData(metric=metric)

    table_name = config["table"]
    if not table_exists(engine, table_name):
        logger.warning(f"Table {table_name} not found")
        return HourlyBreakdownData(metric=metric)

    start_time = datetime.now(UTC) - timedelta(days=period_days)
    value_col = config["value_col"]
    hour_col = config["hour_col"]

    # Build device filter
    device_filter = ""
    params: dict[str, Any] = {"start_time": start_time}
    if device_ids:
        device_filter = "AND DeviceId IN :device_ids"
        params["device_ids"] = device_ids

    query = text(f"""
        SELECT
            {hour_col} as hour,
            AVG(CAST({value_col} AS FLOAT)) as avg_value,
            MIN({value_col}) as min_value,
            MAX({value_col}) as max_value,
            STDEV({value_col}) as std_value,
            COUNT(*) as sample_count
        FROM dbo.{table_name}
        WHERE CollectedDate > :start_time
            {device_filter}
        GROUP BY {hour_col}
        ORDER BY {hour_col}
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
    except Exception as e:
        logger.error(f"Failed to load hourly breakdown: {e}")
        return HourlyBreakdownData(metric=metric)

    if df.empty:
        return HourlyBreakdownData(metric=metric)

    # Build hourly data points
    hourly_data = []
    for _, row in df.iterrows():
        hourly_data.append(
            HourlyDataPoint(
                hour=int(row["hour"]),
                avg_value=float(row["avg_value"]) if pd.notna(row["avg_value"]) else 0,
                min_value=float(row["min_value"]) if pd.notna(row["min_value"]) else 0,
                max_value=float(row["max_value"]) if pd.notna(row["max_value"]) else 0,
                std_value=float(row["std_value"]) if pd.notna(row["std_value"]) else 0,
                sample_count=int(row["sample_count"]),
            )
        )

    # Find peak and low hours
    avg_values = [h.avg_value for h in hourly_data]
    if avg_values:
        mean = np.mean(avg_values)
        std = np.std(avg_values)

        peak_hours = [h.hour for h in hourly_data if h.avg_value > mean + std]
        low_hours = [h.hour for h in hourly_data if h.avg_value < mean - std]

        # Calculate day/night ratio (6am-6pm vs 6pm-6am)
        day_hours = [h for h in hourly_data if 6 <= h.hour < 18]
        night_hours = [h for h in hourly_data if h.hour < 6 or h.hour >= 18]

        day_avg = np.mean([h.avg_value for h in day_hours]) if day_hours else 0
        night_avg = np.mean([h.avg_value for h in night_hours]) if night_hours else 0
        day_night_ratio = day_avg / night_avg if night_avg > 0 else 1.0
    else:
        peak_hours = []
        low_hours = []
        day_night_ratio = 1.0

    return HourlyBreakdownData(
        metric=metric,
        hourly_data=hourly_data,
        peak_hours=peak_hours,
        low_hours=low_hours,
        day_night_ratio=day_night_ratio,
    )


def detect_peaks(
    metric: str,
    period_days: int = 7,
    std_threshold: float = 2.0,
    engine: Engine | None = None,
) -> list[PeakDetection]:
    """
    Detect statistically significant peaks in usage.

    Args:
        metric: Metric type
        period_days: Number of days to analyze
        std_threshold: Number of standard deviations for peak detection
        engine: SQLAlchemy engine

    Returns:
        List of detected peaks
    """
    if engine is None:
        engine = create_dw_engine()

    # Map metric to table and column
    metric_config = {
        "data_usage": {
            "table": "cs_DataUsageByHour",
            "value_col": "Download + Upload",
            "time_col": "CollectedDate",
        },
        "battery_drain": {
            "table": "cs_BatteryLevelDrop",
            "value_col": "BatteryLevel",
            "time_col": "CollectedDate",
        },
    }

    config = metric_config.get(metric)
    if not config:
        return []

    table_name = config["table"]
    if not table_exists(engine, table_name):
        return []

    start_time = datetime.now(UTC) - timedelta(days=period_days)

    # Get hourly aggregates
    query = text(f"""
        SELECT
            DATEADD(HOUR, DATEDIFF(HOUR, 0, {config["time_col"]}), 0) as time_bucket,
            SUM(CAST({config["value_col"]} AS FLOAT)) as total_value
        FROM dbo.{table_name}
        WHERE {config["time_col"]} > :start_time
        GROUP BY DATEADD(HOUR, DATEDIFF(HOUR, 0, {config["time_col"]}), 0)
        ORDER BY time_bucket
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_time": start_time})
    except Exception as e:
        logger.error(f"Failed to detect peaks: {e}")
        return []

    if df.empty or len(df) < 3:
        return []

    # Calculate z-scores
    values = df["total_value"].values
    mean = np.mean(values)
    std = np.std(values)

    peaks = []
    for _, row in df.iterrows():
        value = float(row["total_value"])
        z_score = (value - mean) / std if std > 0 else 0

        if abs(z_score) >= std_threshold:
            ts = row["time_bucket"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            elif hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            peaks.append(
                PeakDetection(
                    timestamp=ts,
                    value=value,
                    z_score=z_score,
                    is_significant=True,
                )
            )

    # Sort by z-score magnitude
    peaks.sort(key=lambda p: abs(p.z_score), reverse=True)

    return peaks[:20]  # Return top 20 peaks


def compare_periods(
    metric: str,
    period_a_start: datetime,
    period_a_end: datetime,
    period_b_start: datetime,
    period_b_end: datetime,
    engine: Engine | None = None,
) -> TemporalComparisonData:
    """
    Compare metrics between two time periods.

    Args:
        metric: Metric type
        period_a_start: Start of first period
        period_a_end: End of first period
        period_b_start: Start of second period
        period_b_end: End of second period
        engine: SQLAlchemy engine

    Returns:
        TemporalComparisonData with statistical comparison
    """
    if engine is None:
        engine = create_dw_engine()

    metric_config = {
        "data_usage": {
            "table": "cs_DataUsageByHour",
            "value_col": "Download + Upload",
            "time_col": "CollectedDate",
        },
        "battery_drain": {
            "table": "cs_BatteryLevelDrop",
            "value_col": "BatteryLevel",
            "time_col": "CollectedDate",
        },
    }

    config = metric_config.get(metric)
    empty_result = TemporalComparisonData(
        metric=metric,
        period_a=PeriodStats(period_a_start, period_a_end, 0, 0, 0, 0),
        period_b=PeriodStats(period_b_start, period_b_end, 0, 0, 0, 0),
        change_percent=0,
        is_significant=False,
        p_value=1.0,
    )

    if not config:
        return empty_result

    table_name = config["table"]
    if not table_exists(engine, table_name):
        return empty_result

    def get_period_data(start: datetime, end: datetime) -> pd.DataFrame:
        query = text(f"""
            SELECT CAST({config["value_col"]} AS FLOAT) as value
            FROM dbo.{table_name}
            WHERE {config["time_col"]} >= :start_time
                AND {config["time_col"]} < :end_time
        """)
        try:
            with engine.connect() as conn:
                return pd.read_sql(query, conn, params={"start_time": start, "end_time": end})
        except Exception as e:
            logger.error(f"Failed to get period data: {e}")
            return pd.DataFrame()

    df_a = get_period_data(period_a_start, period_a_end)
    df_b = get_period_data(period_b_start, period_b_end)

    if df_a.empty or df_b.empty:
        return empty_result

    values_a = df_a["value"].dropna().values
    values_b = df_b["value"].dropna().values

    if len(values_a) < 2 or len(values_b) < 2:
        return empty_result

    # Calculate statistics
    stats_a = PeriodStats(
        start=period_a_start,
        end=period_a_end,
        avg=float(np.mean(values_a)),
        median=float(np.median(values_a)),
        std=float(np.std(values_a)),
        sample_count=len(values_a),
    )

    stats_b = PeriodStats(
        start=period_b_start,
        end=period_b_end,
        avg=float(np.mean(values_b)),
        median=float(np.median(values_b)),
        std=float(np.std(values_b)),
        sample_count=len(values_b),
    )

    # Calculate change
    change_percent = ((stats_b.avg - stats_a.avg) / stats_a.avg * 100) if stats_a.avg != 0 else 0

    # Perform t-test for significance
    try:
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        is_significant = p_value < 0.05
    except Exception:
        p_value = 1.0
        is_significant = False

    return TemporalComparisonData(
        metric=metric,
        period_a=stats_a,
        period_b=stats_b,
        change_percent=change_percent,
        is_significant=is_significant,
        p_value=float(p_value),
    )


def get_day_over_day_comparison(
    metric: str,
    lookback_days: int = 7,
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    """
    Compare each day with the previous day.

    Returns:
        List of daily comparisons with change percentages
    """
    if engine is None:
        engine = create_dw_engine()

    metric_config = {
        "data_usage": {
            "table": "cs_DataUsageByHour",
            "value_col": "Download + Upload",
            "time_col": "CollectedDate",
        },
        "battery_drain": {
            "table": "cs_BatteryLevelDrop",
            "value_col": "BatteryLevel",
            "time_col": "CollectedDate",
        },
    }

    config = metric_config.get(metric)
    if not config:
        return []

    table_name = config["table"]
    if not table_exists(engine, table_name):
        return []

    start_time = datetime.now(UTC) - timedelta(days=lookback_days)

    # Get daily aggregates
    query = text(f"""
        SELECT
            CAST({config["time_col"]} AS DATE) as date,
            AVG(CAST({config["value_col"]} AS FLOAT)) as avg_value,
            COUNT(*) as sample_count
        FROM dbo.{table_name}
        WHERE {config["time_col"]} > :start_time
        GROUP BY CAST({config["time_col"]} AS DATE)
        ORDER BY date
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_time": start_time})
    except Exception as e:
        logger.error(f"Failed to get day over day comparison: {e}")
        return []

    if df.empty:
        return []

    comparisons = []
    prev_value = None

    for _, row in df.iterrows():
        current_value = float(row["avg_value"]) if pd.notna(row["avg_value"]) else 0

        if prev_value is not None and prev_value != 0:
            change = ((current_value - prev_value) / prev_value) * 100
        else:
            change = 0

        comparisons.append(
            {
                "date": str(row["date"]),
                "value": current_value,
                "sample_count": int(row["sample_count"]),
                "change_percent": change,
            }
        )

        prev_value = current_value

    return comparisons


def get_week_over_week_comparison(
    metric: str,
    lookback_weeks: int = 4,
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    """
    Compare each week with the previous week.

    Returns:
        List of weekly comparisons with change percentages
    """
    if engine is None:
        engine = create_dw_engine()

    metric_config = {
        "data_usage": {
            "table": "cs_DataUsageByHour",
            "value_col": "Download + Upload",
            "time_col": "CollectedDate",
        },
        "battery_drain": {
            "table": "cs_BatteryLevelDrop",
            "value_col": "BatteryLevel",
            "time_col": "CollectedDate",
        },
    }

    config = metric_config.get(metric)
    if not config:
        return []

    table_name = config["table"]
    if not table_exists(engine, table_name):
        return []

    start_time = datetime.now(UTC) - timedelta(weeks=lookback_weeks)

    # Get weekly aggregates
    query = text(f"""
        SELECT
            DATEPART(YEAR, {config["time_col"]}) as year,
            DATEPART(WEEK, {config["time_col"]}) as week,
            AVG(CAST({config["value_col"]} AS FLOAT)) as avg_value,
            COUNT(*) as sample_count
        FROM dbo.{table_name}
        WHERE {config["time_col"]} > :start_time
        GROUP BY DATEPART(YEAR, {config["time_col"]}), DATEPART(WEEK, {config["time_col"]})
        ORDER BY year, week
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_time": start_time})
    except Exception as e:
        logger.error(f"Failed to get week over week comparison: {e}")
        return []

    if df.empty:
        return []

    comparisons = []
    prev_value = None

    for _, row in df.iterrows():
        current_value = float(row["avg_value"]) if pd.notna(row["avg_value"]) else 0

        if prev_value is not None and prev_value != 0:
            change = ((current_value - prev_value) / prev_value) * 100
        else:
            change = 0

        comparisons.append(
            {
                "year": int(row["year"]),
                "week": int(row["week"]),
                "value": current_value,
                "sample_count": int(row["sample_count"]),
                "change_percent": change,
            }
        )

        prev_value = current_value

    return comparisons
