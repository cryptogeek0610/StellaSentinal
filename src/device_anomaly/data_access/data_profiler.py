"""
Data profiling module for analyzing SQL Server telemetry data.

Provides comprehensive statistical analysis of DW and MobiControl tables
to understand data distributions, patterns, and baselines for ML training.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from device_anomaly.data_access.db_connection import create_dw_engine, create_mc_engine
from device_anomaly.data_access.db_utils import table_exists

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, label: str) -> str:
    """Ensure SQL identifiers are safe (prevents injection via table/column names)."""
    if not name or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {label} identifier: {name!r}")
    return name


def _column_exists(engine, table_name: str, column_name: str) -> bool:
    query = text(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :table_name
          AND COLUMN_NAME = :column_name
        """
    )
    try:
        with engine.connect() as conn:
            return (
                conn.execute(
                    query,
                    {"table_name": table_name, "column_name": column_name},
                ).first()
                is not None
            )
    except Exception as exc:
        logger.warning(
            "Failed to verify column %s.%s: %s",
            table_name,
            column_name,
            exc,
        )
        return True  # Fail open for non-SQL Server engines.


def _ensure_safe_table(engine, table_name: str) -> str:
    table_name = _validate_identifier(table_name, "table")
    if not table_exists(engine, table_name):
        raise ValueError(f"Unknown table: {table_name}")
    return table_name


def _ensure_safe_column(engine, table_name: str, column_name: str) -> str:
    column_name = _validate_identifier(column_name, "column")
    if not _column_exists(engine, table_name, column_name):
        raise ValueError(f"Unknown column: {table_name}.{column_name}")
    return column_name

# Known telemetry tables in XSight DW (used as fallback if discovery fails)
# This is the LEGACY list - prefer using get_curated_xsight_tables() for dynamic discovery
DW_TELEMETRY_TABLES_LEGACY = [
    "cs_BatteryStat",
    "cs_AppUsage",
    "cs_DataUsage",
    "cs_BatteryAppDrain",
    "cs_Heatmap",
]

# Extended table list including high-value tables discovered via schema analysis
DW_TELEMETRY_TABLES_EXTENDED = [
    # Original 5 tables
    "cs_BatteryStat",
    "cs_AppUsage",
    "cs_DataUsage",
    "cs_BatteryAppDrain",
    "cs_Heatmap",
    # High-volume hourly tables
    "cs_DataUsageByHour",
    "cs_BatteryLevelDrop",
    "cs_AppUsageListed",
    # WiFi and location tables
    "cs_WifiHour",
    "cs_WiFiLocation",
    "cs_LastKnown",
    # App inventory
    "cs_DeviceInstalledApp",
    # Preset apps (high volume)
    "cs_PresetApps",
]

# MobiControl time-series tables (high-value for ML)
MC_TIMESERIES_TABLES = [
    "DeviceStatInt",
    "DeviceStatLocation",
    "DeviceStatString",
    "DeviceStatNetTraffic",
    "MainLog",
    "Alert",
    "DeviceInstalledApp",
]

# For backward compatibility
DW_TELEMETRY_TABLES = DW_TELEMETRY_TABLES_EXTENDED


def get_curated_xsight_tables(
    include_extended: bool = True,
    include_discovered: bool = False,
    min_rows: int = 10000,
    engine=None,
) -> list[str]:
    """
    Get curated list of XSight tables for profiling/training.

    Args:
        include_extended: Include extended table list (default True)
        include_discovered: Also include runtime-discovered tables (slower)
        min_rows: Minimum row count for discovered tables
        engine: SQLAlchemy engine

    Returns:
        List of table names
    """
    tables = set()

    if include_extended:
        tables.update(DW_TELEMETRY_TABLES_EXTENDED)

    if include_discovered:
        try:
            from device_anomaly.data_access.schema_discovery import (
                SourceDatabase,
                get_curated_table_list,
            )
            discovered = get_curated_table_list(
                source_db=SourceDatabase.XSIGHT,
                include_explicit=True,
                include_discovered=True,
                min_rows=min_rows,
            )
            tables.update(discovered)
        except Exception as e:
            logger.warning(f"Failed to discover XSight tables: {e}")

    return sorted(tables)


def get_curated_mc_tables(
    include_discovered: bool = False,
    min_rows: int = 1000,
    engine=None,
) -> list[str]:
    """
    Get curated list of MobiControl tables for profiling/training.

    Args:
        include_discovered: Also include runtime-discovered tables
        min_rows: Minimum row count for discovered tables
        engine: SQLAlchemy engine

    Returns:
        List of table names
    """
    tables = set(MC_TIMESERIES_TABLES)

    if include_discovered:
        try:
            from device_anomaly.data_access.schema_discovery import (
                SourceDatabase,
                get_curated_table_list,
            )
            discovered = get_curated_table_list(
                source_db=SourceDatabase.MOBICONTROL,
                include_explicit=True,
                include_discovered=True,
                min_rows=min_rows,
            )
            tables.update(discovered)
        except Exception as e:
            logger.warning(f"Failed to discover MC tables: {e}")

    return sorted(tables)


# Patterns for time-sliced views that are typically not needed for ML training
TIME_SLICE_SUFFIXES = ["_Last7", "_Last30", "_LastWeek", "_LastMonth", "_Today", "_Yesterday"]


def discover_dw_tables(
    engine=None,
    pattern: str = "cs_%",
    exclude_time_slices: bool = True,
) -> list[str]:
    """
    Dynamically discover all available telemetry tables in the DW database.

    Args:
        engine: SQLAlchemy engine (creates one if not provided)
        pattern: SQL LIKE pattern to filter tables (default: cs_%)
        exclude_time_slices: If True, exclude time-sliced views like _Last7, _LastMonth (default: True)

    Returns:
        List of table names that exist in the database
    """
    if engine is None:
        engine = create_dw_engine()

    query = text("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
          AND TABLE_NAME LIKE :pattern
        ORDER BY TABLE_NAME
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"pattern": pattern})
            tables = [row[0] for row in result.fetchall()]

            # Optionally filter out time-sliced views
            if exclude_time_slices:
                original_count = len(tables)
                tables = [
                    t for t in tables
                    if not any(t.endswith(suffix) for suffix in TIME_SLICE_SUFFIXES)
                ]
                if len(tables) < original_count:
                    logger.debug(
                        f"Filtered {original_count - len(tables)} time-sliced views from discovery"
                    )

            logger.info(f"Discovered {len(tables)} tables matching pattern '{pattern}': {tables}")
            return tables
    except Exception as e:
        logger.error(f"Failed to discover tables: {e}")
        # Fall back to known tables
        return DW_TELEMETRY_TABLES


@dataclass
class ColumnStats:
    """Statistical summary for a single column."""

    column_name: str
    dtype: str
    null_count: int = 0
    null_percent: float = 0.0
    unique_count: int = 0
    min_val: float | None = None
    max_val: float | None = None
    mean: float | None = None
    std: float | None = None
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalPattern:
    """Time-based pattern statistics for a metric."""

    metric_name: str
    hourly_stats: dict[int, dict[str, float]] = field(default_factory=dict)  # hour 0-23
    daily_stats: dict[int, dict[str, float]] = field(default_factory=dict)  # day 0-6 (Mon-Sun)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableProfile:
    """Complete statistical profile of a database table."""

    table_name: str
    row_count: int
    date_range: tuple[str | None, str | None]
    device_count: int
    column_stats: dict[str, ColumnStats] = field(default_factory=dict)
    profiled_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        result = {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "date_range": self.date_range,
            "device_count": self.device_count,
            "profiled_at": self.profiled_at,
            "column_stats": {k: v.to_dict() for k, v in self.column_stats.items()},
        }
        return result


def _get_table_metadata(engine, table_name: str) -> pd.DataFrame:
    """Get column metadata from INFORMATION_SCHEMA."""
    query = text("""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :table_name
        ORDER BY ORDINAL_POSITION
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"table_name": table_name})


def _get_table_row_count(engine, table_name: str) -> int:
    """Get total row count for a table."""
    table_name = _ensure_safe_table(engine, table_name)
    query = text(f"SELECT COUNT(*) as cnt FROM {table_name}")
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
        return result[0] if result else 0


def _get_date_range(engine, table_name: str, date_column: str = "CollectedDate") -> tuple[str | None, str | None]:
    """Get min and max dates from a table."""
    table_name = _ensure_safe_table(engine, table_name)
    date_column = _ensure_safe_column(engine, table_name, date_column)
    query = text(f"""
        SELECT
            MIN({date_column}) as min_date,
            MAX({date_column}) as max_date
        FROM {table_name}
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            if result and result[0] and result[1]:
                return (str(result[0]), str(result[1]))
    except Exception as e:
        logger.warning(f"Could not get date range for {table_name}: {e}")
    return (None, None)


def _get_device_count(engine, table_name: str, device_column: str = "DeviceId") -> int:
    """Get count of unique devices in a table."""
    table_name = _ensure_safe_table(engine, table_name)
    device_column = _ensure_safe_column(engine, table_name, device_column)
    query = text(f"SELECT COUNT(DISTINCT {device_column}) as cnt FROM {table_name}")
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            return result[0] if result else 0
    except Exception:
        return 0


def _compute_column_stats(
    engine,
    table_name: str,
    column_name: str,
    dtype: str,
    sample_limit: int = 100_000,
) -> ColumnStats:
    """Compute statistics for a single numeric column."""
    table_name = _ensure_safe_table(engine, table_name)
    column_name = _ensure_safe_column(engine, table_name, column_name)
    stats = ColumnStats(column_name=column_name, dtype=dtype)

    # Get null counts
    null_query = text(f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) as null_count
        FROM {table_name}
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(null_query).fetchone()
            if result:
                total = result[0] or 1
                stats.null_count = result[1] or 0
                stats.null_percent = round(stats.null_count / total * 100, 2)
    except Exception as e:
        logger.warning(f"Error getting null stats for {table_name}.{column_name}: {e}")

    # For numeric types, compute detailed statistics
    numeric_types = ["int", "bigint", "smallint", "tinyint", "decimal", "numeric", "float", "real", "money"]
    if any(t in dtype.lower() for t in numeric_types):
        # Sample data for percentiles (for large tables)
        sample_query = text(f"""
            SELECT TOP ({sample_limit}) {column_name}
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            ORDER BY NEWID()
        """)
        try:
            with engine.connect() as conn:
                df = pd.read_sql(sample_query, conn)
                if not df.empty:
                    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
                    if len(series) > 0:
                        stats.min_val = float(series.min())
                        stats.max_val = float(series.max())
                        stats.mean = float(series.mean())
                        stats.std = float(series.std())
                        stats.unique_count = int(series.nunique())
                        stats.percentiles = {
                            "p5": float(np.percentile(series, 5)),
                            "p25": float(np.percentile(series, 25)),
                            "p50": float(np.percentile(series, 50)),
                            "p75": float(np.percentile(series, 75)),
                            "p95": float(np.percentile(series, 95)),
                            "p99": float(np.percentile(series, 99)),
                        }
        except Exception as e:
            logger.warning(f"Error computing stats for {table_name}.{column_name}: {e}")

    return stats


def profile_table(
    engine,
    table_name: str,
    date_column: str = "CollectedDate",
    device_column: str = "DeviceId",
    sample_limit: int = 100_000,
) -> TableProfile:
    """
    Generate a complete statistical profile for a single table.

    Args:
        engine: SQLAlchemy engine
        table_name: Name of the table to profile
        date_column: Column containing dates (for date range)
        device_column: Column containing device IDs
        sample_limit: Max rows to sample for percentile calculations

    Returns:
        TableProfile with comprehensive statistics
    """
    logger.info(f"Profiling table: {table_name}")
    table_name = _ensure_safe_table(engine, table_name)
    date_column = _ensure_safe_column(engine, table_name, date_column)
    device_column = _ensure_safe_column(engine, table_name, device_column)

    # Get basic metadata
    row_count = _get_table_row_count(engine, table_name)
    date_range = _get_date_range(engine, table_name, date_column)
    device_count = _get_device_count(engine, table_name, device_column)

    # Get column metadata
    columns_df = _get_table_metadata(engine, table_name)

    # Compute stats for each column
    column_stats: dict[str, ColumnStats] = {}
    for _, row in columns_df.iterrows():
        col_name = row["COLUMN_NAME"]
        dtype = row["DATA_TYPE"]
        try:
            stats = _compute_column_stats(engine, table_name, col_name, dtype, sample_limit)
            column_stats[col_name] = stats
        except Exception as e:
            logger.warning(f"Skipping column {col_name}: {e}")
            column_stats[col_name] = ColumnStats(column_name=col_name, dtype=dtype)

    return TableProfile(
        table_name=table_name,
        row_count=row_count,
        date_range=date_range,
        device_count=device_count,
        column_stats=column_stats,
    )


def profile_dw_tables(
    start_date: str | None = None,
    end_date: str | None = None,
    tables: list[str] | None = None,
    sample_limit: int = 100_000,
    auto_discover: bool = True,
) -> dict[str, TableProfile]:
    """
    Profile all telemetry tables in the XSight DW database.

    Args:
        start_date: Optional filter for date range (currently unused - profiles all data)
        end_date: Optional filter for date range (currently unused - profiles all data)
        tables: Specific tables to profile (defaults to auto-discovered cs_* tables)
        sample_limit: Max rows to sample for percentile calculations
        auto_discover: If True, dynamically discover available tables instead of using hardcoded list

    Returns:
        Dictionary mapping table names to their profiles
    """
    engine = create_dw_engine()

    # Determine which tables to profile
    if tables:
        target_tables = tables
    elif auto_discover:
        target_tables = discover_dw_tables(engine)
    else:
        target_tables = DW_TELEMETRY_TABLES

    profiles: dict[str, TableProfile] = {}

    for table_name in target_tables:
        try:
            logger.info(f"Profiling table: {table_name}")
            profile = profile_table(engine, table_name, sample_limit=sample_limit)
            profiles[table_name] = profile
            logger.info(f"Profiled {table_name}: {profile.row_count:,} rows, {profile.device_count:,} devices")
        except Exception as e:
            logger.error(f"Failed to profile {table_name}: {e}")

    return profiles


def profile_mc_tables(
    tables: list[str] | None = None,
    sample_limit: int = 100_000,
) -> dict[str, TableProfile]:
    """
    Profile tables in the MobiControl database.

    Args:
        tables: Specific tables to profile
        sample_limit: Max rows to sample for percentile calculations

    Returns:
        Dictionary mapping table names to their profiles
    """
    engine = create_mc_engine()
    target_tables = tables or ["Device"]  # Default to main Device table
    profiles: dict[str, TableProfile] = {}

    for table_name in target_tables:
        try:
            profile = profile_table(
                engine,
                table_name,
                date_column="LastCheckInTime",
                sample_limit=sample_limit,
            )
            profiles[table_name] = profile
            logger.info(f"Profiled MC {table_name}: {profile.row_count:,} rows")
        except Exception as e:
            logger.error(f"Failed to profile MC {table_name}: {e}")

    return profiles


def analyze_temporal_patterns(
    df: pd.DataFrame,
    metric_cols: list[str],
    timestamp_col: str = "Timestamp",
) -> dict[str, TemporalPattern]:
    """
    Analyze time-of-day and day-of-week patterns for metrics.

    Args:
        df: DataFrame with timestamp and metric columns
        metric_cols: List of metric column names to analyze
        timestamp_col: Name of the timestamp column

    Returns:
        Dictionary mapping metric names to their temporal patterns
    """
    if df.empty or timestamp_col not in df.columns:
        return {}

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["_hour"] = df[timestamp_col].dt.hour
    df["_dayofweek"] = df[timestamp_col].dt.dayofweek

    patterns: dict[str, TemporalPattern] = {}

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        pattern = TemporalPattern(metric_name=metric)

        # Hourly patterns (0-23)
        for hour in range(24):
            hour_data = df[df["_hour"] == hour][metric].dropna()
            if len(hour_data) > 0:
                pattern.hourly_stats[hour] = {
                    "mean": float(hour_data.mean()),
                    "std": float(hour_data.std()),
                    "count": int(len(hour_data)),
                }

        # Daily patterns (0=Monday, 6=Sunday)
        for day in range(7):
            day_data = df[df["_dayofweek"] == day][metric].dropna()
            if len(day_data) > 0:
                pattern.daily_stats[day] = {
                    "mean": float(day_data.mean()),
                    "std": float(day_data.std()),
                    "count": int(len(day_data)),
                }

        patterns[metric] = pattern

    return patterns


def compute_metric_distribution(
    engine,
    table_name: str,
    column_name: str,
    bins: int = 50,
    sample_limit: int = 500_000,
) -> dict[str, Any]:
    """
    Compute histogram distribution for a specific metric column.

    Args:
        engine: SQLAlchemy engine
        table_name: Table containing the metric
        column_name: Column name of the metric
        bins: Number of histogram bins
        sample_limit: Max rows to sample

    Returns:
        Dictionary with histogram data (bin_edges, counts, stats)
    """
    table_name = _ensure_safe_table(engine, table_name)
    column_name = _ensure_safe_column(engine, table_name, column_name)
    query = text(f"""
        SELECT TOP ({sample_limit}) {column_name}
        FROM {table_name}
        WHERE {column_name} IS NOT NULL
        ORDER BY NEWID()
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return {"bins": [], "counts": [], "stats": {}}

    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if len(series) == 0:
        return {"bins": [], "counts": [], "stats": {}}

    # Compute histogram
    counts, bin_edges = np.histogram(series, bins=bins)

    return {
        "bins": [float(b) for b in bin_edges],
        "counts": [int(c) for c in counts],
        "stats": {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "total_samples": len(series),
        },
    }


def generate_profile_report(
    profiles: dict[str, TableProfile],
    output_path: Path,
    format: str = "json",
) -> None:
    """
    Generate a comprehensive profile report and save to file.

    Args:
        profiles: Dictionary of table profiles
        output_path: Path to save the report
        format: Output format ('json' or 'parquet')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "tables": {name: profile.to_dict() for name, profile in profiles.items()},
        "summary": {
            "total_tables": len(profiles),
            "total_rows": sum(p.row_count for p in profiles.values()),
            "total_devices": max((p.device_count for p in profiles.values()), default=0),
        },
    }

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    elif format == "parquet":
        # Flatten to DataFrame for parquet
        rows = []
        for table_name, profile in profiles.items():
            for col_name, col_stats in profile.column_stats.items():
                rows.append({
                    "table_name": table_name,
                    "column_name": col_name,
                    "dtype": col_stats.dtype,
                    "null_percent": col_stats.null_percent,
                    "min_val": col_stats.min_val,
                    "max_val": col_stats.max_val,
                    "mean": col_stats.mean,
                    "std": col_stats.std,
                    **{f"percentile_{k}": v for k, v in col_stats.percentiles.items()},
                })
        df = pd.DataFrame(rows)
        df.to_parquet(output_path)

    logger.info(f"Profile report saved to {output_path}")


def get_available_metrics(profiles: dict[str, TableProfile]) -> list[dict[str, Any]]:
    """
    Extract list of available numeric metrics from profiles.

    Args:
        profiles: Dictionary of table profiles

    Returns:
        List of metric info dicts with table, column, and basic stats
    """
    metrics = []
    numeric_types = ["int", "bigint", "smallint", "tinyint", "decimal", "numeric", "float", "real", "money"]

    for table_name, profile in profiles.items():
        for col_name, col_stats in profile.column_stats.items():
            if any(t in col_stats.dtype.lower() for t in numeric_types):
                if col_stats.mean is not None:  # Has computed stats
                    metrics.append({
                        "table": table_name,
                        "column": col_name,
                        "dtype": col_stats.dtype,
                        "mean": col_stats.mean,
                        "std": col_stats.std,
                        "min": col_stats.min_val,
                        "max": col_stats.max_val,
                    })

    return metrics
