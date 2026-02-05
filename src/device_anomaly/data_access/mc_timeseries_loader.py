"""
MobiControl Time-Series Data Loader.

Loads time-series data from MobiControl's DeviceStat* tables with
incremental extraction using watermarks.

High-Value Tables:
- DeviceStatInt: 764K rows - Device integer metrics over time
- DeviceStatLocation: 619K rows - GPS location history
- DeviceStatString: 349K rows - Device string metrics
- DeviceStatNetTraffic: 244K rows - Network traffic per app
- MainLog: 1M rows - Activity log
- Alert: 1.3K rows - System alerts

Features:
- Incremental loading with watermarks
- Batch processing with configurable size
- Graceful handling of missing tables/columns
- Canonical event normalization
"""
from __future__ import annotations

import logging
import os
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from device_anomaly.data_access.canonical_events import (
    CanonicalEvent,
    SourceDatabase,
    dataframe_to_canonical_events,
)
from device_anomaly.data_access.db_connection import create_mc_engine
from device_anomaly.data_access.db_utils import table_exists
from device_anomaly.data_access.stat_type_mapper import discover_stat_types
from device_anomaly.data_access.watermark_store import get_watermark_store

logger = logging.getLogger(__name__)

_STAT_TYPES_DISCOVERED = False


def _ensure_stat_types(engine: Engine) -> None:
    """Ensure StatType mappings are discovered once per process."""
    global _STAT_TYPES_DISCOVERED
    if _STAT_TYPES_DISCOVERED:
        return
    try:
        discover_stat_types(engine)
    except Exception as exc:
        logger.debug("StatType discovery failed: %s", exc)
    finally:
        _STAT_TYPES_DISCOVERED = True

# Default batch size for incremental loads
DEFAULT_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "50000"))

# Time-series table configurations
MC_TIMESERIES_TABLES = {
    "DeviceStatInt": {
        "timestamp_col": "ServerDateTime",
        "device_col": "DeviceId",
        "columns": ["DeviceId", "TimeStamp", "StatType", "IntValue", "ServerDateTime"],
        "order_by": "ServerDateTime",
    },
    "DeviceStatLocation": {
        "timestamp_col": "ServerDateTime",
        "device_col": "DeviceId",
        "columns": ["DeviceId", "TimeStamp", "Latitude", "Longitude", "Altitude",
                   "Heading", "Speed", "ServerDateTime"],
        "order_by": "ServerDateTime",
    },
    "DeviceStatString": {
        "timestamp_col": "ServerDateTime",
        "device_col": "DeviceId",
        "columns": ["DeviceId", "TimeStamp", "StatType", "StrValue", "ServerDateTime"],
        "order_by": "ServerDateTime",
    },
    "DeviceStatNetTraffic": {
        "timestamp_col": "ServerDateTime",
        "device_col": "DeviceId",
        "columns": ["DeviceId", "TimeStamp", "StatType", "Upload", "Download",
                   "InterfaceType", "InterfaceID", "Application", "ServerDateTime"],
        "order_by": "ServerDateTime",
    },
    "MainLog": {
        "timestamp_col": "DateTime",
        "device_col": "DeviceId",
        "columns": ["ILogId", "DateTime", "EventId", "Severity", "EventClass",
                   "ResTxt", "DeviceId", "LoginId"],
        "order_by": "DateTime",
    },
    "Alert": {
        "timestamp_col": "SetDateTime",
        "device_col": None,  # DevId is a string, not int
        "columns": ["AlertId", "AlertKey", "AlertName", "AlertSeverity",
                   "DevId", "Status", "SetDateTime", "AckDateTime"],
        "order_by": "SetDateTime",
    },
    "DeviceInstalledApp": {
        "timestamp_col": "LastChangedDate",
        "device_col": "DeviceId",
        "columns": ["DeviceId", "InstalledAppId", "StatusId", "Version",
                   "Size", "DataSize", "IsRunning", "LastChangedDate"],
        "order_by": "LastChangedDate",
    },
}


def _validate_columns(engine: Engine, table_name: str, columns: list[str]) -> list[str]:
    """Validate which columns exist in the table."""
    query = text("""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :table_name
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"table_name": table_name})
            existing = {row[0] for row in result.fetchall()}
            valid = [c for c in columns if c in existing]
            missing = set(columns) - existing
            if missing:
                logger.debug(f"Columns not found in {table_name}: {missing}")
            return valid
    except Exception as e:
        logger.warning(f"Failed to validate columns for {table_name}: {e}")
        return columns  # Return all, let query fail if invalid


def load_mc_timeseries_incremental(
    table_name: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    device_ids: list[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_watermark: bool = True,
    engine: Engine | None = None,
) -> tuple[pd.DataFrame, datetime | None]:
    """
    Load time-series data incrementally from a MobiControl table.

    Args:
        table_name: Name of the table (e.g., DeviceStatInt)
        start_time: Start time filter (uses watermark if None and use_watermark=True)
        end_time: End time filter (uses now if None)
        device_ids: Optional list of device IDs to filter
        batch_size: Maximum rows per batch
        use_watermark: Whether to use stored watermarks
        engine: SQLAlchemy engine (creates one if not provided)

    Returns:
        Tuple of (DataFrame, new_watermark)
    """
    if engine is None:
        engine = create_mc_engine()

    # Validate table exists
    if not table_exists(engine, table_name):
        logger.warning(f"Table {table_name} does not exist in MobiControlDB")
        return pd.DataFrame(), None

    # Get table config
    config = MC_TIMESERIES_TABLES.get(table_name)
    if not config:
        logger.warning(f"No configuration for table {table_name}")
        return pd.DataFrame(), None

    if table_name == "DeviceStatInt":
        _ensure_stat_types(engine)

    ts_col = config["timestamp_col"]
    dev_col = config["device_col"]
    columns = config["columns"]
    order_by = config["order_by"]

    # Validate columns exist
    valid_columns = _validate_columns(engine, table_name, columns)
    if not valid_columns:
        logger.error(f"No valid columns found for {table_name}")
        return pd.DataFrame(), None

    # Get watermark if not explicitly provided
    if start_time is None and use_watermark:
        store = get_watermark_store()
        start_time = store.get_watermark("mobicontrol", table_name)

    if end_time is None:
        end_time = datetime.now(UTC)

    # Build query
    columns_str = ", ".join(valid_columns)

    # Build WHERE clause
    where_parts = [f"{ts_col} > :start_time", f"{ts_col} <= :end_time"]
    params: dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
    }

    if device_ids and dev_col:
        where_parts.append(f"{dev_col} IN :device_ids")
        params["device_ids"] = device_ids

    where_clause = " AND ".join(where_parts)

    sql = f"""
        SELECT TOP ({batch_size}) {columns_str}
        FROM dbo.{table_name}
        WHERE {where_clause}
        ORDER BY {order_by}
    """

    query = text(sql)
    if device_ids:
        query = query.bindparams(bindparam("device_ids", expanding=True))

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            logger.debug(f"No new data in {table_name} since {start_time}")
            return df, None

        # Determine new watermark from max timestamp
        if ts_col in df.columns:
            max_ts = df[ts_col].max()
            if pd.notna(max_ts):
                if isinstance(max_ts, str):
                    new_watermark = datetime.fromisoformat(max_ts.replace("Z", "+00:00"))
                elif hasattr(max_ts, 'to_pydatetime'):
                    new_watermark = max_ts.to_pydatetime()
                else:
                    new_watermark = max_ts

                if new_watermark.tzinfo is None:
                    new_watermark = new_watermark.replace(tzinfo=UTC)
            else:
                new_watermark = None
        else:
            new_watermark = None

        logger.info(
            f"Loaded {len(df)} rows from {table_name} "
            f"({start_time} to {end_time}), new watermark: {new_watermark}"
        )

        return df, new_watermark

    except Exception as e:
        logger.error(f"Failed to load data from {table_name}: {e}")
        return pd.DataFrame(), None


def load_and_update_watermark(
    table_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """
    Load incremental data and update watermark atomically.

    This is the primary method for scheduled ingestion.
    """
    df, new_watermark = load_mc_timeseries_incremental(
        table_name=table_name,
        batch_size=batch_size,
        use_watermark=True,
        engine=engine,
    )

    if not df.empty and new_watermark:
        config = MC_TIMESERIES_TABLES.get(table_name, {})
        store = get_watermark_store()
        success, error = store.set_watermark(
            source_db="mobicontrol",
            table_name=table_name,
            watermark_value=new_watermark,
            watermark_column=config.get("timestamp_col", "ServerDateTime"),
            rows_extracted=len(df),
        )
        if not success:
            logger.warning(f"Failed to update watermark for {table_name}: {error}")

    return df


def load_all_mc_timeseries(
    batch_size: int = DEFAULT_BATCH_SIZE,
    tables: list[str] | None = None,
    engine: Engine | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load incremental data from all configured time-series tables.

    Args:
        batch_size: Maximum rows per table per batch
        tables: Specific tables to load (None = all configured)
        engine: SQLAlchemy engine

    Returns:
        Dictionary mapping table names to DataFrames
    """
    if engine is None:
        engine = create_mc_engine()

    tables_to_load = tables or list(MC_TIMESERIES_TABLES.keys())
    results = {}

    for table_name in tables_to_load:
        try:
            df = load_and_update_watermark(
                table_name=table_name,
                batch_size=batch_size,
                engine=engine,
            )
            results[table_name] = df
            logger.info(f"Loaded {len(df)} rows from {table_name}")
        except Exception as e:
            logger.error(f"Failed to load {table_name}: {e}")
            results[table_name] = pd.DataFrame()

    return results


def load_mc_timeseries_as_events(
    table_name: str,
    tenant_id: str = "default",
    batch_size: int = DEFAULT_BATCH_SIZE,
    engine: Engine | None = None,
) -> list[CanonicalEvent]:
    """
    Load time-series data and normalize to canonical events.

    This method combines loading and normalization for direct pipeline use.
    """
    df = load_and_update_watermark(
        table_name=table_name,
        batch_size=batch_size,
        engine=engine,
    )

    if df.empty:
        return []

    return dataframe_to_canonical_events(
        df=df,
        source_db=SourceDatabase.MOBICONTROL,
        source_table=table_name,
        tenant_id=tenant_id,
    )


def stream_mc_timeseries(
    table_name: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    engine: Engine | None = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream time-series data in batches (for large backfills).

    Yields DataFrames until no more data is available.
    """
    if engine is None:
        engine = create_mc_engine()

    current_watermark = start_time
    if current_watermark is None:
        current_watermark = datetime.now(UTC) - timedelta(days=30)

    while True:
        df, new_watermark = load_mc_timeseries_incremental(
            table_name=table_name,
            start_time=current_watermark,
            end_time=end_time,
            batch_size=batch_size,
            use_watermark=False,
            engine=engine,
        )

        if df.empty or new_watermark is None:
            break

        yield df

        # Safety check - if watermark didn't advance, we're stuck
        if new_watermark <= current_watermark:
            logger.warning(f"Watermark not advancing for {table_name}, stopping")
            break

        # Move watermark forward
        current_watermark = new_watermark


def get_mc_timeseries_stats(
    engine: Engine | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Get statistics for all time-series tables.

    Returns row counts, date ranges, and watermark status.
    """
    if engine is None:
        engine = create_mc_engine()

    store = get_watermark_store()
    stats = {}

    for table_name, config in MC_TIMESERIES_TABLES.items():
        table_stats = {
            "configured": True,
            "exists": False,
            "row_count": 0,
            "min_date": None,
            "max_date": None,
            "watermark": None,
        }

        if not table_exists(engine, table_name):
            stats[table_name] = table_stats
            continue

        table_stats["exists"] = True
        ts_col = config["timestamp_col"]

        # Get row count and date range
        try:
            with engine.connect() as conn:
                # Approximate row count using sys.partitions (avoid COUNT(*) scans)
                result = conn.execute(
                    text("""
                        SELECT SUM(p.rows)
                        FROM sys.tables t
                        INNER JOIN sys.partitions p ON t.object_id = p.object_id
                        WHERE t.name = :table_name AND p.index_id IN (0, 1)
                    """),
                    {"table_name": table_name},
                ).fetchone()
                table_stats["row_count"] = int(result[0]) if result and result[0] else 0

                # Date range
                result = conn.execute(text(f"""
                    SELECT MIN({ts_col}), MAX({ts_col}) FROM dbo.{table_name}
                """)).fetchone()
                if result:
                    table_stats["min_date"] = str(result[0]) if result[0] else None
                    table_stats["max_date"] = str(result[1]) if result[1] else None
        except Exception as e:
            logger.warning(f"Failed to get stats for {table_name}: {e}")

        # Get watermark
        try:
            wm = store.get_watermark("mobicontrol", table_name)
            table_stats["watermark"] = wm.isoformat() if wm else None
        except Exception:
            pass

        stats[table_name] = table_stats

    return stats


def backfill_mc_timeseries(
    table_name: str,
    start_time: datetime,
    end_time: datetime | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    engine: Engine | None = None,
) -> int:
    """
    Backfill historical data for a table.

    Returns total rows loaded.
    """
    total_rows = 0

    for batch_df in stream_mc_timeseries(
        table_name=table_name,
        start_time=start_time,
        end_time=end_time,
        batch_size=batch_size,
        engine=engine,
    ):
        total_rows += len(batch_df)
        logger.info(f"Backfill progress for {table_name}: {total_rows} rows")

    # Update watermark after backfill
    if end_time:
        store = get_watermark_store()
        config = MC_TIMESERIES_TABLES.get(table_name, {})
        success, error = store.set_watermark(
            source_db="mobicontrol",
            table_name=table_name,
            watermark_value=end_time,
            watermark_column=config.get("timestamp_col", "ServerDateTime"),
            rows_extracted=total_rows,
        )
        if not success:
            logger.warning(f"Failed to update watermark after backfill for {table_name}: {error}")

    logger.info(f"Backfill complete for {table_name}: {total_rows} total rows")
    return total_rows
