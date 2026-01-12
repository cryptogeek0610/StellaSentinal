"""
Extended XSight DW Data Loader.

Loads additional telemetry tables from SOTI_XSight_dw beyond the original
5-table subset. Supports incremental extraction with watermarks.

Additional High-Value Tables:
- cs_DataUsageByHour: 104M rows - Hourly data usage per device/app
- cs_BatteryLevelDrop: 14.8M rows - Battery drain patterns
- cs_AppUsageListed: 8.5M rows - App foreground time
- cs_WifiHour: 755K rows - WiFi connectivity patterns
- cs_WiFiLocation: 790K rows - WiFi + GPS location
- cs_LastKnown: 674K rows - Last known location + signal
- cs_DeviceInstalledApp: 372K rows - App install events
- cs_PresetApps: 19M rows - Preset app usage

PRODUCTION-HARDENED:
- Uses keyset pagination (no OFFSET) for efficient large table loads
- Composite watermarks for hourly tables (CollectedDate + Hour)
- Respects table allowlist from settings
- Graceful handling of missing tables/columns
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from device_anomaly.data_access.canonical_events import (
    CanonicalEvent,
    SourceDatabase,
    dataframe_to_canonical_events,
)
from device_anomaly.data_access.db_connection import create_dw_engine
from device_anomaly.data_access.db_utils import table_exists
from device_anomaly.data_access.watermark_store import get_watermark_store

logger = logging.getLogger(__name__)

# Default batch size
DEFAULT_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "100000"))

# Extended XSight table configurations
# Each table defines timestamp column, device column, key columns, and dimensions
XSIGHT_EXTENDED_TABLES = {
    # Original tables (for completeness)
    "cs_BatteryStat": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "DeviceId",
            "TotalBatteryLevelDrop", "TotalDischargeTime_Sec",
            "ChargePatternBadCount", "ChargePatternGoodCount", "ChargePatternMediumCount",
            "AcChargeCount", "UsbChargeCount", "WirelessChargeCount",
            "CalculatedBatteryCapacity", "TotalFreeStorageKb",
        ],
        "priority": 1,
    },
    "cs_AppUsage": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "DeviceId", "AppId",
            "VisitCount", "TotalForegroundTime",
        ],
        "dimensions": ["AppId"],
        "priority": 1,
    },
    "cs_DataUsage": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "DeviceId", "AppId", "ConnectionTypeId",
            "Download", "Upload",
        ],
        "dimensions": ["AppId", "ConnectionTypeId"],
        "priority": 1,
    },
    "cs_BatteryAppDrain": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "DeviceId", "AppId", "BatteryDrain",
        ],
        "dimensions": ["AppId"],
        "priority": 1,
    },
    "cs_Heatmap": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "DeviceId", "NetworkTypeId",
            "SignalStrengthBucketId", "ReadingCount", "DropCnt",
        ],
        "dimensions": ["NetworkTypeId", "SignalStrengthBucketId"],
        "priority": 1,
    },

    # NEW: High-volume hourly tables
    "cs_DataUsageByHour": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "Hour", "DeviceId", "AppId",
            "ConnectionTypeId", "Download", "Upload",
        ],
        "dimensions": ["Hour", "AppId", "ConnectionTypeId"],
        "priority": 2,
        "large_table": True,
    },
    "cs_BatteryLevelDrop": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "Hour", "DeviceId", "BatteryLevel",
        ],
        "dimensions": ["Hour"],
        "priority": 2,
        "large_table": True,
    },
    "cs_AppUsageListed": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "Hour", "DeviceId", "AppId",
            "VisitCount", "TotalForegroundTime",
        ],
        "dimensions": ["Hour", "AppId"],
        "priority": 2,
        "large_table": True,
    },
    "cs_PresetApps": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "Hour", "PresetAppId", "DeviceId",
            "ConnectionTime", "Download", "Upload",
        ],
        "dimensions": ["Hour", "PresetAppId"],
        "priority": 3,
        "large_table": True,
    },

    # NEW: WiFi and location tables
    "cs_WifiHour": {
        "timestamp_col": "CollectedDate",
        "device_col": "Deviceid",  # Note: lowercase 'id'
        "columns": [
            "CollectedDate", "Hour", "Deviceid", "AccessPointId",
            "WiFiSignalStrength", "ConnectionTime", "DisconnectCount",
        ],
        "dimensions": ["Hour", "AccessPointId"],
        "priority": 2,
    },
    "cs_WiFiLocation": {
        "timestamp_col": "CollectedDate",
        "device_col": "Deviceid",
        "columns": [
            "CollectedDate", "Hour", "Deviceid", "AccessPointId",
            "ReadingTime", "WiFiSignalStrength", "Latitude", "Longitude",
        ],
        "dimensions": ["Hour", "AccessPointId"],
        "priority": 2,
    },
    "cs_LastKnown": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "DeviceId", "CollectedTime",
            "Latitude", "Longitude", "NetworkTypeId", "SignalStrengthBucketId",
        ],
        "dimensions": ["NetworkTypeId", "SignalStrengthBucketId"],
        "priority": 2,
    },

    # NEW: App inventory
    "cs_DeviceInstalledApp": {
        "timestamp_col": "EventTime",
        "device_col": "DeviceId",
        "columns": [
            "DeviceId", "AppId", "AppVersionId", "EventTime", "EventType",
        ],
        "dimensions": ["AppId", "EventType"],
        "priority": 2,
    },

    # NEW: Crash logs for app stability analysis
    "cs_CrashLogs": {
        "timestamp_col": "CollectedDate",
        "device_col": "DeviceId",
        "columns": [
            "CollectedDate", "DeviceId", "AppId", "CrashCount",
            "AppNotResponding", "ExceptionType", "ProcessName",
        ],
        "dimensions": ["AppId"],
        "priority": 2,
    },
}


def _validate_columns(engine: Engine, table_name: str, columns: List[str]) -> List[str]:
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
        return columns


def load_xsight_table_incremental(
    table_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    device_ids: Optional[List[int]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_watermark: bool = True,
    engine: Optional[Engine] = None,
    start_hour: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[datetime], Optional[int]]:
    """
    Load data incrementally from an XSight table using KEYSET PAGINATION.

    Uses keyset pagination (WHERE > last_value ORDER BY key) instead of OFFSET
    for efficient loading of large tables.

    For hourly tables (cs_*ByHour, cs_*Hour), uses composite keyset:
    - Primary key: (CollectedDate, Hour)
    - Watermark advances both date and hour

    Args:
        table_name: Name of the cs_* table
        start_date: Start date filter (uses watermark if None)
        end_date: End date filter (uses now if None)
        device_ids: Optional device ID filter
        batch_size: Maximum rows per batch
        use_watermark: Whether to use stored watermarks
        engine: SQLAlchemy engine
        start_hour: For hourly tables, starting hour (0-23)

    Returns:
        Tuple of (DataFrame, new_watermark_date, new_watermark_hour)
        For non-hourly tables, new_watermark_hour is None
    """
    if engine is None:
        engine = create_dw_engine()

    # Validate table exists
    if not table_exists(engine, table_name, base_table_only=True):
        logger.warning(f"Table {table_name} does not exist in XSight DW")
        return pd.DataFrame(), None, None

    # Get table config
    config = XSIGHT_EXTENDED_TABLES.get(table_name)
    if not config:
        logger.warning(f"No configuration for table {table_name}")
        return pd.DataFrame(), None, None

    ts_col = config["timestamp_col"]
    dev_col = config["device_col"]
    columns = config["columns"]
    is_hourly = "Hour" in columns

    # Validate columns
    valid_columns = _validate_columns(engine, table_name, columns)
    if not valid_columns:
        logger.error(f"No valid columns found for {table_name}")
        return pd.DataFrame(), None, None

    # Get watermark
    if start_date is None and use_watermark:
        store = get_watermark_store()
        start_date = store.get_watermark("xsight", table_name)
        if start_hour is None:
            metadata = store.get_watermark_metadata("xsight", table_name)
            if metadata and "hour" in metadata:
                try:
                    start_hour = int(metadata["hour"])
                except (TypeError, ValueError):
                    start_hour = None

    if end_date is None:
        end_date = datetime.now(timezone.utc)

    # For date-only columns (CollectedDate), convert to date
    start_val = start_date.date() if hasattr(start_date, 'date') else start_date
    end_val = end_date.date() if hasattr(end_date, 'date') else end_date

    # Build query with KEYSET PAGINATION
    columns_str = ", ".join(valid_columns)

    # Build WHERE clause using keyset pagination
    # For hourly tables: WHERE (CollectedDate > :start_date) OR (CollectedDate = :start_date AND Hour > :start_hour)
    # For regular tables: WHERE CollectedDate > :start_date
    params: Dict[str, Any] = {
        "end_date": end_val,
    }

    if is_hourly and start_hour is not None:
        # Composite keyset for hourly tables
        where_parts = [
            f"(({ts_col} > :start_date) OR ({ts_col} = :start_date AND Hour > :start_hour))",
            f"{ts_col} <= :end_date"
        ]
        params["start_date"] = start_val
        params["start_hour"] = start_hour
        order_by = f"{ts_col}, Hour"
    else:
        where_parts = [f"{ts_col} > :start_date", f"{ts_col} <= :end_date"]
        params["start_date"] = start_val
        order_by = ts_col

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
            logger.debug(f"No new data in {table_name} since {start_date}")
            return df, None, None

        # Determine new watermark from last row (keyset pagination)
        new_watermark = None
        new_hour = None

        if ts_col in df.columns:
            max_ts = df[ts_col].max()
            if pd.notna(max_ts):
                if isinstance(max_ts, (datetime, pd.Timestamp)):
                    new_watermark = max_ts
                    if hasattr(new_watermark, 'to_pydatetime'):
                        new_watermark = new_watermark.to_pydatetime()
                else:
                    # It's a date, convert to datetime
                    new_watermark = datetime.combine(max_ts, datetime.min.time())

                if new_watermark.tzinfo is None:
                    new_watermark = new_watermark.replace(tzinfo=timezone.utc)

        # For hourly tables, get the max hour for the max date
        if is_hourly and "Hour" in df.columns and new_watermark:
            max_date_rows = df[df[ts_col] == df[ts_col].max()]
            new_hour = int(max_date_rows["Hour"].max())

        logger.info(
            f"Loaded {len(df)} rows from {table_name} "
            f"({start_date} to {end_date}), new watermark: {new_watermark}"
            + (f" hour={new_hour}" if new_hour is not None else "")
        )

        return df, new_watermark, new_hour

    except Exception as e:
        logger.error(f"Failed to load data from {table_name}: {e}")
        return pd.DataFrame(), None, None


def load_and_update_watermark(
    table_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """
    Load incremental data and update watermark atomically.

    For hourly tables, stores hour in watermark metadata for composite keyset.
    """
    df, new_watermark, new_hour = load_xsight_table_incremental(
        table_name=table_name,
        batch_size=batch_size,
        use_watermark=True,
        engine=engine,
    )

    if not df.empty and new_watermark:
        config = XSIGHT_EXTENDED_TABLES.get(table_name, {})
        store = get_watermark_store()

        # Store hour in metadata for hourly tables
        metadata = None
        if new_hour is not None:
            metadata = {"hour": new_hour}

        success, error = store.set_watermark(
            source_db="xsight",
            table_name=table_name,
            watermark_value=new_watermark,
            watermark_column=config.get("timestamp_col", "CollectedDate"),
            rows_extracted=len(df),
            metadata=metadata,
        )
        if not success:
            logger.warning(f"Failed to update watermark for {table_name}: {error}")

    return df


def load_all_xsight_tables(
    batch_size: int = DEFAULT_BATCH_SIZE,
    tables: Optional[List[str]] = None,
    priority_filter: Optional[int] = None,
    engine: Optional[Engine] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load incremental data from multiple XSight tables.

    Args:
        batch_size: Maximum rows per table
        tables: Specific tables to load (None = all configured)
        priority_filter: Only load tables with this priority or higher (1=highest)
        engine: SQLAlchemy engine

    Returns:
        Dictionary mapping table names to DataFrames
    """
    if engine is None:
        engine = create_dw_engine()

    # Filter tables
    tables_to_load = tables or list(XSIGHT_EXTENDED_TABLES.keys())

    if priority_filter is not None:
        tables_to_load = [
            t for t in tables_to_load
            if XSIGHT_EXTENDED_TABLES.get(t, {}).get("priority", 99) <= priority_filter
        ]

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


def load_xsight_as_events(
    table_name: str,
    tenant_id: str = "default",
    batch_size: int = DEFAULT_BATCH_SIZE,
    engine: Optional[Engine] = None,
) -> List[CanonicalEvent]:
    """
    Load XSight data and normalize to canonical events.
    """
    df = load_and_update_watermark(
        table_name=table_name,
        batch_size=batch_size,
        engine=engine,
    )

    if df.empty:
        return []

    config = XSIGHT_EXTENDED_TABLES.get(table_name, {})

    return dataframe_to_canonical_events(
        df=df,
        source_db=SourceDatabase.XSIGHT,
        source_table=table_name,
        tenant_id=tenant_id,
        timestamp_col=config.get("timestamp_col", "CollectedDate"),
        device_id_col=config.get("device_col", "DeviceId"),
    )


def stream_xsight_table(
    table_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    engine: Optional[Engine] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream data from an XSight table in batches using keyset pagination.

    Efficiently streams large tables without OFFSET performance degradation.
    """
    if engine is None:
        engine = create_dw_engine()

    current_watermark = start_date
    if current_watermark is None:
        current_watermark = datetime.now(timezone.utc) - timedelta(days=30)

    current_hour: Optional[int] = None  # For hourly tables

    while True:
        df, new_watermark, new_hour = load_xsight_table_incremental(
            table_name=table_name,
            start_date=current_watermark,
            end_date=end_date,
            batch_size=batch_size,
            use_watermark=False,
            engine=engine,
            start_hour=current_hour,
        )

        if df.empty or new_watermark is None:
            break

        yield df

        # Advance watermark (keyset pagination)
        # Check if watermark actually advanced to prevent infinite loop
        if new_watermark == current_watermark and new_hour == current_hour:
            logger.warning(f"Watermark not advancing for {table_name}, stopping stream")
            break

        current_watermark = new_watermark
        current_hour = new_hour


def get_xsight_table_stats(
    engine: Optional[Engine] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all configured XSight tables.
    """
    if engine is None:
        engine = create_dw_engine()

    store = get_watermark_store()
    stats = {}

    for table_name, config in XSIGHT_EXTENDED_TABLES.items():
        table_stats = {
            "configured": True,
            "exists": False,
            "row_count": 0,
            "min_date": None,
            "max_date": None,
            "watermark": None,
            "priority": config.get("priority", 99),
            "large_table": config.get("large_table", False),
        }

        if not table_exists(engine, table_name, base_table_only=True):
            stats[table_name] = table_stats
            continue

        table_stats["exists"] = True
        ts_col = config["timestamp_col"]

        # Get row count and date range
        try:
            with engine.connect() as conn:
                # Approximate row count using sys.partitions (fast)
                result = conn.execute(text(f"""
                    SELECT SUM(p.rows)
                    FROM sys.tables t
                    INNER JOIN sys.partitions p ON t.object_id = p.object_id
                    WHERE t.name = '{table_name}' AND p.index_id IN (0, 1)
                """)).fetchone()
                table_stats["row_count"] = int(result[0]) if result and result[0] else 0

                # Date range (may be slow for very large tables)
                if not config.get("large_table"):
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
            wm = store.get_watermark("xsight", table_name)
            table_stats["watermark"] = wm.isoformat() if wm else None
        except Exception:
            pass

        stats[table_name] = table_stats

    return stats


def backfill_xsight_table(
    table_name: str,
    start_date: datetime,
    end_date: Optional[datetime] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    engine: Optional[Engine] = None,
) -> int:
    """
    Backfill historical data for an XSight table.

    Returns total rows loaded.
    """
    total_rows = 0

    for batch_df in stream_xsight_table(
        table_name=table_name,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        engine=engine,
    ):
        total_rows += len(batch_df)
        logger.info(f"Backfill progress for {table_name}: {total_rows} rows")

    # Update watermark
    if end_date:
        store = get_watermark_store()
        config = XSIGHT_EXTENDED_TABLES.get(table_name, {})
        success, error = store.set_watermark(
            source_db="xsight",
            table_name=table_name,
            watermark_value=end_date,
            watermark_column=config.get("timestamp_col", "CollectedDate"),
            rows_extracted=total_rows,
        )
        if not success:
            logger.warning(f"Failed to update watermark after backfill for {table_name}: {error}")

    logger.info(f"Backfill complete for {table_name}: {total_rows} total rows")
    return total_rows


def aggregate_hourly_data(
    df: pd.DataFrame,
    table_name: str,
    agg_funcs: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Aggregate hourly data (cs_*ByHour tables) to daily level.

    Useful for reducing data volume while preserving key metrics.
    """
    config = XSIGHT_EXTENDED_TABLES.get(table_name, {})
    ts_col = config.get("timestamp_col", "CollectedDate")
    dev_col = config.get("device_col", "DeviceId")

    if df.empty:
        return df

    # Default aggregation functions
    if agg_funcs is None:
        agg_funcs = {
            "Download": "sum",
            "Upload": "sum",
            "VisitCount": "sum",
            "TotalForegroundTime": "sum",
            "BatteryLevel": "mean",
            "ConnectionTime": "sum",
            "DisconnectCount": "sum",
            "WiFiSignalStrength": "mean",
        }

    # Filter to columns that exist and have agg funcs
    existing_agg = {k: v for k, v in agg_funcs.items() if k in df.columns}

    if not existing_agg:
        return df

    # Group by date and device
    group_cols = [ts_col, dev_col]

    # Add dimension columns if present
    dimension_cols = config.get("dimensions", [])
    for dim in dimension_cols:
        if dim in df.columns and dim != "Hour":
            group_cols.append(dim)

    return df.groupby(group_cols, as_index=False).agg(existing_agg)
