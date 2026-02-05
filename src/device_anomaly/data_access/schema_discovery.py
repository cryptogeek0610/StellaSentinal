"""
Runtime schema discovery for SQL Server databases.

Discovers all tables and views from SOTI_XSight_dw and MobiControlDB at runtime
using INFORMATION_SCHEMA and sys.* metadata. NO table scans, NO COUNT(*).

Features:
- Dynamic table/view discovery with approximate row counts from sys.partitions
- Automatic high-value table identification using patterns
- Column schema inspection for time-series detection
- Cache with TTL keyed by (host, db_name, schema_version_hash)
- Graceful degradation on errors
- Startup logging of discovered schema summary

Production-Safe Design:
- All metadata queries use INFORMATION_SCHEMA and sys.* views only
- Row counts are approximations from sys.partitions (no table scans)
- Allowlist filtering respects settings.xsight_table_allowlist / mc_table_allowlist
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Regex for validating SQL identifiers
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SourceDatabase(StrEnum):
    """Source database identifiers."""

    XSIGHT = "xsight"
    MOBICONTROL = "mobicontrol"


@dataclass
class ColumnInfo:
    """Column metadata from INFORMATION_SCHEMA."""

    name: str
    data_type: str
    max_length: int | None = None
    is_nullable: bool = True
    ordinal_position: int = 0

    @property
    def is_numeric(self) -> bool:
        """Check if column is numeric type."""
        numeric_types = {
            "int",
            "bigint",
            "smallint",
            "tinyint",
            "decimal",
            "numeric",
            "float",
            "real",
            "money",
            "smallmoney",
        }
        return self.data_type.lower() in numeric_types

    @property
    def is_datetime(self) -> bool:
        """Check if column is datetime type."""
        datetime_types = {
            "datetime",
            "datetime2",
            "date",
            "time",
            "datetimeoffset",
            "smalldatetime",
        }
        return self.data_type.lower() in datetime_types

    @property
    def is_string(self) -> bool:
        """Check if column is string type."""
        string_types = {"varchar", "nvarchar", "char", "nchar", "text", "ntext"}
        return self.data_type.lower() in string_types


@dataclass
class TableInfo:
    """Table/View metadata with schema information."""

    name: str
    schema_name: str = "dbo"
    table_type: str = "BASE TABLE"  # or "VIEW"
    row_count: int = 0
    columns: list[ColumnInfo] = field(default_factory=list)
    has_device_id: bool = False
    has_timestamp: bool = False
    timestamp_column: str | None = None
    device_id_column: str | None = None
    priority_score: float = 0.0
    is_high_value: bool = False
    source_db: SourceDatabase = SourceDatabase.XSIGHT
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def full_name(self) -> str:
        """Get fully qualified table name."""
        return f"{self.schema_name}.{self.name}"

    @property
    def is_view(self) -> bool:
        """Check if this is a view."""
        return self.table_type == "VIEW"

    @property
    def is_time_series(self) -> bool:
        """Check if table appears to be time-series data."""
        return self.has_device_id and self.has_timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["source_db"] = self.source_db.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableInfo:
        """Create from dictionary."""
        if "source_db" in data and isinstance(data["source_db"], str):
            data["source_db"] = SourceDatabase(data["source_db"])
        if "columns" in data:
            data["columns"] = [
                ColumnInfo(**c) if isinstance(c, dict) else c for c in data["columns"]
            ]
        return cls(**data)


@dataclass
class DatabaseSchema:
    """Complete schema for a database."""

    database_name: str
    source_db: SourceDatabase
    tables: dict[str, TableInfo] = field(default_factory=dict)
    views: dict[str, TableInfo] = field(default_factory=dict)
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    discovery_duration_ms: int = 0

    @property
    def all_objects(self) -> dict[str, TableInfo]:
        """Get all tables and views."""
        return {**self.tables, **self.views}

    @property
    def high_value_tables(self) -> list[TableInfo]:
        """Get tables marked as high value."""
        return [t for t in self.all_objects.values() if t.is_high_value]

    @property
    def time_series_tables(self) -> list[TableInfo]:
        """Get tables that appear to be time-series."""
        return [t for t in self.all_objects.values() if t.is_time_series]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "database_name": self.database_name,
            "source_db": self.source_db.value,
            "tables": {k: v.to_dict() for k, v in self.tables.items()},
            "views": {k: v.to_dict() for k, v in self.views.items()},
            "discovered_at": self.discovered_at,
            "discovery_duration_ms": self.discovery_duration_ms,
        }


# High-value table patterns for automatic identification
HIGH_VALUE_PATTERNS = {
    SourceDatabase.XSIGHT: {
        # Primary telemetry tables (cs_* prefix)
        "patterns": [
            r"^cs_.*",  # All cs_* telemetry tables
            r"^sb_.*",  # Smart battery tables
            r"^vw_mc.*",  # MobiControl views
            r"^vw_Device.*",  # Device views
        ],
        # Explicit high-value tables
        "explicit": {
            "cs_BatteryStat",
            "cs_AppUsage",
            "cs_DataUsage",
            "cs_BatteryAppDrain",
            "cs_Heatmap",
            "cs_DataUsageByHour",
            "cs_BatteryLevelDrop",
            "cs_AppUsageListed",
            "cs_WifiHour",
            "cs_WiFiLocation",
            "cs_LastKnown",
            "cs_DeviceInstalledApp",
            "cs_PresetApps",
        },
        # Tables to exclude (time-sliced views, staging, etc.)
        "exclude_suffixes": [
            "_Last7",
            "_Last30",
            "_LastWeek",
            "_LastMonth",
            "_Today",
            "_Yesterday",
            "_staging",
            "_temp",
        ],
    },
    SourceDatabase.MOBICONTROL: {
        "patterns": [
            r"^DeviceStat.*",  # Time-series device stats
            r"^Device.*",  # Device tables
            r"^Alert.*",  # Alert tables
            r"^Event.*",  # Event tables
        ],
        "explicit": {
            "DeviceStatInt",
            "DeviceStatLocation",
            "DeviceStatString",
            "DeviceStatNetTraffic",
            "DeviceInstalledApp",
            "MainLog",
            "Alert",
            "Events",
            "DevInfo",
            "DeviceLastKnownLocation",
            "AndroidDevice",
            "iOSDevice",
            "WindowsDevice",
            "MacDevice",
            "LinuxDevice",
            "ZebraAndroidDevice",
        },
        "exclude_suffixes": ["_backup", "_old", "_staging", "_temp"],
    },
}

# Known timestamp columns for each database
TIMESTAMP_COLUMNS = {
    SourceDatabase.XSIGHT: [
        "CollectedDate",
        "CollectedTime",
        "EventTime",
        "Timestamp",
        "ReadingTime",
    ],
    SourceDatabase.MOBICONTROL: [
        "ServerDateTime",
        "TimeStamp",
        "DateTime",
        "CollectedDate",
        "LastCheckInTime",
        "SetDateTime",
        "ModifiedTime",
    ],
}

# Known device ID columns
DEVICE_ID_COLUMNS = ["DeviceId", "DevId", "Deviceid"]


class SchemaDiscoveryCache:
    """Thread-safe cache for discovered schemas.

    Cache keys include:
    - host: SQL Server hostname
    - database_name: Database name
    - schema_signature: Hash of table/view names and modify dates

    This ensures cache is invalidated when schema changes (DDL operations).
    """

    def __init__(self, ttl_seconds: int = 3600, cache_dir: Path | None = None):
        self._cache: dict[str, tuple[DatabaseSchema, float]] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds
        self._cache_dir = cache_dir or Path("data/schema_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_cache_key(host: str, database_name: str, schema_signature: str) -> str:
        """Build a stable cache key from host, db, and schema signature.

        The schema_signature should be a hash of table names + modify dates.
        """
        # Normalize host (remove port, lowercase)
        normalized_host = host.split(":")[0].split("\\")[0].lower()
        # Create short hash of signature
        sig_hash = hashlib.sha256(schema_signature.encode()).hexdigest()[:12]
        return f"{normalized_host}_{database_name}_{sig_hash}"

    def get(self, key: str) -> DatabaseSchema | None:
        """Get cached schema if not expired."""
        with self._lock:
            if key in self._cache:
                schema, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return schema
                else:
                    del self._cache[key]

            # Try file cache - use safe filename
            safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
            cache_file = self._cache_dir / f"{safe_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        data = json.load(f)
                    # Check file age
                    file_age = time.time() - cache_file.stat().st_mtime
                    if file_age < self._ttl:
                        schema = self._deserialize_schema(data)
                        self._cache[key] = (schema, time.time())
                        return schema
                except Exception as e:
                    logger.warning(f"Failed to load cached schema from {cache_file}: {e}")

        return None

    def set(self, key: str, schema: DatabaseSchema) -> None:
        """Cache schema in memory and file."""
        with self._lock:
            self._cache[key] = (schema, time.time())

            # Also persist to file - use safe filename
            safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
            cache_file = self._cache_dir / f"{safe_key}.json"
            try:
                with open(cache_file, "w") as f:
                    json.dump(schema.to_dict(), f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to persist schema cache to {cache_file}: {e}")

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cached schema(s)."""
        with self._lock:
            if key:
                self._cache.pop(key, None)
                safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
                cache_file = self._cache_dir / f"{safe_key}.json"
                if cache_file.exists():
                    cache_file.unlink()
            else:
                self._cache.clear()
                for f in self._cache_dir.glob("*.json"):
                    f.unlink()

    def _deserialize_schema(self, data: dict[str, Any]) -> DatabaseSchema:
        """Deserialize schema from dictionary."""
        tables = {k: TableInfo.from_dict(v) for k, v in data.get("tables", {}).items()}
        views = {k: TableInfo.from_dict(v) for k, v in data.get("views", {}).items()}
        return DatabaseSchema(
            database_name=data["database_name"],
            source_db=SourceDatabase(data["source_db"]),
            tables=tables,
            views=views,
            discovered_at=data.get("discovered_at", ""),
            discovery_duration_ms=data.get("discovery_duration_ms", 0),
        )


# Global cache instance
_schema_cache = SchemaDiscoveryCache()


def _validate_identifier(name: str) -> bool:
    """Validate SQL identifier is safe."""
    return bool(name and _IDENTIFIER_RE.match(name))


def _get_schema_signature(engine: Engine) -> str:
    """Get a schema signature based on table names and modify dates.

    This uses sys.tables.modify_date which changes when schema is altered.
    The signature is used to invalidate cache when DDL operations occur.
    """
    query = text("""
        SELECT
            t.name,
            CONVERT(VARCHAR, t.modify_date, 126) AS modify_date
        FROM sys.tables t
        ORDER BY t.name
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            # Create signature from sorted table names and dates
            signature_parts = [f"{row[0]}:{row[1]}" for row in rows]
            return "|".join(signature_parts)
    except Exception as e:
        logger.warning(f"Failed to get schema signature, using timestamp: {e}")
        # Fallback to current hour (cache invalidates hourly)
        return datetime.utcnow().strftime("%Y%m%d%H")


def _get_table_row_counts(engine: Engine, limit: int = 1000) -> dict[str, int]:
    """
    Get approximate row counts for all tables using sys.partitions.
    This is fast and doesn't require table scans.
    """
    query = text("""
        SELECT
            SCHEMA_NAME(t.schema_id) AS schema_name,
            t.name AS table_name,
            SUM(p.rows) AS row_count
        FROM sys.tables t
        INNER JOIN sys.partitions p
            ON t.object_id = p.object_id
        WHERE p.index_id IN (0, 1)  -- heap or clustered index
        GROUP BY t.schema_id, t.name
        ORDER BY SUM(p.rows) DESC
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            return {f"{row[0]}.{row[1]}": int(row[2]) for row in result.fetchall()}
    except Exception as e:
        logger.warning(f"Failed to get row counts: {e}")
        return {}


def _get_table_columns(
    engine: Engine, table_name: str, schema_name: str = "dbo"
) -> list[ColumnInfo]:
    """Get column metadata for a specific table."""
    if not _validate_identifier(table_name) or not _validate_identifier(schema_name):
        logger.warning(f"Invalid identifier: {schema_name}.{table_name}")
        return []

    query = text("""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :table_name
          AND TABLE_SCHEMA = :schema_name
        ORDER BY ORDINAL_POSITION
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"table_name": table_name, "schema_name": schema_name})
            return [
                ColumnInfo(
                    name=row[0],
                    data_type=row[1],
                    max_length=row[2],
                    is_nullable=row[3] == "YES",
                    ordinal_position=row[4],
                )
                for row in result.fetchall()
            ]
    except Exception as e:
        logger.warning(f"Failed to get columns for {schema_name}.{table_name}: {e}")
        return []


def _identify_time_columns(
    columns: list[ColumnInfo], source_db: SourceDatabase
) -> tuple[bool, str | None]:
    """Identify if table has timestamp column and which one."""
    known_ts_cols = TIMESTAMP_COLUMNS.get(source_db, [])

    # First check for known timestamp columns
    for col in columns:
        if col.name in known_ts_cols:
            return True, col.name

    # Then check for any datetime column
    for col in columns:
        if col.is_datetime:
            return True, col.name

    return False, None


def _identify_device_columns(columns: list[ColumnInfo]) -> tuple[bool, str | None]:
    """Identify if table has device ID column and which one."""
    for col in columns:
        if col.name in DEVICE_ID_COLUMNS:
            return True, col.name
    return False, None


def _is_high_value_table(table_name: str, source_db: SourceDatabase, row_count: int = 0) -> bool:
    """Determine if table is high-value based on patterns and heuristics."""
    config = HIGH_VALUE_PATTERNS.get(source_db, {})

    # Check exclude suffixes
    for suffix in config.get("exclude_suffixes", []):
        if table_name.endswith(suffix):
            return False

    # Check explicit list
    if table_name in config.get("explicit", set()):
        return True

    # Check patterns
    for pattern in config.get("patterns", []):
        if re.match(pattern, table_name, re.IGNORECASE):
            # Also require minimum row count for pattern matches
            if row_count >= 1000:
                return True

    return False


def _calculate_priority_score(table_info: TableInfo) -> float:
    """Calculate priority score for table (higher = more valuable for ML)."""
    score = 0.0

    # Base score from row count (log scale)
    if table_info.row_count > 0:
        import math

        score += math.log10(table_info.row_count + 1) * 10

    # Bonus for time-series capability
    if table_info.is_time_series:
        score += 50

    # Bonus for high-value flag
    if table_info.is_high_value:
        score += 30

    # Bonus for numeric columns (ML features)
    numeric_cols = sum(1 for c in table_info.columns if c.is_numeric)
    score += numeric_cols * 2

    # Small penalty for views (may be slower)
    if table_info.is_view:
        score -= 10

    return round(score, 2)


def discover_database_schema(
    engine: Engine,
    source_db: SourceDatabase,
    database_name: str,
    host: str | None = None,
    include_columns: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> DatabaseSchema:
    """
    Discover complete schema for a database.

    Args:
        engine: SQLAlchemy engine
        source_db: Source database identifier
        database_name: Name of the database
        host: SQL Server hostname (for cache key)
        include_columns: Whether to fetch column details (slower but more complete)
        use_cache: Whether to use cached results
        force_refresh: Force refresh even if cached

    Returns:
        DatabaseSchema with all discovered tables and views
    """
    # Get host from engine URL if not provided
    if host is None:
        try:
            host = str(engine.url.host or "localhost")
        except Exception:
            host = "localhost"

    # Build cache key with schema signature for proper invalidation
    schema_signature = _get_schema_signature(engine) if use_cache else ""
    cache_key = SchemaDiscoveryCache.build_cache_key(host, database_name, schema_signature)

    # Check cache
    if use_cache and not force_refresh:
        cached = _schema_cache.get(cache_key)
        if cached:
            logger.debug(f"Using cached schema for {database_name} (key={cache_key[:30]}...)")
            return cached

    start_time = time.time()
    logger.info(f"Discovering schema for {source_db.value}:{database_name} (host={host})...")

    # Get all tables and views
    tables_query = text("""
        SELECT
            TABLE_SCHEMA,
            TABLE_NAME,
            TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        ORDER BY TABLE_TYPE, TABLE_NAME
    """)

    tables: dict[str, TableInfo] = {}
    views: dict[str, TableInfo] = {}

    try:
        # Get row counts first (fast)
        row_counts = _get_table_row_counts(engine)

        with engine.connect() as conn:
            result = conn.execute(tables_query)
            all_objects = result.fetchall()

        logger.info(f"Found {len(all_objects)} objects in {database_name}")

        for row in all_objects:
            schema_name, table_name, table_type = row[0], row[1], row[2]
            full_name = f"{schema_name}.{table_name}"

            # Get columns if requested
            columns = []
            if include_columns:
                columns = _get_table_columns(engine, table_name, schema_name)

            # Identify time and device columns
            has_ts, ts_col = _identify_time_columns(columns, source_db)
            has_dev, dev_col = _identify_device_columns(columns)

            # Get row count
            row_count = row_counts.get(full_name, 0)

            # Check if high-value
            is_high_value = _is_high_value_table(table_name, source_db, row_count)

            table_info = TableInfo(
                name=table_name,
                schema_name=schema_name,
                table_type=table_type,
                row_count=row_count,
                columns=columns,
                has_device_id=has_dev,
                has_timestamp=has_ts,
                timestamp_column=ts_col,
                device_id_column=dev_col,
                is_high_value=is_high_value,
                source_db=source_db,
            )

            # Calculate priority score
            table_info.priority_score = _calculate_priority_score(table_info)

            if table_type == "VIEW":
                views[table_name] = table_info
            else:
                tables[table_name] = table_info

        duration_ms = int((time.time() - start_time) * 1000)

        schema = DatabaseSchema(
            database_name=database_name,
            source_db=source_db,
            tables=tables,
            views=views,
            discovery_duration_ms=duration_ms,
        )

        logger.info(
            f"Discovered {len(tables)} tables, {len(views)} views in {duration_ms}ms. "
            f"High-value: {len(schema.high_value_tables)}, "
            f"Time-series: {len(schema.time_series_tables)}"
        )

        # Cache result
        if use_cache:
            _schema_cache.set(cache_key, schema)

        return schema

    except Exception as e:
        logger.error(f"Failed to discover schema for {database_name}: {e}")
        # Return empty schema on error
        return DatabaseSchema(
            database_name=database_name,
            source_db=source_db,
        )


def discover_xsight_schema(
    engine: Engine | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> DatabaseSchema:
    """
    Discover XSight DW database schema.

    Args:
        engine: SQLAlchemy engine (creates one if not provided)
        use_cache: Whether to use cached results
        force_refresh: Force refresh even if cached

    Returns:
        DatabaseSchema for XSight
    """
    if engine is None:
        from device_anomaly.data_access.db_connection import create_dw_engine

        engine = create_dw_engine()

    from device_anomaly.config.settings import get_settings

    settings = get_settings()

    return discover_database_schema(
        engine=engine,
        source_db=SourceDatabase.XSIGHT,
        database_name=settings.dw.database,
        use_cache=use_cache,
        force_refresh=force_refresh,
    )


def discover_mobicontrol_schema(
    engine: Engine | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> DatabaseSchema:
    """
    Discover MobiControlDB database schema.

    Args:
        engine: SQLAlchemy engine (creates one if not provided)
        use_cache: Whether to use cached results
        force_refresh: Force refresh even if cached

    Returns:
        DatabaseSchema for MobiControl
    """
    if engine is None:
        from device_anomaly.data_access.db_connection import create_mc_engine

        engine = create_mc_engine()

    from device_anomaly.config.settings import get_settings

    settings = get_settings()

    return discover_database_schema(
        engine=engine,
        source_db=SourceDatabase.MOBICONTROL,
        database_name=settings.mc.database,
        use_cache=use_cache,
        force_refresh=force_refresh,
    )


def get_high_value_tables(
    source_db: SourceDatabase,
    min_rows: int = 1000,
    require_time_series: bool = False,
    engine: Engine | None = None,
) -> list[TableInfo]:
    """
    Get list of high-value tables for a database.

    Args:
        source_db: Which database to query
        min_rows: Minimum row count filter
        require_time_series: Only return time-series tables
        engine: Optional engine (creates one if not provided)

    Returns:
        List of high-value TableInfo objects sorted by priority
    """
    if source_db == SourceDatabase.XSIGHT:
        schema = discover_xsight_schema(engine=engine)
    else:
        schema = discover_mobicontrol_schema(engine=engine)

    tables = [t for t in schema.high_value_tables if t.row_count >= min_rows]

    if require_time_series:
        tables = [t for t in tables if t.is_time_series]

    # Sort by priority score descending
    tables.sort(key=lambda t: t.priority_score, reverse=True)

    return tables


def get_curated_table_list(
    source_db: SourceDatabase,
    include_explicit: bool = True,
    include_discovered: bool = True,
    min_rows: int = 10000,
    apply_allowlist: bool = True,
) -> list[str]:
    """
    Get curated list of table names for ingestion.

    Combines explicit high-value list with runtime discovery.
    Respects allowlists from settings if apply_allowlist=True.

    Args:
        source_db: Which database
        include_explicit: Include explicitly configured tables
        include_discovered: Include auto-discovered high-value tables
        min_rows: Minimum rows for discovered tables
        apply_allowlist: If True, filter by settings allowlist (if set)

    Returns:
        List of table names
    """
    from device_anomaly.config.settings import get_settings

    settings = get_settings()

    # Check if allowlist is configured - if so, use it exclusively
    if apply_allowlist:
        if source_db == SourceDatabase.XSIGHT and settings.xsight_table_allowlist:
            logger.info(f"Using XSight allowlist: {settings.xsight_table_allowlist}")
            return sorted(settings.xsight_table_allowlist)
        elif source_db == SourceDatabase.MOBICONTROL and settings.mc_table_allowlist:
            logger.info(f"Using MobiControl allowlist: {settings.mc_table_allowlist}")
            return sorted(settings.mc_table_allowlist)

    tables: set[str] = set()

    # Add explicit tables
    if include_explicit:
        explicit = HIGH_VALUE_PATTERNS.get(source_db, {}).get("explicit", set())
        tables.update(explicit)

    # Add discovered tables
    if include_discovered:
        try:
            high_value = get_high_value_tables(
                source_db=source_db,
                min_rows=min_rows,
                require_time_series=False,
            )
            tables.update(t.name for t in high_value)
        except Exception as e:
            logger.warning(f"Failed to discover tables for {source_db}: {e}")

    return sorted(tables)


def invalidate_schema_cache(source_db: SourceDatabase | None = None) -> None:
    """Invalidate cached schemas."""
    if source_db:
        from device_anomaly.config.settings import get_settings

        settings = get_settings()
        if source_db == SourceDatabase.XSIGHT:
            _schema_cache.invalidate(f"{source_db.value}_{settings.dw.database}")
        else:
            _schema_cache.invalidate(f"{source_db.value}_{settings.mc.database}")
    else:
        _schema_cache.invalidate()


def discover_training_tables(
    min_rows: int = 10000,
    require_time_series: bool = True,
    include_xsight: bool = True,
    include_mc: bool = True,
) -> dict[str, list[TableInfo]]:
    """
    Discover all tables suitable for ML training.

    This function identifies high-value tables from both XSight and MobiControl
    databases that are suitable for training anomaly detection models.

    Criteria for training tables:
    - Has sufficient row count (>= min_rows)
    - Has DeviceId column for device-level analysis
    - Has timestamp column for temporal analysis (if require_time_series=True)
    - Is marked as high-value based on patterns (cs_*, DeviceStat*, etc.)

    Args:
        min_rows: Minimum row count for a table to be included
        require_time_series: If True, only include tables with timestamp columns
        include_xsight: Whether to discover XSight tables
        include_mc: Whether to discover MobiControl tables

    Returns:
        Dictionary with keys "xsight" and "mobicontrol", each containing
        a list of TableInfo objects suitable for training, sorted by priority.

    Example:
        tables = discover_training_tables(min_rows=10000)
        print(f"XSight tables: {[t.name for t in tables['xsight']]}")
        print(f"MC tables: {[t.name for t in tables['mobicontrol']]}")
    """
    result: dict[str, list[TableInfo]] = {
        "xsight": [],
        "mobicontrol": [],
    }

    if include_xsight:
        try:
            schema = discover_xsight_schema(use_cache=True)
            xsight_tables = [
                t
                for t in schema.high_value_tables
                if t.row_count >= min_rows
                and t.has_device_id
                and (not require_time_series or t.has_timestamp)
            ]
            # Sort by priority score (higher = more valuable for ML)
            xsight_tables.sort(key=lambda t: t.priority_score, reverse=True)
            result["xsight"] = xsight_tables

            logger.info(
                f"Discovered {len(xsight_tables)} XSight training tables "
                f"(min_rows={min_rows}, time_series={require_time_series})"
            )
            if xsight_tables:
                top_tables = [t.name for t in xsight_tables[:10]]
                logger.info(f"Top XSight tables: {top_tables}")

        except Exception as e:
            logger.warning(f"Failed to discover XSight training tables: {e}")

    if include_mc:
        try:
            schema = discover_mobicontrol_schema(use_cache=True)
            mc_tables = [
                t
                for t in schema.high_value_tables
                if t.row_count >= min_rows
                and t.has_device_id
                and (not require_time_series or t.has_timestamp)
            ]
            # Sort by priority score
            mc_tables.sort(key=lambda t: t.priority_score, reverse=True)
            result["mobicontrol"] = mc_tables

            logger.info(
                f"Discovered {len(mc_tables)} MobiControl training tables "
                f"(min_rows={min_rows}, time_series={require_time_series})"
            )
            if mc_tables:
                top_tables = [t.name for t in mc_tables[:10]]
                logger.info(f"Top MC tables: {top_tables}")

        except Exception as e:
            logger.warning(f"Failed to discover MobiControl training tables: {e}")

    return result


def get_training_table_summary(tables: dict[str, list[TableInfo]]) -> dict[str, Any]:
    """
    Get a summary of discovered training tables.

    Args:
        tables: Output from discover_training_tables()

    Returns:
        Dictionary with summary statistics
    """
    xsight_tables = tables.get("xsight", [])
    mc_tables = tables.get("mobicontrol", [])

    return {
        "total_tables": len(xsight_tables) + len(mc_tables),
        "xsight": {
            "count": len(xsight_tables),
            "total_rows": sum(t.row_count for t in xsight_tables),
            "time_series_count": sum(1 for t in xsight_tables if t.is_time_series),
            "tables": [
                {
                    "name": t.name,
                    "rows": t.row_count,
                    "priority": t.priority_score,
                    "timestamp_col": t.timestamp_column,
                }
                for t in xsight_tables[:20]
            ],
        },
        "mobicontrol": {
            "count": len(mc_tables),
            "total_rows": sum(t.row_count for t in mc_tables),
            "time_series_count": sum(1 for t in mc_tables if t.is_time_series),
            "tables": [
                {
                    "name": t.name,
                    "rows": t.row_count,
                    "priority": t.priority_score,
                    "timestamp_col": t.timestamp_column,
                }
                for t in mc_tables[:20]
            ],
        },
    }


def log_startup_schema_summary(
    discover_xsight: bool = True,
    discover_mc: bool = True,
) -> dict[str, Any]:
    """
    Log schema discovery summary at application startup.

    This function should be called once at startup to:
    1. Discover and cache database schemas
    2. Log summary of tables/views found
    3. Log curated table list that will be used for ingestion
    4. Return summary dict for programmatic access

    Args:
        discover_xsight: Whether to discover XSight schema
        discover_mc: Whether to discover MobiControl schema

    Returns:
        Dictionary with schema summary statistics
    """
    from device_anomaly.config.settings import get_settings

    settings = get_settings()

    summary: dict[str, Any] = {
        "xsight": None,
        "mobicontrol": None,
        "feature_flags": {
            "enable_schema_discovery": settings.enable_schema_discovery,
            "enable_xsight_extended": settings.enable_xsight_extended,
            "enable_xsight_hourly": settings.enable_xsight_hourly,
            "enable_mc_timeseries": settings.enable_mc_timeseries,
        },
    }

    if not settings.enable_schema_discovery:
        logger.info("Schema discovery is disabled (ENABLE_SCHEMA_DISCOVERY=false)")
        return summary

    logger.info("=" * 60)
    logger.info("STARTUP SCHEMA DISCOVERY")
    logger.info("=" * 60)

    # XSight discovery
    if discover_xsight:
        try:
            schema = discover_xsight_schema(use_cache=True)
            curated = get_curated_table_list(
                SourceDatabase.XSIGHT,
                apply_allowlist=True,
            )

            summary["xsight"] = {
                "database": schema.database_name,
                "tables_count": len(schema.tables),
                "views_count": len(schema.views),
                "high_value_count": len(schema.high_value_tables),
                "time_series_count": len(schema.time_series_tables),
                "curated_tables": curated,
                "discovery_ms": schema.discovery_duration_ms,
            }

            logger.info(
                f"XSight ({schema.database_name}): "
                f"{len(schema.tables)} tables, {len(schema.views)} views, "
                f"{len(schema.high_value_tables)} high-value, "
                f"{len(schema.time_series_tables)} time-series"
            )
            logger.info(f"XSight curated tables ({len(curated)}): {curated}")

            if settings.xsight_table_allowlist:
                logger.info(f"XSight ALLOWLIST active: {settings.xsight_table_allowlist}")

        except Exception as e:
            logger.warning(f"Failed to discover XSight schema: {e}")
            summary["xsight"] = {"error": str(e)}

    # MobiControl discovery
    if discover_mc:
        try:
            schema = discover_mobicontrol_schema(use_cache=True)
            curated = get_curated_table_list(
                SourceDatabase.MOBICONTROL,
                apply_allowlist=True,
            )

            summary["mobicontrol"] = {
                "database": schema.database_name,
                "tables_count": len(schema.tables),
                "views_count": len(schema.views),
                "high_value_count": len(schema.high_value_tables),
                "time_series_count": len(schema.time_series_tables),
                "curated_tables": curated,
                "discovery_ms": schema.discovery_duration_ms,
            }

            logger.info(
                f"MobiControl ({schema.database_name}): "
                f"{len(schema.tables)} tables, {len(schema.views)} views, "
                f"{len(schema.high_value_tables)} high-value, "
                f"{len(schema.time_series_tables)} time-series"
            )
            logger.info(f"MobiControl curated tables ({len(curated)}): {curated}")

            if settings.mc_table_allowlist:
                logger.info(f"MobiControl ALLOWLIST active: {settings.mc_table_allowlist}")

        except Exception as e:
            logger.warning(f"Failed to discover MobiControl schema: {e}")
            summary["mobicontrol"] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("Feature flags:")
    logger.info(f"  ENABLE_XSIGHT_EXTENDED={settings.enable_xsight_extended}")
    logger.info(f"  ENABLE_XSIGHT_HOURLY={settings.enable_xsight_hourly}")
    logger.info(f"  ENABLE_MC_TIMESERIES={settings.enable_mc_timeseries}")
    logger.info("=" * 60)

    return summary
