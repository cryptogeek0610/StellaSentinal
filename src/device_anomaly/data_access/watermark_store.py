"""
Watermark store for incremental data extraction.

Stores per-table watermarks (last extracted timestamp) for incremental loading.

PRODUCTION-HARDENED:
- PostgreSQL is the authoritative source (single source of truth)
- Watermarks are MONOTONIC: never move backward unless explicitly reset
- Redis is write-through cache only (for fast reads)
- File fallback DISABLED by default (enable_file_watermark_fallback)
- All updates go through Postgres first, then cache

Features:
- Atomic watermark updates with transaction support
- Monotonic enforcement (rejects backward moves)
- Backfill support via explicit reset_watermark() call
- Metrics: total rows extracted per table
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class WatermarkRecord(Base):
    """SQLAlchemy model for watermark storage."""
    __tablename__ = "ingestion_watermarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_db = Column(String(50), nullable=False)
    table_name = Column(String(255), nullable=False)
    watermark_column = Column(String(100), nullable=False)
    watermark_value = Column(DateTime(timezone=True), nullable=False)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    rows_extracted = Column(Integer, default=0)
    metadata_json = Column(Text, nullable=True)

    __table_args__ = (
        {"schema": "public"},  # Use public schema for PostgreSQL
    )


class WatermarkStore:
    """
    Production-hardened watermark storage.

    Architecture:
    - PostgreSQL is the SINGLE SOURCE OF TRUTH
    - Redis is a write-through cache (optional, for fast reads)
    - File fallback is DISABLED by default (dev only)

    Monotonic Enforcement:
    - set_watermark() REJECTS any watermark < current watermark
    - To move backward, use reset_watermark() explicitly
    - This prevents data loss from bugs or race conditions
    """

    def __init__(
        self,
        postgres_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        file_path: Optional[Path] = None,
        lookback_hours: int = 24,
        enable_file_fallback: Optional[bool] = None,
    ):
        """
        Initialize watermark store.

        Args:
            postgres_url: PostgreSQL connection URL
            redis_url: Redis connection URL (optional)
            file_path: Fallback file path
            lookback_hours: Default lookback for new tables (from env or default 24)
            enable_file_fallback: Enable file fallback (default: from settings)
        """
        self._postgres_url = postgres_url
        self._redis_url = redis_url
        self._file_path = file_path or Path("data/watermarks.json")

        # Get settings
        try:
            from device_anomaly.config.settings import get_settings
            settings = get_settings()
            self._enable_file_fallback = (
                enable_file_fallback
                if enable_file_fallback is not None
                else settings.enable_file_watermark_fallback
            )
        except Exception:
            self._enable_file_fallback = enable_file_fallback or False

        if self._enable_file_fallback:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

        # Get lookback from environment
        self._lookback_hours = int(os.getenv("INGEST_LOOKBACK_HOURS", str(lookback_hours)))

        self._lock = Lock()
        self._pg_session = None
        self._pg_engine = None
        self._redis_client = None
        self._file_cache: Dict[str, Dict[str, Any]] = {}

        # Initialize backends in priority order
        self._init_postgres()
        self._init_redis()
        if self._enable_file_fallback:
            self._load_file_cache()
        else:
            logger.debug("File watermark fallback disabled")

    def _init_postgres(self) -> None:
        """Initialize PostgreSQL backend (authoritative source)."""
        if not self._postgres_url:
            try:
                from device_anomaly.config.settings import get_settings
                settings = get_settings()
                self._postgres_url = settings.backend_db.url
            except Exception as e:
                logger.debug(f"Could not get PostgreSQL URL from settings: {e}")
                return

        try:
            self._pg_engine = create_engine(self._postgres_url, pool_pre_ping=True)

            # Create table if not exists with index on (source_db, table_name)
            with self._pg_engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ingestion_watermarks (
                        id SERIAL PRIMARY KEY,
                        source_db VARCHAR(50) NOT NULL,
                        table_name VARCHAR(255) NOT NULL,
                        watermark_column VARCHAR(100) NOT NULL,
                        watermark_value TIMESTAMP WITH TIME ZONE NOT NULL,
                        last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        rows_extracted BIGINT DEFAULT 0,
                        metadata_json TEXT,
                        UNIQUE(source_db, table_name)
                    )
                """))
                conn.commit()

            Session = sessionmaker(bind=self._pg_engine)
            self._pg_session = Session
            logger.info("Watermark store: PostgreSQL backend initialized (authoritative)")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL watermark store: {e}")
            self._pg_session = None
            self._pg_engine = None

    def _init_redis(self) -> None:
        """Initialize Redis backend."""
        if not self._redis_url:
            self._redis_url = os.getenv("REDIS_URL")

        if not self._redis_url:
            return

        try:
            import redis
            self._redis_client = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=5,
            )
            # Test connection
            self._redis_client.ping()
            logger.info("Watermark store: Redis backend initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis watermark store: {e}")
            self._redis_client = None

    def _load_file_cache(self) -> None:
        """Load watermarks from file cache."""
        if self._file_path.exists():
            try:
                with open(self._file_path) as f:
                    self._file_cache = json.load(f)
                logger.debug(f"Loaded {len(self._file_cache)} watermarks from file cache")
            except Exception as e:
                logger.warning(f"Failed to load file cache: {e}")
                self._file_cache = {}

    def _save_file_cache(self) -> None:
        """Save watermarks to file cache."""
        try:
            with open(self._file_path, "w") as f:
                json.dump(self._file_cache, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save file cache: {e}")

    def _make_key(self, source_db: str, table_name: str) -> str:
        """Create cache key."""
        return f"watermark:{source_db}:{table_name}"

    def get_watermark(
        self,
        source_db: str,
        table_name: str,
        default_lookback_hours: Optional[int] = None,
    ) -> datetime:
        """
        Get last watermark for a table.

        Args:
            source_db: Source database identifier (xsight/mobicontrol)
            table_name: Table name
            default_lookback_hours: Override default lookback for new tables

        Returns:
            Last watermark timestamp (or default based on lookback)
        """
        lookback = default_lookback_hours or self._lookback_hours
        default_watermark = datetime.now(timezone.utc) - timedelta(hours=lookback)

        key = self._make_key(source_db, table_name)

        # Try PostgreSQL first
        if self._pg_session:
            try:
                session = self._pg_session()
                result = session.execute(
                    text("""
                        SELECT watermark_value
                        FROM ingestion_watermarks
                        WHERE source_db = :source_db AND table_name = :table_name
                    """),
                    {"source_db": source_db, "table_name": table_name}
                ).fetchone()
                session.close()

                if result and result[0]:
                    watermark = result[0]
                    if watermark.tzinfo is None:
                        watermark = watermark.replace(tzinfo=timezone.utc)
                    return watermark
            except Exception as e:
                logger.warning(f"PostgreSQL watermark read failed: {e}")

        # Try Redis
        if self._redis_client:
            try:
                value = self._redis_client.get(key)
                if value:
                    return datetime.fromisoformat(value)
            except Exception as e:
                logger.debug(f"Redis watermark read failed: {e}")

        # Try file cache
        with self._lock:
            if key in self._file_cache:
                try:
                    return datetime.fromisoformat(self._file_cache[key]["watermark_value"])
                except Exception:
                    pass

        return default_watermark

    def _get_stored_watermark(
        self,
        source_db: str,
        table_name: str,
    ) -> Optional[datetime]:
        """Return persisted watermark if it exists, otherwise None."""
        key = self._make_key(source_db, table_name)

        if self._pg_session:
            try:
                session = self._pg_session()
                result = session.execute(
                    text("""
                        SELECT watermark_value
                        FROM ingestion_watermarks
                        WHERE source_db = :source_db AND table_name = :table_name
                    """),
                    {"source_db": source_db, "table_name": table_name},
                ).fetchone()
                session.close()

                if result and result[0]:
                    watermark = result[0]
                    if watermark.tzinfo is None:
                        watermark = watermark.replace(tzinfo=timezone.utc)
                    return watermark
            except Exception as e:
                logger.debug(f"PostgreSQL stored watermark read failed: {e}")

        if self._redis_client:
            try:
                value = self._redis_client.get(key)
                if value:
                    return datetime.fromisoformat(value)
            except Exception as e:
                logger.debug(f"Redis stored watermark read failed: {e}")

        with self._lock:
            if key in self._file_cache:
                try:
                    return datetime.fromisoformat(self._file_cache[key]["watermark_value"])
                except Exception:
                    return None

        return None

    def get_watermark_metadata(
        self,
        source_db: str,
        table_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get stored metadata JSON for a watermark record."""
        # Try PostgreSQL first
        if self._pg_session:
            try:
                session = self._pg_session()
                result = session.execute(
                    text("""
                        SELECT metadata_json
                        FROM ingestion_watermarks
                        WHERE source_db = :source_db AND table_name = :table_name
                    """),
                    {"source_db": source_db, "table_name": table_name},
                ).fetchone()
                session.close()
                if result and result[0]:
                    return json.loads(result[0])
            except Exception as e:
                logger.debug(f"PostgreSQL watermark metadata read failed: {e}")

        # Try file cache
        with self._lock:
            key = self._make_key(source_db, table_name)
            if key in self._file_cache:
                metadata_json = self._file_cache[key].get("metadata_json")
                if metadata_json:
                    try:
                        return json.loads(metadata_json)
                    except Exception:
                        return None

        return None

    def set_watermark(
        self,
        source_db: str,
        table_name: str,
        watermark_value: datetime,
        watermark_column: str = "ServerDateTime",
        rows_extracted: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Set watermark for a table (MONOTONIC - only moves forward).

        Args:
            source_db: Source database identifier
            table_name: Table name
            watermark_value: New watermark timestamp
            watermark_column: Column name used for watermarking
            rows_extracted: Number of rows extracted in this batch
            metadata: Optional metadata to store
            force: If True, skip monotonic check (USE RESET_WATERMARK INSTEAD)

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
            - (True, None) on success
            - (False, "reason") on failure
        """
        if watermark_value.tzinfo is None:
            watermark_value = watermark_value.replace(tzinfo=timezone.utc)

        key = self._make_key(source_db, table_name)

        # MONOTONIC CHECK: Get stored watermark and reject backward moves
        if not force:
            stored_watermark = self._get_stored_watermark(source_db, table_name)
            if stored_watermark and watermark_value < stored_watermark:
                msg = (
                    f"Monotonic violation: {source_db}.{table_name} "
                    f"new={watermark_value.isoformat()} < current={stored_watermark.isoformat()}. "
                    f"Use reset_watermark() to move backward."
                )
                logger.warning(msg)
                return (False, msg)

            if stored_watermark and watermark_value == stored_watermark:
                # No change, skip write
                logger.debug(f"Watermark unchanged for {source_db}.{table_name}")
                return (True, None)

        # PostgreSQL is authoritative - must succeed for overall success
        pg_success = False
        if self._pg_session:
            try:
                session = self._pg_session()
                # Use GREATEST to ensure monotonic at database level too
                session.execute(
                    text("""
                        INSERT INTO ingestion_watermarks
                            (source_db, table_name, watermark_column, watermark_value,
                             last_updated, rows_extracted, metadata_json)
                        VALUES
                            (:source_db, :table_name, :watermark_column, :watermark_value,
                             NOW(), :rows_extracted, :metadata_json)
                        ON CONFLICT (source_db, table_name) DO UPDATE SET
                            watermark_column = EXCLUDED.watermark_column,
                            watermark_value = GREATEST(
                                ingestion_watermarks.watermark_value,
                                EXCLUDED.watermark_value
                            ),
                            last_updated = NOW(),
                            rows_extracted = ingestion_watermarks.rows_extracted + EXCLUDED.rows_extracted,
                            metadata_json = COALESCE(EXCLUDED.metadata_json, ingestion_watermarks.metadata_json)
                    """),
                    {
                        "source_db": source_db,
                        "table_name": table_name,
                        "watermark_column": watermark_column,
                        "watermark_value": watermark_value,
                        "rows_extracted": rows_extracted,
                        "metadata_json": json.dumps(metadata) if metadata else None,
                    }
                )
                session.commit()
                session.close()
                pg_success = True
                logger.debug(f"PostgreSQL watermark (monotonic): {source_db}.{table_name} = {watermark_value}")
            except Exception as e:
                logger.error(f"PostgreSQL watermark write FAILED: {e}")
                return (False, f"PostgreSQL write failed: {e}")
        else:
            # No PostgreSQL = no authoritative source
            if not self._enable_file_fallback:
                return (False, "No PostgreSQL backend and file fallback disabled")

        # Write-through to Redis cache (best effort)
        if self._redis_client and pg_success:
            try:
                self._redis_client.set(
                    key,
                    watermark_value.isoformat(),
                    ex=86400 * 7,  # 7 day TTL
                )
            except Exception as e:
                logger.debug(f"Redis cache write failed (non-fatal): {e}")

        # File fallback (only if enabled and no Postgres)
        if self._enable_file_fallback and not pg_success:
            with self._lock:
                self._file_cache[key] = {
                    "source_db": source_db,
                    "table_name": table_name,
                    "watermark_column": watermark_column,
                    "watermark_value": watermark_value.isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "rows_extracted": rows_extracted,
                    "metadata_json": json.dumps(metadata) if metadata else None,
                }
                self._save_file_cache()
                return (True, None)

        return (pg_success, None)

    def get_all_watermarks(self, source_db: Optional[str] = None) -> Dict[str, datetime]:
        """
        Get all watermarks, optionally filtered by source database.

        Returns:
            Dictionary mapping "source_db.table_name" to watermark datetime
        """
        watermarks: Dict[str, datetime] = {}

        # Try PostgreSQL
        if self._pg_session:
            try:
                session = self._pg_session()
                query = "SELECT source_db, table_name, watermark_value FROM ingestion_watermarks"
                params = {}
                if source_db:
                    query += " WHERE source_db = :source_db"
                    params["source_db"] = source_db

                result = session.execute(text(query), params).fetchall()
                session.close()

                for row in result:
                    key = f"{row[0]}.{row[1]}"
                    wm = row[2]
                    if wm.tzinfo is None:
                        wm = wm.replace(tzinfo=timezone.utc)
                    watermarks[key] = wm

                return watermarks
            except Exception as e:
                logger.warning(f"Failed to get watermarks from PostgreSQL: {e}")

        # Fall back to file cache
        with self._lock:
            for key, data in self._file_cache.items():
                if source_db and not key.startswith(f"watermark:{source_db}:"):
                    continue
                try:
                    parts = key.replace("watermark:", "").split(":", 1)
                    if len(parts) == 2:
                        watermarks[f"{parts[0]}.{parts[1]}"] = datetime.fromisoformat(
                            data["watermark_value"]
                        )
                except Exception:
                    pass

        return watermarks

    def reset_watermark(
        self,
        source_db: str,
        table_name: str,
        lookback_hours: Optional[int] = None,
        to_datetime: Optional[datetime] = None,
    ) -> Tuple[datetime, bool]:
        """
        Reset watermark to trigger backfill (EXPLICIT backward move).

        This is the ONLY way to move a watermark backward. Use for:
        - Triggering full backfill
        - Recovering from data issues
        - Re-processing specific time ranges

        Args:
            source_db: Source database identifier
            table_name: Table name
            lookback_hours: How far back to reset (None = use default)
            to_datetime: Explicit datetime to reset to (overrides lookback_hours)

        Returns:
            Tuple of (new_watermark, success)
        """
        if to_datetime:
            new_watermark = to_datetime
            if new_watermark.tzinfo is None:
                new_watermark = new_watermark.replace(tzinfo=timezone.utc)
        else:
            lookback = lookback_hours or self._lookback_hours
            new_watermark = datetime.now(timezone.utc) - timedelta(hours=lookback)

        key = self._make_key(source_db, table_name)

        # Direct write to PostgreSQL bypassing monotonic check
        pg_success = False
        if self._pg_session:
            try:
                session = self._pg_session()
                session.execute(
                    text("""
                        INSERT INTO ingestion_watermarks
                            (source_db, table_name, watermark_column, watermark_value,
                             last_updated, rows_extracted, metadata_json)
                        VALUES
                            (:source_db, :table_name, 'reset', :watermark_value,
                             NOW(), 0, :metadata_json)
                        ON CONFLICT (source_db, table_name) DO UPDATE SET
                            watermark_value = :watermark_value,
                            last_updated = NOW(),
                            rows_extracted = 0,
                            metadata_json = :metadata_json
                    """),
                    {
                        "source_db": source_db,
                        "table_name": table_name,
                        "watermark_value": new_watermark,
                        "metadata_json": json.dumps({
                            "reset": True,
                            "reset_at": datetime.now(timezone.utc).isoformat(),
                            "lookback_hours": lookback_hours,
                        }),
                    }
                )
                session.commit()
                session.close()
                pg_success = True
            except Exception as e:
                logger.error(f"Failed to reset watermark in PostgreSQL: {e}")

        # Clear Redis cache
        if self._redis_client:
            try:
                self._redis_client.delete(key)
            except Exception:
                pass

        # Update file cache if enabled
        if self._enable_file_fallback:
            with self._lock:
                self._file_cache[key] = {
                    "source_db": source_db,
                    "table_name": table_name,
                    "watermark_column": "reset",
                    "watermark_value": new_watermark.isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "rows_extracted": 0,
                    "metadata_json": json.dumps({
                        "reset": True,
                        "reset_at": datetime.now(timezone.utc).isoformat(),
                        "lookback_hours": lookback_hours,
                    }),
                }
                self._save_file_cache()

        success = pg_success or self._enable_file_fallback
        if success:
            logger.info(f"RESET watermark for {source_db}.{table_name} to {new_watermark}")
        else:
            logger.error(f"FAILED to reset watermark for {source_db}.{table_name}")

        return (new_watermark, success)


    def get_last_scoring_watermark(self) -> Optional[datetime]:
        """
        Get the watermark for the last scoring run.

        Returns:
            Last scoring timestamp, or None if never scored.
        """
        return self._get_stored_watermark("scoring", "last_run")

    def set_last_scoring_watermark(self, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """
        Record the timestamp of a scoring run.

        Args:
            timestamp: The timestamp to record

        Returns:
            Tuple of (success, error_message)
        """
        return self.set_watermark(
            source_db="scoring",
            table_name="last_run",
            watermark_value=timestamp,
            watermark_column="scoring_timestamp",
        )

    def check_source_data_freshness(
        self,
        source_tables: Optional[list] = None,
    ) -> Tuple[bool, Optional[datetime], str]:
        """
        Check if source data has new records since the last scoring run.

        This prevents anomaly score inflation when running scoring on
        static/backup databases by skipping scoring when no new data exists.

        Args:
            source_tables: List of (source_db, table_name) tuples to check.
                          Defaults to common XSight and MobiControl tables.

        Returns:
            Tuple of:
            - has_new_data: True if there's data newer than last scoring
            - max_source_timestamp: Latest timestamp in source data
            - reason: Human-readable explanation
        """
        if source_tables is None:
            source_tables = [
                ("xsight", "cs_DeviceHourlyTelemetry"),
                ("xsight", "cs_Battery"),
                ("xsight", "cs_DataUsageByHour"),
                ("mobicontrol", "Device"),
            ]

        last_scoring = self.get_last_scoring_watermark()

        # Find the most recent watermark across source tables
        max_source_timestamp: Optional[datetime] = None
        checked_tables = []

        for source_db, table_name in source_tables:
            wm = self._get_stored_watermark(source_db, table_name)
            if wm is not None:
                checked_tables.append(f"{source_db}.{table_name}")
                if max_source_timestamp is None or wm > max_source_timestamp:
                    max_source_timestamp = wm

        if max_source_timestamp is None:
            # No source watermarks found - first run or no ingestion yet
            return (
                True,
                None,
                "No source watermarks found - proceeding with scoring",
            )

        if last_scoring is None:
            # Never scored before - definitely should score
            return (
                True,
                max_source_timestamp,
                f"First scoring run. Source data up to {max_source_timestamp.isoformat()}",
            )

        # Compare latest source data to last scoring time
        if max_source_timestamp > last_scoring:
            delta = max_source_timestamp - last_scoring
            return (
                True,
                max_source_timestamp,
                f"New data available. Source: {max_source_timestamp.isoformat()}, "
                f"last scoring: {last_scoring.isoformat()}, delta: {delta}",
            )
        else:
            return (
                False,
                max_source_timestamp,
                f"No new data since last scoring at {last_scoring.isoformat()}. "
                f"Latest source: {max_source_timestamp.isoformat()}. "
                f"Checked: {', '.join(checked_tables)}",
            )


# Global singleton
_watermark_store: Optional[WatermarkStore] = None
_store_lock = Lock()


def get_watermark_store() -> WatermarkStore:
    """Get or create global watermark store instance."""
    global _watermark_store
    if _watermark_store is None:
        with _store_lock:
            if _watermark_store is None:
                _watermark_store = WatermarkStore()
    return _watermark_store


def reset_watermark_store() -> None:
    """Reset global watermark store (for testing)."""
    global _watermark_store
    _watermark_store = None
