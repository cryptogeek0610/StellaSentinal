"""
Ingestion pipeline entrypoint.

Wires schema discovery, allowlists, orchestrator throttling, and metrics
into a single batch ingestion flow.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sqlalchemy import BigInteger, Column, DateTime, MetaData, String, Table, Text, inspect, text

from device_anomaly.config.settings import get_settings
from device_anomaly.data_access.canonical_events import (
    CanonicalEvent,
    SourceDatabase as CanonicalSourceDatabase,
    dataframe_to_canonical_events,
    dedupe_events,
)
from device_anomaly.data_access.data_profiler import DW_TELEMETRY_TABLES_LEGACY
from device_anomaly.data_access.db_connection import create_dw_engine, create_mc_engine
from device_anomaly.data_access.mc_timeseries_loader import (
    MC_TIMESERIES_TABLES,
    load_mc_timeseries_incremental,
    _validate_columns as _mc_validate_columns,
)
from device_anomaly.data_access.db_utils import table_exists as _mc_table_exists
from device_anomaly.data_access.schema_discovery import (
    SourceDatabase as SchemaSourceDatabase,
    get_curated_table_list,
)
from device_anomaly.data_access.watermark_store import get_watermark_store
from device_anomaly.data_access.xsight_loader_extended import (
    XSIGHT_EXTENDED_TABLES,
    load_xsight_table_incremental,
    _validate_columns as _xsight_validate_columns,
)
# Import table_exists alias for xsight - uses same db_utils helper
_xsight_table_exists = _mc_table_exists
from device_anomaly.db.session import DatabaseSession
from device_anomaly.services.ingestion_metrics import record_ingestion_metric
from device_anomaly.services.ingestion_orchestrator import (
    IngestionBatchResult,
    create_table_list_for_ingestion,
    get_table_weight,
    run_batch_sync,
)

logger = logging.getLogger(__name__)


@dataclass
class SkippedTable:
    source_db: str
    table_name: str
    reason: str


def _is_hourly_table(table_name: str) -> bool:
    config = XSIGHT_EXTENDED_TABLES.get(table_name, {})
    columns = config.get("columns", [])
    return "Hour" in columns or config.get("large_table", False)


def _xsight_table_enabled(table_name: str, settings) -> Tuple[bool, Optional[str]]:
    base_tables = set(DW_TELEMETRY_TABLES_LEGACY)
    if table_name in base_tables:
        return True, None
    if _is_hourly_table(table_name):
        return settings.enable_xsight_hourly, "disabled_xsight_hourly"
    return settings.enable_xsight_extended, "disabled_xsight_extended"


def _select_candidate_tables(
    source_db: SchemaSourceDatabase,
    include_discovered: bool,
    fallback: Iterable[str],
) -> List[str]:
    try:
        candidates = get_curated_table_list(
            source_db=source_db,
            include_explicit=True,
            include_discovered=include_discovered,
            apply_allowlist=True,
        )
    except Exception as exc:
        logger.warning("Failed to load curated tables for %s: %s", source_db.value, exc)
        candidates = []
    if not candidates:
        candidates = list(fallback)
    return sorted(set(candidates))


def _build_ingestion_plan(
    xsight_tables: Optional[List[str]] = None,
    mc_tables: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, str]], List[SkippedTable]]:
    settings = get_settings()
    include_discovered = settings.enable_schema_discovery

    xsight_candidates = xsight_tables or _select_candidate_tables(
        SchemaSourceDatabase.XSIGHT,
        include_discovered=include_discovered,
        fallback=XSIGHT_EXTENDED_TABLES.keys(),
    )
    mc_candidates = mc_tables or _select_candidate_tables(
        SchemaSourceDatabase.MOBICONTROL,
        include_discovered=include_discovered,
        fallback=MC_TIMESERIES_TABLES.keys(),
    )

    tasks: List[Dict[str, str]] = []
    skipped: List[SkippedTable] = []

    for table_name in xsight_candidates:
        if table_name not in XSIGHT_EXTENDED_TABLES:
            skipped.append(SkippedTable("xsight", table_name, "missing_config"))
            continue
        enabled, reason = _xsight_table_enabled(table_name, settings)
        if not enabled:
            skipped.append(SkippedTable("xsight", table_name, reason or "disabled"))
            continue
        tasks.append({"table_name": table_name, "source_db": "xsight"})

    for table_name in mc_candidates:
        if table_name not in MC_TIMESERIES_TABLES:
            skipped.append(SkippedTable("mobicontrol", table_name, "missing_config"))
            continue
        if not settings.enable_mc_timeseries:
            skipped.append(SkippedTable("mobicontrol", table_name, "disabled_mc_timeseries"))
            continue
        tasks.append({"table_name": table_name, "source_db": "mobicontrol"})

    return tasks, skipped


def _mock_dataframe(table_name: str, source_db: str, rows: int = 3) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    if source_db == "xsight":
        config = XSIGHT_EXTENDED_TABLES.get(table_name, {})
    else:
        config = MC_TIMESERIES_TABLES.get(table_name, {})
    columns = config.get("columns", [])
    timestamp_col = config.get("timestamp_col")
    device_col = config.get("device_col") or "DeviceId"

    data = []
    for i in range(rows):
        row = {}
        for col in columns:
            if col == timestamp_col or col in {"TimeStamp", "DateTime", "SetDateTime", "LastChangedDate"}:
                row[col] = now - timedelta(minutes=rows - i)
            elif col == "Hour":
                row[col] = (i + 8) % 24
            elif col in {"DeviceId", "Deviceid"}:
                row[col] = 100 + i
            elif col == device_col:
                row[col] = 100 + i
            elif col == "StatType":
                row[col] = 1
            elif col in {"IntValue", "Upload", "Download", "BatteryLevel", "BatteryDrain"}:
                row[col] = 10 + i
            elif col.endswith("Id") or col.endswith("ID"):
                row[col] = i + 1
            else:
                row[col] = i
        data.append(row)
    return pd.DataFrame(data)


def _ensure_canonical_events_table(engine, auto_create: bool) -> bool:
    inspector = inspect(engine)
    if inspector.has_table("canonical_events"):
        return True
    if not auto_create:
        return False

    metadata = MetaData()
    Table(
        "canonical_events",
        metadata,
        Column("event_id", String(64), primary_key=True),
        Column("tenant_id", String(50), nullable=False),
        Column("source_db", String(50), nullable=False),
        Column("source_table", String(255), nullable=False),
        Column("device_id", BigInteger, nullable=False),
        Column("event_time", DateTime(timezone=True), nullable=False),
        Column("metric_name", String(255), nullable=False),
        Column("metric_type", String(20), nullable=False),
        Column("metric_value_json", Text),
        Column("dimensions_json", Text),
        Column("created_at", DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP")),
    )
    metadata.create_all(engine)
    return True


def _insert_canonical_events(engine, events: List[CanonicalEvent]) -> int:
    if not events:
        return 0

    rows = []
    for event in events:
        rows.append(
            {
                "event_id": event.event_id,
                "tenant_id": event.tenant_id,
                "source_db": event.source_db.value,
                "source_table": event.source_table,
                "device_id": event.device_id,
                "event_time": event.event_time,
                "metric_name": event.metric_name,
                "metric_type": event.metric_type.value,
                "metric_value_json": json.dumps(event.metric_value, default=str),
                "dimensions_json": json.dumps(event.dimensions, default=str),
            }
        )

    if engine.dialect.name == "sqlite":
        insert_sql = """
            INSERT OR IGNORE INTO canonical_events (
                event_id, tenant_id, source_db, source_table, device_id, event_time,
                metric_name, metric_type, metric_value_json, dimensions_json
            ) VALUES (
                :event_id, :tenant_id, :source_db, :source_table, :device_id, :event_time,
                :metric_name, :metric_type, :metric_value_json, :dimensions_json
            )
        """
    else:
        insert_sql = """
            INSERT INTO canonical_events (
                event_id, tenant_id, source_db, source_table, device_id, event_time,
                metric_name, metric_type, metric_value_json, dimensions_json
            ) VALUES (
                :event_id, :tenant_id, :source_db, :source_table, :device_id, :event_time,
                :metric_name, :metric_type, :metric_value_json, :dimensions_json
            )
            ON CONFLICT (event_id) DO NOTHING
        """

    with engine.begin() as conn:
        result = conn.execute(text(insert_sql), rows)
    if result.rowcount and result.rowcount > 0:
        return result.rowcount
    return len(rows)


def _record_skip(
    skip: SkippedTable,
    batch_id: str,
    record_metric: Callable[..., None],
) -> None:
    record_metric(
        source_db=skip.source_db,
        table_name=skip.table_name,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        rows_skipped=1,
        success=True,
        warning=skip.reason,
        batch_id=batch_id,
        weight=get_table_weight(skip.table_name),
    )
    logger.warning(
        "INGESTION_SKIP: %s",
        json.dumps({"source_db": skip.source_db, "table": skip.table_name, "reason": skip.reason}),
    )


def _ingest_table(
    table_name: str,
    source_db: str,
    *,
    batch_id: str,
    dry_run: bool,
    record_metric: Callable[..., None],
    watermark_store,
) -> Dict[str, int]:
    settings = get_settings()
    started_at = datetime.now(timezone.utc)
    rows_fetched = 0
    rows_inserted = 0
    rows_deduped = 0
    rows_skipped = 0
    success = True
    error = None
    warning = None
    watermark_start = None
    watermark_end = None
    query_time_ms = None
    transform_time_ms = None
    write_time_ms = None

    try:
        if source_db == "xsight":
            engine = None if dry_run else create_dw_engine()
            config = XSIGHT_EXTENDED_TABLES.get(table_name)
            if not config:
                rows_skipped = 1
                warning = "missing_config"
                return {"rows_fetched": 0, "rows_inserted": 0, "rows_deduped": 0}

            if not dry_run and engine and not _xsight_table_exists(engine, table_name):
                rows_skipped = 1
                warning = "missing_table"
                return {"rows_fetched": 0, "rows_inserted": 0, "rows_deduped": 0}

            if not dry_run and engine:
                valid_columns = _xsight_validate_columns(engine, table_name, config["columns"])
                if not valid_columns:
                    rows_skipped = 1
                    warning = "missing_columns"
                    return {"rows_fetched": 0, "rows_inserted": 0, "rows_deduped": 0}

            watermark_start = watermark_store.get_watermark("xsight", table_name)
            start_hour = None
            metadata = watermark_store.get_watermark_metadata("xsight", table_name)
            if metadata and "hour" in metadata:
                try:
                    start_hour = int(metadata["hour"])
                except (TypeError, ValueError):
                    start_hour = None

            query_start = time.perf_counter()
            if dry_run:
                df = _mock_dataframe(table_name, source_db)
                new_watermark = datetime.now(timezone.utc)
                new_hour = None
            else:
                df, new_watermark, new_hour = load_xsight_table_incremental(
                    table_name=table_name,
                    start_date=watermark_start,
                    end_date=datetime.now(timezone.utc),
                    batch_size=settings.ingest_batch_size,
                    use_watermark=False,
                    engine=engine,
                    start_hour=start_hour,
                )
            query_time_ms = (time.perf_counter() - query_start) * 1000.0

            rows_fetched = len(df)
            watermark_end = new_watermark

            transform_start = time.perf_counter()
            events = dataframe_to_canonical_events(
                df=df,
                source_db=CanonicalSourceDatabase.XSIGHT,
                source_table=table_name,
                tenant_id="default",
            )
            deduped, _ = dedupe_events(events)
            rows_deduped = max(0, len(events) - len(deduped))
            transform_time_ms = (time.perf_counter() - transform_start) * 1000.0

            if not dry_run and settings.enable_canonical_event_storage:
                engine = DatabaseSession().engine
                if _ensure_canonical_events_table(engine, settings.auto_create_canonical_events_tables):
                    write_start = time.perf_counter()
                    rows_inserted = _insert_canonical_events(engine, deduped)
                    write_time_ms = (time.perf_counter() - write_start) * 1000.0
                else:
                    warning = "canonical_events_table_missing"
            else:
                if dry_run:
                    warning = "dry_run"

            if new_watermark:
                metadata = {"hour": new_hour} if new_hour is not None else None
                watermark_store.set_watermark(
                    source_db="xsight",
                    table_name=table_name,
                    watermark_value=new_watermark,
                    watermark_column=config.get("timestamp_col", "CollectedDate"),
                    rows_extracted=rows_fetched,
                    metadata=metadata,
                )

        elif source_db == "mobicontrol":
            engine = None if dry_run else create_mc_engine()
            config = MC_TIMESERIES_TABLES.get(table_name)
            if not config:
                rows_skipped = 1
                warning = "missing_config"
                return {"rows_fetched": 0, "rows_inserted": 0, "rows_deduped": 0}

            if not dry_run and engine and not _mc_table_exists(engine, table_name):
                rows_skipped = 1
                warning = "missing_table"
                return {"rows_fetched": 0, "rows_inserted": 0, "rows_deduped": 0}

            if not dry_run and engine:
                valid_columns = _mc_validate_columns(engine, table_name, config["columns"])
                if not valid_columns:
                    rows_skipped = 1
                    warning = "missing_columns"
                    return {"rows_fetched": 0, "rows_inserted": 0, "rows_deduped": 0}

            watermark_start = watermark_store.get_watermark("mobicontrol", table_name)

            query_start = time.perf_counter()
            if dry_run:
                df = _mock_dataframe(table_name, source_db)
                new_watermark = datetime.now(timezone.utc)
            else:
                df, new_watermark = load_mc_timeseries_incremental(
                    table_name=table_name,
                    start_time=watermark_start,
                    end_time=datetime.now(timezone.utc),
                    batch_size=settings.ingest_batch_size,
                    use_watermark=False,
                    engine=engine,
                )
            query_time_ms = (time.perf_counter() - query_start) * 1000.0

            rows_fetched = len(df)
            watermark_end = new_watermark

            transform_start = time.perf_counter()
            events = dataframe_to_canonical_events(
                df=df,
                source_db=CanonicalSourceDatabase.MOBICONTROL,
                source_table=table_name,
                tenant_id="default",
            )
            deduped, _ = dedupe_events(events)
            rows_deduped = max(0, len(events) - len(deduped))
            transform_time_ms = (time.perf_counter() - transform_start) * 1000.0

            if not dry_run and settings.enable_canonical_event_storage:
                engine = DatabaseSession().engine
                if _ensure_canonical_events_table(engine, settings.auto_create_canonical_events_tables):
                    write_start = time.perf_counter()
                    rows_inserted = _insert_canonical_events(engine, deduped)
                    write_time_ms = (time.perf_counter() - write_start) * 1000.0
                else:
                    warning = "canonical_events_table_missing"
            else:
                if dry_run:
                    warning = "dry_run"

            if new_watermark:
                watermark_store.set_watermark(
                    source_db="mobicontrol",
                    table_name=table_name,
                    watermark_value=new_watermark,
                    watermark_column=config.get("timestamp_col", "ServerDateTime"),
                    rows_extracted=rows_fetched,
                )
        else:
            rows_skipped = 1
            warning = "unknown_source_db"

    except Exception as exc:
        success = False
        error = str(exc)
        logger.exception("Ingestion failed for %s.%s", source_db, table_name)

    finally:
        record_metric(
            source_db=source_db,
            table_name=table_name,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            rows_fetched=rows_fetched,
            rows_inserted=rows_inserted,
            rows_deduped=rows_deduped,
            rows_skipped=rows_skipped,
            watermark_start=watermark_start,
            watermark_end=watermark_end,
            success=success,
            error=error,
            warning=warning,
            batch_id=batch_id,
            query_time_ms=query_time_ms,
            transform_time_ms=transform_time_ms,
            write_time_ms=write_time_ms,
            weight=get_table_weight(table_name),
        )

    return {
        "rows_fetched": rows_fetched,
        "rows_inserted": rows_inserted,
        "rows_deduped": rows_deduped,
    }


def run_ingestion_batch(
    xsight_tables: Optional[List[str]] = None,
    mc_tables: Optional[List[str]] = None,
    dry_run: bool = False,
    max_weight: Optional[int] = None,
    record_metric: Callable[..., None] = record_ingestion_metric,
    watermark_store=None,
) -> IngestionBatchResult:
    """
    Run a full ingestion batch using the weighted orchestrator.

    Returns an IngestionBatchResult for reporting.
    """
    settings = get_settings()
    batch_id = datetime.now(timezone.utc).strftime("ingest_%Y%m%d_%H%M%S")
    watermark_store = watermark_store or get_watermark_store()

    tasks, skipped = _build_ingestion_plan(xsight_tables=xsight_tables, mc_tables=mc_tables)

    for skip in skipped:
        _record_skip(skip, batch_id=batch_id, record_metric=record_metric)

    if not tasks:
        now = datetime.now(timezone.utc)
        logger.warning("No ingestion tasks scheduled (dry_run=%s).", dry_run)
        return IngestionBatchResult(started_at=now, completed_at=now, tasks=[])

    def loader(table_name: str, source_db: str) -> Dict[str, int]:
        return _ingest_table(
            table_name,
            source_db,
            batch_id=batch_id,
            dry_run=dry_run,
            record_metric=record_metric,
            watermark_store=watermark_store,
        )

    if max_weight is None:
        max_weight = settings.ingest_max_tables_parallel

    return run_batch_sync(
        tables=create_table_list_for_ingestion(
            xsight_tables=[t["table_name"] for t in tasks if t["source_db"] == "xsight"],
            mc_tables=[t["table_name"] for t in tasks if t["source_db"] == "mobicontrol"],
        ),
        loader_func=loader,
        max_weight=max_weight,
    )
