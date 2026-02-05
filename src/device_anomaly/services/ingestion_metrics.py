"""
Ingestion Metrics and Coverage Reporting.

Provides observability for the data ingestion pipeline:
- Per-table metrics (rows fetched, inserted, deduped, lag, duration)
- Daily coverage reports to PostgreSQL
- Structured logging with metric tags

Tables:
- ingestion_metrics: Per-table per-run metrics
- telemetry_coverage_report: Daily aggregated coverage report
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from device_anomaly.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TableIngestionMetric:
    """Metrics for a single table ingestion run."""
    source_db: str
    table_name: str
    started_at: datetime
    completed_at: datetime | None = None

    # Row counts
    rows_fetched: int = 0
    rows_inserted: int = 0
    rows_deduped: int = 0
    rows_skipped: int = 0  # Skipped due to allowlist, missing table, etc.

    # Watermarks
    watermark_start: datetime | None = None
    watermark_end: datetime | None = None

    # Performance
    query_time_ms: float | None = None
    transform_time_ms: float | None = None
    write_time_ms: float | None = None

    # Status
    success: bool = True
    error: str | None = None
    warning: str | None = None

    # Context
    batch_id: str | None = None
    weight: int = 1

    @property
    def duration_ms(self) -> float | None:
        """Total duration in milliseconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def lag_seconds(self) -> float | None:
        """Data lag in seconds (now - watermark_end)."""
        if self.watermark_end:
            return (datetime.now(UTC) - self.watermark_end).total_seconds()
        return None

    @property
    def dedupe_ratio(self) -> float:
        """Ratio of deduped rows to fetched rows."""
        if self.rows_fetched > 0:
            return self.rows_deduped / self.rows_fetched
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source_db": self.source_db,
            "table_name": self.table_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "rows_fetched": self.rows_fetched,
            "rows_inserted": self.rows_inserted,
            "rows_deduped": self.rows_deduped,
            "rows_skipped": self.rows_skipped,
            "watermark_start": self.watermark_start.isoformat() if self.watermark_start else None,
            "watermark_end": self.watermark_end.isoformat() if self.watermark_end else None,
            "lag_seconds": self.lag_seconds,
            "dedupe_ratio": self.dedupe_ratio,
            "query_time_ms": self.query_time_ms,
            "transform_time_ms": self.transform_time_ms,
            "write_time_ms": self.write_time_ms,
            "success": self.success,
            "error": self.error,
            "warning": self.warning,
            "batch_id": self.batch_id,
            "weight": self.weight,
        }


@dataclass
class DailyCoverageReport:
    """Daily aggregated coverage report."""
    report_date: datetime
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Source database stats
    xsight_tables_configured: int = 0
    xsight_tables_loaded: int = 0
    xsight_tables_failed: int = 0
    xsight_total_rows: int = 0

    mc_tables_configured: int = 0
    mc_tables_loaded: int = 0
    mc_tables_failed: int = 0
    mc_total_rows: int = 0

    # Quality metrics
    total_rows_fetched: int = 0
    total_rows_inserted: int = 0
    total_rows_deduped: int = 0
    avg_lag_seconds: float | None = None
    max_lag_seconds: float | None = None

    # Table-level details
    table_stats: list[dict[str, Any]] = field(default_factory=list)

    # Issues
    tables_with_errors: list[str] = field(default_factory=list)
    tables_with_high_lag: list[str] = field(default_factory=list)  # > 1 hour
    tables_with_high_dedupe: list[str] = field(default_factory=list)  # > 20% dedupe

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "report_date": self.report_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "xsight_tables_configured": self.xsight_tables_configured,
            "xsight_tables_loaded": self.xsight_tables_loaded,
            "xsight_tables_failed": self.xsight_tables_failed,
            "xsight_total_rows": self.xsight_total_rows,
            "mc_tables_configured": self.mc_tables_configured,
            "mc_tables_loaded": self.mc_tables_loaded,
            "mc_tables_failed": self.mc_tables_failed,
            "mc_total_rows": self.mc_total_rows,
            "total_rows_fetched": self.total_rows_fetched,
            "total_rows_inserted": self.total_rows_inserted,
            "total_rows_deduped": self.total_rows_deduped,
            "avg_lag_seconds": self.avg_lag_seconds,
            "max_lag_seconds": self.max_lag_seconds,
            "table_stats": self.table_stats,
            "tables_with_errors": self.tables_with_errors,
            "tables_with_high_lag": self.tables_with_high_lag,
            "tables_with_high_dedupe": self.tables_with_high_dedupe,
        }


class IngestionMetricsStore:
    """
    Store and retrieve ingestion metrics in PostgreSQL.

    Tables created:
    - ingestion_metrics: Per-table per-run metrics
    - telemetry_coverage_report: Daily aggregated reports
    """

    def __init__(self, postgres_url: str | None = None):
        self._postgres_url = postgres_url
        self._engine: Engine | None = None
        self._lock = Lock()
        self._buffer: list[TableIngestionMetric] = []
        self._buffer_size = 100  # Flush every N metrics
        self._tables_ready = False
        self._auto_create_tables = False

        self._init_postgres()

    def _init_postgres(self) -> None:
        """Initialize PostgreSQL connection and create tables."""
        if not self._postgres_url:
            try:
                settings = get_settings()
                self._postgres_url = settings.backend_db.url
            except Exception as e:
                logger.debug(f"Could not get PostgreSQL URL: {e}")
                return

        try:
            self._engine = create_engine(self._postgres_url, pool_pre_ping=True)
            settings = get_settings()
            self._auto_create_tables = settings.auto_create_metrics_tables

            if self._auto_create_tables:
                with self._engine.connect() as conn:
                    # Per-table metrics table
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS ingestion_metrics (
                            id SERIAL PRIMARY KEY,
                            source_db VARCHAR(50) NOT NULL,
                            table_name VARCHAR(255) NOT NULL,
                            started_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            completed_at TIMESTAMP WITH TIME ZONE,
                            duration_ms FLOAT,
                            rows_fetched BIGINT DEFAULT 0,
                            rows_inserted BIGINT DEFAULT 0,
                            rows_deduped BIGINT DEFAULT 0,
                            rows_skipped BIGINT DEFAULT 0,
                            watermark_start TIMESTAMP WITH TIME ZONE,
                            watermark_end TIMESTAMP WITH TIME ZONE,
                            lag_seconds FLOAT,
                            success BOOLEAN DEFAULT TRUE,
                            error TEXT,
                            batch_id VARCHAR(100),
                            metadata_json TEXT,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        )
                    """))

                    # Index for querying by date and table
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_ingestion_metrics_source_table
                        ON ingestion_metrics (source_db, table_name, started_at DESC)
                    """))

                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_ingestion_metrics_date
                        ON ingestion_metrics (DATE(started_at))
                    """))

                    # Daily coverage report table
                    conn.execute(text("""
                        CREATE TABLE IF NOT EXISTS telemetry_coverage_report (
                            id SERIAL PRIMARY KEY,
                            report_date DATE NOT NULL UNIQUE,
                            generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            xsight_tables_configured INT DEFAULT 0,
                            xsight_tables_loaded INT DEFAULT 0,
                            xsight_tables_failed INT DEFAULT 0,
                            xsight_total_rows BIGINT DEFAULT 0,
                            mc_tables_configured INT DEFAULT 0,
                            mc_tables_loaded INT DEFAULT 0,
                            mc_tables_failed INT DEFAULT 0,
                            mc_total_rows BIGINT DEFAULT 0,
                            total_rows_fetched BIGINT DEFAULT 0,
                            total_rows_inserted BIGINT DEFAULT 0,
                            total_rows_deduped BIGINT DEFAULT 0,
                            avg_lag_seconds FLOAT,
                            max_lag_seconds FLOAT,
                            report_json TEXT,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        )
                    """))

                    conn.commit()
                    logger.info("Ingestion metrics tables created")

                self._tables_ready = True
            else:
                self._tables_ready = self._check_tables_ready()
                if not self._tables_ready:
                    logger.warning(
                        "Ingestion metrics tables missing. "
                        "Set AUTO_CREATE_METRICS_TABLES=true to create them."
                    )

        except Exception as e:
            logger.error(f"Failed to initialize metrics store: {e}")
            self._engine = None

    def record_metric(self, metric: TableIngestionMetric) -> bool:
        """
        Record a single table ingestion metric.

        Metrics are buffered and flushed periodically for efficiency.
        """
        with self._lock:
            self._buffer.append(metric)

            if len(self._buffer) >= self._buffer_size:
                return self._flush_buffer()

        # Also log structured metric for real-time observability
        self._log_metric(metric)
        return True

    def _log_metric(self, metric: TableIngestionMetric) -> None:
        """Log metric in structured format for log aggregation."""
        log_data = {
            "event": "ingestion_metric",
            "source_db": metric.source_db,
            "table": metric.table_name,
            "rows_fetched": metric.rows_fetched,
            "rows_inserted": metric.rows_inserted,
            "rows_deduped": metric.rows_deduped,
            "rows_skipped": metric.rows_skipped,
            "duration_ms": metric.duration_ms,
            "lag_seconds": metric.lag_seconds,
            "success": metric.success,
        }
        if metric.error:
            log_data["error"] = metric.error
        if metric.warning:
            log_data["warning"] = metric.warning

        if metric.success:
            logger.info(f"METRIC: {json.dumps(log_data)}")
        else:
            logger.warning(f"METRIC: {json.dumps(log_data)}")

    def _check_tables_ready(self) -> bool:
        """Check whether metrics tables already exist."""
        if not self._engine:
            return False
        try:
            with self._engine.connect() as conn:
                if self._engine.dialect.name == "sqlite":
                    result = conn.execute(
                        text("""
                            SELECT name
                            FROM sqlite_master
                            WHERE type = 'table'
                              AND name IN ('ingestion_metrics', 'telemetry_coverage_report')
                        """)
                    ).fetchall()
                else:
                    result = conn.execute(
                        text("""
                            SELECT table_name
                            FROM information_schema.tables
                            WHERE table_name IN ('ingestion_metrics', 'telemetry_coverage_report')
                        """)
                    ).fetchall()
            table_names = {row[0] for row in result}
            return {"ingestion_metrics", "telemetry_coverage_report"} <= table_names
        except Exception as e:
            logger.debug(f"Metrics table existence check failed: {e}")
            return False

    def _flush_buffer(self) -> bool:
        """Flush buffered metrics to PostgreSQL."""
        if not self._engine or not self._buffer:
            return False
        if not self._tables_ready:
            self._buffer.clear()
            return False

        metrics_to_flush = self._buffer.copy()
        self._buffer.clear()

        try:
            with self._engine.connect() as conn:
                for metric in metrics_to_flush:
                    conn.execute(
                        text("""
                            INSERT INTO ingestion_metrics (
                                source_db, table_name, started_at, completed_at,
                                duration_ms, rows_fetched, rows_inserted, rows_deduped,
                                rows_skipped, watermark_start, watermark_end, lag_seconds,
                                success, error, batch_id, metadata_json
                            ) VALUES (
                                :source_db, :table_name, :started_at, :completed_at,
                                :duration_ms, :rows_fetched, :rows_inserted, :rows_deduped,
                                :rows_skipped, :watermark_start, :watermark_end, :lag_seconds,
                                :success, :error, :batch_id, :metadata_json
                            )
                        """),
                        {
                            "source_db": metric.source_db,
                            "table_name": metric.table_name,
                            "started_at": metric.started_at,
                            "completed_at": metric.completed_at,
                            "duration_ms": metric.duration_ms,
                            "rows_fetched": metric.rows_fetched,
                            "rows_inserted": metric.rows_inserted,
                            "rows_deduped": metric.rows_deduped,
                            "rows_skipped": metric.rows_skipped,
                            "watermark_start": metric.watermark_start,
                            "watermark_end": metric.watermark_end,
                            "lag_seconds": metric.lag_seconds,
                            "success": metric.success,
                            "error": metric.error,
                            "batch_id": metric.batch_id,
                            "metadata_json": json.dumps({
                                "query_time_ms": metric.query_time_ms,
                                "transform_time_ms": metric.transform_time_ms,
                                "write_time_ms": metric.write_time_ms,
                                "weight": metric.weight,
                                "warning": metric.warning,
                            }),
                        }
                    )
                conn.commit()
            logger.debug(f"Flushed {len(metrics_to_flush)} metrics to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            # Put metrics back in buffer for retry
            self._buffer.extend(metrics_to_flush)
            return False

    def flush(self) -> bool:
        """Force flush all buffered metrics."""
        with self._lock:
            return self._flush_buffer()

    def generate_daily_report(
        self,
        report_date: datetime | None = None,
    ) -> DailyCoverageReport | None:
        """
        Generate daily coverage report from metrics.

        Args:
            report_date: Date to generate report for (default: yesterday)

        Returns:
            DailyCoverageReport or None if generation fails
        """
        if not self._engine:
            logger.warning("Cannot generate report: no PostgreSQL connection")
            return None

        # Flush any pending metrics first
        self.flush()

        if report_date is None:
            report_date = datetime.now(UTC).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=1)

        report_date_str = report_date.strftime("%Y-%m-%d")
        logger.info(f"Generating daily coverage report for {report_date_str}")

        try:
            with self._engine.connect() as conn:
                # Get aggregated metrics for the day
                result = conn.execute(
                    text("""
                        SELECT
                            source_db,
                            COUNT(DISTINCT table_name) as tables_loaded,
                            COUNT(DISTINCT CASE WHEN NOT success THEN table_name END) as tables_failed,
                            SUM(rows_fetched) as total_fetched,
                            SUM(rows_inserted) as total_inserted,
                            SUM(rows_deduped) as total_deduped,
                            AVG(lag_seconds) as avg_lag,
                            MAX(lag_seconds) as max_lag
                        FROM ingestion_metrics
                        WHERE DATE(started_at) = :report_date
                        GROUP BY source_db
                    """),
                    {"report_date": report_date_str}
                ).fetchall()

                report = DailyCoverageReport(report_date=report_date)

                for row in result:
                    source_db = row[0]
                    if source_db == "xsight":
                        report.xsight_tables_loaded = row[1] or 0
                        report.xsight_tables_failed = row[2] or 0
                        report.xsight_total_rows = row[3] or 0
                    elif source_db == "mobicontrol":
                        report.mc_tables_loaded = row[1] or 0
                        report.mc_tables_failed = row[2] or 0
                        report.mc_total_rows = row[3] or 0

                    report.total_rows_fetched += row[3] or 0
                    report.total_rows_inserted += row[4] or 0
                    report.total_rows_deduped += row[5] or 0

                # Get table-level stats
                table_stats = conn.execute(
                    text("""
                        SELECT
                            source_db,
                            table_name,
                            SUM(rows_fetched) as rows_fetched,
                            SUM(rows_inserted) as rows_inserted,
                            SUM(rows_deduped) as rows_deduped,
                            AVG(lag_seconds) as avg_lag,
                            MAX(lag_seconds) as max_lag,
                            BOOL_OR(NOT success) as had_error,
                            MAX(error) as last_error
                        FROM ingestion_metrics
                        WHERE DATE(started_at) = :report_date
                        GROUP BY source_db, table_name
                        ORDER BY source_db, table_name
                    """),
                    {"report_date": report_date_str}
                ).fetchall()

                for row in table_stats:
                    table_stat = {
                        "source_db": row[0],
                        "table_name": row[1],
                        "rows_fetched": row[2] or 0,
                        "rows_inserted": row[3] or 0,
                        "rows_deduped": row[4] or 0,
                        "avg_lag_seconds": row[5],
                        "max_lag_seconds": row[6],
                        "had_error": row[7],
                        "last_error": row[8],
                    }
                    report.table_stats.append(table_stat)

                    table_key = f"{row[0]}.{row[1]}"

                    if row[7]:  # had_error
                        report.tables_with_errors.append(table_key)

                    if row[6] and row[6] > 3600:  # max_lag > 1 hour
                        report.tables_with_high_lag.append(table_key)

                    # Check dedupe ratio
                    if row[2] and row[2] > 0:
                        dedupe_ratio = (row[4] or 0) / row[2]
                        if dedupe_ratio > 0.2:  # > 20% dedupe
                            report.tables_with_high_dedupe.append(table_key)

                # Calculate overall lag
                all_lags = [s["avg_lag_seconds"] for s in report.table_stats if s["avg_lag_seconds"]]
                if all_lags:
                    report.avg_lag_seconds = sum(all_lags) / len(all_lags)
                    report.max_lag_seconds = max(s["max_lag_seconds"] or 0 for s in report.table_stats)

                # Save report to database
                self._save_daily_report(report)

                logger.info(
                    f"Coverage report for {report_date_str}: "
                    f"XSight={report.xsight_tables_loaded} tables/{report.xsight_total_rows} rows, "
                    f"MC={report.mc_tables_loaded} tables/{report.mc_total_rows} rows, "
                    f"errors={len(report.tables_with_errors)}, "
                    f"high_lag={len(report.tables_with_high_lag)}"
                )

                return report

        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            return None

    def _save_daily_report(self, report: DailyCoverageReport) -> bool:
        """Save daily report to PostgreSQL."""
        if not self._engine:
            return False

        try:
            with self._engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO telemetry_coverage_report (
                            report_date, generated_at,
                            xsight_tables_configured, xsight_tables_loaded, xsight_tables_failed, xsight_total_rows,
                            mc_tables_configured, mc_tables_loaded, mc_tables_failed, mc_total_rows,
                            total_rows_fetched, total_rows_inserted, total_rows_deduped,
                            avg_lag_seconds, max_lag_seconds, report_json
                        ) VALUES (
                            :report_date, :generated_at,
                            :xsight_tables_configured, :xsight_tables_loaded, :xsight_tables_failed, :xsight_total_rows,
                            :mc_tables_configured, :mc_tables_loaded, :mc_tables_failed, :mc_total_rows,
                            :total_rows_fetched, :total_rows_inserted, :total_rows_deduped,
                            :avg_lag_seconds, :max_lag_seconds, :report_json
                        )
                        ON CONFLICT (report_date) DO UPDATE SET
                            generated_at = EXCLUDED.generated_at,
                            xsight_tables_loaded = EXCLUDED.xsight_tables_loaded,
                            xsight_tables_failed = EXCLUDED.xsight_tables_failed,
                            xsight_total_rows = EXCLUDED.xsight_total_rows,
                            mc_tables_loaded = EXCLUDED.mc_tables_loaded,
                            mc_tables_failed = EXCLUDED.mc_tables_failed,
                            mc_total_rows = EXCLUDED.mc_total_rows,
                            total_rows_fetched = EXCLUDED.total_rows_fetched,
                            total_rows_inserted = EXCLUDED.total_rows_inserted,
                            total_rows_deduped = EXCLUDED.total_rows_deduped,
                            avg_lag_seconds = EXCLUDED.avg_lag_seconds,
                            max_lag_seconds = EXCLUDED.max_lag_seconds,
                            report_json = EXCLUDED.report_json
                    """),
                    {
                        "report_date": report.report_date.date(),
                        "generated_at": report.generated_at,
                        "xsight_tables_configured": report.xsight_tables_configured,
                        "xsight_tables_loaded": report.xsight_tables_loaded,
                        "xsight_tables_failed": report.xsight_tables_failed,
                        "xsight_total_rows": report.xsight_total_rows,
                        "mc_tables_configured": report.mc_tables_configured,
                        "mc_tables_loaded": report.mc_tables_loaded,
                        "mc_tables_failed": report.mc_tables_failed,
                        "mc_total_rows": report.mc_total_rows,
                        "total_rows_fetched": report.total_rows_fetched,
                        "total_rows_inserted": report.total_rows_inserted,
                        "total_rows_deduped": report.total_rows_deduped,
                        "avg_lag_seconds": report.avg_lag_seconds,
                        "max_lag_seconds": report.max_lag_seconds,
                        "report_json": json.dumps(report.to_dict()),
                    }
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save daily report: {e}")
            return False

    def get_daily_report(self, report_date: datetime) -> DailyCoverageReport | None:
        """Retrieve a daily coverage report."""
        if not self._engine:
            return None

        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT report_json FROM telemetry_coverage_report
                        WHERE report_date = :report_date
                    """),
                    {"report_date": report_date.date()}
                ).fetchone()

                if result and result[0]:
                    data = json.loads(result[0])
                    return DailyCoverageReport(
                        report_date=datetime.fromisoformat(data["report_date"]),
                        generated_at=datetime.fromisoformat(data["generated_at"]),
                        xsight_tables_configured=data.get("xsight_tables_configured", 0),
                        xsight_tables_loaded=data.get("xsight_tables_loaded", 0),
                        xsight_tables_failed=data.get("xsight_tables_failed", 0),
                        xsight_total_rows=data.get("xsight_total_rows", 0),
                        mc_tables_configured=data.get("mc_tables_configured", 0),
                        mc_tables_loaded=data.get("mc_tables_loaded", 0),
                        mc_tables_failed=data.get("mc_tables_failed", 0),
                        mc_total_rows=data.get("mc_total_rows", 0),
                        total_rows_fetched=data.get("total_rows_fetched", 0),
                        total_rows_inserted=data.get("total_rows_inserted", 0),
                        total_rows_deduped=data.get("total_rows_deduped", 0),
                        avg_lag_seconds=data.get("avg_lag_seconds"),
                        max_lag_seconds=data.get("max_lag_seconds"),
                        table_stats=data.get("table_stats", []),
                        tables_with_errors=data.get("tables_with_errors", []),
                        tables_with_high_lag=data.get("tables_with_high_lag", []),
                        tables_with_high_dedupe=data.get("tables_with_high_dedupe", []),
                    )
        except Exception as e:
            logger.error(f"Failed to get daily report: {e}")

        return None

    def get_table_metrics(
        self,
        source_db: str,
        table_name: str,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Get recent metrics for a specific table."""
        if not self._engine:
            return []

        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT
                            started_at, completed_at, duration_ms,
                            rows_fetched, rows_inserted, rows_deduped,
                            watermark_start, watermark_end, lag_seconds,
                            success, error, batch_id
                        FROM ingestion_metrics
                        WHERE source_db = :source_db
                          AND table_name = :table_name
                          AND started_at > NOW() - INTERVAL ':days days'
                        ORDER BY started_at DESC
                        LIMIT 100
                    """.replace(":days", str(days))),
                    {"source_db": source_db, "table_name": table_name}
                ).fetchall()

                return [
                    {
                        "started_at": row[0].isoformat() if row[0] else None,
                        "completed_at": row[1].isoformat() if row[1] else None,
                        "duration_ms": row[2],
                        "rows_fetched": row[3],
                        "rows_inserted": row[4],
                        "rows_deduped": row[5],
                        "watermark_start": row[6].isoformat() if row[6] else None,
                        "watermark_end": row[7].isoformat() if row[7] else None,
                        "lag_seconds": row[8],
                        "success": row[9],
                        "error": row[10],
                        "batch_id": row[11],
                    }
                    for row in result
                ]
        except Exception as e:
            logger.error(f"Failed to get table metrics: {e}")
            return []


# Global singleton
_metrics_store: IngestionMetricsStore | None = None
_metrics_lock = Lock()


def get_metrics_store() -> IngestionMetricsStore:
    """Get or create global metrics store instance."""
    global _metrics_store
    if _metrics_store is None:
        with _metrics_lock:
            if _metrics_store is None:
                settings = get_settings()
                if settings.enable_ingestion_metrics:
                    _metrics_store = IngestionMetricsStore()
                else:
                    # Return a no-op store when metrics disabled
                    _metrics_store = IngestionMetricsStore(postgres_url="disabled")
                    _metrics_store._engine = None
    return _metrics_store


def record_ingestion_metric(
    source_db: str,
    table_name: str,
    started_at: datetime,
    completed_at: datetime | None = None,
    rows_fetched: int = 0,
    rows_inserted: int = 0,
    rows_deduped: int = 0,
    rows_skipped: int = 0,
    watermark_start: datetime | None = None,
    watermark_end: datetime | None = None,
    success: bool = True,
    error: str | None = None,
    warning: str | None = None,
    batch_id: str | None = None,
    query_time_ms: float | None = None,
    transform_time_ms: float | None = None,
    write_time_ms: float | None = None,
    weight: int = 1,
) -> None:
    """
    Convenience function to record a single ingestion metric.

    Use this in loader functions after completing an ingestion run.
    """
    settings = get_settings()
    if not settings.enable_ingestion_metrics:
        return

    metric = TableIngestionMetric(
        source_db=source_db,
        table_name=table_name,
        started_at=started_at,
        completed_at=completed_at or datetime.now(UTC),
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
        weight=weight,
    )

    store = get_metrics_store()
    store.record_metric(metric)


def generate_daily_coverage_report(
    report_date: datetime | None = None,
) -> DailyCoverageReport | None:
    """
    Generate and store daily coverage report.

    Should be called once per day, typically by a scheduler.
    """
    settings = get_settings()
    if not settings.enable_daily_coverage_report:
        logger.info("Daily coverage report disabled")
        return None

    store = get_metrics_store()
    return store.generate_daily_report(report_date)
