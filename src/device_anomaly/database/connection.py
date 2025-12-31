"""Database connection utilities for anomaly results storage."""
from __future__ import annotations

import os
from threading import Lock

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from device_anomaly.database.schema import Base


def get_results_db_url() -> str:
    """Get the database URL for storing anomaly results."""
    # Check for explicit results DB URL
    results_db_url = os.getenv("RESULTS_DB_URL")
    if results_db_url:
        return results_db_url

    # Use BACKEND_DB_* environment variables if set (Docker/production)
    backend_host = os.getenv("BACKEND_DB_HOST")
    if backend_host:
        user = os.getenv("BACKEND_DB_USER", "postgres")
        password = os.getenv("BACKEND_DB_PASS", "postgres")
        port = os.getenv("BACKEND_DB_PORT", "5432")
        database = os.getenv("BACKEND_DB_NAME", "anomaly_detection")
        return f"postgresql://{user}:{password}@{backend_host}:{port}/{database}"

    # Default to SQLite for local development
    db_path = os.getenv("RESULTS_DB_PATH", "anomaly_results.db")
    return f"sqlite:///{db_path}"


_ENGINE = None
_SESSION_FACTORY = None
_ENGINE_LOCK = Lock()


def _build_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, echo=False, connect_args=connect_args)


def _migrate_results_schema(engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    def _table_exists(conn, table: str) -> bool:
        row = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table"),
            {"table": table},
        ).fetchone()
        return row is not None

    def _column_names(conn, table: str) -> set[str]:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return {row[1] for row in rows}

    def _ensure_column(
        conn,
        table: str,
        column: str,
        column_type: str,
        default_sql: str | None = None,
    ) -> None:
        columns = _column_names(conn, table)
        if column in columns:
            return
        ddl = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
        if default_sql is not None:
            ddl += f" DEFAULT {default_sql}"
        conn.execute(text(ddl))

    def _ensure_index(conn, index_name: str, ddl: str) -> None:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {ddl}"))

    def _ensure_unique_index(conn, index_name: str, ddl: str) -> None:
        conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {ddl}"))

    with engine.begin() as conn:
        if _table_exists(conn, "anomaly_results"):
            _ensure_column(conn, "anomaly_results", "tenant_id", "TEXT", "'default'")
            conn.execute(text("UPDATE anomaly_results SET tenant_id = 'default' WHERE tenant_id IS NULL"))
            _ensure_index(conn, "idx_anomaly_results_tenant_id", "anomaly_results(tenant_id)")

        if _table_exists(conn, "device_metadata"):
            _ensure_column(conn, "device_metadata", "tenant_id", "TEXT", "'default'")
            conn.execute(text("UPDATE device_metadata SET tenant_id = 'default' WHERE tenant_id IS NULL"))
            _ensure_index(conn, "idx_device_metadata_tenant_id", "device_metadata(tenant_id)")

        if _table_exists(conn, "investigation_notes"):
            _ensure_column(conn, "investigation_notes", "tenant_id", "TEXT", "'default'")
            conn.execute(text("UPDATE investigation_notes SET tenant_id = 'default' WHERE tenant_id IS NULL"))
            _ensure_index(conn, "idx_investigation_notes_tenant_id", "investigation_notes(tenant_id)")

        if _table_exists(conn, "troubleshooting_cache"):
            _ensure_column(conn, "troubleshooting_cache", "tenant_id", "TEXT", "'default'")
            conn.execute(text("UPDATE troubleshooting_cache SET tenant_id = 'default' WHERE tenant_id IS NULL"))

            index_rows = conn.execute(text("PRAGMA index_list(troubleshooting_cache)")).fetchall()
            needs_rebuild = False
            for _, index_name, is_unique, *_ in index_rows:
                if not is_unique:
                    continue
                info_rows = conn.execute(text(f"PRAGMA index_info({index_name})")).fetchall()
                cols = [row[2] for row in info_rows]
                if cols == ["error_signature"]:
                    needs_rebuild = True
                    break

            if needs_rebuild:
                conn.execute(text("ALTER TABLE troubleshooting_cache RENAME TO troubleshooting_cache_old"))
                conn.execute(
                    text(
                        """
                        CREATE TABLE troubleshooting_cache (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            tenant_id TEXT NOT NULL,
                            error_signature TEXT NOT NULL,
                            error_pattern TEXT NOT NULL,
                            advice TEXT NOT NULL,
                            summary TEXT,
                            service_type TEXT,
                            use_count INTEGER NOT NULL DEFAULT 0,
                            last_used DATETIME NOT NULL,
                            created_at DATETIME NOT NULL
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO troubleshooting_cache (
                            tenant_id,
                            error_signature,
                            error_pattern,
                            advice,
                            summary,
                            service_type,
                            use_count,
                            last_used,
                            created_at
                        )
                        SELECT
                            'default',
                            error_signature,
                            error_pattern,
                            advice,
                            summary,
                            service_type,
                            use_count,
                            last_used,
                            created_at
                        FROM troubleshooting_cache_old
                        """
                    )
                )
                conn.execute(text("DROP TABLE troubleshooting_cache_old"))

            _ensure_index(conn, "idx_troubleshooting_cache_tenant_id", "troubleshooting_cache(tenant_id)")
            _ensure_unique_index(
                conn,
                "idx_tc_tenant_signature",
                "troubleshooting_cache(tenant_id, error_signature)",
            )

        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    model_version TEXT,
                    status TEXT NOT NULL,
                    config_json TEXT,
                    metrics_json TEXT,
                    artifacts_json TEXT,
                    started_at DATETIME,
                    completed_at DATETIME,
                    error TEXT,
                    created_at DATETIME NOT NULL
                )
                """
            )
        )
        _ensure_index(conn, "idx_training_runs_run_id", "training_runs(run_id)")
        _ensure_index(conn, "idx_training_runs_tenant_id", "training_runs(tenant_id)")


def get_results_db_engine():
    """Create and return a cached database engine for anomaly results."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                database_url = get_results_db_url()
                _ENGINE = _build_engine(database_url)
                use_migrations = os.getenv("RESULTS_DB_USE_MIGRATIONS", "false").lower() == "true"
                if not use_migrations:
                    Base.metadata.create_all(_ENGINE)
                    _migrate_results_schema(_ENGINE)
                try:
                    from device_anomaly.observability.db_metrics import instrument_engine as instrument_db_metrics
                    from device_anomaly.observability.sqlalchemy import instrument_engine as instrument_otel

                    instrument_db_metrics(_ENGINE, "results")
                    instrument_otel(_ENGINE)
                except Exception:
                    pass
                _SESSION_FACTORY = sessionmaker(bind=_ENGINE, autocommit=False, autoflush=False)
    return _ENGINE


def get_results_db_session() -> Session:
    """Get a database session for anomaly results."""
    global _SESSION_FACTORY
    engine = get_results_db_engine()
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _SESSION_FACTORY()
