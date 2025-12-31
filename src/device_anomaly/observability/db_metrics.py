from __future__ import annotations

import os
import time

from sqlalchemy import event

from device_anomaly.api.request_context import get_request_context

try:
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover
    Counter = None
    Histogram = None


_QUERY_DURATION = None
_QUERY_ERRORS = None
_TENANT_METRICS_ENABLED = os.getenv("ENABLE_TENANT_METRICS", "false").lower() == "true"
_TENANT_ALLOWLIST = {
    tenant.strip()
    for tenant in os.getenv("TENANT_METRICS_ALLOWLIST", "").split(",")
    if tenant.strip()
}


def _get_metrics():
    global _QUERY_DURATION, _QUERY_ERRORS
    if Counter is None or Histogram is None:
        return None, None

    label_names = ["engine", "operation"]
    if _TENANT_METRICS_ENABLED:
        label_names.append("tenant")

    if _QUERY_DURATION is None:
        _QUERY_DURATION = Histogram(
            "db_query_duration_seconds",
            "Database query duration in seconds",
            label_names,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
        )
    if _QUERY_ERRORS is None:
        _QUERY_ERRORS = Counter(
            "db_query_errors_total",
            "Database query errors",
            label_names,
        )
    return _QUERY_DURATION, _QUERY_ERRORS


def _operation_from_statement(statement: str | None) -> str:
    if not statement:
        return "UNKNOWN"
    return statement.lstrip().split(None, 1)[0].upper()


def _resolve_tenant_label() -> str:
    if not _TENANT_METRICS_ENABLED:
        return ""
    if not _TENANT_ALLOWLIST:
        return "unknown"
    tenant_id = get_request_context().tenant_id or "unknown"
    if tenant_id in _TENANT_ALLOWLIST:
        return tenant_id
    return "other"


def _build_labels(engine: str, operation: str) -> dict[str, str]:
    labels = {"engine": engine, "operation": operation}
    if _TENANT_METRICS_ENABLED:
        labels["tenant"] = _resolve_tenant_label()
    return labels


def instrument_engine(engine, name: str) -> bool:
    enabled = os.getenv("ENABLE_DB_METRICS", "true").lower() == "true"
    if not enabled:
        return False
    if getattr(engine, "_db_metrics_instrumented", False):
        return False

    duration_hist, error_counter = _get_metrics()
    if duration_hist is None or error_counter is None:
        return False

    @event.listens_for(engine, "before_cursor_execute")
    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info["query_start_time"] = time.perf_counter()
        conn.info["query_operation"] = _operation_from_statement(statement)

    @event.listens_for(engine, "after_cursor_execute")
    def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        start = conn.info.pop("query_start_time", None)
        operation = conn.info.pop("query_operation", _operation_from_statement(statement))
        if start is None:
            return
        duration_hist.labels(**_build_labels(name, operation)).observe(time.perf_counter() - start)

    @event.listens_for(engine, "handle_error")
    def _handle_error(exception_context):
        operation = _operation_from_statement(exception_context.statement)
        error_counter.labels(**_build_labels(name, operation)).inc()

    engine._db_metrics_instrumented = True
    return True
