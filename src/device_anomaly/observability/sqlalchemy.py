from __future__ import annotations

import os

from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor


def instrument_engine(engine) -> bool:
    enabled = os.getenv("ENABLE_OTEL", "false").lower() == "true" or bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if not enabled:
        return False
    if getattr(engine, "_otel_instrumented", False):
        return False
    SQLAlchemyInstrumentor().instrument(engine=engine)
    engine._otel_instrumented = True
    return True
