from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str) -> bool:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    enabled = os.getenv("ENABLE_OTEL", "false").lower() == "true" or bool(endpoint)
    if not enabled:
        return False

    if not endpoint:
        endpoint = "http://localhost:4317"

    insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
    service_name = os.getenv("OTEL_SERVICE_NAME", service_name)
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing enabled (endpoint=%s, insecure=%s)", endpoint, insecure)
    return True
