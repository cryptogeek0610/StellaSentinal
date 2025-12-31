"""
Real-time Streaming Module for Device Anomaly Detection.

This module provides real-time telemetry ingestion, streaming feature computation,
and instant anomaly detection - moving beyond batch-only processing.

Components:
- StreamingEngine: Core pub/sub infrastructure using Redis
- TelemetryStream: Real-time telemetry ingestion and buffering
- StreamingFeatureComputer: Incremental feature computation
- AnomalyStreamProcessor: Real-time anomaly scoring
- WebSocketManager: Push anomaly alerts to connected clients
"""

from device_anomaly.streaming.engine import (
    StreamingEngine,
    StreamConfig,
    StreamMessage,
    MessageType,
)
from device_anomaly.streaming.telemetry_stream import (
    TelemetryStream,
    TelemetryEvent,
    TelemetryBuffer,
)
from device_anomaly.streaming.feature_computer import (
    StreamingFeatureComputer,
    IncrementalStats,
)
from device_anomaly.streaming.anomaly_processor import (
    AnomalyStreamProcessor,
    StreamingAnomalyResult,
)

__all__ = [
    "StreamingEngine",
    "StreamConfig",
    "StreamMessage",
    "MessageType",
    "TelemetryStream",
    "TelemetryEvent",
    "TelemetryBuffer",
    "StreamingFeatureComputer",
    "IncrementalStats",
    "AnomalyStreamProcessor",
    "StreamingAnomalyResult",
]
