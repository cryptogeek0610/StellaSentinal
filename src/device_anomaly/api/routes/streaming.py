"""
Streaming API Routes - Real-time telemetry ingestion and WebSocket endpoints.

Provides:
- POST /streaming/telemetry - Ingest real-time telemetry
- WebSocket /streaming/ws/anomalies - Real-time anomaly alerts
- GET /streaming/status - Streaming system status
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streaming", tags=["streaming"])


# =============================================================================
# Request/Response Models
# =============================================================================


class TelemetryPayload(BaseModel):
    """Real-time telemetry data from a device."""

    device_id: int = Field(..., description="Device identifier")
    timestamp: Optional[datetime] = Field(None, description="Event timestamp (defaults to now)")
    metrics: dict[str, float] = Field(..., description="Metric name to value mapping")

    class Config:
        json_schema_extra = {
            "example": {
                "device_id": 12345,
                "timestamp": "2024-01-15T10:30:00Z",
                "metrics": {
                    "TotalBatteryLevelDrop": 15.0,
                    "TotalDischargeTime_Sec": 3600.0,
                    "AppForegroundTime": 7200.0,
                    "Download": 50000000.0,
                    "Upload": 5000000.0,
                    "AvgSignalStrength": -75.0,
                    "TotalDropCnt": 3.0,
                },
            }
        }


class TelemetryBatchPayload(BaseModel):
    """Batch of telemetry events."""

    events: list[TelemetryPayload] = Field(..., description="List of telemetry events")


class TelemetryResponse(BaseModel):
    """Response for telemetry ingestion."""

    success: bool
    message: str
    device_id: int
    processed_at: datetime


class StreamingStatusResponse(BaseModel):
    """Streaming system status."""

    streaming_enabled: bool
    redis_connected: bool
    active_devices: int
    total_events_buffered: int
    websocket_connections: int
    anomalies_detected_last_hour: int
    components: dict[str, dict]


# =============================================================================
# Streaming State (initialized on startup)
# =============================================================================


class StreamingState:
    """Global streaming state, initialized on app startup."""

    engine = None
    telemetry_stream = None
    feature_computer = None
    anomaly_processor = None
    websocket_manager = None
    initialized = False


_state = StreamingState()


async def get_streaming_state() -> StreamingState:
    """Dependency to get streaming state."""
    return _state


async def initialize_streaming() -> None:
    """Initialize the streaming system. Call this on app startup."""
    from device_anomaly.streaming.engine import StreamingEngine, StreamConfig
    from device_anomaly.streaming.telemetry_stream import TelemetryStream, TelemetryBuffer
    from device_anomaly.streaming.feature_computer import StreamingFeatureComputer
    from device_anomaly.streaming.anomaly_processor import AnomalyStreamProcessor
    from device_anomaly.streaming.websocket_manager import WebSocketManager

    try:
        # Initialize engine
        config = StreamConfig.from_env()
        _state.engine = StreamingEngine(config)
        await _state.engine.connect()

        # Initialize components
        buffer = TelemetryBuffer()
        _state.telemetry_stream = TelemetryStream(_state.engine, buffer)
        _state.feature_computer = StreamingFeatureComputer(_state.engine, buffer)
        _state.anomaly_processor = AnomalyStreamProcessor(_state.engine)
        _state.websocket_manager = WebSocketManager(_state.engine)

        # Start all components
        await _state.telemetry_stream.start()
        await _state.feature_computer.start()
        await _state.anomaly_processor.start()
        await _state.websocket_manager.start()

        # Start engine listener
        await _state.engine.start_listening()

        _state.initialized = True
        logger.info("Streaming system initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize streaming: %s", e, exc_info=True)
        _state.initialized = False


async def shutdown_streaming() -> None:
    """Shutdown the streaming system. Call this on app shutdown."""
    if not _state.initialized:
        return

    try:
        await _state.engine.stop_listening()
        await _state.websocket_manager.stop()
        await _state.anomaly_processor.stop()
        await _state.feature_computer.stop()
        await _state.telemetry_stream.stop()
        await _state.engine.disconnect()

        _state.initialized = False
        logger.info("Streaming system shutdown complete")

    except Exception as e:
        logger.error("Error during streaming shutdown: %s", e, exc_info=True)


# =============================================================================
# API Routes
# =============================================================================


@router.post(
    "/telemetry",
    response_model=TelemetryResponse,
    summary="Ingest real-time telemetry",
    description="Ingest a single telemetry event for real-time processing and anomaly detection.",
)
async def ingest_telemetry(
    payload: TelemetryPayload,
    state: StreamingState = Depends(get_streaming_state),
) -> TelemetryResponse:
    """Ingest a single telemetry event."""
    if not state.initialized:
        raise HTTPException(
            status_code=503,
            detail="Streaming system not initialized",
        )

    try:
        await state.telemetry_stream.ingest(
            device_id=payload.device_id,
            metrics=payload.metrics,
            timestamp=payload.timestamp,
        )

        return TelemetryResponse(
            success=True,
            message="Telemetry ingested successfully",
            device_id=payload.device_id,
            processed_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error("Failed to ingest telemetry: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/telemetry/batch",
    response_model=dict,
    summary="Ingest batch telemetry",
    description="Ingest multiple telemetry events in a single request.",
)
async def ingest_telemetry_batch(
    payload: TelemetryBatchPayload,
    state: StreamingState = Depends(get_streaming_state),
) -> dict:
    """Ingest a batch of telemetry events."""
    if not state.initialized:
        raise HTTPException(
            status_code=503,
            detail="Streaming system not initialized",
        )

    success_count = 0
    error_count = 0

    for event in payload.events:
        try:
            await state.telemetry_stream.ingest(
                device_id=event.device_id,
                metrics=event.metrics,
                timestamp=event.timestamp,
            )
            success_count += 1
        except Exception as e:
            logger.warning("Failed to ingest event for device %d: %s", event.device_id, e)
            error_count += 1

    return {
        "success": error_count == 0,
        "message": f"Processed {success_count}/{len(payload.events)} events",
        "success_count": success_count,
        "error_count": error_count,
        "processed_at": datetime.utcnow().isoformat(),
    }


@router.get(
    "/status",
    response_model=StreamingStatusResponse,
    summary="Get streaming system status",
    description="Get the current status of the streaming system and its components.",
)
async def get_streaming_status(
    state: StreamingState = Depends(get_streaming_state),
) -> StreamingStatusResponse:
    """Get streaming system status."""
    components = {}

    if state.initialized:
        components = {
            "telemetry_stream": state.telemetry_stream.get_stats(),
            "anomaly_processor": state.anomaly_processor.get_stats(),
            "websocket_manager": state.websocket_manager.get_stats(),
        }

    return StreamingStatusResponse(
        streaming_enabled=state.initialized,
        redis_connected=state.initialized and state.engine is not None,
        active_devices=components.get("telemetry_stream", {}).get("device_count", 0),
        total_events_buffered=components.get("telemetry_stream", {}).get("total_events", 0),
        websocket_connections=components.get("websocket_manager", {}).get("connection_count", 0),
        anomalies_detected_last_hour=components.get("anomaly_processor", {}).get("alerts_sent", 0),
        components=components,
    )


@router.get(
    "/device/{device_id}/buffer",
    summary="Get device telemetry buffer",
    description="Get the buffered telemetry data for a specific device.",
)
async def get_device_buffer(
    device_id: int,
    state: StreamingState = Depends(get_streaming_state),
) -> dict:
    """Get buffered telemetry for a device."""
    if not state.initialized:
        raise HTTPException(
            status_code=503,
            detail="Streaming system not initialized",
        )

    buffer = state.telemetry_stream.get_buffer_for_device(device_id)
    if not buffer:
        return {
            "device_id": device_id,
            "found": False,
            "message": "No buffer found for this device",
        }

    return {
        "device_id": device_id,
        "found": True,
        "event_count": len(buffer.events),
        "last_update": buffer.last_update.isoformat(),
        "metrics_tracked": list(buffer._counts.keys()),
        "rolling_stats": {
            metric: buffer.get_rolling_stats(metric)
            for metric in list(buffer._counts.keys())[:10]  # Limit to first 10
        },
    }


@router.websocket("/ws/anomalies")
async def websocket_anomalies(
    websocket: WebSocket,
    state: StreamingState = Depends(get_streaming_state),
) -> None:
    """
    WebSocket endpoint for real-time anomaly alerts.

    Connect to receive real-time anomaly notifications.

    Client commands:
    - {"command": "subscribe_device", "device_id": 12345}
    - {"command": "unsubscribe_device", "device_id": 12345}
    - {"command": "set_severity_filter", "severities": ["high", "critical"]}
    - {"command": "ping"}
    - {"command": "get_stats"}

    Server messages:
    - {"type": "connected", "client_id": "...", "message": "..."}
    - {"type": "anomaly", "device_id": 12345, "severity": "high", ...}
    - {"type": "pong", "timestamp": "..."}
    """
    if not state.initialized:
        await websocket.close(code=1013, reason="Streaming system not initialized")
        return

    try:
        client_id = await state.websocket_manager.connect(websocket)
        await state.websocket_manager.handle_client_messages(client_id)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


@router.get(
    "/cohorts",
    summary="Get cohort statistics",
    description="Get running statistics for device cohorts (model + OS combinations).",
)
async def get_cohort_stats(
    state: StreamingState = Depends(get_streaming_state),
) -> dict:
    """Get statistics for all cohorts."""
    if not state.initialized:
        raise HTTPException(
            status_code=503,
            detail="Streaming system not initialized",
        )

    cohort_ids = state.feature_computer.get_all_cohort_ids()

    return {
        "cohort_count": len(cohort_ids),
        "cohorts": [
            state.feature_computer.get_cohort_stats(cid)
            for cid in cohort_ids[:50]  # Limit to first 50
        ],
    }
