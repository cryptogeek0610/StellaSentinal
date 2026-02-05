"""
Telemetry Stream - Real-time telemetry ingestion and buffering.

Handles incoming device telemetry with:
- Per-device buffering for rolling window calculations
- Device metadata enrichment
- Cohort identification
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from device_anomaly.streaming.engine import (
    MessageType,
    StreamingEngine,
    StreamMessage,
)

logger = logging.getLogger(__name__)


@dataclass
class TelemetryEvent:
    """A single telemetry event from a device."""

    device_id: int
    timestamp: datetime
    metrics: dict[str, float]

    # Device context (enriched from MobiControl)
    model_id: int | None = None
    manufacturer_id: int | None = None
    os_version_id: int | None = None
    firmware_version: str | None = None
    tenant_id: str | None = None

    @property
    def cohort_id(self) -> str:
        """Get cohort identifier for this device."""
        parts = [
            str(self.manufacturer_id or "unk"),
            str(self.model_id or "unk"),
            str(self.os_version_id or "unk"),
        ]
        if self.firmware_version:
            parts.append(self.firmware_version)
        return "_".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "model_id": self.model_id,
            "manufacturer_id": self.manufacturer_id,
            "os_version_id": self.os_version_id,
            "firmware_version": self.firmware_version,
            "tenant_id": self.tenant_id,
            "cohort_id": self.cohort_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TelemetryEvent:
        """Create from dictionary."""
        return cls(
            device_id=data["device_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=data["metrics"],
            model_id=data.get("model_id"),
            manufacturer_id=data.get("manufacturer_id"),
            os_version_id=data.get("os_version_id"),
            firmware_version=data.get("firmware_version"),
            tenant_id=data.get("tenant_id"),
        )


@dataclass
class DeviceBuffer:
    """Rolling buffer of telemetry for a single device."""

    device_id: int
    max_events: int = 100  # Keep last N events
    max_age_hours: int = 168  # 7 days

    events: list[TelemetryEvent] = field(default_factory=list)
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Running statistics for incremental computation
    _running_sums: dict[str, float] = field(default_factory=dict)
    _running_sq_sums: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def add_event(self, event: TelemetryEvent) -> None:
        """Add a new telemetry event to the buffer."""
        self.events.append(event)
        self.last_update = datetime.now(UTC)

        # Update running statistics
        for metric, value in event.metrics.items():
            if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                self._running_sums[metric] = self._running_sums.get(metric, 0) + value
                self._running_sq_sums[metric] = self._running_sq_sums.get(metric, 0) + value ** 2
                self._counts[metric] = self._counts.get(metric, 0) + 1

        # Prune old events
        self._prune()

    def _prune(self) -> None:
        """Remove old events from buffer."""
        cutoff = datetime.now(UTC) - timedelta(hours=self.max_age_hours)

        # Remove by age (handle timezone-naive timestamps by treating them as UTC)
        def _ensure_aware(ts: datetime) -> datetime:
            if ts.tzinfo is None:
                return ts.replace(tzinfo=UTC)
            return ts

        while self.events and _ensure_aware(self.events[0].timestamp) < cutoff:
            removed = self.events.pop(0)
            self._subtract_from_running_stats(removed)

        # Remove by count
        while len(self.events) > self.max_events:
            removed = self.events.pop(0)
            self._subtract_from_running_stats(removed)

    def _subtract_from_running_stats(self, event: TelemetryEvent) -> None:
        """Subtract an event from running statistics."""
        for metric, value in event.metrics.items():
            if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                self._running_sums[metric] = self._running_sums.get(metric, 0) - value
                self._running_sq_sums[metric] = self._running_sq_sums.get(metric, 0) - value ** 2
                self._counts[metric] = max(0, self._counts.get(metric, 0) - 1)

    def get_rolling_mean(self, metric: str) -> float | None:
        """Get rolling mean for a metric."""
        count = self._counts.get(metric, 0)
        if count == 0:
            return None
        return self._running_sums.get(metric, 0) / count

    def get_rolling_std(self, metric: str) -> float | None:
        """Get rolling standard deviation for a metric."""
        count = self._counts.get(metric, 0)
        if count < 2:
            return None

        mean = self.get_rolling_mean(metric)
        if mean is None:
            return None

        variance = (self._running_sq_sums.get(metric, 0) / count) - (mean ** 2)
        if variance < 0:
            variance = 0  # Numerical stability
        return np.sqrt(variance)

    def get_rolling_stats(self, metric: str) -> dict[str, float | None]:
        """Get all rolling statistics for a metric."""
        values = [
            e.metrics.get(metric)
            for e in self.events
            if metric in e.metrics and e.metrics[metric] is not None
        ]

        if not values:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
                "count": 0,
            }

        return {
            "mean": self.get_rolling_mean(metric),
            "std": self.get_rolling_std(metric),
            "min": min(values),
            "max": max(values),
            "median": float(np.median(values)),
            "count": len(values),
        }

    def get_delta(self, metric: str) -> float | None:
        """Get the change from previous event."""
        recent = [e for e in self.events if metric in e.metrics]
        if len(recent) < 2:
            return None

        prev_value = recent[-2].metrics.get(metric)
        curr_value = recent[-1].metrics.get(metric)

        if prev_value is None or curr_value is None:
            return None

        return curr_value - prev_value

    def to_dict(self) -> dict[str, Any]:
        """Serialize buffer state for persistence."""
        return {
            "device_id": self.device_id,
            "max_events": self.max_events,
            "max_age_hours": self.max_age_hours,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceBuffer:
        """Rehydrate buffer from serialized state."""
        buffer = cls(
            device_id=int(data["device_id"]),
            max_events=int(data.get("max_events", 100)),
            max_age_hours=int(data.get("max_age_hours", 168)),
        )
        for event_data in data.get("events", []):
            buffer.add_event(TelemetryEvent.from_dict(event_data))
        return buffer


class TelemetryBuffer:
    """
    Manages per-device telemetry buffers.

    Provides:
    - Per-device rolling windows
    - Efficient incremental statistics
    - Memory-bounded storage
    """

    def __init__(
        self,
        max_devices: int = 10000,
        buffer_size_per_device: int = 100,
        max_age_hours: int = 168,
    ):
        self.max_devices = max_devices
        self.buffer_size = buffer_size_per_device
        self.max_age_hours = max_age_hours
        self._buffers: dict[int, DeviceBuffer] = {}
        self._lock = asyncio.Lock()
        self.restored_at: datetime | None = None
        self.restore_source: str | None = None

    async def add_event(self, event: TelemetryEvent) -> DeviceBuffer:
        """Add a telemetry event to the appropriate buffer."""
        async with self._lock:
            if event.device_id not in self._buffers:
                # Check if we need to evict old buffers
                if len(self._buffers) >= self.max_devices:
                    self._evict_oldest()

                self._buffers[event.device_id] = DeviceBuffer(
                    device_id=event.device_id,
                    max_events=self.buffer_size,
                    max_age_hours=self.max_age_hours,
                )

            buffer = self._buffers[event.device_id]
            buffer.add_event(event)
            return buffer

    def _evict_oldest(self) -> None:
        """Evict the oldest device buffer."""
        if not self._buffers:
            return

        oldest_device = min(
            self._buffers.keys(),
            key=lambda d: self._buffers[d].last_update,
        )
        del self._buffers[oldest_device]
        logger.debug("Evicted buffer for device %d", oldest_device)

    def get_buffer(self, device_id: int) -> DeviceBuffer | None:
        """Get the buffer for a device."""
        return self._buffers.get(device_id)

    def get_device_count(self) -> int:
        """Get the number of devices with active buffers."""
        return len(self._buffers)

    def get_total_events(self) -> int:
        """Get total number of events across all buffers."""
        return sum(len(b.events) for b in self._buffers.values())

    def snapshot(self) -> dict[str, Any]:
        """Serialize all device buffers for persistence."""
        return {
            "max_devices": self.max_devices,
            "buffer_size": self.buffer_size,
            "max_age_hours": self.max_age_hours,
            "buffers": {str(device_id): buf.to_dict() for device_id, buf in self._buffers.items()},
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> TelemetryBuffer:
        """Restore TelemetryBuffer from a snapshot."""
        buffer = cls(
            max_devices=int(snapshot.get("max_devices", 10000)),
            buffer_size_per_device=int(snapshot.get("buffer_size", 100)),
            max_age_hours=int(snapshot.get("max_age_hours", 168)),
        )
        buffers = snapshot.get("buffers", {})
        for device_id, data in buffers.items():
            device_buffer = DeviceBuffer.from_dict(data)
            buffer._buffers[int(device_id)] = device_buffer
        return buffer

    def save_snapshot(self, path: Path, max_bytes: int) -> bool:
        """Persist buffer snapshot to disk using an atomic write."""
        payload = self.snapshot()
        data = json.dumps(payload, indent=2, default=float)
        data_bytes = data.encode("utf-8")
        if max_bytes > 0 and len(data_bytes) > max_bytes:
            logger.warning(
                "Streaming state snapshot too large (%d bytes > %d); skipping save.",
                len(data_bytes),
                max_bytes,
            )
            return False

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_bytes(data_bytes)
            os.replace(tmp_path, path)
            return True
        except Exception as exc:
            logger.warning("Failed to save streaming state to %s: %s", path, exc)
            return False

    @classmethod
    def load_snapshot_path(cls, path: Path, max_bytes: int) -> TelemetryBuffer | None:
        """Load buffer snapshot from disk with size/corruption guards."""
        if not path.exists():
            return None

        try:
            size = path.stat().st_size
            if max_bytes > 0 and size > max_bytes:
                logger.warning(
                    "Streaming state snapshot too large (%d bytes > %d); skipping load.",
                    size,
                    max_bytes,
                )
                return None
            payload = json.loads(path.read_text())
            buffer = cls.from_snapshot(payload)
            buffer.restored_at = datetime.now(UTC)
            buffer.restore_source = str(path)
            return buffer
        except Exception as exc:
            logger.warning("Failed to load streaming state from %s: %s", path, exc)
            return None


class TelemetryStream:
    """
    Real-time telemetry stream processor.

    Ingests raw telemetry, enriches with device metadata,
    and forwards to feature computation.

    Usage:
        stream = TelemetryStream(engine)
        await stream.start()

        # Ingest telemetry
        await stream.ingest(device_id=12345, metrics={"battery_level": 85})
    """

    def __init__(
        self,
        engine: StreamingEngine,
        buffer: TelemetryBuffer | None = None,
    ):
        self.engine = engine
        self.buffer = buffer or TelemetryBuffer()
        self._device_metadata_cache: dict[int, dict] = {}
        self._running = False

    async def start(self) -> None:
        """Start the telemetry stream processor."""
        # Subscribe to raw telemetry
        await self.engine.subscribe(
            MessageType.TELEMETRY_RAW,
            self._handle_raw_telemetry,
        )
        self._running = True
        logger.info("TelemetryStream started")

    async def stop(self) -> None:
        """Stop the telemetry stream processor."""
        self._running = False
        await self.engine.unsubscribe(MessageType.TELEMETRY_RAW)
        logger.info("TelemetryStream stopped")

    async def ingest(
        self,
        device_id: int,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """
        Ingest raw telemetry and publish to stream.

        Args:
            device_id: Device identifier
            metrics: Dictionary of metric names to values
            timestamp: Event timestamp (defaults to now)
            tenant_id: Tenant identifier for multi-tenancy
        """
        await self.engine.publish(StreamMessage(
            message_type=MessageType.TELEMETRY_RAW,
            payload={
                "metrics": metrics,
                "timestamp": (timestamp or datetime.now(UTC)).isoformat(),
            },
            device_id=device_id,
            tenant_id=tenant_id,
        ))

    async def _handle_raw_telemetry(self, message: StreamMessage) -> None:
        """Process raw telemetry message."""
        try:
            # Create telemetry event
            event = TelemetryEvent(
                device_id=message.device_id,
                timestamp=datetime.fromisoformat(message.payload["timestamp"]),
                metrics=message.payload["metrics"],
                tenant_id=message.tenant_id,
            )

            # Enrich with device metadata
            event = await self._enrich_with_metadata(event)

            # Add to buffer
            buffer = await self.buffer.add_event(event)

            # Publish enriched telemetry
            await self.engine.publish(StreamMessage(
                message_type=MessageType.TELEMETRY_ENRICHED,
                payload={
                    "event": event.to_dict(),
                    "buffer_stats": {
                        "event_count": len(buffer.events),
                        "metrics": list(buffer._counts.keys()),
                    },
                },
                device_id=message.device_id,
                tenant_id=message.tenant_id,
            ))

            logger.debug(
                "Processed telemetry for device %d (%d metrics)",
                message.device_id,
                len(event.metrics),
            )

        except Exception as e:
            logger.error(
                "Error processing telemetry for device %d: %s",
                message.device_id,
                e,
                exc_info=True,
            )

    async def _enrich_with_metadata(self, event: TelemetryEvent) -> TelemetryEvent:
        """Enrich event with device metadata from cache or database."""
        # Check cache first
        if event.device_id in self._device_metadata_cache:
            metadata = self._device_metadata_cache[event.device_id]
            event.model_id = metadata.get("model_id")
            event.manufacturer_id = metadata.get("manufacturer_id")
            event.os_version_id = metadata.get("os_version_id")
            event.firmware_version = metadata.get("firmware_version")
            return event

        # Fetch from database (async)
        try:
            metadata = await self._fetch_device_metadata(event.device_id)
            if metadata:
                self._device_metadata_cache[event.device_id] = metadata
                event.model_id = metadata.get("model_id")
                event.manufacturer_id = metadata.get("manufacturer_id")
                event.os_version_id = metadata.get("os_version_id")
                event.firmware_version = metadata.get("firmware_version")
        except Exception as e:
            logger.warning(
                "Could not fetch metadata for device %d: %s",
                event.device_id,
                e,
            )

        return event

    async def _fetch_device_metadata(self, device_id: int) -> dict | None:
        """
        Fetch device metadata from MobiControl cache for streaming enrichment.

        This provides cohort information (model, manufacturer, OS version, firmware)
        for real-time anomaly detection without querying the database on every event.

        The cache is populated by device_metadata_sync.load_mc_device_metadata_cache()
        which should be called at startup and periodically refreshed.
        """
        try:
            from device_anomaly.services.device_metadata_sync import (
                get_mc_device_metadata,
                refresh_mc_cache_if_stale,
            )

            # Refresh cache if stale (non-blocking check)
            refresh_mc_cache_if_stale(max_age_minutes=30)

            # Look up device in cache
            metadata = get_mc_device_metadata(device_id)
            if metadata:
                return metadata

            # Device not in cache - this is expected for new devices
            # They'll be picked up on the next cache refresh
            return None

        except ImportError:
            logger.debug("device_metadata_sync not available for streaming enrichment")
            return None
        except Exception as e:
            logger.debug("Error fetching metadata for device %d: %s", device_id, e)
            return None

    def get_buffer_for_device(self, device_id: int) -> DeviceBuffer | None:
        """Get the telemetry buffer for a specific device."""
        return self.buffer.get_buffer(device_id)

    def get_stats(self) -> dict[str, Any]:
        """Get telemetry stream statistics."""
        return {
            "running": self._running,
            "device_count": self.buffer.get_device_count(),
            "total_events": self.buffer.get_total_events(),
            "metadata_cache_size": len(self._device_metadata_cache),
            "restored_at": self.buffer.restored_at.isoformat() if self.buffer.restored_at else None,
            "restore_source": Path(self.buffer.restore_source).name if self.buffer.restore_source else None,
        }
