"""
Streaming Engine - Core pub/sub infrastructure for real-time processing.

Uses Redis for message passing between components, enabling:
- Real-time telemetry ingestion
- Streaming feature computation
- Instant anomaly detection and alerting
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages in the streaming system."""

    TELEMETRY_RAW = "telemetry.raw"           # Raw telemetry from devices
    TELEMETRY_ENRICHED = "telemetry.enriched"  # Enriched with device metadata
    FEATURES_COMPUTED = "features.computed"    # Features ready for scoring
    ANOMALY_DETECTED = "anomaly.detected"      # Anomaly alert
    DEVICE_ONLINE = "device.online"            # Device came online
    DEVICE_OFFLINE = "device.offline"          # Device went offline
    MODEL_UPDATED = "model.updated"            # ML model was updated


@dataclass
class StreamConfig:
    """Configuration for the streaming engine."""

    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0

    # Channel prefixes
    channel_prefix: str = "anomaly:"

    # Buffer settings
    buffer_size: int = 1000
    flush_interval_ms: int = 100

    # Processing settings
    max_concurrent_handlers: int = 10
    handler_timeout_seconds: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 100

    @classmethod
    def from_env(cls) -> StreamConfig:
        """Load configuration from environment variables."""
        import os
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            buffer_size=int(os.getenv("STREAM_BUFFER_SIZE", "1000")),
            flush_interval_ms=int(os.getenv("STREAM_FLUSH_INTERVAL_MS", "100")),
        )


@dataclass
class StreamMessage:
    """A message in the streaming system."""

    message_type: MessageType
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    device_id: Optional[int] = None
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_json(self) -> str:
        """Serialize message to JSON."""
        return json.dumps({
            "type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
        })

    @classmethod
    def from_json(cls, data: str) -> StreamMessage:
        """Deserialize message from JSON."""
        parsed = json.loads(data)
        return cls(
            message_type=MessageType(parsed["type"]),
            payload=parsed["payload"],
            timestamp=datetime.fromisoformat(parsed["timestamp"]),
            device_id=parsed.get("device_id"),
            tenant_id=parsed.get("tenant_id"),
            correlation_id=parsed.get("correlation_id"),
        )


# Type alias for message handlers
MessageHandler = Callable[[StreamMessage], Coroutine[Any, Any, None]]


class StreamingEngine:
    """
    Core streaming engine using Redis pub/sub.

    Provides:
    - Publish/subscribe messaging
    - Channel management
    - Message routing
    - Error handling and retries

    Usage:
        engine = StreamingEngine(config)
        await engine.connect()

        # Subscribe to messages
        async def handle_telemetry(msg: StreamMessage):
            print(f"Received: {msg.payload}")

        await engine.subscribe(MessageType.TELEMETRY_RAW, handle_telemetry)

        # Publish messages
        await engine.publish(StreamMessage(
            message_type=MessageType.TELEMETRY_RAW,
            payload={"battery_level": 85},
            device_id=12345
        ))
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig.from_env()
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._handlers: dict[MessageType, list[MessageHandler]] = {}
        self._running = False
        self._listener_task: Optional[asyncio.Task] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def connect(self) -> None:
        """Connect to Redis and initialize pub/sub."""
        logger.info("Connecting to Redis: %s", self.config.redis_url)

        self._redis = redis.from_url(
            self.config.redis_url,
            db=self.config.redis_db,
            decode_responses=True,
        )

        # Test connection
        await self._redis.ping()
        logger.info("Connected to Redis successfully")

        self._pubsub = self._redis.pubsub()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_handlers)

    async def disconnect(self) -> None:
        """Disconnect from Redis and cleanup."""
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        logger.info("Disconnected from Redis")

    def _get_channel(self, message_type: MessageType) -> str:
        """Get the Redis channel name for a message type."""
        return f"{self.config.channel_prefix}{message_type.value}"

    async def publish(self, message: StreamMessage) -> int:
        """
        Publish a message to the appropriate channel.

        Args:
            message: The message to publish

        Returns:
            Number of subscribers that received the message
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        channel = self._get_channel(message.message_type)
        data = message.to_json()

        subscribers = await self._redis.publish(channel, data)

        logger.debug(
            "Published %s to %s (%d subscribers)",
            message.message_type.value,
            channel,
            subscribers,
        )

        return subscribers

    async def subscribe(
        self,
        message_type: MessageType,
        handler: MessageHandler,
    ) -> None:
        """
        Subscribe to a message type with a handler.

        Args:
            message_type: Type of messages to receive
            handler: Async function to handle messages
        """
        if message_type not in self._handlers:
            self._handlers[message_type] = []

            # Subscribe to Redis channel
            channel = self._get_channel(message_type)
            await self._pubsub.subscribe(channel)
            logger.info("Subscribed to channel: %s", channel)

        self._handlers[message_type].append(handler)
        logger.info(
            "Added handler for %s (total: %d)",
            message_type.value,
            len(self._handlers[message_type]),
        )

    async def unsubscribe(
        self,
        message_type: MessageType,
        handler: Optional[MessageHandler] = None,
    ) -> None:
        """
        Unsubscribe from a message type.

        Args:
            message_type: Type to unsubscribe from
            handler: Specific handler to remove, or None to remove all
        """
        if message_type not in self._handlers:
            return

        if handler:
            self._handlers[message_type].remove(handler)
        else:
            self._handlers[message_type] = []

        # Unsubscribe from channel if no handlers left
        if not self._handlers[message_type]:
            channel = self._get_channel(message_type)
            await self._pubsub.unsubscribe(channel)
            del self._handlers[message_type]
            logger.info("Unsubscribed from channel: %s", channel)

    async def start_listening(self) -> None:
        """Start the message listener loop."""
        if self._running:
            return

        self._running = True
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info("Started streaming engine listener")

    async def stop_listening(self) -> None:
        """Stop the message listener loop."""
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped streaming engine listener")

    async def _listen_loop(self) -> None:
        """Main loop for receiving messages."""
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message and message["type"] == "message":
                    await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in listener loop: %s", e, exc_info=True)
                await asyncio.sleep(0.1)

    async def _handle_message(self, raw_message: dict) -> None:
        """Handle a received message."""
        try:
            # Parse the message
            stream_message = StreamMessage.from_json(raw_message["data"])

            # Get handlers for this message type
            handlers = self._handlers.get(stream_message.message_type, [])

            if not handlers:
                logger.warning(
                    "No handlers for message type: %s",
                    stream_message.message_type.value,
                )
                return

            # Execute handlers concurrently with semaphore
            tasks = [
                self._execute_handler(handler, stream_message)
                for handler in handlers
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(
                "Error handling message: %s",
                e,
                exc_info=True,
            )

    async def _execute_handler(
        self,
        handler: MessageHandler,
        message: StreamMessage,
    ) -> None:
        """Execute a single handler with timeout and retry."""
        async with self._semaphore:
            retries = 0

            while retries <= self.config.max_retries:
                try:
                    await asyncio.wait_for(
                        handler(message),
                        timeout=self.config.handler_timeout_seconds,
                    )
                    return

                except asyncio.TimeoutError:
                    logger.warning(
                        "Handler timeout for %s (attempt %d/%d)",
                        message.message_type.value,
                        retries + 1,
                        self.config.max_retries + 1,
                    )
                except Exception as e:
                    logger.error(
                        "Handler error for %s: %s (attempt %d/%d)",
                        message.message_type.value,
                        e,
                        retries + 1,
                        self.config.max_retries + 1,
                    )

                retries += 1
                if retries <= self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_ms / 1000)

    # Convenience methods for specific message types

    async def publish_telemetry(
        self,
        device_id: int,
        telemetry: dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> int:
        """Publish raw telemetry data."""
        return await self.publish(StreamMessage(
            message_type=MessageType.TELEMETRY_RAW,
            payload=telemetry,
            device_id=device_id,
            tenant_id=tenant_id,
        ))

    async def publish_anomaly(
        self,
        device_id: int,
        anomaly_score: float,
        details: dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> int:
        """Publish an anomaly detection."""
        return await self.publish(StreamMessage(
            message_type=MessageType.ANOMALY_DETECTED,
            payload={
                "anomaly_score": anomaly_score,
                **details,
            },
            device_id=device_id,
            tenant_id=tenant_id,
        ))


# Singleton instance for convenience
_engine: Optional[StreamingEngine] = None


async def get_streaming_engine() -> StreamingEngine:
    """Get or create the global streaming engine instance."""
    global _engine
    if _engine is None:
        _engine = StreamingEngine()
        await _engine.connect()
    return _engine


async def shutdown_streaming_engine() -> None:
    """Shutdown the global streaming engine."""
    global _engine
    if _engine:
        await _engine.disconnect()
        _engine = None
