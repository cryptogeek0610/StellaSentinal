"""
WebSocket Manager - Real-time anomaly alerts to connected clients.

Pushes anomaly alerts to connected WebSocket clients for real-time
dashboard updates and notifications.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from device_anomaly.streaming.engine import (
    MessageType,
    StreamingEngine,
    StreamMessage,
)

logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""

    websocket: WebSocket
    client_id: str
    tenant_id: str | None = None
    subscribed_devices: set[int] = field(default_factory=set)
    subscribed_severity: set[str] = field(
        default_factory=lambda: {"low", "medium", "high", "critical"}
    )
    connected_at: datetime = field(default_factory=datetime.utcnow)
    messages_sent: int = 0


class WebSocketManager:
    """
    Manages WebSocket connections for real-time anomaly alerts.

    Features:
    - Multi-client support
    - Per-client filtering (by device, severity, tenant)
    - Automatic reconnection handling
    - Message queuing for slow clients

    Usage:
        manager = WebSocketManager(engine)
        await manager.start()

        # In FastAPI route:
        @app.websocket("/ws/anomalies")
        async def websocket_endpoint(websocket: WebSocket):
            await manager.connect(websocket)
    """

    def __init__(
        self,
        engine: StreamingEngine,
        max_queue_size: int = 100,
    ):
        self.engine = engine
        self.max_queue_size = max_queue_size
        self._connections: dict[str, ClientConnection] = {}
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start listening for anomaly alerts."""
        await self.engine.subscribe(
            MessageType.ANOMALY_DETECTED,
            self._handle_anomaly,
        )
        self._running = True
        logger.info("WebSocketManager started")

    async def stop(self) -> None:
        """Stop the WebSocket manager."""
        self._running = False
        await self.engine.unsubscribe(MessageType.ANOMALY_DETECTED)

        # Close all connections
        async with self._lock:
            for client in list(self._connections.values()):
                with contextlib.suppress(Exception):
                    await client.websocket.close()
            self._connections.clear()

        logger.info("WebSocketManager stopped")

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str | None = None,
        tenant_id: str | None = None,
    ) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Optional client identifier
            tenant_id: Optional tenant for multi-tenant filtering

        Returns:
            The assigned client ID
        """
        await websocket.accept()

        client_id = client_id or f"client_{len(self._connections)}_{datetime.utcnow().timestamp()}"

        connection = ClientConnection(
            websocket=websocket,
            client_id=client_id,
            tenant_id=tenant_id,
        )

        async with self._lock:
            self._connections[client_id] = connection

        logger.info("Client connected: %s (tenant: %s)", client_id, tenant_id)

        # Send welcome message
        await self._send_to_client(
            connection,
            {
                "type": "connected",
                "client_id": client_id,
                "message": "Connected to anomaly stream",
            },
        )

        return client_id

    async def disconnect(self, client_id: str) -> None:
        """Disconnect a client."""
        async with self._lock:
            if client_id in self._connections:
                del self._connections[client_id]
                logger.info("Client disconnected: %s", client_id)

    async def handle_client_messages(self, client_id: str) -> None:
        """
        Handle incoming messages from a client.

        Call this in a loop after connecting to handle client commands.
        """
        connection = self._connections.get(client_id)
        if not connection:
            return

        try:
            while self._running:
                message = await connection.websocket.receive_text()
                await self._process_client_message(connection, message)

        except WebSocketDisconnect:
            await self.disconnect(client_id)
        except Exception as e:
            logger.error("Error handling client %s: %s", client_id, e)
            await self.disconnect(client_id)

    async def _process_client_message(
        self,
        connection: ClientConnection,
        message: str,
    ) -> None:
        """Process a message from a client."""
        try:
            data = json.loads(message)
            command = data.get("command")

            if command == "subscribe_device":
                device_id = data.get("device_id")
                if device_id:
                    connection.subscribed_devices.add(int(device_id))
                    await self._send_to_client(
                        connection,
                        {
                            "type": "subscribed",
                            "device_id": device_id,
                        },
                    )

            elif command == "unsubscribe_device":
                device_id = data.get("device_id")
                if device_id:
                    connection.subscribed_devices.discard(int(device_id))
                    await self._send_to_client(
                        connection,
                        {
                            "type": "unsubscribed",
                            "device_id": device_id,
                        },
                    )

            elif command == "set_severity_filter":
                severities = data.get("severities", [])
                connection.subscribed_severity = set(severities)
                await self._send_to_client(
                    connection,
                    {
                        "type": "filter_updated",
                        "severities": list(connection.subscribed_severity),
                    },
                )

            elif command == "ping":
                await self._send_to_client(
                    connection,
                    {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

            elif command == "get_stats":
                await self._send_to_client(
                    connection,
                    {
                        "type": "stats",
                        "connected_at": connection.connected_at.isoformat(),
                        "messages_sent": connection.messages_sent,
                        "subscribed_devices": list(connection.subscribed_devices),
                        "subscribed_severity": list(connection.subscribed_severity),
                    },
                )

        except json.JSONDecodeError:
            await self._send_to_client(
                connection,
                {
                    "type": "error",
                    "message": "Invalid JSON",
                },
            )
        except Exception as e:
            await self._send_to_client(
                connection,
                {
                    "type": "error",
                    "message": str(e),
                },
            )

    async def _handle_anomaly(self, message: StreamMessage) -> None:
        """Handle incoming anomaly alert and broadcast to clients."""
        anomaly = message.payload
        device_id = anomaly.get("device_id")
        severity = anomaly.get("severity", "medium")
        tenant_id = message.tenant_id

        # Broadcast to matching clients
        async with self._lock:
            for connection in list(self._connections.values()):
                if self._should_send_to_client(connection, device_id, severity, tenant_id):
                    await self._send_to_client(
                        connection,
                        {
                            "type": "anomaly",
                            **anomaly,
                        },
                    )

    def _should_send_to_client(
        self,
        connection: ClientConnection,
        device_id: int,
        severity: str,
        tenant_id: str | None,
    ) -> bool:
        """Check if an anomaly should be sent to a client."""
        # Tenant filter
        if connection.tenant_id and connection.tenant_id != tenant_id:
            return False

        # Severity filter
        if severity not in connection.subscribed_severity:
            return False

        # Device filter (empty = all devices)
        return not (
            connection.subscribed_devices and device_id not in connection.subscribed_devices
        )

    async def _send_to_client(
        self,
        connection: ClientConnection,
        data: dict,
    ) -> bool:
        """Send a message to a specific client."""
        try:
            await connection.websocket.send_json(data)
            connection.messages_sent += 1
            return True
        except Exception as e:
            logger.warning(
                "Failed to send to client %s: %s",
                connection.client_id,
                e,
            )
            return False

    async def broadcast(
        self,
        data: dict,
        tenant_id: str | None = None,
    ) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            data: Message to send
            tenant_id: Optional tenant filter

        Returns:
            Number of clients that received the message
        """
        sent_count = 0

        async with self._lock:
            for connection in list(self._connections.values()):
                if tenant_id and connection.tenant_id != tenant_id:
                    continue

                if await self._send_to_client(connection, data):
                    sent_count += 1

        return sent_count

    def get_connection_count(self) -> int:
        """Get number of connected clients."""
        return len(self._connections)

    def get_stats(self) -> dict[str, Any]:
        """Get WebSocket manager statistics."""
        return {
            "running": self._running,
            "connection_count": len(self._connections),
            "total_messages_sent": sum(c.messages_sent for c in self._connections.values()),
            "connections": [
                {
                    "client_id": c.client_id,
                    "tenant_id": c.tenant_id,
                    "connected_at": c.connected_at.isoformat(),
                    "messages_sent": c.messages_sent,
                    "subscribed_devices": len(c.subscribed_devices),
                }
                for c in self._connections.values()
            ],
        }
