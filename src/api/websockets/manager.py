import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import redis.asyncio as redis
import structlog
from fastapi import WebSocket
from prometheus_client import Counter, Gauge  # Import Prometheus client metrics

from .codec import ProtocolType, WebSocketCodec

logger = structlog.get_logger()

# Prometheus Metrics
WEBSOCKET_CONNECTIONS_TOTAL = Counter(
    "websocket_connections_total", "Total number of WebSocket connections"
)
WEBSOCKET_DISCONNECTIONS_TOTAL = Counter(
    "websocket_disconnections_total", "Total number of WebSocket disconnections"
)
WEBSOCKET_ACTIVE_CONNECTIONS = Gauge(
    "websocket_active_connections", "Current number of active WebSocket connections"
)
WEBSOCKET_MESSAGES_SENT_TOTAL = Counter(
    "websocket_messages_sent_total", "Total number of messages sent over WebSockets"
)


@dataclass
class ConnectionMetadata:
    user_id: str | None = None
    protocol: ProtocolType = ProtocolType.JSON
    subscriptions: set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)

    def update_heartbeat(self):
        self.last_heartbeat = datetime.utcnow()


class ConnectionManager:
    """
    High-performance WebSocket connection manager for C100k.
    OPTIMIZED: O(1) connection management and binary-aware delivery.
    """

    def __init__(self):
        # Store active connections: { "AAPL": {ws1, ws2}, "GOOG": {ws3} }
        self.active_connections: dict[str, set[WebSocket]] = {}

        # Redis setup for cross-worker communication
        # Use service name 'redis' if running inside docker, otherwise 'localhost'
        is_docker = os.getenv("INSIDE_DOCKER") == "1"
        redis_fallback = (
            "redis://redis:6379/0" if is_docker else "redis://localhost:6379/0"
        )

        redis_url = os.environ.get("REDIS_URL") or redis_fallback
        self.redis = redis.from_url(redis_url, encoding=None, decode_responses=False)
        self.pubsub = self.redis.pubsub()
        self._listener_task = None

    async def _listen_to_redis(self):
        """Background task to listen for Redis messages and broadcast locally."""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    symbol = (
                        channel.decode("utf-8")
                        if isinstance(channel, bytes)
                        else channel
                    )
                    raw_data = message["data"]
                    await self.broadcast_to_symbol(
                        symbol, raw_data, from_redis=True, is_raw=True
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("ws_redis_listener_error", error=str(e))

    async def connect(self, websocket: WebSocket, symbol: str):
        """Accept connection and subscribe to symbol updates."""
        await websocket.accept()

        # Start listener if not running
        if self._listener_task is None:
            self._listener_task = asyncio.create_task(self._listen_to_redis())

        # Ensure metadata exists
        if not hasattr(websocket, "metadata"):
            websocket.metadata = ConnectionMetadata(protocol=ProtocolType.MSGPACK)

        if symbol not in self.active_connections:
            self.active_connections[symbol] = set()
            await self.pubsub.subscribe(symbol)

        self.active_connections[symbol].add(websocket)
        logger.info(
            "ws_connected", symbol=symbol, total=len(self.active_connections[symbol])
        )
        WEBSOCKET_CONNECTIONS_TOTAL.inc()
        WEBSOCKET_ACTIVE_CONNECTIONS.inc()

    def disconnect(self, websocket: WebSocket, symbol: str):
        """Handle disconnection and cleanup in O(1)."""
        if symbol in self.active_connections:
            self.active_connections[symbol].discard(websocket)
            if not self.active_connections[symbol]:
                # In production, we might want to delay unsubscribe to avoid thrashing
                del self.active_connections[symbol]

        WEBSOCKET_DISCONNECTIONS_TOTAL.inc()
        WEBSOCKET_ACTIVE_CONNECTIONS.dec()

    async def broadcast_to_symbol(
        self, symbol: str, message: Any, from_redis: bool = False, is_raw: bool = False
    ):
        """
        Send message to all users watching a specific ticker.
        OPTIMIZED: Multi-protocol delivery with minimal serialization overhead.
        """
        if not from_redis:
            # Originating locally: Encode to binary once and push to Redis
            payload = WebSocketCodec.encode(message, ProtocolType.MSGPACK)
            await self.redis.publish(symbol, payload)
            return

        if symbol not in self.active_connections:
            return

        connections = self.active_connections[symbol]
        if not connections:
            return

        #  GOD MODE: Deliver to clients
        by_protocol: dict[ProtocolType, list[WebSocket]] = {}
        for conn in connections:
            proto = getattr(conn, "metadata", ConnectionMetadata()).protocol
            if proto not in by_protocol:
                by_protocol[proto] = []
            by_protocol[proto].append(conn)

        tasks = []
        for proto, conns in by_protocol.items():
            try:
                # Optimized Encoding
                if is_raw and proto == ProtocolType.MSGPACK:
                    encoded = message  # Pass-through bytes
                else:
                    data = (
                        WebSocketCodec.decode(message, ProtocolType.MSGPACK)
                        if is_raw
                        else message
                    )
                    encoded = WebSocketCodec.encode(data, proto)

                for conn in conns:
                    # Send as bytes regardless of protocol for maximum speed
                    tasks.append(conn.send_bytes(encoded))

                WEBSOCKET_MESSAGES_SENT_TOTAL.inc(len(conns))
            except Exception as e:
                logger.error(
                    "ws_encode_error", symbol=symbol, protocol=proto, error=str(e)
                )
                continue

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Global manager instance for reuse across routes
manager = ConnectionManager()
