import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import orjson
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
    Optimized for broadcast using orjson and Redis Pub/Sub.
    """

    def __init__(self):
        # Store active connections: { "AAPL": [ws1, ws2], "GOOG": [ws3] }
        self.active_connections: dict[str, list[WebSocket]] = {}

        # Redis setup for cross-worker communication
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.redis = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        self.pubsub = self.redis.pubsub()
        self._listener_task = None

    async def _listen_to_redis(self):
        """Background task to listen for Redis messages and broadcast locally."""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    symbol = message["channel"]
                    data = orjson.loads(message["data"])
                    await self.broadcast_to_symbol(symbol, data, from_redis=True)
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
            websocket.metadata = ConnectionMetadata(protocol=ProtocolType.JSON)

        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
            # Subscribe to Redis topic for this symbol
            await self.pubsub.subscribe(symbol)

        self.active_connections[symbol].append(websocket)
        logger.info(
            "ws_connected", symbol=symbol, total=len(self.active_connections[symbol])
        )
        WEBSOCKET_CONNECTIONS_TOTAL.inc()  # Increment counter
        WEBSOCKET_ACTIVE_CONNECTIONS.inc()  # Increment gauge

    def disconnect(self, websocket: WebSocket, symbol: str):
        """Handle disconnection and cleanup."""
        if symbol in self.active_connections:
            if websocket in self.active_connections[symbol]:
                self.active_connections[symbol].remove(websocket)

            # Cleanup empty lists
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]
                # In production, we'd also unsubscribe from Redis if no workers are listening

        logger.info("ws_disconnected", symbol=symbol)
        WEBSOCKET_DISCONNECTIONS_TOTAL.inc()  # Increment counter
        WEBSOCKET_ACTIVE_CONNECTIONS.dec()  # Decrement gauge

    async def broadcast_to_symbol(
        self, symbol: str, message: Any, from_redis: bool = False
    ):
        """
        Send message to all users watching a specific ticker.
        Supports multi-protocol broadcasting (JSON, Proto, MsgPack).
        """
        # If message originates locally, publish to Redis for other workers
        if not from_redis:
            await self.redis.publish(symbol, orjson.dumps(message))
            # If we broadcast locally now, we'll get it back from Redis listener too
            # unless we add logic to skip. For now, let Redis be the source of truth.
            return

        if symbol not in self.active_connections:
            return

        connections = self.active_connections[symbol]
        if not connections:
            return

        # Group by protocol to encode ONCE per protocol
        by_protocol: dict[ProtocolType, list[WebSocket]] = {}
        for conn in connections:
            proto = getattr(conn, "metadata", ConnectionMetadata()).protocol
            if proto not in by_protocol:
                by_protocol[proto] = []
            by_protocol[proto].append(conn)

        tasks = []
        for proto, conns in by_protocol.items():
            try:
                encoded = WebSocketCodec.encode(message, proto)
                for conn in conns:
                    if isinstance(encoded, str):
                        tasks.append(conn.send_text(encoded))
                    else:
                        tasks.append(conn.send_bytes(encoded))
                WEBSOCKET_MESSAGES_SENT_TOTAL.inc(len(conns))
            except Exception as e:
                logger.error(
                    "ws_encode_error", symbol=symbol, protocol=proto, error=str(e)
                )
                continue

        if tasks:
            # Execute in chunks to avoid blocking the event loop for too long
            chunk_size = 100
            for i in range(0, len(tasks), chunk_size):
                await asyncio.gather(*tasks[i : i + chunk_size], return_exceptions=True)


# Global manager instance for reuse across routes
manager = ConnectionManager()
