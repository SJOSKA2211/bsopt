import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import redis.asyncio as redis
import structlog
from fastapi import WebSocket
from prometheus_client import REGISTRY, Counter, Gauge  # Import Prometheus client metrics

from .codec import ProtocolType, WebSocketCodec

logger = structlog.get_logger()

# Prometheus Metrics (Idempotent for tests)
def _get_metric(cls, name, documentation):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return cls(name, documentation)

WEBSOCKET_CONNECTIONS_TOTAL = _get_metric(Counter,
    "websocket_connections_total", "Total number of WebSocket connections"
)
WEBSOCKET_DISCONNECTIONS_TOTAL = _get_metric(Counter,
    "websocket_disconnections_total", "Total number of WebSocket disconnections"
)
WEBSOCKET_ACTIVE_CONNECTIONS = _get_metric(Gauge,
    "websocket_active_connections", "Current number of active WebSocket connections"
)
WEBSOCKET_MESSAGES_SENT_TOTAL = _get_metric(Counter,
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
        self._listener_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._pubsub = None
        self._lock = asyncio.Lock()

    async def _get_pubsub(self):
        if self._pubsub is None:
            from src.utils.cache import get_redis
            redis_client = get_redis()
            if redis_client:
                self._pubsub = redis_client.pubsub()
        return self._pubsub

    async def _listen_to_redis(self):
        """Background task to listen for Redis messages and broadcast locally."""
        pubsub = await self._get_pubsub()
        if not pubsub:
            logger.error("ws_redis_pubsub_unavailable")
            return

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    symbol = channel.decode("utf-8") if isinstance(channel, bytes) else channel
                    raw_data = message["data"]
                    # Offload broadcast to avoid blocking listener
                    asyncio.create_task(self.broadcast_to_symbol(symbol, raw_data, from_redis=True, is_raw=True))
        except asyncio.CancelledError:
            logger.info("ws_redis_listener_cancelled")
        except Exception as e:
            logger.error("ws_redis_listener_error", error=str(e))
        finally:
            self._listener_task = None

    async def _heartbeat_monitor(self):
        """Prune dead connections based on last heartbeat."""
        while True:
            await asyncio.sleep(30)  # Check every 30s
            now = datetime.utcnow()
            to_prune = []

            async with self._lock:
                for symbol, connections in self.active_connections.items():
                    for ws in list(connections):
                        meta = getattr(ws, "metadata", ConnectionMetadata())
                        if (now - meta.last_heartbeat).total_seconds() > 60:
                            to_prune.append((ws, symbol))

            for ws, symbol in to_prune:
                logger.warning("ws_heartbeat_timeout", symbol=symbol)
                try:
                    await ws.close(code=1001, reason="Heartbeat timeout")
                except Exception:
                    pass
                await self.disconnect(ws, symbol)

    async def connect(self, websocket: WebSocket, symbol: str):
        """Accept connection and subscribe to symbol updates."""
        await websocket.accept()

        # Ensure metadata exists
        if not hasattr(websocket, "metadata"):
            websocket.metadata = ConnectionMetadata(protocol=ProtocolType.MSGPACK)

        async with self._lock:
            # Lazy init background tasks
            if self._listener_task is None or self._listener_task.done():
                self._listener_task = asyncio.create_task(self._listen_to_redis())
            
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            pubsub = await self._get_pubsub()
            if pubsub:
                if symbol not in self.active_connections:
                    self.active_connections[symbol] = set()
                    await pubsub.subscribe(symbol)
            else:
                logger.warning("ws_running_without_redis_synchronization", symbol=symbol)
                if symbol not in self.active_connections:
                    self.active_connections[symbol] = set()

            self.active_connections[symbol].add(websocket)
        
        logger.info("ws_connected", symbol=symbol, total=len(self.active_connections[symbol]))
        WEBSOCKET_CONNECTIONS_TOTAL.inc()
        WEBSOCKET_ACTIVE_CONNECTIONS.inc()

    async def disconnect(self, websocket: WebSocket, symbol: str):
        """Handle disconnection and cleanup in O(1)."""
        async with self._lock:
            if symbol in self.active_connections:
                self.active_connections[symbol].discard(websocket)
                if not self.active_connections[symbol]:
                    del self.active_connections[symbol]
                    pubsub = await self._get_pubsub()
                    if pubsub:
                        try:
                            await pubsub.unsubscribe(symbol)
                        except Exception as e:
                            logger.warning("ws_unsubscribe_failed", symbol=symbol, error=str(e))

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
            from src.utils.cache import get_redis
            redis_client = get_redis()
            if redis_client:
                payload = WebSocketCodec.encode(message, ProtocolType.MSGPACK)
                await redis_client.publish(symbol, payload)
            # Local broadcast will happen via Redis Pub/Sub listener to ensure consistency
            return

        # Use local copy of connections to minimize lock contention
        async with self._lock:
            if symbol not in self.active_connections:
                return
            connections = list(self.active_connections[symbol])

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
                        WebSocketCodec.decode(message, ProtocolType.MSGPACK) if is_raw else message
                    )
                    encoded = WebSocketCodec.encode(data, proto)

                for conn in conns:
                    # Send as bytes regardless of protocol for maximum speed
                    tasks.append(conn.send_bytes(encoded))

                WEBSOCKET_MESSAGES_SENT_TOTAL.inc(len(conns))
            except Exception as e:
                logger.error("ws_encode_error", symbol=symbol, protocol=proto, error=str(e))
                continue

        if tasks:
            # Gather with return_exceptions to prevent one bad connection from killing the broadcast
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.debug("ws_send_failed", symbol=symbol, error=str(res))
                    # Connection likely dead, will be pruned by heartbeat or explicit disconnect


# Global manager instance for reuse across routes
manager = ConnectionManager()
