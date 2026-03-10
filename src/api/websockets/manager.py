import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

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


WEBSOCKET_CONNECTIONS_TOTAL = _get_metric(
    Counter, "websocket_connections_total", "Total number of WebSocket connections"
)
WEBSOCKET_DISCONNECTIONS_TOTAL = _get_metric(
    Counter, "websocket_disconnections_total", "Total number of WebSocket disconnections"
)
WEBSOCKET_ACTIVE_CONNECTIONS = _get_metric(
    Gauge, "websocket_active_connections", "Current number of active WebSocket connections"
)
WEBSOCKET_MESSAGES_SENT_TOTAL = _get_metric(
    Counter, "websocket_messages_sent_total", "Total number of messages sent over WebSockets"
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
        self._shm_task: asyncio.Task | None = None
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

    async def _listen_to_shm(self):
        """High-frequency polling of SHM for market data."""
        import os

        if os.getenv("USE_SHM") != "1":
            return

        logger.info("ws_shm_listener_started")
        try:
            from src.shared.shm_manager import SHMManager

            shm = SHMManager("market_mesh", dict, size=50 * 1024 * 1024)
            last_seq = 0
            while True:
                try:
                    seq = shm.get_sequence()
                    if seq != last_seq:
                        last_seq = seq
                        data = shm.read()
                        for symbol, item in data.items():
                            if symbol in self.active_connections:
                                # Bypass Redis for SHM
                                asyncio.create_task(
                                    self.broadcast_to_symbol(
                                        symbol, item, from_redis=True, is_raw=False
                                    )
                                )
                        await asyncio.sleep(0)  # Yield to event loop, immediate poll
                    else:
                        await asyncio.sleep(0.001)  # 1ms poll when idle
                except FileNotFoundError:
                    await asyncio.sleep(0.05)
                except Exception:
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("ws_shm_listener_cancelled")

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
                    asyncio.create_task(
                        self.broadcast_to_symbol(symbol, raw_data, from_redis=True, is_raw=True)
                    )
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

    async def connect(self, websocket: WebSocket):
        """Accept connection and initialize metadata."""
        await websocket.accept()

        # Ensure metadata exists
        if not hasattr(websocket, "metadata"):
            websocket.metadata = ConnectionMetadata(protocol=ProtocolType.JSON)

        async with self._lock:
            # Lazy init background tasks
            if self._listener_task is None or self._listener_task.done():
                self._listener_task = asyncio.create_task(self._listen_to_redis())
            if self._shm_task is None or self._shm_task.done():
                self._shm_task = asyncio.create_task(self._listen_to_shm())

            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

        logger.info("ws_connected", client=str(websocket.client))
        WEBSOCKET_CONNECTIONS_TOTAL.inc()
        WEBSOCKET_ACTIVE_CONNECTIONS.inc()

    async def disconnect(self, websocket: WebSocket):
        """Handle disconnection and cleanup all symbol subscriptions for this websocket."""
        meta = getattr(websocket, "metadata", ConnectionMetadata())
        symbols = list(meta.subscriptions)

        for symbol in symbols:
            await self.unsubscribe_from_symbol(websocket, symbol)

        WEBSOCKET_DISCONNECTIONS_TOTAL.inc()
        WEBSOCKET_ACTIVE_CONNECTIONS.dec()
        logger.info("ws_disconnected", client=str(websocket.client))

    async def subscribe_to_symbol(self, websocket: WebSocket, symbol: str):
        """Subscribe a connection to a specific symbol updates."""
        symbol = symbol.upper()
        async with self._lock:
            if symbol not in self.active_connections:
                self.active_connections[symbol] = set()
                pubsub = await self._get_pubsub()
                if pubsub:
                    await pubsub.subscribe(symbol)

            self.active_connections[symbol].add(websocket)

            meta = getattr(websocket, "metadata", ConnectionMetadata())
            meta.subscriptions.add(symbol)

        logger.debug("ws_subscribed", symbol=symbol, client=str(websocket.client))

    async def unsubscribe_from_symbol(self, websocket: WebSocket, symbol: str):
        """Unsubscribe a connection from a specific symbol."""
        symbol = symbol.upper()
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

            meta = getattr(websocket, "metadata", ConnectionMetadata())
            meta.subscriptions.discard(symbol)

        logger.debug("ws_unsubscribed", symbol=symbol, client=str(websocket.client))

    async def close(self):
        """Shutdown the manager and cleanup all resources."""
        logger.info("ws_manager_shutting_down")

        if self._listener_task:
            self._listener_task.cancel()
        if self._shm_task:
            self._shm_task.cancel()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self._pubsub:
            try:
                await self._pubsub.close()
            except Exception:
                pass

        # Close all active connections
        async with self._lock:
            for symbol, connections in self.active_connections.items():
                for ws in list(connections):
                    try:
                        await ws.close(code=1001, reason="Server shutting down")
                    except Exception:
                        pass
            self.active_connections.clear()

        logger.info("ws_manager_shutdown_complete")

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

        #  HIGH-PERFORMANCE: Deliver to clients
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
