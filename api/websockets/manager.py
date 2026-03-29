import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import structlog
from fastapi import WebSocket
from prometheus_client import REGISTRY, Counter, Gauge  # Import Prometheus client metrics

from .codec import ProtocolType, WebSocketCodec
from src.shared.config import settings

logger = structlog.get_logger()

# Prometheus Metrics (Idempotent for tests)
def _get_metric(cls: Any, name: str, documentation: str) -> Any:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return cls(name, documentation)

WEBSOCKET_CONNECTIONS_TOTAL: Counter = _get_metric(
    Counter, "websocket_connections_total", "Total number of WebSocket connections"
)
WEBSOCKET_DISCONNECTIONS_TOTAL: Counter = _get_metric(
    Counter, "websocket_disconnections_total", "Total number of WebSocket disconnections"
)
WEBSOCKET_ACTIVE_CONNECTIONS: Gauge = _get_metric(
    Gauge, "websocket_active_connections", "Current number of active WebSocket connections"
)
WEBSOCKET_MESSAGES_SENT_TOTAL: Counter = _get_metric(
    Counter, "websocket_messages_sent_total", "Total number of messages sent over WebSockets"
)

@dataclass
class ConnectionMetadata:
    user_id: str | None = None
    protocol: ProtocolType = ProtocolType.JSON
    subscriptions: set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)

    def update_heartbeat(self) -> None:
        self.last_heartbeat = datetime.utcnow()

class ConnectionManager:
    """
    High-performance WebSocket connection manager for C100k.
    OPTIMIZED: O(1) connection management and binary-aware delivery.
    """

    def __init__(self) -> None:
        # Store active connections: { "AAPL": {ws1, ws2}, "GOOG": {ws3} }
        self.active_connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._listener_task: asyncio.Task[None] | None = None
        self._shm_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._pubsub: Any | None = None
        self._lock = asyncio.Lock()

    async def _get_pubsub(self) -> Any:
        if self._pubsub is None:
            from src.shared.utils.cache import get_redis

            redis_client = get_redis()
            if redis_client:
                self._pubsub = redis_client.pubsub()
        return self._pubsub

    async def _listen_to_shm(self) -> None:
        """High-frequency polling of Ring Buffers for market and Greeks data."""
        if not settings.USE_SHM:
            return

        logger.info("ws_shm_listener_started_multi_ring")
        try:
            from src.shared.shm_mesh import GreeksBuffer, SharedMemoryRingBuffer

            mesh = SharedMemoryRingBuffer(create=False)
            g_mesh = GreeksBuffer(create=False)
            last_head = 0
            last_g_head = 0

            while True:
                try:
                    # 1. Market Data Poll
                    ticks, new_head = mesh.read_latest_msgspec(last_head)
                    if new_head > last_head:
                        last_head = new_head
                        for tick in ticks:
                            if tick.symbol in self.active_connections:
                                asyncio.create_task(
                                    self.broadcast_to_symbol(tick.symbol, tick, from_redis=True)
                                )

                    # 2. Greeks Data Poll
                    import struct

                    g_head_tuple = struct.unpack_from("q", g_mesh.buf, 0)
                    g_head = cast(int, g_head_tuple[0])
                    if g_head > last_g_head:
                        # Read new Greeks
                        for i in range(last_g_head, g_head):
                            idx = i % 1000  # GREEKS_BUFFER_CAPACITY
                            data = g_mesh.view[idx]
                            symbol = data["symbol"].decode("ascii").strip("\x00")
                            g_channel = f"GREEKS:{symbol}"
                            if g_channel in self.active_connections:
                                asyncio.create_task(
                                    self.broadcast_to_symbol(
                                        g_channel,
                                        {
                                            "delta": float(data["delta"]),
                                            "gamma": float(data["gamma"]),
                                            "theta": float(data["theta"]),
                                            "vega": float(data["vega"]),
                                            "rho": float(data["rho"]),
                                            "timestamp": int(data["calc_ts_ns"]),
                                        },
                                        from_redis=True,
                                    )
                                )
                        last_g_head = g_head

                    await asyncio.sleep(
                        0 if new_head > last_head or g_head > last_g_head else 0.001
                    )
                except Exception as e:
                    logger.debug("shm_multi_poll_error", error=str(e))
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("ws_shm_listener_cancelled")

    async def _listen_to_redis(self) -> None:
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

    async def _heartbeat_monitor(self) -> None:
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
                except Exception as e:
                    logger.debug("ws_close_error", error=str(e))
                await self.disconnect(ws, symbol)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept connection and initialize metadata."""
        await websocket.accept()

        # Ensure metadata exists
        if not hasattr(websocket, "metadata"):
            setattr(websocket, "metadata", ConnectionMetadata(protocol=ProtocolType.JSON))

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

    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle disconnection and cleanup all symbol subscriptions for this websocket."""
        meta = cast(ConnectionMetadata, getattr(websocket, "metadata", ConnectionMetadata()))
        symbols = list(meta.subscriptions)

        for symbol in symbols:
            await self.unsubscribe_from_symbol(websocket, symbol)

        WEBSOCKET_DISCONNECTIONS_TOTAL.inc()
        WEBSOCKET_ACTIVE_CONNECTIONS.dec()
        logger.info("ws_disconnected", client=str(websocket.client))

    async def subscribe_to_symbol(self, websocket: WebSocket, symbol: str) -> None:
        """Subscribe a connection to a specific symbol updates."""
        symbol = symbol.upper()
        async with self._lock:
            if symbol not in self.active_connections:
                pubsub = await self._get_pubsub()
                if pubsub:
                    await pubsub.subscribe(symbol)

            self.active_connections[symbol].add(websocket)

            meta = cast(ConnectionMetadata, getattr(websocket, "metadata", ConnectionMetadata()))
            meta.subscriptions.add(symbol)

        logger.debug("ws_subscribed", symbol=symbol, client=str(websocket.client))

    async def unsubscribe_from_symbol(self, websocket: WebSocket, symbol: str) -> None:
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

            meta = cast(ConnectionMetadata, getattr(websocket, "metadata", ConnectionMetadata()))
            meta.subscriptions.discard(symbol)

        logger.debug("ws_unsubscribed", symbol=symbol, client=str(websocket.client))

    async def close(self) -> None:
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
            except Exception as e:
                logger.debug("ws_resource_cleanup_error", error=str(e))

        # Close all active connections
        async with self._lock:
            for symbol, connections in self.active_connections.items():
                for ws in list(connections):
                    try:
                        await ws.close(code=1001, reason="Server shutting down")
                    except Exception as e:
                        logger.debug("ws_connection_close_error", error=str(e))
            self.active_connections.clear()

        logger.info("ws_manager_shutdown_complete")

    async def broadcast_to_symbol(
        self, symbol: str, message: Any, from_redis: bool = False, is_raw: bool = False
    ) -> None:
        """
        Send message to all users watching a specific ticker.
        OPTIMIZED: Multi-protocol delivery with minimal serialization overhead.
        """
        if not from_redis:
            # Originating locally: Encode to binary once and push to Redis
            from src.shared.utils.cache import get_redis

            redis_client = get_redis()
            if redis_client:
                payload = WebSocketCodec.encode(message, ProtocolType.MSGPACK)
                await redis_client.publish(symbol, payload)
            # Local broadcast will happen via Redis Pub/Sub listener to ensure consistency
            return

        connections = self.active_connections.get(symbol)
        if not connections:
            return

        targets = list(connections)
        if not targets:
            return

        #  HIGH-PERFORMANCE: Group by protocol to avoid redundant encoding
        by_protocol: dict[ProtocolType, list[WebSocket]] = {}
        for conn in targets:
            meta = cast(ConnectionMetadata, getattr(conn, "metadata", ConnectionMetadata()))
            proto = meta.protocol
            if proto not in by_protocol:
                by_protocol[proto] = []
            by_protocol[proto].append(conn)

        tasks = []
        decoded_data = None

        for proto, conns in by_protocol.items():
            try:
                # Optimized Encoding: Encode once per protocol
                if is_raw and proto == ProtocolType.MSGPACK:
                    encoded = message  # Pass-through bytes
                elif proto == ProtocolType.PROTO:
                    # Dynamically select message type based on symbol or channel
                    from src.protos import market_data_pb2

                    if decoded_data is None and is_raw:
                        decoded_data = WebSocketCodec.decode(message, ProtocolType.MSGPACK)

                    data = decoded_data if is_raw else message

                    # Assume TickerUpdate for simplicity, or handle based on symbol/prefix
                    pb_msg = market_data_pb2.TickerUpdate(
                        symbol=data.get("symbol", symbol), price=float(data.get("price", 0.0))
                    )
                    encoded = WebSocketCodec.encode(pb_msg, ProtocolType.PROTO)
                else:
                    if decoded_data is None and is_raw:
                        decoded_data = WebSocketCodec.decode(message, ProtocolType.MSGPACK)

                    data = decoded_data if is_raw else message
                    encoded = WebSocketCodec.encode(data, proto)

                for conn in conns:
                    # Exponential Backoff with Jitter for sending messages
                    async def send_with_backoff(c: WebSocket, d: bytes) -> None:
                        import random

                        retries = 3
                        base_delay = 0.1
                        for attempt in range(retries):
                            try:
                                await c.send_bytes(d)
                                return
                            except Exception as ex:
                                if attempt == retries - 1:
                                    raise ex
                                # Exponential backoff with jitter: (2^attempt * base_delay) + random_jitter
                                jitter = random.uniform(0, 0.1)
                                await asyncio.sleep((2**attempt * base_delay) + jitter)

                    tasks.append(send_with_backoff(conn, encoded))

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

# Global manager instance for reuse across routes
manager = ConnectionManager()
