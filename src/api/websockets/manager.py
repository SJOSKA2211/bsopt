from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import WebSocket
import orjson
import redis.asyncio as redis
import asyncio
import structlog
import os
from .codec import ProtocolType, WebSocketCodec
from prometheus_client import Counter, Gauge # Import Prometheus client metrics

logger = structlog.get_logger()

# Prometheus Metrics
WEBSOCKET_CONNECTIONS_TOTAL = Counter('websocket_connections_total', 'Total number of WebSocket connections')
WEBSOCKET_DISCONNECTIONS_TOTAL = Counter('websocket_disconnections_total', 'Total number of WebSocket disconnections')
WEBSOCKET_ACTIVE_CONNECTIONS = Gauge('websocket_active_connections', 'Current number of active WebSocket connections')
WEBSOCKET_MESSAGES_SENT_TOTAL = Counter('websocket_messages_sent_total', 'Total number of messages sent over WebSockets')

@dataclass
class ConnectionMetadata:
    user_id: Optional[str] = None
    protocol: ProtocolType = ProtocolType.JSON
    subscriptions: Set[str] = field(default_factory=set)
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
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
        # Redis setup for cross-worker communication
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.redis = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        self.pubsub = self.redis.pubsub()

    async def connect(self, websocket: WebSocket, symbol: str):
        """Accept connection and subscribe to symbol updates."""
        await websocket.accept()
        
        # Ensure metadata exists
        if not hasattr(websocket, "metadata"):
             websocket.metadata = ConnectionMetadata(protocol=ProtocolType.JSON)

        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
            # Subscribe to Redis topic for this symbol
            await self.pubsub.subscribe(symbol)
            
        self.active_connections[symbol].append(websocket)
        logger.info("ws_connected", symbol=symbol, total=len(self.active_connections[symbol]))
        WEBSOCKET_CONNECTIONS_TOTAL.inc() # Increment counter
        WEBSOCKET_ACTIVE_CONNECTIONS.inc() # Increment gauge

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
        WEBSOCKET_DISCONNECTIONS_TOTAL.inc() # Increment counter
        WEBSOCKET_ACTIVE_CONNECTIONS.dec() # Decrement gauge

    async def broadcast_to_symbol(self, symbol: str, message: Any):
        """
        Send message to all users watching a specific ticker.
        Supports multi-protocol broadcasting (JSON, Proto, MsgPack).
        """
        if symbol not in self.active_connections:
            return

        connections = self.active_connections[symbol]
        if not connections:
            return

        # Group by protocol to encode ONCE per protocol
        by_protocol: Dict[ProtocolType, List[WebSocket]] = {}
        for conn in connections:
            # Assume metadata exists (set in connect)
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
                WEBSOCKET_MESSAGES_SENT_TOTAL.inc(len(conns)) # Increment counter by number of recipients
            except Exception as e:
                logger.error("ws_encode_error", symbol=symbol, protocol=proto, error=str(e))
                continue

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# Global manager instance for reuse across routes
manager = ConnectionManager()