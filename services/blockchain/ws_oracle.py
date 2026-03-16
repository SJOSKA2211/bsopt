import asyncio
import time

import msgspec
import structlog
import websockets
from redis.asyncio import Redis

from core.shared.cache import get_redis

logger = structlog.get_logger(__name__)


class DexWebSocketOracle:
    """
    High-Frequency DEX Oracle via WebSockets.
    Subscribes to multiple DEX pairs and updates Redis for sub-millisecond access.
    """

    def __init__(self, feed_url: str = "wss://stream.binance.com:9443/ws"):
        self.feed_url = feed_url
        self._running = False
        self._pairs = []
        self.redis: Redis | None = None
        self._decoder = msgspec.json.Decoder()
        self._reconnect_attempt = 0

    async def start(self, pairs: list[str]):
        """Start the WebSocket listener for specified pairs with buffered updates."""
        self._pairs = pairs
        self._running = True
        self.redis = get_redis()

        # Initialize SHM for ultra-low latency local broadcast
        try:
            from core.shared.shm_mesh import SharedMemoryRingBuffer

            mesh = SharedMemoryRingBuffer(create=False)
        except Exception:
            mesh = None
            logger.warning("shm_mesh_unavailable_for_oracle_broadcast")

        while self._running:
            try:
                async with websockets.connect(self.feed_url) as ws:
                    subscribe_msg = {
                        "method": "SUBSCRIBE",
                        "params": [f"{p.lower()}@aggTrade" for p in pairs],
                        "id": 1,
                    }
                    await ws.send(msgspec.json.encode(subscribe_msg))
                    self._reconnect_attempt = 0  # Reset on successful connection

                    # Buffer for pipeline updates
                    buffer = []
                    last_flush = time.time()

                    async for message in ws:
                        try:
                            data = self._decoder.decode(message)
                            if "s" in data and "p" in data:
                                symbol = data["s"]
                                price = float(data["p"])

                                # 1.  ULTRA-SPEED: Local SHM Mesh
                                if mesh:
                                    mesh.write_tick(
                                        symbol, price, int(data.get("q", 0)), time.time()
                                    )

                                # 2.  PIPELINE BUFFER: Global Redis
                                buffer.append((symbol, price))

                                if len(buffer) >= 10 or (time.time() - last_flush) > 0.1:
                                    if self.redis:
                                        pipe = self.redis.pipeline()
                                        for s, p in buffer:
                                            pipe.set(f"price:ws:{s}", str(p))
                                            pipe.set(f"price:ws:{s}:ts", str(time.time()))
                                        await pipe.execute()
                                    buffer = []
                                    last_flush = time.time()

                        except Exception as e:
                            logger.error("dex_oracle_parse_error", error=str(e))
            except Exception as e:
                logger.error("dex_oracle_ws_error", error=str(e))
                self._reconnect_attempt += 1
                await asyncio.sleep(min(30, 2**self._reconnect_attempt))  # Exponential backoff

    async def stop(self):
        self._running = False
