import asyncio
import json
import time

import structlog
import websockets
from redis.asyncio import Redis

from src.utils.cache import get_redis

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

    async def start(self, pairs: list[str]):
        """Start the WebSocket listener for specified pairs."""
        self._pairs = pairs
        self._running = True
        self.redis = get_redis()

        while self._running:
            try:
                async with websockets.connect(self.feed_url) as ws:
                    # Binance example: subscribe to aggregate trades
                    subscribe_msg = {
                        "method": "SUBSCRIBE",
                        "params": [f"{p.lower()}@aggTrade" for p in pairs],
                        "id": 1,
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info("dex_oracle_subscribed", pairs=pairs)

                    async for message in ws:
                        data = json.loads(message)
                        if "s" in data and "p" in data:
                            symbol = data["s"]
                            price = data["p"]
                            # Update Redis for Hybrid Oracle consumption
                            if self.redis:
                                await self.redis.set(f"price:ws:{symbol}", price)
                                await self.redis.set(f"price:ws:{symbol}:ts", str(time.time()))
            except Exception as e:
                logger.error("dex_oracle_ws_error", error=str(e))
                await asyncio.sleep(5)  # Backoff

    async def stop(self):
        self._running = False
