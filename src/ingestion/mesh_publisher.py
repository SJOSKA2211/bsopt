"""
Market Mesh Publisher (RabbitMQ Bridge)

Publishes real-time scraped market data to RabbitMQ for decoupled processing.
"""

import time

import structlog

from src.shared.rabbitmq import get_rabbitmq

logger = structlog.get_logger(__name__)

class MarketMeshPublisher:
    """
    Publishes market data to RabbitMQ.
    """

    def __init__(self):
        self.rmq = get_rabbitmq()

    async def publish(self, data: dict):
        """Write ticker data to RabbitMQ."""
        try:
            # data is { symbol: {price, volume, ...} }
            count = 0
            for symbol, tick in data.items():
                payload = {
                    "symbol": symbol,
                    "price": float(tick.get("price", 0.0)),
                    "volume": int(tick.get("volume", 0)),
                    "time": tick.get("time", time.time()),
                    "side": tick.get("side", 0) # 0: Unknown, 1: Buy, 2: Sell
                }
                await self.rmq.publish_tick(payload)
                count += 1
            logger.debug("market_data_published_to_queue", count=count)
        except Exception as e:
            logger.error("market_publish_failed", error=str(e))

_publisher = None

def get_market_publisher():
    global _publisher
    if _publisher is None:
        _publisher = MarketMeshPublisher()
    return _publisher
