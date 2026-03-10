"""
Market Mesh Publisher

Writes real-time scraped market data to shared memory for zero-copy access.
"""

import time
import structlog

from src.shared.shm_mesh import SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)


class MarketMeshPublisher:
    """
    Publishes market data to the lock-free SharedMemoryRingBuffer.
    """

    def __init__(self):
        # Ultra-high-performance ring buffer for market ticks
        self.mesh = SharedMemoryRingBuffer(create=True)

    def publish(self, data: dict):
        """Write ticker data to the Ring Buffer."""
        try:
            # data is { symbol: {price, volume, ...} }
            count = 0
            for symbol, tick in data.items():
                self.mesh.write_tick(
                    symbol,
                    float(tick.get("price", 0.0)),
                    int(tick.get("volume", 0)),
                    float(time.time()),  # Use current server time for the mesh
                )
                count += 1
            logger.debug("market_data_published_to_ring", count=count)
        except Exception as e:
            logger.error("market_publish_failed", error=str(e))


_publisher = None


def get_market_publisher():
    global _publisher
    if _publisher is None:
        _publisher = MarketMeshPublisher()
    return _publisher
