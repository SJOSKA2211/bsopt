"""
Market Mesh Publisher
=====================

Writes real-time scraped market data to shared memory for zero-copy access.
"""

from src.shared.shm_manager import SHMManager
import structlog
from typing import Dict

logger = structlog.get_logger(__name__)

class MarketMeshPublisher:
    """
    Publishes market data to the 'market_mesh' shared memory segment.
    """
    def __init__(self):
        # 50MB buffer for market ticks
        self.shm = SHMManager("market_mesh", dict, size=50 * 1024 * 1024)
        self.shm.create()

    def publish(self, data: Dict):
        """Write ticker data to SHM."""
        try:
            self.shm.write(data)
            logger.debug("market_data_published", count=len(data))
        except Exception as e:
            logger.error("market_publish_failed", error=str(e))

_publisher = None

def get_market_publisher():
    global _publisher
    if _publisher is None:
        _publisher = MarketMeshPublisher()
    return _publisher
