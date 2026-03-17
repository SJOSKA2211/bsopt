"""
Market Data Consumer Service

Consumes real-time market ticks from Redis Streams and persists them to TimescaleDB.
"""

import asyncio
import os
import signal
from datetime import datetime

import structlog

from core.database import get_async_db_context
from core.database.crud import bulk_insert_market_ticks
from core.shared.redis_streams import RedisStreamManager

logger = structlog.get_logger(__name__)


class MarketDataConsumer:
    """
    High-performance consumer for market ticks.
    """

    def __init__(self, stream_name: str = "market_ticks_stream"):
        self.manager = RedisStreamManager(stream_name, "market_data_ingestors")
        self.consumer_name = f"consumer_{os.uname().nodename}_{os.getpid()}"
        self.batch: list[dict] = []
        self.batch_size = 100
        self.flush_interval = 2.0  # seconds
        self._last_flush = datetime.now()

    async def handle_tick(self, msg_id: str, data: dict) -> None:
        """
        Process a single market tick.
        Batches ticks for high-throughput persistence to TimescaleDB.
        """
        # Ensure time is a datetime object
        if "time" in data and isinstance(data["time"], str):
            data["time"] = datetime.fromisoformat(data["time"])

        self.batch.append(data)

        if (
            len(self.batch) >= self.batch_size
            or (datetime.now() - self._last_flush).total_seconds() >= self.flush_interval
        ):
            await self.flush_batch()

    async def flush_batch(self) -> None:
        """Persist batched ticks to PostgreSQL/TimescaleDB."""
        if not self.batch:
            return

        current_batch = self.batch
        self.batch = []
        self._last_flush = datetime.now()

        try:
            async with get_async_db_context() as db:
                count = await bulk_insert_market_ticks(db, current_batch)
                logger.info("market_ticks_persisted", count=count)
        except Exception as e:
            logger.error("market_ticks_persistence_failed", error=str(e))
            # Put back in batch or move to retry buffer
            self.batch.extend(current_batch)

    async def run(self) -> None:
        """Start the consumption loop."""
        logger.info("starting_market_data_consumer", consumer=self.consumer_name)
        await self.manager.consume(self.consumer_name, self.handle_tick)


if __name__ == "__main__":
    # Standalone execution logic
    consumer = MarketDataConsumer()

    loop = asyncio.get_event_loop()

    def shutdown():
        logger.info("shutdown_signal_received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, shutdown)

    try:
        loop.run_until_complete(consumer.run())
    except asyncio.CancelledError:
        pass
    finally:
        loop.run_until_complete(consumer.flush_batch())
        loop.close()
