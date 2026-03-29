import asyncio
import logging
import os
import signal
from datetime import datetime

import structlog

from src.database import get_async_db_context
from src.database.crud import bulk_insert_market_ticks
from src.shared.rabbitmq import get_rabbitmq

logger = structlog.get_logger(__name__)

class MarketDataConsumer:
    """
    High-performance RabbitMQ consumer for market ticks.
    """

    def __init__(self):
        self.rmq = get_rabbitmq()
        self.consumer_name = f"consumer_{os.uname().nodename}_{os.getpid()}"
        self.batch: list[dict] = []
        self.batch_size = 500  # Increased for RabbitMQ throughput
        self.flush_interval = 1.0  # seconds
        self._last_flush = datetime.now()
        self._lock = asyncio.Lock()

    async def handle_tick(self, data: dict) -> None:
        """Process tick data from RabbitMQ."""
        async with self._lock:
            # Ensure time is a datetime object
            if "time" in data and isinstance(data["time"], str):
                data["time"] = datetime.fromisoformat(data["time"].replace("Z", "+00:00"))
            
            self.batch.append(data)

            if len(self.batch) >= self.batch_size or \
               (datetime.now() - self._last_flush).total_seconds() >= self.flush_interval:
                await self.flush_batch()

    async def flush_batch(self) -> None:
        """Persist batched ticks to TimescaleDB via PgBouncer."""
        if not self.batch:
            return

        current_batch = self.batch
        self.batch = []
        self._last_flush = datetime.now()

        try:
            async with get_async_db_context() as db:
                # bulk_insert_market_ticks handles the mapping to Model
                count = await bulk_insert_market_ticks(db, current_batch)
                logger.info("market_ticks_persisted", count=count, consumer=self.consumer_name)
        except Exception as e:
            logger.error("persistence_failed", error=str(e))
            # Critical: In a real HFT system, we'd spill to local disk or a local cache here
            self.batch.extend(current_batch)

    async def run(self) -> None:
        """Start consumption loop."""
        logger.info("starting_rabbitmq_consumer", consumer=self.consumer_name)
        await self.rmq.connect()
        await self.rmq.consume_ticks(self.handle_tick)

async def main():
    consumer = MarketDataConsumer()
    
    def shutdown(sig, frame):
        logger.info("shutdown_signal_received")
        loop = asyncio.get_event_loop()
        loop.stop()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        await consumer.run()
    except Exception as e:
        logger.error("consumer_runtime_error", error=str(e))
    finally:
        await consumer.flush_batch()

if __name__ == "__main__":
    asyncio.run(main())
