import asyncio
import signal

import structlog

from src.database import db_manager
from src.database.crud import bulk_insert_market_ticks
from src.streaming.rabbitmq_consumer import RabbitMQMarketDataConsumer

logger = structlog.get_logger(__name__)


async def persist_ticks(data: dict):
    """
    Process a batch of ticks and persist to TimescaleDB.
    Expected data format: { symbol: {price, volume, timestamp, ...}, ... }
    """
    if not data:
        return

    # Convert dict format to list of dicts for bulk insert
    ticks_list = []
    for symbol, tick in data.items():
        # Handle both 'time' and 'timestamp' keys for backward compatibility
        ts = tick.get("time") or tick.get("timestamp")

        ticks_list.append(
            {
                "symbol": symbol,
                "price": float(tick.get("price", 0.0)),
                "volume": int(tick.get("volume", 0)),
                "time": ts,  # TimescaleDB primary time column
                "market": tick.get("market", "NSE"),
            }
        )

    async with db_manager.get_async_session() as db:
        try:
            count = await bulk_insert_market_ticks(db, ticks_list)
            logger.info("market_ticks_persisted", count=count)
        except Exception as e:
            logger.error("market_ticks_persistence_failed", error=str(e))
            raise  # Re-raise to trigger RabbitMQ NACK/DLQ


async def main():
    consumer = RabbitMQMarketDataConsumer()

    # Graceful shutdown handling
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def shutdown():
        logger.info("shutdown_signal_received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown)

    logger.info("persistence_worker_starting")

    try:
        # Start consumption task
        consume_task = asyncio.create_task(consumer.consume(persist_ticks))

        # Wait for stop signal
        await stop_event.wait()

        # Cancel consumption
        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

    finally:
        await consumer.close()
        logger.info("persistence_worker_stopped")


if __name__ == "__main__":
    asyncio.run(main())
