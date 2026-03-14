import asyncio
import signal
from datetime import datetime

import structlog

from src.database import db_manager
from src.database.crud import bulk_insert_market_ticks
from src.streaming.kafka_consumer import MarketDataConsumer

logger = structlog.get_logger(__name__)

async def persist_ticks(batch: list[dict], topic: str):
    """
    Process a batch of ticks from Kafka and persist to TimescaleDB.
    """
    if not batch:
        return

    ticks_list = []
    for tick in batch:
        try:
            # Map Kafka MarketData schema to DB Tick schema with Lineage
            ticks_list.append(
                {
                    "symbol": tick.get("symbol"),
                    "price": float(tick.get("last") or tick.get("bid") or 0.0),
                    "volume": int(tick.get("volume") or 0),
                    "time": datetime.fromtimestamp(float(tick.get("time") or 0.0)),
                    "market": tick.get("source") or "NSE",
                    "source_id": f"kafka-{topic}-{tick.get('symbol')}",
                    "audit_trail": {
                        "original_source": tick.get("source"),
                        "raw_payload_checksum": hash(str(tick)) # Simple hash for lineage
                    }
                }
            )
        except Exception as e:
            from src.streaming.dlq import dlq_manager
            await dlq_manager.send_to_dlq(tick, str(e), topic)

    async with db_manager.get_async_session() as db:
        try:
            count = await bulk_insert_market_ticks(db, ticks_list)
            logger.info("kafka_ticks_persisted", count=count, topic=topic)
        except Exception as e:
            logger.error("kafka_persistence_failed", error=str(e), topic=topic)
            # In case of bulk failure, we could retry individually or DLQ the batch
            from src.streaming.dlq import dlq_manager
            for tick in batch:
                await dlq_manager.send_to_dlq(tick, f"Bulk failure: {str(e)}", topic)

async def main():
    consumer = MarketDataConsumer(
        bootstrap_servers="kafka-1:9092",
        group_id="persistence-group",
        topics=["market-data"]
    )

    # Graceful shutdown handling
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def shutdown():
        logger.info("shutdown_signal_received")
        stop_event.set()
        consumer.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown)

    logger.info("kafka_persistence_worker_starting")

    try:
        # Start consumption task
        await consumer.consume_messages(persist_ticks, batch_size=200)
    except Exception as e:
        logger.error("persistence_worker_runtime_error", error=str(e))
    finally:
        logger.info("kafka_persistence_worker_stopped")

if __name__ == "__main__":
    asyncio.run(main())
