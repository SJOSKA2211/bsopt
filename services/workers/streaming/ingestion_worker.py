import asyncio
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI

from services.api.responses import MsgspecJSONResponse
from services.api.websockets.manager import manager as ws_manager
from core.data.xdp_ingest import XDPIngester
from core.database.pipeliner import db_engine
from core.shared.eternal_ledger import EternalLedger
from core.shared.observability import setup_logging, tune_gc
from services.workers.streaming.kafka_consumer import MarketDataConsumer

# Optimized event loop
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


setup_logging()
logger = structlog.get_logger(__name__)
tune_gc(mode="high_frequency")  # Optimized for high-frequency trading workers


class BroadcastWorker:
    """The Voice: Dedicated to zero-latency WebSocket broadcasting."""

    def __init__(self):
        self.queue = asyncio.Queue(maxsize=1000)
        self.running = False

    async def run(self):
        self.running = True
        logger.info("broadcast_worker_active")
        while self.running:
            batch = await self.queue.get()
            try:
                # OPTIMIZED: Parallel broadcast for the entire batch
                tasks = [ws_manager.broadcast_to_symbol(item["symbol"], item) for item in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error("broadcast_batch_failed", error=str(e))
            finally:
                self.queue.task_done()


class PersistenceWorker:
    """The Scribe: Dedicated to high-throughput DB persistence using Vectorized COPY."""

    def __init__(self):
        self.market_queue = asyncio.Queue(maxsize=5000)
        self.audit_queue = asyncio.Queue(maxsize=5000)
        self.running = False
        self.ledger = EternalLedger()

    async def run(self):
        self.running = True
        logger.info("persistence_worker_active")

        # Launch specialized persistence tasks
        await asyncio.gather(
            self._persist_market_data(),
            self._persist_audit_logs(),
        )

    async def _persist_market_data(self):
        async with db_engine as db:
            while self.running:
                batch = await self.market_queue.get()
                try:
                    # 1. HOT PATH: Persistent Binary Ledger (Zero-latency local disk)
                    self.ledger.write_batch(batch)

                    # 2. Historical DB Persistence (Binary COPY protocol)
                    now_utc = datetime.now(UTC)
                    today_date = now_utc.date()

                    # OPTIMIZED: Use list comprehension for faster transformation
                    transformed = [
                        (
                            item.get("timestamp") or now_utc,
                            item["symbol"],
                            float(item.get("strike", 0.0)),
                            item.get("expiry") or today_date,
                            item.get("option_type", "call"),
                            float(item["price"]),
                            float(item["delta"]) if item.get("delta") is not None else None,
                            float(item["gamma"]) if item.get("gamma") is not None else None,
                            float(item.get("implied_volatility", 0.0)),
                        )
                        for item in batch
                    ]

                    await db.insert_prices_vectorized(transformed)
                except Exception as e:
                    logger.error("persistence_market_batch_failed", error=str(e))
                finally:
                    self.market_queue.task_done()

    async def _persist_audit_logs(self):
        from core.database import get_async_db_context
        from core.database.crud import bulk_insert_audit_logs

        while self.running:
            batch = await self.audit_queue.get()
            try:
                # Optimized for high-throughput audit ingestion
                async with get_async_db_context() as session:
                    await bulk_insert_audit_logs(session, batch)
            except Exception as e:
                logger.error("persistence_audit_batch_failed", error=str(e))
            finally:
                self.audit_queue.task_done()


class IngestionWorker:
    """
    Advanced Multi-Path Dispatcher.
    Splits ingestion into:
    1. Pulse (XDP/SHM) - Internal to XDPIngester
    2. Voice (WebSockets) - Dispatched to BroadcastWorker
    3. Scribe (Postgres) - Dispatched to PersistenceWorker (Market & Audit)
    """

    def __init__(self, topics: list[str] | None = None):
        self.topics = topics or ["market-data", "audit-logs"]
        self.consumer = MarketDataConsumer(topics=self.topics)
        self.running = False
        self.xdp_ingester = XDPIngester()
        self.broadcaster = BroadcastWorker()
        self.scribe = PersistenceWorker()

    async def _dispatch_batch(self, batch: list[dict], topic: str = "market-data"):
        """Non-blocking dispatch to specialized workers based on topic."""
        if topic == "market-data":
            # Market data is loss-tolerant if queue is full (latest value matters)
            try:
                self.broadcaster.queue.put_nowait(batch)
            except asyncio.QueueFull:
                pass

            try:
                self.scribe.market_queue.put_nowait(batch)
            except asyncio.QueueFull:
                pass

        elif topic == "audit-logs":
            # Audit logs are NOT loss-tolerant. Wait for space.
            try:
                await self.scribe.audit_queue.put(batch)
            except Exception as e:
                logger.error("audit_dispatch_failed", error=str(e))

    async def run(self):
        """Unified entry point for starting all components."""
        self.running = True
        # Launch specialized threads/tasks
        asyncio.create_task(self.scribe.run())
        asyncio.create_task(self.broadcaster.run())

        # Start consuming (blocking)
        while self.running:
            try:
                # Updated consumer callback to pass topic context
                await self.consumer.consume_messages(callback=self._dispatch_batch)
            except Exception as e:
                logger.error("dispatcher_crash", error=str(e))
                await asyncio.sleep(1)

    def stop(self):
        """Graceful shutdown of all subsystems."""
        self.running = False
        self.consumer.stop()
        self.broadcaster.running = False
        self.scribe.running = False
        self.xdp_ingester.stop()
        logger.info("ingestion_worker_shutdown_complete")


# FastAPI for monitoring the worker
app = FastAPI(title="BS-Opt Ingestion Worker", default_response_class=MsgspecJSONResponse)
worker = IngestionWorker()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker.run())


@app.get("/health")
async def health():
    return {"status": "running", "consumer_active": worker.consumer.running}
