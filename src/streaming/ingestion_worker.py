import asyncio
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from src.database import get_async_db_context
from src.shared.observability import setup_logging, tune_gc
from src.streaming.kafka_consumer import MarketDataConsumer

# Optimized event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


setup_logging()
logger = structlog.get_logger(__name__)
tune_gc(mode="high_frequency") # Optimized for high-frequency trading workers

from src.api.websockets.manager import manager as ws_manager
from src.data.xdp_ingest import XDPIngester


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
                for item in batch:
                    await ws_manager.broadcast_to_symbol(item['symbol'], item)
            except Exception as e:
                logger.error("broadcast_batch_failed", error=str(e))
            finally:
                self.queue.task_done()

from src.shared.eternal_ledger import EternalLedger


class PersistenceWorker:
    """The Scribe: Dedicated to high-throughput DB persistence and Binary Ledger."""
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=1000)
        self.running = False
        self.ledger = EternalLedger()

    async def run(self):
        from src.database.crud import bulk_insert_option_prices
        self.running = True
        logger.info("persistence_worker_active")
        while self.running:
            batch = await self.queue.get()
            try:
                #  HOT PATH: Persistent Binary Logging (Zero-latency)
                self.ledger.write_batch(batch)
                
                # 2. Historical DB Persistence (Asynchronous Scribe)
                now_utc = datetime.now(UTC)
                today_date = now_utc.date()
                transformed = [
                    {
                        "time": item.get('timestamp', now_utc),
                        "symbol": item['symbol'],
                        "strike": item.get('strike', 0.0),
                        "expiry": item.get('expiry', today_date),
                        "option_type": item.get('option_type', 'call'),
                        "last": item['price'],
                        "bid": item.get('bid', 0.0),
                        "ask": item.get('ask', 0.0),
                        "volume": item.get('volume', 0),
                        "open_interest": item.get('open_interest', 0),
                        "implied_volatility": item.get('implied_volatility', 0.0),
                        "delta": item.get('delta'),
                        "gamma": item.get('gamma'),
                        "vega": item.get('vega'),
                        "theta": item.get('theta'),
                        "rho": item.get('rho')
                    }
                    for item in batch
                ]
                # Note: In a real singularity, we'd batch these even further
                async with get_async_db_context() as db:
                    await bulk_insert_option_prices(db, transformed)
            except Exception as e:
                logger.error("persistence_batch_failed", error=str(e))
            finally:
                self.queue.task_done()

class IngestionWorker:
    """
    Advanced Multi-Path Dispatcher.
    Splits ingestion into:
    1. Pulse (XDP/SHM) - Internal to XDPIngester
    2. Voice (WebSockets) - Dispatched to BroadcastWorker
    3. Scribe (Postgres) - Dispatched to PersistenceWorker
    """
    def __init__(self, topics: list[str] = ["market-data"]):
        self.consumer = MarketDataConsumer(topics=topics)
        self.running = False
        self.xdp_ingester = XDPIngester()
        self.broadcaster = BroadcastWorker()
        self.scribe = PersistenceWorker()

    async def _dispatch_batch(self, batch: list[dict]):
        """Non-blocking dispatch to specialized workers."""
        try:
            self.broadcaster.queue.put_nowait(batch)
        except asyncio.QueueFull:
            pass
            
        try:
            self.scribe.queue.put_nowait(batch)
        except asyncio.QueueFull:
            pass

    async def run_dispatcher(self, cpu_core: int):
        """Pure dispatcher: Kafka -> Queues."""
        self.running = True
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("dispatcher_pinned", core=cpu_core)
        except Exception:
            pass

        while self.running:
            try:
                await self.consumer.consume_messages(callback=self._dispatch_batch)
                break
            except Exception as e:
                logger.error("dispatcher_crash", error=str(e))
                await asyncio.sleep(1)

    async def run_broadcaster(self, cpu_core: int):
        """Voice: Dedicated thread for WS broadcasting."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("broadcaster_pinned", core=cpu_core)
        except Exception:
            pass
        await self.broadcaster.run()

    async def run_scribe(self, cpu_core: int):
        """Scribe: Dedicated thread for DB persistence."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("scribe_pinned", core=cpu_core)
        except Exception:
            pass
        await self.scribe.run()

    def stop(self):
        self.running = False
        self.consumer.stop()
        self.broadcaster.running = False
        self.scribe.running = False
        self.xdp_ingester.stop()

    def stop(self):
        self.running = False
        self.consumer.stop()

# FastAPI for monitoring the worker
app = FastAPI(
    title="BS-Opt Ingestion Worker",
    default_response_class=ORJSONResponse
)
worker = IngestionWorker()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker.run())

@app.get("/health")
async def health():
    return {"status": "running", "consumer_active": worker.consumer.running}
