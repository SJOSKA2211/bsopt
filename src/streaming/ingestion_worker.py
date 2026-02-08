import asyncio
import time
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

from src.shared.shm_mesh import SharedMemoryRingBuffer

setup_logging()
logger = structlog.get_logger(__name__)
tune_gc(mode="high_frequency") # Optimized for high-frequency trading workers

from src.data.xdp_ingest import XDPIngester
from src.shared.shm_mesh import SharedMemoryRingBuffer

from src.api.websockets.manager import manager as ws_manager

class IngestionWorker:
    """
    Asynchronous ingestion worker that bridges Kafka to Postgres,
    SHM Mesh, and real-time WebSocket clients via Redis.
    """
    def __init__(self, topics: list[str] = ["market-data"]):
        self.consumer = MarketDataConsumer(topics=topics)
        self.running = False
        self.shm_mesh = SharedMemoryRingBuffer(create=True)
        self.xdp_ingester = XDPIngester(self.shm_mesh)

    async def _ingest_batch_callback(self, batch: list[dict]):
        """
        Multi-path ingestion:
        1. DB Persistence (Historical)
        2. WebSocket Broadcast via Redis (Real-time)
        Note: SHM is handled by XDPIngester thread.
        """
        from src.database.crud import bulk_insert_option_prices
        start_time = time.time()
        try:
            # 1. Broadast to WebSockets (Async, non-blocking)
            for item in batch:
                await ws_manager.broadcast_to_symbol(item['symbol'], item)

            # 2. Transformation for DB
            now_utc = datetime.now(UTC)
            today_date = now_utc.date()
            
            transformed_batch = [
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

            async with get_async_db_context() as db:
                count = await bulk_insert_option_prices(db, transformed_batch)
            
            duration = time.time() - start_time
            logger.info("ingestion_complete", db_count=count, ws_broadcasts=len(batch), duration_ms=duration*1000)
        except Exception as e:
            logger.error("ingestion_batch_failed", error=str(e))

    async def run(self):
        self.running = True
        self.xdp_ingester.start()
        
        retry_delay = 1
        while self.running:
            try:
                await self.consumer.consume_messages(callback=self._ingest_batch_callback)
                break
            except Exception as e:
                logger.error("ingestion_worker_crash", error=str(e), next_retry_s=retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(60, retry_delay * 2)
        
        self.xdp_ingester.stop()
        self.shm_mesh.close()
        logger.info("ingestion_worker_stop")

    def stop(self):
        self.running = False
        self.consumer.stop()

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
