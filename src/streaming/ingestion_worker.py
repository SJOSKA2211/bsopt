import asyncio
import structlog
import time
from typing import Dict, List
from src.streaming.kafka_consumer import MarketDataConsumer
from src.database import get_async_db_context
from src.database.models import OptionPrice
from src.shared.observability import tune_gc, setup_logging
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

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

class IngestionWorker:
    """
    Asynchronous ingestion worker that bridges Kafka to both Postgres AND 
    the Zero-Copy Shared Memory Mesh for ultra-low latency internal paths.
    """
    def __init__(self, topics: List[str] = ["market-data"]):
        self.consumer = MarketDataConsumer(topics=topics)
        self.running = False
        # Initialize the high-performance SHM ring buffer
        self.shm_mesh = SharedMemoryRingBuffer(create=True)

    async def _ingest_batch_callback(self, batch: List[Dict]):
        """
        Optimized batch processing:
        1. Writes to Zero-Copy SHM Mesh for immediate consumption by Pricing/Trading.
        2. Async bulk writes to Postgres for historical persistence.
        """
        from src.database.crud import bulk_insert_option_prices
        start_time = time.time()
        try:
            # 1. Update Zero-Copy Shared Memory Mesh immediately (Process Loop)
            current_time = time.time()
            for item in batch:
                self.shm_mesh.write_tick(
                    symbol=item['symbol'],
                    price=item['price'],
                    volume=item.get('volume', 0),
                    timestamp=current_time
                )

            # 2. Vectorized Transformation for DB (List Comprehension)
            # Pre-calculate common values
            now_utc = datetime.now(timezone.utc)
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
            logger.info("ingestion_complete", shm_updates=len(batch), db_count=count, duration_ms=duration*1000)
        except Exception as e:
            logger.error("ingestion_batch_failed", error=str(e), duration_ms=(time.time() - start_time) * 1000)

    _ingest_batch_callback._is_batch_aware = True

    async def run(self):
        self.running = True
        retry_delay = 1
        max_delay = 60

        while self.running:
            logger.info("ingestion_worker_iteration_start", retry_delay=retry_delay)
            try:
                await self.consumer.consume_messages(callback=self._ingest_batch_callback)
                # If it exits cleanly, we might want to stop
                break
            except Exception as e:
                logger.error("ingestion_worker_crash", error=str(e), next_retry_s=retry_delay)
                await asyncio.sleep(retry_delay)
                # Exponential backoff
                retry_delay = min(max_delay, retry_delay * 2)
        
        self.shm_mesh.close()
        logger.info("ingestion_worker_stop")

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
