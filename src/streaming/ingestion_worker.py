import asyncio
import structlog
import time
from typing import Dict, List
from src.streaming.kafka_consumer import MarketDataConsumer
from src.database import get_async_db_context
from src.database.models import OptionPrice
from src.shared.observability import tune_gc
from datetime import datetime, timezone
from fastapi import FastAPI

logger = structlog.get_logger(__name__)
tune_gc()

class IngestionWorker:
    """
    Asynchronous ingestion worker that bridges Kafka and PostgreSQL (Async).
    Optimized for high-throughput batch writes.
    """
    def __init__(self, topics: List[str] = ["market-data"]):
        self.consumer = MarketDataConsumer(topics=topics)
        self.running = False
        
        # Mark callback as batch-aware for optimized consumer processing
        self._ingest_batch_callback._is_batch_aware = True

    async def _ingest_batch_callback(self, batch: List[Dict]):
        """
        Optimized batch write to database using AsyncSession and bulk upsert.
        """
        from src.database.crud import bulk_insert_option_prices
        start_time = time.time()
        try:
            # Transform batch to match database schema expectations
            transformed_batch = []
            for item in batch:
                transformed_batch.append({
                    "time": item.get('timestamp', datetime.now(timezone.utc)),
                    "symbol": item['symbol'],
                    "strike": item.get('strike', 0.0),
                    "expiry": item.get('expiry', datetime.now(timezone.utc).date()),
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
                })

            async with get_async_db_context() as db:
                count = await bulk_insert_option_prices(db, transformed_batch)
            
            duration = time.time() - start_time
            logger.info("ingestion_batch_complete", count=count, duration_ms=duration*1000)
        except Exception as e:
            logger.error("ingestion_batch_failed", error=str(e))

    async def start(self):
        self.running = True
        logger.info("ingestion_worker_start")
        await self.consumer.consume_messages(callback=self._ingest_batch_callback)

    def stop(self):
        self.running = False
        self.consumer.stop()

# FastAPI for monitoring the worker
app = FastAPI(title="BS-Opt Ingestion Worker")
worker = IngestionWorker()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker.start())

@app.get("/health")
async def health():
    return {"status": "running", "consumer_active": worker.consumer.running}
