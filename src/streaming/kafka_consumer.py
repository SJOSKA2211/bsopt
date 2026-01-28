from confluent_kafka import Consumer, KafkaError
import asyncio
import ujson
import structlog
from typing import Dict, List, Callable
import time
from pydantic import BaseModel, ValidationError, TypeAdapter
from datetime import datetime

class MarketDataSchema(BaseModel):
    symbol: str
    price: float
    timestamp: datetime

logger = structlog.get_logger()
market_data_adapter = TypeAdapter(MarketDataSchema)

class MarketDataConsumer:
    """
    High-performance Kafka consumer for real-time processing.
    Features:
    - Consumer group for load balancing
    - Automatic offset management
    - Bulk fetching via consume()
    - ujson serialization
    """
    def __init__(
        self,
        bootstrap_servers: str = "kafka-1:9092,kafka-2:9092,kafka-3:9092",
        group_id: str = "market-data-consumers",
        topics: List[str] = ["market-data"]
    ):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            # Performance tuning
            'fetch.min.bytes': 1024,
            'fetch.wait.max.ms': 100,
            'max.partition.fetch.bytes': 1048576,
            # Enable auto-commit
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
            # Session management
            'session.timeout.ms': 10000,
            'heartbeat.interval.ms': 3000,
        }
        self.consumer = Consumer(self.config)
        self.consumer.subscribe(topics)
        self.running = False

    async def consume_messages(
        self,
        callback: Callable[[Dict], None],
        batch_size: int = 100
    ):
        """
        Consume messages in batches and process with callback.
        Uses bulk fetching for high throughput.
        """
        self.running = True
        try:
            while self.running:
                # Bulk fetch for efficiency
                msgs = self.consumer.consume(num_messages=batch_size, timeout=0.1)
                
                if not msgs:
                    await asyncio.sleep(0.01) # Yield
                    continue

                batch = []
                for msg in msgs:
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        logger.error("kafka_consumer_error", error=str(msg.error()))
                        continue

                    # Optimized processing
                    try:
                        raw_data = msg.value() # Bytes
                        data = ujson.loads(raw_data)
                        # Fast validation
                        validated_data = market_data_adapter.validate_python(data)
                        batch.append(validated_data.model_dump())
                    except Exception as e:
                        logger.error("message_processing_error", error=str(e))

                if batch:
                    await self._process_batch(batch, callback)
                    
        finally:
            self.consumer.close()

    async def _process_batch(self, batch: List[Dict], callback: Callable):
        """Process batch of messages efficiently."""
        start_time = time.time()
        try:
            # Check if callback explicitly handles batches
            if getattr(callback, "_is_batch_aware", False):
                await callback(batch)
            else:
                # Process in parallel for standard callbacks
                tasks = [callback(msg) for msg in batch]
                await asyncio.gather(*tasks)
            
            duration = time.time() - start_time
            if duration <= 0:
                duration = 0.001

            logger.info(
                "batch_processed",
                batch_size=len(batch),
                duration_ms=duration * 1000,
                throughput=len(batch) / duration
            )
        except Exception as e:
            logger.error("batch_processing_error", error=str(e))

    def stop(self):
        """Stop consuming messages"""
        self.running = False
