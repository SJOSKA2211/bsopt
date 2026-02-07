import asyncio
import time
from collections.abc import Callable
from datetime import datetime

import structlog
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from pydantic import BaseModel, ConfigDict, TypeAdapter


class MarketDataSchema(BaseModel):
    model_config = ConfigDict(slots=True)
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
    - Avro deserialization (matches producer)
    """
    def __init__(
        self,
        bootstrap_servers: str = "kafka-1:9092,kafka-2:9092,kafka-3:9092",
        schema_registry_url: str = "http://schema-registry:8081",
        group_id: str = "market-data-consumers",
        topics: list[str] = ["market-data"]
    ):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            # Performance tuning
            'fetch.min.bytes': 524288, # 512KB (Optimized)
            'fetch.wait.max.ms': 100,
            'max.partition.fetch.bytes': 104857600, # Increased to 100MB
            # Disable auto-commit for at-least-once delivery reliability
            'enable.auto.commit': False,
            # Session management: Increased timeout to avoid rebalances during heavy processing
            'session.timeout.ms': 45000,
            'heartbeat.interval.ms': 15000,
        }
        self.consumer = Consumer(self.config)
        self.consumer.subscribe(topics)
        self.running = False

        # Schema Registry for Avro deserialization
        self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        with open("src/streaming/schemas/market_data.avsc") as f:
            self.market_data_schema = f.read()
        self.avro_deserializer = AvroDeserializer(
            self.schema_registry,
            self.market_data_schema
        )

    async def consume_messages(
        self,
        callback: Callable[[dict], None],
        batch_size: int = 100
    ):
        """
        Consume messages in adaptive batches and process with callback.
        Uses bulk fetching with dynamic batch sizing for high throughput.
        """
        self.running = True
        current_batch_size = batch_size
        min_batch = 10
        max_batch = 1000
        target_duration = 0.5 # Aim for 500ms processing cycles
        
        try:
            while self.running:
                # Bulk fetch for efficiency
                msgs = self.consumer.consume(num_messages=current_batch_size, timeout=0.1)
                
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

                    try:
                        data = self.avro_deserializer(msg.value(), None)
                        batch.append(data)
                    except Exception as e:
                        logger.error("message_processing_error", error=str(e))

                if batch:
                    start_time = time.time()
                    await self._process_batch(batch, callback)
                    duration = time.time() - start_time
                    
                    # Commit offsets manually (Async for performance)
                    try:
                        self.consumer.commit(asynchronous=True)
                    except Exception as e:
                        logger.error("offset_commit_failed", error=str(e))
                    
                    # Adaptive batch sizing: 
                    # If we processed too fast, increase batch size.
                    # If we processed too slow, decrease batch size.
                    if duration < target_duration * 0.8:
                        current_batch_size = min(max_batch, int(current_batch_size * 1.2))
                    elif duration > target_duration * 1.2:
                        current_batch_size = max(min_batch, int(current_batch_size * 0.8))
                    
                    logger.debug("adaptive_batching", next_batch_size=current_batch_size, last_duration=duration)
                    
        finally:
            self.consumer.close()

    async def _process_batch(self, batch: list[dict], callback: Callable):
        """Process batch of messages efficiently."""
        start_time = time.time()
        try:
            # Check if callback explicitly handles batches
            if hasattr(callback, "_is_batch_aware") and getattr(callback, "_is_batch_aware"):
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
