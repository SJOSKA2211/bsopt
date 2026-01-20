from confluent_kafka import Consumer, KafkaError
import asyncio
import json
import structlog
from typing import Dict, List, Callable
import time
from pydantic import BaseModel, ValidationError
from datetime import datetime

class MarketDataSchema(BaseModel):
    symbol: str
    price: float
    timestamp: datetime

class MarketDataSchema(BaseModel):
    symbol: str
    price: float
    timestamp: datetime

logger = structlog.get_logger()

class MarketDataConsumer:
    """
    High-performance Kafka consumer for real-time processing.
    Features:
    - Consumer group for load balancing
    - Automatic offset management
    - Error handling and retry
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
        Args:
            callback: Function to process each message
            batch_size: Number of messages to batch before processing
        """
        self.running = True
        batch = []
        try:
            while self.running:
                # Poll for messages
                msg = self.consumer.poll(timeout=0.1)
                
                await asyncio.sleep(0) # Yield to event loop
                
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info("kafka_partition_eof", partition=msg.partition())
                    else:
                        logger.error("kafka_consumer_error", error=str(msg.error()))
                    continue

                # Deserialize and validate message
                try:
                    raw_data = msg.value().decode('utf-8')
                    # Optional: Add size limit check before json.loads for very large messages
                    # if len(raw_data) > MAX_MESSAGE_SIZE:
                    #     raise ValueError("Message size exceeds limit")
                    
                    data = json.loads(raw_data)
                    validated_data = MarketDataSchema(**data) # Validate with Pydantic
                    batch.append(validated_data.dict()) # Append validated data as dict
                    # Process batch when full
                    if len(batch) >= batch_size:
                        await self._process_batch(batch, callback)
                        batch = []
                except (json.JSONDecodeError, ValidationError, ValueError) as e:
                    logger.error("message_processing_error", error=str(e), raw_message=raw_data[:200])
                except Exception as e:
                    logger.error("unexpected_message_processing_error", error=str(e))
            # Process remaining messages if batch:
            if batch:
                await self._process_batch(batch, callback)
        finally:
            self.consumer.close()

    async def _process_batch(self, batch: List[Dict], callback: Callable):
        """Process batch of messages"""
        start_time = time.time()
        try:
            # Process in parallel
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
