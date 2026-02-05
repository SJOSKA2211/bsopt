import asyncio
import time
from collections.abc import Callable

import structlog
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer

logger = structlog.get_logger()

class MarketDataConsumer:
    """
    High-performance Kafka consumer for real-time processing.
    Features:
    - Consumer group for load balancing
    - Automatic offset management
    - Batch processing with async callbacks
    - Error handling and logging
    """
    def __init__(
        self, 
        bootstrap_servers: str = "localhost:9092", 
        group_id: str = "market-data-consumers", 
        topics: list[str] = ["market-data"],
        schema_registry_url: str = "http://localhost:8081",
        batch_size: int = 100
    ):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            # High-throughput performance tuning
            'fetch.min.bytes': 102400, # 100KB minimum fetch
            'fetch.wait.max.ms': 200,   # Increased wait for larger batches
            'max.partition.fetch.bytes': 10485760, # 10MB per partition
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
            'session.timeout.ms': 30000,
            'heartbeat.interval.ms': 10000,
        }
        self.consumer = Consumer(self.config)
        self.consumer.subscribe(topics)
        
        # Schema Registry for Avro deserialization
        self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        self.avro_deserializer = AvroDeserializer(self.schema_registry)
        
        self.batch_size = batch_size
        self.running = False

    async def consume_messages(
        self, 
        callback: Callable[[dict], None]
    ):
        """
        Consume messages in batches and process with async callback.
        Uses bulk fetching via consume() for maximum throughput.
        """
        self.running = True
        try:
            while self.running:
                # Bulk fetch messages for better performance than polling individually
                msgs = self.consumer.consume(num_messages=self.batch_size, timeout=0.1)
                
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
                        logger.error("deserialization_error", error=str(e))

                if batch:
                    await self._process_batch(batch, callback)
                    
        finally:
            self.stop()

    async def _process_batch(self, batch: list[dict], callback: Callable):
        """Process batch of messages in parallel using gather"""
        start_time = time.time()
        
        tasks = [callback(msg) for msg in batch]
        # Use return_exceptions=True to allow individual task failures without stopping the whole batch
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_count = 0
        failed_count = 0
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                failed_count += 1
                logger.error("streaming_message_processing_failed", error=str(res), message=batch[i])
            else:
                processed_count += 1
        
        duration = time.time() - start_time
        if duration <= 0:
            duration = 0.001
            
        logger.info(
            "batch_processed_summary", 
            batch_size=len(batch),
            processed_ok=processed_count,
            failed=failed_count,
            duration_ms=duration * 1000,
            throughput=len(batch) / duration
        )
    
        if failed_count > 0:
            logger.warning("streaming_batch_partial_failure", failed_count=failed_count, total_count=len(batch))
    def stop(self):
        """Stop consuming messages"""
        self.running = False
        try:
            self.consumer.close()
        except Exception:
            pass # nosec B110
