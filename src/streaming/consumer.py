from confluent_kafka import Consumer, KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
import asyncio
from typing import Dict, Callable, List
import structlog
import time

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
        topics: List[str] = ["market-data"],
        schema_registry_url: str = "http://localhost:8081",
        batch_size: int = 100
    ):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000,
            'session.timeout.ms': 10000,
            'heartbeat.interval.ms': 3000,
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
        callback: Callable[[Dict], None]
    ):
        """
        Consume messages in batches and process with async callback.
        """
        self.running = True
        batch = []
        try:
            while self.running:
                # Poll for messages
                # Smaller timeout for better responsiveness to stop signal
                msg = self.consumer.poll(timeout=0.1)
                
                await asyncio.sleep(0) # Yield to other tasks
                
                if not self.running:
                    break
                    
                if msg is None:
                    if batch:
                        await self._process_batch(batch, callback)
                        batch = []
                    continue
                    
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info("kafka_partition_eof", partition=msg.partition())
                    else:
                        logger.error("kafka_consumer_error", error=str(msg.error()))
                    continue

                # Deserialize message
                try:
                    data = self.avro_deserializer(msg.value(), None)
                    batch.append(data)
                    
                    # Process batch when full
                    if len(batch) >= self.batch_size:
                        await self._process_batch(batch, callback)
                        batch = []
                except Exception as e:
                    logger.error("deserialization_error", error=str(e))
                    # In production, send to DLQ here
                    
        finally:
            self.stop()

    async def _process_batch(self, batch: List[Dict], callback: Callable):
        """Process batch of messages in parallel using gather"""
        start_time = time.time()
        try:
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
        try:
            self.consumer.close()
        except Exception:
            pass
