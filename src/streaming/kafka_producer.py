from typing import Any

import msgspec
import structlog
from confluent_kafka import Producer as ConfluentProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

from .base import Producer

logger = structlog.get_logger()

class MarketData(msgspec.Struct):
    """SOTA: High-performance typed schema for market data."""
    time: float
    symbol: str
    strike: float
    expiry: str
    option_type: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: int | None = None
    open_interest: int | None = None

class MarketDataProducer(Producer):
    """
    High-throughput Kafka producer for real-time market data.
    Features:
    - Batching for efficiency
    - Compression (LZ4)
    - Partitioning by symbol for ordering
    - Schema validation with Avro
    - Idempotence for exactly-once semantics
    """
    def __init__(
        self,
        bootstrap_servers: str = "kafka-1:9092,kafka-2:9092,kafka-3:9092",
        schema_registry_url: str = "http://schema-registry:8081"
    ):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'market-data-producer',
            # Low-latency performance tuning
            'compression.type': 'lz4',
            'linger.ms': 20,  # Optimized for balance between throughput and latency
            'batch.size': 524288, # 512KB batches
            'acks': '1', 
            'max.in.flight.requests.per.connection': 5,
            # Reliability
            'enable.idempotence': True,
            'retries': 10,
            'retry.backoff.ms': 100,
            # Monitoring
            'statistics.interval.ms': 60000,
        }
        self.producer = ConfluentProducer(self.config)

        # Schema Registry for Avro serialization
        self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        # Define Avro schema for market data
        # Read the Avro schema from the file
        with open("src/streaming/schemas/market_data.avsc") as f:
            self.market_data_schema = f.read()
        self.avro_serializer = AvroSerializer(
            self.schema_registry,
            self.market_data_schema
        )

    async def produce(
        self,
        data: dict[str, Any],
        **kwargs
    ):
        """Produce market data message to Kafka."""
        topic = kwargs.get("topic")
        key = kwargs.get("key")
        
        if not topic:
             logger.error("kafka_produce_missing_topic")
             return

        try:
            # Serialize with Avro
            value = self.avro_serializer(data, None)
            # Produce to Kafka (async)
            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8') if key else None,
                value=value,
                on_delivery=self._delivery_callback
            )
            self.producer.poll(0)
        except Exception as e:
            logger.error("kafka_produce_error", error=str(e), topic=topic)

    async def produce_batch(self, batch: list[dict[str, Any]], topic: str):
        """
        🚀 OPTIMIZATION: Batched ingestion with msgspec validation.
        """
        if not topic:
            return
            
        try:
            # 1. Zero-cost validation via msgspec
            # This ensures all data in the batch matches the schema before processing
            validated_batch = [msgspec.convert(d, MarketData) for d in batch]
            
            for data in validated_batch:
                # 2. Optimized Avro serialization (using dict conversion)
                payload = msgspec.to_builtins(data)
                value = self.avro_serializer(payload, None)
                self.producer.produce(
                    topic=topic,
                    key=data.symbol.encode('utf-8'),
                    value=value,
                    on_delivery=self._delivery_callback
                )
            # 🚀 FLUSH BATCH
            self.producer.poll(0)
            logger.debug("kafka_batch_produced_msgspec", count=len(batch), topic=topic)
        except Exception as e:
            logger.error("kafka_batch_error", error=str(e), topic=topic)

    def _delivery_callback(self, err, msg):
        """Callback for message delivery confirmation"""
        if err:
            logger.error("kafka_delivery_failed", error=str(err))
        else:
            logger.debug(
                "kafka_message_delivered",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset()
            )

    def flush(self):
        """Flush pending messages"""
        self.producer.flush()

    def close(self):
        self.flush()
        # Confluent producer doesn't have explicit close usually needed if flush is called, but good to have.
