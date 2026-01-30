from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
import structlog
from typing import Dict

logger = structlog.get_logger()

class MarketDataProducer:
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
            # High-throughput performance tuning
            'compression.type': 'lz4',
            'linger.ms': 20,  # Increased for better batching
            'batch.size': 65536, # 64KB batches
            'acks': 'all', 
            'max.in.flight.requests.per.connection': 5,
            # Reliability
            'enable.idempotence': True,
            'retries': 10,
            'retry.backoff.ms': 100,
            # Monitoring
            'statistics.interval.ms': 60000,
        }
        self.producer = Producer(self.config)

        # Schema Registry for Avro serialization
        self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        # Define Avro schema for market data
        # Read the Avro schema from the file
        with open("src/streaming/schemas/market_data.avsc", "r") as f:
            self.market_data_schema = f.read()
        self.avro_serializer = AvroSerializer(
            self.schema_registry,
            self.market_data_schema
        )

    async def produce_market_data(
        self,
        topic: str,
        data: Dict,
        key: str = None
    ):
        """
        Produce market data message to Kafka.
        Args:
            topic: Kafka topic name
            data: Market data dictionary
            key: Partition key (typically symbol for ordering)
        """
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
            # Trigger send (non-blocking)
            self.producer.poll(0)
        except Exception as e:
            logger.error("kafka_produce_error", error=str(e), topic=topic)

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

    async def flush(self):
        """Flush pending messages"""
        self.producer.flush()
