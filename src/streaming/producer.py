from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from typing import Dict, Optional
import structlog
import os

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
        bootstrap_servers: str = "localhost:9092", 
        schema_registry_url: str = "http://localhost:8081"
    ):
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'market-data-producer',
            # High-throughput performance tuning
            'compression.type': 'lz4',
            'linger.ms': 100,  # Optimized for higher batching efficiency
            'batch.size': 1048576, # 1MB batches for high throughput
            'acks': 'all',  # Strongest reliability for v4.0
            'max.in.flight.requests.per.connection': 5,
            'enable.idempotence': True,
            'retries': 10,
            'statistics.interval.ms': 60000,
        }
        self.producer = Producer(self.config)
        
        # Schema Registry for Avro serialization
        self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        
        # Load Avro schema
        schema_path = os.path.join(os.path.dirname(__file__), "../shared/schemas/market_data.avsc")
        with open(schema_path, 'r') as f:
            self.market_data_schema = f.read()
            
        self.avro_serializer = AvroSerializer(
            self.schema_registry, 
            self.market_data_schema
        )

    async def produce_market_data(
        self, 
        topic: str, 
        data: Dict, 
        key: Optional[str] = None
    ):
        """
        Produce market data message to Kafka.
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
            raise

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

    def flush(self, timeout: float = 10.0):
        """Flush pending messages"""
        self.producer.flush(timeout)
