from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
import structlog
from typing import Dict
import os

logger = structlog.get_logger()

class MarketDataMeshProducer:
    """
    Kafka producer for the Hybrid Data Mesh.
    Standardizes disparate sources into a unified Kafka stream.
    """
    def __init__(
        self,
        bootstrap_servers: str = None,
        schema_registry_url: str = None
    ):
        bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka-1:9092")
        schema_registry_url = schema_registry_url or os.environ.get("SCHEMA_REGISTRY_URL", "http://schema-registry:8081")

        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'market-data-mesh-producer',
            'compression.type': 'lz4',
            'enable.idempotence': True,
        }
        self.producer = Producer(self.config)

        # Schema Registry for Avro serialization
        self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        
        # Load Mesh Schema
        schema_path = os.path.join(os.path.dirname(__file__), "schemas/market_data_mesh.avsc")
        with open(schema_path, "r") as f:
            self.mesh_schema = f.read()
            
        self.avro_serializer = AvroSerializer(
            self.schema_registry,
            self.mesh_schema
        )

    async def produce_mesh_data(
        self,
        data: Dict,
        topic: str = "market-data-mesh"
    ):
        """
        Produce normalized mesh data to Kafka.
        """
        try:
            symbol = data.get("symbol", "UNKNOWN")
            # Serialize with Avro
            value = self.avro_serializer(data, None)
            
            # Produce to Kafka (async)
            self.producer.produce(
                topic=topic,
                key=symbol.encode('utf-8'),
                value=value,
                on_delivery=self._delivery_callback
            )
            # Trigger send (non-blocking)
            self.producer.poll(0)
        except Exception as e:
            logger.error("mesh_kafka_produce_error", error=str(e), symbol=data.get("symbol"))

    def _delivery_callback(self, err, msg):
        if err:
            logger.error("mesh_kafka_delivery_failed", error=str(err))
        else:
            logger.debug("mesh_kafka_message_delivered", topic=msg.topic(), partition=msg.partition())

    def flush(self):
        self.producer.flush()
