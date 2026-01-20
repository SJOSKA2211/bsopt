from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
import structlog
from typing import Dict, List
import os
import urllib.parse

logger = structlog.get_logger()

# Define allowed hosts for internal streaming services
_ALLOWED_KAFKA_HOSTS = [
    "kafka-1",
    "kafka-2",
    "kafka-3",
    "localhost",
    "127.0.0.1",
]
_ALLOWED_KAFKA_HOSTS = [h for h in _ALLOWED_KAFKA_HOSTS if h is not None]

_ALLOWED_SCHEMA_REGISTRY_HOSTS = [
    "schema-registry",
    "localhost",
    "127.0.0.1",
]
_ALLOWED_SCHEMA_REGISTRY_HOSTS = [h for h in _ALLOWED_SCHEMA_REGISTRY_HOSTS if h is not None]

def _validate_streaming_url(url: str, allowed_schemes: List[str], allowed_hosts: List[str]) -> str:
    """
    Validates a URL for streaming services against a list of allowed schemes and hosts to prevent SSRF.
    CRITICAL: This is a placeholder. A robust implementation requires
    comprehensive URL parsing, IP-address resolution to prevent DNS rebinding,
    and matching against a strict allowlist of internal/trusted endpoints.
    """
    if not url:
        raise ValueError("URL cannot be empty.")

    # Handle multiple bootstrap servers separated by commas
    urls_to_validate = url.split(',')
    validated_urls = []
    
    # Clean allowed_hosts to ensure no None values are present
    cleaned_allowed_hosts = [h for h in allowed_hosts if h is not None]

    for single_url in urls_to_validate:
        # Prepend a dummy scheme if none is present to help urlparse
        temp_url = single_url.strip()
        if '://' not in temp_url:
            temp_url = f"dummy://{temp_url}"
            
        parsed_url = urllib.parse.urlparse(temp_url)

        # Validate scheme
        # If the original url had no scheme and we prepended 'dummy://', the scheme will be 'dummy'
        # In this case, we default to the first allowed scheme (which is often PLAINTEXT for Kafka)
        actual_scheme = parsed_url.scheme if parsed_url.scheme != 'dummy' else allowed_schemes[0]
        if actual_scheme not in allowed_schemes:
            raise ValueError(f"URL scheme '{actual_scheme}' not allowed for URL: {single_url}")
        
        # Validate hostname
        effective_hostname = parsed_url.hostname
        if not effective_hostname: # Fallback for cases like "hostname:port" without scheme
            effective_hostname = single_url.split(':')[0]
            
        if effective_hostname not in cleaned_allowed_hosts:
            raise ValueError(f"URL host '{effective_hostname}' not allowed for URL: {single_url}")
        
        validated_urls.append(single_url.strip())
            
    return ','.join(validated_urls)


class MarketDataMeshProducer:
    """
    Kafka producer for the Hybrid Data Mesh.
    Standardizes disparate sources into a unified Kafka stream.
    """
    def __init__(
        self,
        bootstrap_servers: str = None,
        schema_registry_url: str = None,
        allowed_kafka_hosts: List[str] = None,
        allowed_schema_registry_hosts: List[str] = None
    ):
        bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka-1:9092")
        schema_registry_url = schema_registry_url or os.environ.get("SCHEMA_REGISTRY_URL", "http://schema-registry:8081")

        # Use provided allowed_hosts for testing, or default to module-level lists
        final_allowed_kafka_hosts = allowed_kafka_hosts if allowed_kafka_hosts is not None else _ALLOWED_KAFKA_HOSTS
        final_allowed_schema_registry_hosts = allowed_schema_registry_hosts if allowed_schema_registry_hosts is not None else _ALLOWED_SCHEMA_REGISTRY_HOSTS

        # --- SECURITY: SSRF Prevention ---
        try:
            self.bootstrap_servers = _validate_streaming_url(bootstrap_servers, ["kafka", "PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"], final_allowed_kafka_hosts)
            self.schema_registry_url = _validate_streaming_url(schema_registry_url, ["http", "https"], final_allowed_schema_registry_hosts)
        except ValueError as e:
            logger.critical(f"SSRF Prevention: Invalid streaming service URL configured. Shutting down. Error: {e}")
            raise

        self.config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': 'market-data-mesh-producer',
            'compression.type': 'lz4',
            'enable.idempotence': True,
        }
        self.producer = Producer(self.config)

        # Schema Registry for Avro serialization
        self.schema_registry = SchemaRegistryClient({'url': self.schema_registry_url})
        
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
            # --- SECURITY: Sanitize symbol for logging to prevent Log Injection ---
            sanitized_symbol = symbol.replace('\n', '\\n').replace('\r', '\\r')
            logger.debug("produce_mesh_data", topic=topic, key=sanitized_symbol)
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
            raise

    def _delivery_callback(self, err, msg):
        if err:
            logger.error("mesh_kafka_delivery_failed", error=str(err))
        else:
            logger.debug("mesh_kafka_message_delivered", topic=msg.topic(), partition=msg.partition())

    def flush(self):
        self.producer.flush()

