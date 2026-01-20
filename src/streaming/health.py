"""
Health Check Utilities for Streaming Service
"""
import requests
from confluent_kafka.admin import AdminClient
import structlog
from typing import List
import urllib.parse
import os

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
    "http://schema-registry:8081",
]
_ALLOWED_SCHEMA_REGISTRY_HOSTS = [h for h in _ALLOWED_SCHEMA_REGISTRY_HOSTS if h is not None]


_ALLOWED_KSQLDB_HOSTS = [
    "ksqldb-server",
    "localhost",
    "127.0.0.1",
]
_ALLOWED_KSQLDB_HOSTS = [h for h in _ALLOWED_KSQLDB_HOSTS if h is not None]

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
    urls_to_validate = url.split(',') if "bootstrap.servers" in url else [url]
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

class KafkaHealthCheck:
    """
    Utility class to check health of Kafka infrastructure.
    """
    def __init__(self, bootstrap_servers: str, schema_registry_url: str, ksqldb_url: str):
        # --- SECURITY: SSRF Prevention ---
        try:
            self.bootstrap_servers = _validate_streaming_url(bootstrap_servers, ["kafka", "PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"], _ALLOWED_KAFKA_HOSTS)
            self.schema_registry_url = _validate_streaming_url(schema_registry_url, ["http", "https"], _ALLOWED_SCHEMA_REGISTRY_HOSTS)
            self.ksqldb_url = _validate_streaming_url(ksqldb_url, ["http", "https"], _ALLOWED_KSQLDB_HOSTS)
        except ValueError as e:
            logger.critical(f"SSRF Prevention: Invalid health check URL configured. Shutting down. Error: {e}")
            raise

    def check_brokers(self) -> bool:
        try:
            admin_client = AdminClient({'bootstrap.servers': self.bootstrap_servers, 'socket.timeout.ms': 5000})
            metadata = admin_client.list_topics(timeout=5)
            return len(metadata.brokers) > 0
        except Exception as e:
            logger.error("kafka_broker_health_check_failed", error=str(e))
            return False

    def check_schema_registry(self) -> bool:
        try:
            response = requests.get(f"{self.schema_registry_url}/subjects", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error("schema_registry_health_check_failed", error=str(e))
            return False

    def check_ksqldb(self) -> bool:
        try:
            # Try /healthcheck then /info
            response = requests.get(f"{self.ksqldb_url}/healthcheck", timeout=5)
            if response.status_code == 200:
                return True
            response = requests.get(f"{self.ksqldb_url}/info", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error("ksqldb_health_check_failed", error=str(e))
            return False

    def is_healthy(self) -> bool:
        return all([
            self.check_brokers(),
            self.check_schema_registry(),
            self.check_ksqldb()
        ])
