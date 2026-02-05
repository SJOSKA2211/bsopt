import os

import pytest
import requests
from confluent_kafka.admin import AdminClient


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping infra check in CI")
def test_kafka_brokers_available():
    """Verify that Kafka brokers are reachable on localhost:9092."""
    conf = {'bootstrap.servers': 'localhost:9092'}
    admin = AdminClient(conf)
    try:
        metadata = admin.list_topics(timeout=10)
        assert len(metadata.brokers) >= 1
    except Exception as e:
        pytest.fail(f"Kafka brokers not available: {e}")

@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping infra check in CI")
def test_schema_registry_available():
    """Verify that Schema Registry is reachable on http://localhost:8081."""
    try:
        response = requests.get("http://localhost:8081/subjects", timeout=5)
        assert response.status_code == 200
    except Exception as e:
        pytest.fail(f"Schema Registry not available: {e}")

@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping infra check in CI")
def test_ksqldb_available():
    """Verify that ksqlDB is reachable on http://localhost:8088."""
    # Try /healthcheck or /info
    try:
        response = requests.get("http://localhost:8088/info", timeout=5)
        assert response.status_code == 200
    except Exception as e:
        pytest.fail(f"ksqlDB not available: {e}")
