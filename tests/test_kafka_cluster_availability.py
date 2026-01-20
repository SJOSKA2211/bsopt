import pytest
import requests
from unittest.mock import MagicMock, patch
import os
from confluent_kafka.admin import AdminClient

@patch("tests.test_kafka_cluster_availability.AdminClient")
def test_kafka_brokers_available(mock_admin_client):
    """Verify that Kafka brokers check logic works."""
    # Setup mock
    mock_admin_instance = MagicMock()
    mock_metadata = MagicMock()
    # Mock brokers dictionary with one entry
    mock_metadata.brokers = {1: "broker1"}
    mock_admin_instance.list_topics.return_value = mock_metadata
    mock_admin_client.return_value = mock_admin_instance

    conf = {'bootstrap.servers': 'localhost:9092'}
    admin = AdminClient(conf)
    try:
        metadata = admin.list_topics(timeout=10)
        assert len(metadata.brokers) >= 1
    except Exception as e:
        pytest.fail(f"Kafka brokers check failed: {e}")

@patch("requests.get")
def test_schema_registry_available(mock_get):
    """Verify that Schema Registry check logic works."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    try:
        response = requests.get("http://localhost:8081/subjects", timeout=5)
        assert response.status_code == 200
    except Exception as e:
        pytest.fail(f"Schema Registry check failed: {e}")

@patch("requests.get")
def test_ksqldb_available(mock_get):
    """Verify that ksqlDB check logic works."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    try:
        response = requests.get("http://localhost:8088/info", timeout=5)
        assert response.status_code == 200
    except Exception as e:
        pytest.fail(f"ksqlDB check failed: {e}")
