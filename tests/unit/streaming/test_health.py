import pytest
from unittest.mock import MagicMock, patch
from src.streaming.health import KafkaHealthCheck, _validate_streaming_url
import os

@pytest.fixture
def mock_kafka_health_dependencies():
    with patch('src.streaming.health.requests') as mock_requests, \
         patch('src.streaming.health.AdminClient') as mock_admin_client_cls, \
         patch('src.streaming.health.logger') as mock_logger, \
         patch.dict(os.environ, {
             "KAFKA_BOOTSTRAP_SERVERS": "mock_kafka_server:9092",
             "SCHEMA_REGISTRY_URL": "http://mock_schema_registry:8081",
             "KSQLDB_URL": "http://mock_ksqldb:8088"
         }):
        
        mock_admin_client_instance = MagicMock()
        mock_admin_client_instance.list_topics.return_value.brokers = {"broker1": MagicMock()}
        mock_admin_client_cls.return_value = mock_admin_client_instance

        mock_requests.get.return_value.status_code = 200
        mock_requests.get.return_value.json.return_value = {} # Default for /info
        
        yield mock_requests, mock_admin_client_instance, mock_logger

# Test _validate_streaming_url
@pytest.mark.parametrize("url,allowed_schemes,allowed_hosts,expected", [
    ("kafka://localhost:9092", ["kafka", "PLAINTEXT"], ["localhost"], "kafka://localhost:9092"),
    ("http://schema-registry:8081", ["http"], ["schema-registry"], "http://schema-registry:8081"),
    ("kafka-1:9092", ["PLAINTEXT"], ["kafka-1"], "kafka-1:9092"), # No scheme, defaults to PLAINTEXT
    ("kafka-1:9092,kafka-2:9092", ["PLAINTEXT"], ["kafka-1", "kafka-2"], "kafka-1:9092,kafka-2:9092"),
])
def test_validate_streaming_url_success(url, allowed_schemes, allowed_hosts, expected):
    assert _validate_streaming_url(url, allowed_schemes, allowed_hosts) == expected

@pytest.mark.parametrize("url,allowed_schemes,allowed_hosts,error_msg_part", [
    ("", ["kafka"], ["localhost"], "URL cannot be empty"),
    ("ftp://localhost:9092", ["kafka"], ["localhost"], "URL scheme 'ftp' not allowed"),
    ("kafka://badhost:9092", ["kafka"], ["localhost"], "URL host 'badhost' not allowed"),
])
def test_validate_streaming_url_failure(url, allowed_schemes, allowed_hosts, error_msg_part):
    with pytest.raises(ValueError, match=error_msg_part):
        _validate_streaming_url(url, allowed_schemes, allowed_hosts)

# Test KafkaHealthCheck
def test_kafka_health_check_init_success(mock_kafka_health_dependencies):
    _, _, mock_logger = mock_kafka_health_dependencies
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    assert health_check_instance.bootstrap_servers == "mock_kafka_server:9092"
    assert health_check_instance.schema_registry_url == "http://mock_schema_registry:8081"
    assert health_check_instance.ksqldb_url == "http://mock_ksqldb:8088"
    mock_logger.critical.assert_not_called()

def test_kafka_health_check_init_failure(mock_kafka_health_dependencies):
    _, _, mock_logger = mock_kafka_health_dependencies
    with pytest.raises(ValueError, match="URL host 'bad_kafka' not allowed for URL: bad_kafka:9092"):
        KafkaHealthCheck(
            bootstrap_servers="bad_kafka:9092",
            schema_registry_url="http://mock_schema_registry:8081",
            ksqldb_url="http://mock_ksqldb:8088"
        )
    mock_logger.critical.assert_called_once()

def test_check_brokers_success(mock_kafka_health_dependencies):
    _, mock_admin_client_instance, _ = mock_kafka_health_dependencies
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    assert health_check_instance.check_brokers() is True
    mock_admin_client_instance.list_topics.assert_called_once_with(timeout=5)

def test_check_brokers_failure(mock_kafka_health_dependencies):
    _, mock_admin_client_instance, mock_logger = mock_kafka_health_dependencies
    mock_admin_client_instance.list_topics.side_effect = Exception("Kafka error")
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    assert health_check_instance.check_brokers() is False
    mock_logger.error.assert_called_with("kafka_broker_health_check_failed", error="Kafka error")

def test_check_schema_registry_success(mock_kafka_health_dependencies):
    mock_requests, _, _ = mock_kafka_health_dependencies
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    assert health_check_instance.check_schema_registry() is True
    mock_requests.get.assert_called_with("http://mock_schema_registry:8081/subjects", timeout=5)

def test_check_schema_registry_failure(mock_kafka_health_dependencies):
    mock_requests, _, mock_logger = mock_kafka_health_dependencies
    mock_requests.get.side_effect = Exception("Schema Registry error")
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    assert health_check_instance.check_schema_registry() is False
    mock_logger.error.assert_called_with("schema_registry_health_check_failed", error="Schema Registry error")

def test_check_ksqldb_success(mock_kafka_health_dependencies):
    mock_requests, _, _ = mock_kafka_health_dependencies
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    assert health_check_instance.check_ksqldb() is True
    mock_requests.get.assert_any_call("http://mock_ksqldb:8088/healthcheck", timeout=5)

def test_check_ksqldb_failure(mock_kafka_health_dependencies):
    mock_requests, _, mock_logger = mock_kafka_health_dependencies
    mock_requests.get.side_effect = [MagicMock(status_code=500), Exception("KSQLDB error")] # First call fails, second raises exception
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    assert health_check_instance.check_ksqldb() is False
    mock_requests.get.assert_any_call("http://mock_ksqldb:8088/healthcheck", timeout=5)
    mock_requests.get.assert_any_call("http://mock_ksqldb:8088/info", timeout=5)
    mock_logger.error.assert_called_with("ksqldb_health_check_failed", error="KSQLDB error")

def test_is_healthy_success(mock_kafka_health_dependencies):
    mock_requests, mock_admin_client_instance, _ = mock_kafka_health_dependencies
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    # Ensure all sub-checks pass
    with patch.object(health_check_instance, 'check_brokers', return_value=True), \
         patch.object(health_check_instance, 'check_schema_registry', return_value=True), \
         patch.object(health_check_instance, 'check_ksqldb', return_value=True):
        assert health_check_instance.is_healthy() is True

def test_is_healthy_failure_brokers(mock_kafka_health_dependencies):
    mock_requests, mock_admin_client_instance, _ = mock_kafka_health_dependencies
    health_check_instance = KafkaHealthCheck(
        bootstrap_servers="mock_kafka_server:9092",
        schema_registry_url="http://mock_schema_registry:8081",
        ksqldb_url="http://mock_ksqldb:8088"
    )
    # Ensure at least one sub-check fails
    with patch.object(health_check_instance, 'check_brokers', return_value=False), \
         patch.object(health_check_instance, 'check_schema_registry', return_value=True), \
         patch.object(health_check_instance, 'check_ksqldb', return_value=True):
        assert health_check_instance.is_healthy() is False

