import pytest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from src.streaming.mesh_producer import MarketDataMeshProducer, _validate_streaming_url, _ALLOWED_KAFKA_HOSTS, _ALLOWED_SCHEMA_REGISTRY_HOSTS
from confluent_kafka import KafkaError
import asyncio
import os
import urllib.parse
import json

# Mock environment variables for consistent testing
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "KAFKA_BOOTSTRAP_SERVERS": "env_kafka:9092",
        "SCHEMA_REGISTRY_URL": "http://env_schema_registry:8081"
    }):
        yield

@pytest.fixture
def mock_mesh_producer_dependencies():
    with patch('src.streaming.mesh_producer.Producer') as mock_producer_cls, \
         patch('src.streaming.mesh_producer.SchemaRegistryClient') as mock_schema_registry_client_cls, \
         patch('src.streaming.mesh_producer.AvroSerializer') as mock_avro_serializer_cls, \
         patch("builtins.open", mock_open(read_data='{"type": "record", "name": "MarketDataMesh"}')) as mock_file_open, \
         patch('os.path.join', return_value='mock/path/market_data_mesh.avsc'), \
         patch('src.streaming.mesh_producer.logger') as mock_logger:
        
        mock_producer_instance = mock_producer_cls.return_value
        mock_avro_serializer_instance = mock_avro_serializer_cls.return_value
        
        yield mock_producer_instance, mock_avro_serializer_instance, mock_file_open, mock_logger

# --- Tests for _validate_streaming_url ---

@pytest.mark.parametrize("url,allowed_schemes,allowed_hosts,expected_result", [
    ("kafka://localhost:9092", ["kafka", "PLAINTEXT"], ["localhost"], "kafka://localhost:9092"),
    ("http://schema-registry:8081", ["http"], ["schema-registry"], "http://schema-registry:8081"),
    ("kafka-1:9092", ["PLAINTEXT"], ["kafka-1"], "kafka-1:9092"),
    ("kafka-1:9092,kafka-2:9092", ["PLAINTEXT"], ["kafka-1", "kafka-2"], "kafka-1:9092,kafka-2:9092"),
])
def test_validate_streaming_url_success(url, allowed_schemes, allowed_hosts, expected_result):
    assert _validate_streaming_url(url, allowed_schemes, allowed_hosts) == expected_result

@pytest.mark.parametrize("url,allowed_schemes,allowed_hosts,error_msg_part", [
    ("", ["kafka"], ["localhost"], "URL cannot be empty"),
    ("ftp://localhost:9092", ["kafka"], ["localhost"], "URL scheme 'ftp' not allowed"),
    ("kafka://badhost:9092", ["kafka"], ["localhost"], "URL host 'badhost' not allowed"),
])
def test_validate_streaming_url_failure(url, allowed_schemes, allowed_hosts, error_msg_part):
    with pytest.raises(ValueError, match=error_msg_part):
        _validate_streaming_url(url, allowed_schemes, allowed_hosts)

# --- Tests for MarketDataMeshProducer ---

def test_mesh_producer_init_default(mock_mesh_producer_dependencies):
    mock_producer_instance, mock_avro_serializer_instance, mock_file_open, mock_logger = mock_mesh_producer_dependencies
    
    # Define test-specific allowed hosts
    test_allowed_kafka_hosts = _ALLOWED_KAFKA_HOSTS + ["env_kafka"]
    test_allowed_schema_registry_hosts = _ALLOWED_SCHEMA_REGISTRY_HOSTS + ["env_schema_registry"]

    producer = MarketDataMeshProducer(
        allowed_kafka_hosts=test_allowed_kafka_hosts,
        allowed_schema_registry_hosts=test_allowed_schema_registry_hosts
    )
    assert producer.bootstrap_servers == "env_kafka:9092"
    assert producer.schema_registry_url == "http://env_schema_registry:8081"
    assert producer.producer is mock_producer_instance
    assert producer.avro_serializer is mock_avro_serializer_instance
    mock_file_open.assert_called_with('mock/path/market_data_mesh.avsc', "r")
    assert producer.mesh_schema == '{"type": "record", "name": "MarketDataMesh"}'
    mock_logger.critical.assert_not_called()

def test_mesh_producer_init_custom_params(mock_mesh_producer_dependencies):
    mock_producer_instance, mock_avro_serializer_instance, mock_file_open, mock_logger = mock_mesh_producer_dependencies
    
    # Define test-specific allowed hosts
    test_allowed_kafka_hosts = _ALLOWED_KAFKA_HOSTS + ["custom_kafka"]
    test_allowed_schema_registry_hosts = _ALLOWED_SCHEMA_REGISTRY_HOSTS + ["custom_schema_registry"]

    producer = MarketDataMeshProducer(
        bootstrap_servers="custom_kafka:9092",
        schema_registry_url="http://custom_schema_registry:8081",
        allowed_kafka_hosts=test_allowed_kafka_hosts,
        allowed_schema_registry_hosts=test_allowed_schema_registry_hosts
    )
    assert producer.bootstrap_servers == "custom_kafka:9092"
    assert producer.schema_registry_url == "http://custom_schema_registry:8081"
    assert producer.producer is mock_producer_instance
    assert producer.avro_serializer is mock_avro_serializer_instance
    mock_file_open.assert_called_with('mock/path/market_data_mesh.avsc', "r")
    assert producer.mesh_schema == '{"type": "record", "name": "MarketDataMesh"}'
    mock_logger.critical.assert_not_called()

def test_mesh_producer_init_validation_failure(mock_mesh_producer_dependencies):
    _, _, _, mock_logger = mock_mesh_producer_dependencies
    
    # Define test-specific allowed hosts that do NOT include "bad_kafka"
    test_allowed_kafka_hosts = _ALLOWED_KAFKA_HOSTS
    
    with pytest.raises(ValueError, match="URL host 'bad_kafka' not allowed for URL: bad_kafka:9092"):
        MarketDataMeshProducer(
            bootstrap_servers="bad_kafka:9092",
            allowed_kafka_hosts=test_allowed_kafka_hosts
        )
    mock_logger.critical.assert_called_once()


@pytest.mark.asyncio
async def test_produce_mesh_data_success(mock_mesh_producer_dependencies):
    mock_producer_instance, mock_avro_serializer_instance, _, mock_logger = mock_mesh_producer_dependencies
    
    # Define test-specific allowed hosts for producer initialization
    test_allowed_kafka_hosts = _ALLOWED_KAFKA_HOSTS + ["env_kafka"]
    test_allowed_schema_registry_hosts = _ALLOWED_SCHEMA_REGISTRY_HOSTS + ["env_schema_registry"]

    producer = MarketDataMeshProducer(
        allowed_kafka_hosts=test_allowed_kafka_hosts,
        allowed_schema_registry_hosts=test_allowed_schema_registry_hosts
    )
    producer.producer = mock_producer_instance
    producer.avro_serializer = mock_avro_serializer_instance
    
    mock_avro_serializer_instance.return_value = b'serialized_avro_data'
    
    data = {"symbol": "MSFT", "timestamp": "2026-01-19T12:00:00Z", "open": 200.0, "high": 205.0, "low": 199.0, "close": 204.0, "volume": 1000000, "market": "NASDAQ", "source_type": "mock"}
    topic = "market-data-mesh"
    
    await producer.produce_mesh_data(data, topic)
    
    mock_avro_serializer_instance.assert_called_once_with(data, None)
    mock_producer_instance.produce.assert_called_once_with(
        topic=topic,
        key=data["symbol"].encode('utf-8'),
        value=b'serialized_avro_data',
        on_delivery=producer._delivery_callback
    )
    mock_producer_instance.poll.assert_called_once_with(0)
    mock_logger.debug.assert_called_with("produce_mesh_data", topic=topic, key=data["symbol"])
    mock_logger.error.assert_not_called()

@pytest.mark.asyncio
async def test_produce_mesh_data_error(mock_mesh_producer_dependencies):
    mock_producer_instance, _, _, mock_logger = mock_mesh_producer_dependencies
    
    # Define test-specific allowed hosts for producer initialization
    test_allowed_kafka_hosts = _ALLOWED_KAFKA_HOSTS + ["env_kafka"]
    test_allowed_schema_registry_hosts = _ALLOWED_SCHEMA_REGISTRY_HOSTS + ["env_schema_registry"]

    producer = MarketDataMeshProducer(
        allowed_kafka_hosts=test_allowed_kafka_hosts,
        allowed_schema_registry_hosts=test_allowed_schema_registry_hosts
    )
    producer.producer = mock_producer_instance
    
    mock_producer_instance.produce.side_effect = Exception("Kafka produce error")
    
    data = {"symbol": "MSFT", "timestamp": "2026-01-19T12:00:00Z", "open": 200.0, "high": 205.0, "low": 199.0, "close": 204.0, "volume": 1000000, "market": "NASDAQ", "source_type": "mock"}
    topic = "market-data-mesh"
    
    with pytest.raises(Exception, match="Kafka produce error"):
        await producer.produce_mesh_data(data, topic)
    
    mock_logger.error.assert_called_with("mesh_kafka_produce_error", error="Kafka produce error", symbol=data["symbol"])

def test_delivery_callback_error(mock_mesh_producer_dependencies):
    _, _, _, mock_logger = mock_mesh_producer_dependencies
    
    # Define test-specific allowed hosts for producer initialization
    test_allowed_kafka_hosts = _ALLOWED_KAFKA_HOSTS + ["env_kafka"]
    test_allowed_schema_registry_hosts = _ALLOWED_SCHEMA_REGISTRY_HOSTS + ["env_schema_registry"]

    producer = MarketDataMeshProducer(
        allowed_kafka_hosts=test_allowed_kafka_hosts,
        allowed_schema_registry_hosts=test_allowed_schema_registry_hosts
    )
    mock_err = KafkaError._FAIL
    mock_msg = MagicMock()
    
    producer._delivery_callback(mock_err, mock_msg)
    mock_logger.error.assert_called_with("mesh_kafka_delivery_failed", error=str(mock_err))

def test_delivery_callback_success(mock_mesh_producer_dependencies):
    _, _, _, mock_logger = mock_mesh_producer_dependencies
    
    # Define test-specific allowed hosts for producer initialization
    test_allowed_kafka_hosts = _ALLOWED_KAFKA_HOSTS + ["env_kafka"]
    test_allowed_schema_registry_hosts = _ALLOWED_SCHEMA_REGISTRY_HOSTS + ["env_schema_registry"]

    producer = MarketDataMeshProducer(
        allowed_kafka_hosts=test_allowed_kafka_hosts,
        allowed_schema_registry_hosts=test_allowed_schema_registry_hosts
    )
    mock_err = None
    mock_msg = MagicMock()
    mock_msg.topic.return_value = "market-data-mesh"
    mock_msg.partition.return_value = 0
    
    producer._delivery_callback(mock_err, mock_msg)
    mock_logger.debug.assert_called_with(
        "mesh_kafka_message_delivered",
        topic="market-data-mesh",
        partition=0
    )

def test_flush(mock_mesh_producer_dependencies):
    mock_producer_instance, _, _, _ = mock_mesh_producer_dependencies
    
    # Define test-specific allowed hosts for producer initialization
    test_allowed_kafka_hosts = _ALLOWED_KAFKA_HOSTS + ["env_kafka"]
    test_allowed_schema_registry_hosts = _ALLOWED_SCHEMA_REGISTRY_HOSTS + ["env_schema_registry"]

    producer = MarketDataMeshProducer(
        allowed_kafka_hosts=test_allowed_kafka_hosts,
        allowed_schema_registry_hosts=test_allowed_schema_registry_hosts
    )
    producer.producer = mock_producer_instance
    producer.flush()
    mock_producer_instance.flush.assert_called_once()