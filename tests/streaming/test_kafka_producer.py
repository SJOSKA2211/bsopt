import os
import pytest
from unittest.mock import MagicMock, patch

# Assuming MarketDataProducer will be importable from src.streaming.kafka_producer
# This import will likely fail initially, leading to the "Red" phase.
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from streaming.kafka_producer import MarketDataProducer
except ImportError:
    MarketDataProducer = None # Will be None if import fails


PRODUCER_PATH = "src/streaming/kafka_producer.py"
TEST_BOOTSTRAP_SERVERS = "localhost:9092"
TEST_SCHEMA_REGISTRY_URL = "http://localhost:8081"
TEST_TOPIC = "market-data"
TEST_DATA = {"symbol": "AAPL", "timestamp": 123456789, "bid": 150.0, "ask": 150.5, "last": 150.25, "volume": 1000, "open_interest": 500, "implied_volatility": 0.2, "delta": 0.5, "gamma": 0.1, "vega": 0.2, "theta": -0.1}
TEST_KEY = "AAPL"


def test_kafka_producer_file_exists():
    """
    Test that the kafka_producer.py file exists.
    """
    assert os.path.exists(PRODUCER_PATH), f"MarketDataProducer file not found at {PRODUCER_PATH}"


def test_market_data_producer_class_exists():
    """
    Test that the MarketDataProducer class can be imported.
    This test will fail if the class is not yet defined or importable.
    """
    assert MarketDataProducer is not None, "MarketDataProducer class is not defined or importable."


@patch('streaming.kafka_producer.Producer')
@patch('streaming.kafka_producer.SchemaRegistryClient')
@patch('streaming.kafka_producer.AvroSerializer')
def test_market_data_producer_init(mock_avro_serializer, mock_schema_registry_client, mock_producer):
    """
    Test that MarketDataProducer constructor initializes Producer, SchemaRegistryClient, and AvroSerializer.
    """
    # Mock instances
    mock_producer_instance = MagicMock()
    mock_producer.return_value = mock_producer_instance
    mock_schema_registry_client_instance = MagicMock()
    mock_schema_registry_client.return_value = mock_schema_registry_client_instance
    mock_avro_serializer_instance = MagicMock()
    mock_avro_serializer.return_value = mock_avro_serializer_instance

    producer = MarketDataProducer(TEST_BOOTSTRAP_SERVERS, TEST_SCHEMA_REGISTRY_URL)

    # Assertions for constructor calls
    mock_producer.assert_called_once_with({
        'bootstrap.servers': TEST_BOOTSTRAP_SERVERS,
        'client.id': 'market-data-producer',
        'compression.type': 'lz4',
        'linger.ms': 10,
        'batch.size': 16384,
        'acks': 1,
        'max.in.flight.requests.per.connection': 5,
        'enable.idempotence': True,
        'retries': 10,
        'statistics.interval.ms': 60000,
    })
    mock_schema_registry_client.assert_called_once_with({'url': TEST_SCHEMA_REGISTRY_URL})
    mock_avro_serializer.assert_called_once() # We don't have the schema string here, so just check it's called
    assert producer.producer == mock_producer_instance
    assert producer.schema_registry == mock_schema_registry_client_instance
    assert producer.avro_serializer == mock_avro_serializer_instance


@pytest.mark.asyncio
@patch('streaming.kafka_producer.Producer')
@patch('streaming.kafka_producer.SchemaRegistryClient')
@patch('streaming.kafka_producer.AvroSerializer')
async def test_produce_market_data(mock_avro_serializer, mock_schema_registry_client, mock_producer):
    """
    Test that produce_market_data method calls the mocked producer.produce with correct arguments.
    """
    mock_producer_instance = MagicMock()
    mock_producer.return_value = mock_producer_instance
    mock_producer_instance.poll.return_value = None # Ensure poll doesn't block
    mock_avro_serializer_instance = MagicMock()
    mock_avro_serializer_instance.return_value = b'serialized_data'
    mock_avro_serializer.return_value = mock_avro_serializer_instance

    producer = MarketDataProducer(TEST_BOOTSTRAP_SERVERS, TEST_SCHEMA_REGISTRY_URL)
    await producer.produce_market_data(TEST_TOPIC, TEST_DATA, TEST_KEY)

    # Assert that AvroSerializer was used to serialize the data
    mock_avro_serializer_instance.assert_called_once_with(TEST_DATA, None)

    # Assert that producer.produce was called with correct arguments
    mock_producer_instance.produce.assert_called_once_with(
        topic=TEST_TOPIC,
        key=TEST_KEY.encode('utf-8'),
        value=b'serialized_data',
        on_delivery=producer._delivery_callback # Assert the callback is passed
    )
    # Assert that producer.poll is called
    mock_producer_instance.poll.assert_called_once_with(0)

@pytest.mark.asyncio
@patch('streaming.kafka_producer.Producer')
@patch('streaming.kafka_producer.SchemaRegistryClient')
@patch('streaming.kafka_producer.AvroSerializer')
async def test_flush_producer(mock_avro_serializer, mock_schema_registry_client, mock_producer):
    """
    Test that the flush method calls producer.flush().
    """
    mock_producer_instance = MagicMock()
    mock_producer.return_value = mock_producer_instance

    producer = MarketDataProducer(TEST_BOOTSTRAP_SERVERS, TEST_SCHEMA_REGISTRY_URL)
    await producer.flush()

    mock_producer_instance.flush.assert_called_once()
