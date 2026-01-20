import pytest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from src.streaming.kafka_producer import MarketDataProducer
from confluent_kafka import KafkaError
import asyncio
import json

@pytest.fixture
def mock_kafka_producer():
    with patch('src.streaming.kafka_producer.Producer') as mock_producer_cls, \
         patch('src.streaming.kafka_producer.SchemaRegistryClient') as mock_schema_registry_client_cls, \
         patch('src.streaming.kafka_producer.AvroSerializer') as mock_avro_serializer_cls, \
         patch("builtins.open", mock_open(read_data='{"type": "record"}')) as mock_file_open:
        
        mock_producer_instance = mock_producer_cls.return_value
        mock_avro_serializer_instance = mock_avro_serializer_cls.return_value
        
        yield mock_producer_instance, mock_avro_serializer_instance, mock_file_open

def test_init(mock_kafka_producer):
    mock_producer_instance, mock_avro_serializer_instance, mock_file_open = mock_kafka_producer
    producer = MarketDataProducer()
    assert producer.producer is mock_producer_instance
    assert producer.avro_serializer is mock_avro_serializer_instance
    mock_file_open.assert_called_with("src/streaming/schemas/market_data.avsc", "r")
    assert producer.market_data_schema == '{"type": "record"}'

@pytest.mark.asyncio
async def test_produce_market_data_success(mock_kafka_producer):
    mock_producer_instance, mock_avro_serializer_instance, _ = mock_kafka_producer
    
    producer = MarketDataProducer()
    producer.producer = mock_producer_instance # Ensure the mock is used
    producer.avro_serializer = mock_avro_serializer_instance
    
    mock_avro_serializer_instance.return_value = b'serialized_avro_data'
    
    topic = "test_topic"
    data = {"symbol": "AAPL", "price": 150.0}
    key = "AAPL"
    
    await producer.produce_market_data(topic, data, key)
    
    mock_avro_serializer_instance.assert_called_with(data, None)
    mock_producer_instance.produce.assert_called_once_with(
        topic=topic,
        key=key.encode('utf-8'),
        value=b'serialized_avro_data',
        on_delivery=producer._delivery_callback
    )
    mock_producer_instance.poll.assert_called_once_with(0)

@pytest.mark.asyncio
async def test_produce_market_data_no_key(mock_kafka_producer):
    mock_producer_instance, mock_avro_serializer_instance, _ = mock_kafka_producer
    
    producer = MarketDataProducer()
    producer.producer = mock_producer_instance
    producer.avro_serializer = mock_avro_serializer_instance
    
    mock_avro_serializer_instance.return_value = b'serialized_avro_data'
    
    topic = "test_topic"
    data = {"symbol": "AAPL", "price": 150.0}
    
    await producer.produce_market_data(topic, data)
    
    mock_producer_instance.produce.assert_called_once_with(
        topic=topic,
        key=None,
        value=b'serialized_avro_data',
        on_delivery=producer._delivery_callback
    )

@pytest.mark.asyncio
async def test_produce_market_data_error(mock_kafka_producer):
    mock_producer_instance, _, _ = mock_kafka_producer
    
    producer = MarketDataProducer()
    producer.producer = mock_producer_instance
    
    mock_producer_instance.produce.side_effect = Exception("Kafka produce error")
    
    topic = "test_topic"
    data = {"symbol": "AAPL", "price": 150.0}
    
    with patch('src.streaming.kafka_producer.logger') as mock_logger:
        await producer.produce_market_data(topic, data)
        mock_logger.error.assert_called_with("kafka_produce_error", error="Kafka produce error", topic=topic)

def test_delivery_callback_error():
    producer = MarketDataProducer()
    mock_err = KafkaError._FAIL
    mock_msg = MagicMock()
    
    with patch('src.streaming.kafka_producer.logger') as mock_logger:
        producer._delivery_callback(mock_err, mock_msg)
        mock_logger.error.assert_called_with("kafka_delivery_failed", error=str(mock_err))

def test_delivery_callback_success():
    producer = MarketDataProducer()
    mock_err = None
    mock_msg = MagicMock()
    mock_msg.topic.return_value = "test_topic"
    mock_msg.partition.return_value = 0
    mock_msg.offset.return_value = 123
    
    with patch('src.streaming.kafka_producer.logger') as mock_logger:
        producer._delivery_callback(mock_err, mock_msg)
        mock_logger.debug.assert_called_with(
            "kafka_message_delivered",
            topic="test_topic",
            partition=0,
            offset=123
        )

@pytest.mark.asyncio
async def test_flush(mock_kafka_producer):
    mock_producer_instance, _, _ = mock_kafka_producer
    producer = MarketDataProducer()
    producer.producer = mock_producer_instance
    await producer.flush()
    mock_producer_instance.flush.assert_called_once()
