import pytest
from unittest.mock import MagicMock, patch, mock_open
from src.streaming.kafka_producer import MarketDataProducer
import json

@pytest.fixture
def mock_kafka_deps():
    with patch("src.streaming.kafka_producer.Producer") as mock_producer, \
         patch("src.streaming.kafka_producer.SchemaRegistryClient") as mock_schema_reg, \
         patch("src.streaming.kafka_producer.AvroSerializer") as mock_serializer, \
         patch("builtins.open", mock_open(read_data='{"type": "record", "name": "MarketData", "fields": []}')):
        
        yield mock_producer, mock_schema_reg, mock_serializer

@pytest.mark.asyncio
async def test_kafka_producer_init(mock_kafka_deps):
    mock_p, mock_sr, mock_ser = mock_kafka_deps
    producer = MarketDataProducer()
    assert producer.config['bootstrap.servers'] == "kafka-1:9092,kafka-2:9092,kafka-3:9092"
    mock_p.assert_called_once()

@pytest.mark.asyncio
async def test_produce_market_data(mock_kafka_deps):
    mock_p, mock_sr, mock_ser = mock_kafka_deps
    producer = MarketDataProducer()
    
    mock_ser_instance = mock_ser.return_value
    mock_ser_instance.return_value = b"serialized_data"
    
    await producer.produce_market_data("test-topic", {"price": 100}, key="AAPL")
    
    # Check if producer.produce was called
    producer.producer.produce.assert_called_once()
    args, kwargs = producer.producer.produce.call_args
    assert kwargs['topic'] == "test-topic"
    assert kwargs['key'] == b"AAPL"
    assert kwargs['value'] == b"serialized_data"

@pytest.mark.asyncio
async def test_produce_market_data_error(mock_kafka_deps):
    mock_p, mock_sr, mock_ser = mock_kafka_deps
    producer = MarketDataProducer()
    
    producer.producer.produce.side_effect = Exception("Kafka error")
    
    with patch("src.streaming.kafka_producer.logger") as mock_logger:
        await producer.produce_market_data("test-topic", {"price": 100})
        mock_logger.error.assert_called_with("kafka_produce_error", error="Kafka error", topic="test-topic")

def test_delivery_callback(mock_kafka_deps):
    mock_p, mock_sr, mock_ser = mock_kafka_deps
    producer = MarketDataProducer()
    
    mock_msg = MagicMock()
    mock_msg.topic.return_value = "topic"
    mock_msg.partition.return_value = 0
    mock_msg.offset.return_value = 10
    
    # Test success
    producer._delivery_callback(None, mock_msg)
    
    # Test error
    with patch("src.streaming.kafka_producer.logger") as mock_logger:
        producer._delivery_callback("Error", mock_msg)
        mock_logger.error.assert_called_with("kafka_delivery_failed", error="Error")

@pytest.mark.asyncio
async def test_flush(mock_kafka_deps):
    mock_p, mock_sr, mock_ser = mock_kafka_deps
    producer = MarketDataProducer()
    await producer.flush()
    producer.producer.flush.assert_called_once()