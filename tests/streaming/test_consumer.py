import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.streaming.consumer import MarketDataConsumer
from confluent_kafka import KafkaError
import json
import asyncio

@pytest.fixture
def mock_consumer_deps():
    with patch("src.streaming.consumer.Consumer") as mock_c, \
         patch("src.streaming.consumer.SchemaRegistryClient") as mock_sr, \
         patch("src.streaming.consumer.AvroDeserializer") as mock_deser:
        yield mock_c, mock_sr, mock_deser

def test_consumer_init(mock_consumer_deps):
    mock_c, mock_sr, mock_deser = mock_consumer_deps
    consumer = MarketDataConsumer()
    assert consumer.batch_size == 100
    mock_c.assert_called_once()

@pytest.mark.asyncio
async def test_consume_messages_success(mock_consumer_deps):
    mock_c_cls, mock_sr, mock_deser = mock_consumer_deps
    mock_c = mock_c_cls.return_value
    consumer = MarketDataConsumer(batch_size=1)
    
    msg = MagicMock()
    msg.error.return_value = None
    msg.value.return_value = b"data"
    
    mock_deser.return_value.return_value = {"price": 100}
    
    call_count = 0
    def poll_side_effect(timeout):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return msg
        # After one message, stop
        consumer.running = False
        return None
    
    mock_c.poll.side_effect = poll_side_effect
    callback = AsyncMock()
    
    await consumer.consume_messages(callback)
    callback.assert_called_once_with({"price": 100})

@pytest.mark.asyncio
async def test_consume_messages_error(mock_consumer_deps):
    mock_c_cls, mock_sr, mock_deser = mock_consumer_deps
    mock_c = mock_c_cls.return_value
    consumer = MarketDataConsumer()
    
    error_msg = MagicMock()
    error_obj = MagicMock()
    error_obj.code.return_value = KafkaError._PARTITION_EOF
    error_msg.error.return_value = error_obj
    
    call_count = 0
    def poll_side_effect(timeout):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return error_msg
        consumer.stop()
        return None
        
    mock_c.poll.side_effect = poll_side_effect
    
    with patch("src.streaming.consumer.logger") as mock_logger:
        await consumer.consume_messages(AsyncMock())
        mock_logger.info.assert_called_with("kafka_partition_eof", partition=error_msg.partition())

@pytest.mark.asyncio
async def test_deserialization_error(mock_consumer_deps):
    mock_c_cls, mock_sr, mock_deser = mock_consumer_deps
    mock_c = mock_c_cls.return_value
    consumer = MarketDataConsumer()
    
    msg = MagicMock()
    msg.error.return_value = None
    mock_deser.return_value.side_effect = Exception("Deser error")
    
    call_count = 0
    def poll_side_effect(timeout):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return msg
        consumer.stop()
        return None
    
    mock_c.poll.side_effect = poll_side_effect
    
    with patch("src.streaming.consumer.logger") as mock_logger:
        await consumer.consume_messages(AsyncMock())
        mock_logger.error.assert_called_with("deserialization_error", error="Deser error")
