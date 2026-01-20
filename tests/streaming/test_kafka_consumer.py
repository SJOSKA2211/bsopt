import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.streaming.kafka_consumer import MarketDataConsumer
from confluent_kafka import KafkaError
import json
import asyncio

@pytest.fixture
def mock_consumer_deps():
    with patch("src.streaming.kafka_consumer.Consumer") as mock_consumer:
        yield mock_consumer

def test_kafka_consumer_init(mock_consumer_deps):
    mock_c = mock_consumer_deps
    consumer = MarketDataConsumer()
    assert consumer.config['group.id'] == "market-data-consumers"
    mock_c.assert_called_once()
    mock_c.return_value.subscribe.assert_called_once()

@pytest.mark.asyncio
async def test_consume_messages_success(mock_consumer_deps):
    mock_c = mock_consumer_deps.return_value
    consumer = MarketDataConsumer()
    
    # Mock messages
    msg1 = MagicMock()
    msg1.error.return_value = None
    msg1.value.return_value = json.dumps({"price": 100}).encode('utf-8')
    
    msg2 = MagicMock()
    msg2.error.return_value = None
    msg2.value.return_value = json.dumps({"price": 101}).encode('utf-8')
    
    # To stop the loop, we use a side effect that sets running=False
    call_count = 0
    def poll_side_effect(timeout):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return msg1
        if call_count == 1:
            call_count += 1
            return msg2
        consumer.stop()
        return None
    
    mock_c.poll.side_effect = poll_side_effect
    
    callback = AsyncMock()
    
    await consumer.consume_messages(callback, batch_size=2)
    
    assert callback.call_count == 2
    mock_c.close.assert_called_once()

@pytest.mark.asyncio
async def test_consume_messages_error(mock_consumer_deps):
    mock_c = mock_consumer_deps.return_value
    consumer = MarketDataConsumer()
    
    error_msg = MagicMock()
    error_obj = MagicMock()
    error_obj.code.return_value = KafkaError._PARTITION_EOF
    error_msg.error.return_value = error_obj
    
    # Another error
    error_msg2 = MagicMock()
    error_obj2 = MagicMock()
    error_obj2.code.return_value = KafkaError.REBALANCE_IN_PROGRESS
    error_msg2.error.return_value = error_obj2
    
    call_count = 0
    def poll_side_effect(timeout):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return error_msg
        if call_count == 1:
            call_count += 1
            return error_msg2
        consumer.stop()
        return None
        
    mock_c.poll.side_effect = poll_side_effect
    
    with patch("src.streaming.kafka_consumer.logger") as mock_logger:
        await consumer.consume_messages(AsyncMock())
        mock_logger.info.assert_any_call("kafka_partition_eof", partition=error_msg.partition())
        mock_logger.error.assert_any_call("kafka_consumer_error", error=str(error_msg2.error()))

@pytest.mark.asyncio
async def test_process_batch_error(mock_consumer_deps):
    consumer = MarketDataConsumer()
    
    callback = AsyncMock(side_effect=Exception("Process error"))
    batch = [{"data": 1}]
    
    with patch("src.streaming.kafka_consumer.logger") as mock_logger:
        await consumer._process_batch(batch, callback)
        mock_logger.error.assert_called_with("batch_processing_error", error="Process error")
