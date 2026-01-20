import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.streaming.consumer import MarketDataConsumer
from confluent_kafka import KafkaError

@pytest.fixture
def mock_kafka():
    with patch('src.streaming.consumer.Consumer') as mock_consumer, \
         patch('src.streaming.consumer.SchemaRegistryClient') as mock_registry, \
         patch('src.streaming.consumer.AvroDeserializer') as mock_deserializer:
        yield mock_consumer, mock_registry, mock_deserializer

def test_init(mock_kafka):
    consumer = MarketDataConsumer()
    assert consumer.consumer is not None
    assert consumer.schema_registry is not None

@pytest.mark.asyncio
async def test_consume_messages_success(mock_kafka):
    mock_consumer_cls, _, mock_deserializer_cls = mock_kafka
    mock_consumer = mock_consumer_cls.return_value
    mock_deserializer = mock_deserializer_cls.return_value
    
    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = b'avro_data'
    
    # Return message once then None to trigger batch processing
    mock_consumer.poll.side_effect = [mock_msg, None]
    
    mock_deserializer.return_value = {'symbol': 'AAPL', 'price': 150.0}
    
    consumer = MarketDataConsumer(batch_size=1)
    
    # We need a way to stop the loop
    async def stop_soon(*args, **kwargs):
        consumer.running = False
        return None
        
    callback = AsyncMock(side_effect=stop_soon)
    
    await consumer.consume_messages(callback)
    
    assert callback.called
    mock_consumer.close.called

@pytest.mark.asyncio
async def test_consume_messages_error(mock_kafka):
    mock_consumer_cls, _, _ = mock_kafka
    mock_consumer = mock_consumer_cls.return_value
    
    mock_error = MagicMock()
    mock_error.code.return_value = KafkaError._RESOLVE_FLAGS
    mock_msg = MagicMock()
    mock_msg.error.return_value = mock_error
    
    # Return message then None to keep loop running until stopped
    mock_consumer.poll.side_effect = [mock_msg, None, None, None, None]
    
    consumer = MarketDataConsumer()
    
    async def run_and_stop():
        task = asyncio.create_task(consumer.consume_messages(AsyncMock()))
        await asyncio.sleep(0.1)
        consumer.stop()
        await task

    await run_and_stop()
    
    assert mock_consumer.poll.called
def test_stop(mock_kafka):
    consumer = MarketDataConsumer()
    consumer.running = True
    consumer.stop()
    assert consumer.running is False
    assert consumer.consumer.close.called
