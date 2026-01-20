import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.streaming.kafka_consumer import MarketDataConsumer
from confluent_kafka import KafkaError
import asyncio
import json

@pytest.fixture
def mock_kafka():
    with patch('src.streaming.kafka_consumer.Consumer') as mock_consumer_cls, \
         patch('src.streaming.kafka_consumer.SchemaRegistryClient') as mock_schema_registry_client_cls, \
         patch('src.streaming.kafka_consumer.AvroDeserializer') as mock_avro_deserializer_cls, \
         patch('src.streaming.kafka_consumer.json') as mock_json:
        
        mock_schema_registry_client_instance = MagicMock()
        mock_avro_deserializer_instance = MagicMock()
        
        mock_schema_registry_client_cls.return_value = mock_schema_registry_client_instance
        mock_avro_deserializer_cls.return_value = mock_avro_deserializer_instance
        
        yield mock_consumer_cls, mock_json, mock_schema_registry_client_instance, mock_avro_deserializer_instance

def test_init(mock_kafka):
    mock_consumer_cls, _, mock_schema_registry_client_instance, mock_avro_deserializer_instance = mock_kafka
    consumer = MarketDataConsumer(schema_registry_client=mock_schema_registry_client_instance, avro_deserializer=mock_avro_deserializer_instance)
    assert consumer.consumer is not None
    assert consumer.schema_registry == mock_schema_registry_client_instance
    assert consumer.avro_deserializer == mock_avro_deserializer_instance
    mock_consumer_cls.assert_called_once()

@pytest.mark.asyncio
async def test_consume_messages_success(mock_kafka):
    mock_consumer_cls, mock_json, mock_schema_registry_client_instance, mock_avro_deserializer_instance = mock_kafka
    mock_consumer = mock_consumer_cls.return_value
    
    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = b'{"symbol": "TEST", "price": 100.0}'
    
    # Simulate two messages, then no more to allow the loop to naturally exhaust
    mock_consumer.poll.side_effect = [mock_msg, mock_msg, None, None]
    
    mock_json.loads.return_value = {'symbol': 'TEST', 'price': 100.0}
    
    consumer = MarketDataConsumer(schema_registry_client=mock_schema_registry_client_instance, avro_deserializer=mock_avro_deserializer_instance)
    callback = AsyncMock(return_value=None)
    
    # Run consume_messages and allow it to complete naturally
    async def run_and_stop():
        task = asyncio.create_task(consumer.consume_messages(callback, batch_size=2))
        await asyncio.sleep(0.5) # Give consumer enough time to process messages and exit loop
        consumer.stop() # Ensure consumer loop is explicitly stopped
        await task # Await the consumer task to finish its loop
        
    await run_and_stop()
    
    assert callback.call_count == 2
    assert consumer.consumer.close.called # consumer.close should be called in finally

@pytest.mark.asyncio
async def test_consume_messages_kafka_error(mock_kafka):
    mock_consumer_cls, mock_json, mock_schema_registry_client_instance, mock_avro_deserializer_instance = mock_kafka
    mock_consumer = mock_consumer_cls.return_value
    
    mock_error = MagicMock()
    mock_error.code.return_value = KafkaError._PARTITION_EOF
    mock_msg = MagicMock()
    mock_msg.error.return_value = mock_error
    
    # Simulate an error message, followed by None to allow loop to terminate
    mock_consumer.poll.side_effect = [mock_msg, None, None, None, None]
    
    consumer = MarketDataConsumer(schema_registry_client=mock_schema_registry_client_instance, avro_deserializer=mock_avro_deserializer_instance)
    callback = AsyncMock()
    
    async def run_and_stop():
        task = asyncio.create_task(consumer.consume_messages(callback))
        await asyncio.sleep(0.1) # Give consumer a chance to process error and re-poll
        consumer.stop()
        await task

    with patch('src.streaming.kafka_consumer.logger') as mock_logger:
        await run_and_stop()
    
    callback.assert_not_called()
    assert mock_consumer.poll.called
    mock_logger.info.assert_called_with("kafka_partition_eof", partition=mock_msg.partition())

@pytest.mark.asyncio
async def test_consume_messages_deserialization_error(mock_kafka):
    mock_consumer_cls, mock_json, mock_schema_registry_client_instance, mock_avro_deserializer_instance = mock_kafka
    mock_consumer = mock_consumer_cls.return_value
    
    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = b'invalid json'
    
    # Simulate a deserialization error, followed by None to allow loop to terminate
    mock_consumer.poll.side_effect = [mock_msg, None, None, None, None]
    mock_json.loads.side_effect = json.JSONDecodeError("mock error", "doc", 0)
    
    consumer = MarketDataConsumer(schema_registry_client=mock_schema_registry_client_instance, avro_deserializer=mock_avro_deserializer_instance)
    callback = AsyncMock()
    
    async def run_and_stop():
        task = asyncio.create_task(consumer.consume_messages(callback))
        await asyncio.sleep(0.1) # Give consumer a chance to process error and re-poll
        consumer.stop()
        await task

    with patch('src.streaming.kafka_consumer.logger') as mock_logger:
        await run_and_stop()
    
    callback.assert_not_called()
    assert mock_consumer.poll.called
    mock_logger.error.assert_called_with("message_processing_error", error='mock error: line 1 column 1 (char 0)')

@pytest.mark.asyncio
async def test_process_batch_error(mock_kafka):
    mock_consumer_cls, mock_json, mock_schema_registry_client_instance, mock_avro_deserializer_instance = mock_kafka
    mock_consumer = mock_consumer_cls.return_value
    
    consumer = MarketDataConsumer(schema_registry_client=mock_schema_registry_client_instance, avro_deserializer=mock_avro_deserializer_instance)
    mock_callback = AsyncMock(side_effect=Exception("Batch processing error"))
    
    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = b'{"symbol": "TEST", "price": 100.0}'
    
    # Simulate a message with valid data, then None to allow loop to terminate
    mock_consumer.poll.side_effect = [mock_msg, None, None, None, None]
    
    with patch('src.streaming.kafka_consumer.json.loads', return_value={'symbol': 'TEST', 'price': 100.0}):
        with patch('src.streaming.kafka_consumer.logger') as mock_logger:
            async def run_and_stop():
                task = asyncio.create_task(consumer.consume_messages(mock_callback))
                await asyncio.sleep(0.1) # Give consumer a chance to process batch error and re-poll
                consumer.stop()
                await task
            
            await run_and_stop()
    
    mock_callback.assert_called_once()
    assert mock_consumer.poll.called
    mock_logger.error.assert_called_with("batch_processing_error", error="Batch processing error")

def test_stop(mock_kafka):
    _, _, mock_schema_registry_client_instance, mock_avro_deserializer_instance = mock_kafka
    consumer = MarketDataConsumer(schema_registry_client=mock_schema_registry_client_instance, avro_deserializer=mock_avro_deserializer_instance)
    consumer.running = True
    consumer.stop()
    assert consumer.running is False
