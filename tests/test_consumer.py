import pytest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
import time
import asyncio

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def mock_schema_file():
    with patch("builtins.open", mock_open(read_data='{"type": "record", "name": "MarketData"}')) as m:
        yield m

def test_consumer_initialization(mock_schema_file):
    with patch('streaming.kafka_consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.kafka_consumer.SchemaRegistryClient'), \
         patch('streaming.kafka_consumer.AvroDeserializer'):
        
        from streaming.kafka_consumer import MarketDataConsumer
        
        consumer = MarketDataConsumer(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
            topics=["test-topic"]
        )
        
        assert consumer.config['bootstrap.servers'] == "localhost:9092"
        assert consumer.config['group.id'] == "test-group"
        
        mock_kafka_consumer.assert_called_once()
        mock_kafka_consumer.return_value.subscribe.assert_called_once_with(["test-topic"])

@pytest.mark.asyncio
async def test_consume_messages_batch(mock_schema_file):
    with patch('streaming.kafka_consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.kafka_consumer.SchemaRegistryClient'), \
         patch('streaming.kafka_consumer.AvroDeserializer') as mock_avro_deserializer:
        
        from streaming.kafka_consumer import MarketDataConsumer
        
        mock_instance = MagicMock()
        mock_kafka_consumer.return_value = mock_instance
        
        # Mock messages
        mock_msg1 = MagicMock()
        mock_msg1.value.return_value = b"data1"
        mock_msg1.error.return_value = None
        
        mock_msg2 = MagicMock()
        mock_msg2.value.return_value = b"data2"
        mock_msg2.error.return_value = None
        
        # consume returns list of messages
        # First call returns [msg1, msg2], subsequent calls return []
        import itertools
        mock_instance.consume.side_effect = itertools.chain([[mock_msg1, mock_msg2]], itertools.repeat([]))
        
        # Deserializer returns dicts
        mock_deserializer_instance = MagicMock()
        mock_deserializer_instance.side_effect = [{"symbol": "A"}, {"symbol": "B"}]
        mock_avro_deserializer.return_value = mock_deserializer_instance
        
        consumer = MarketDataConsumer()
        
        processed_data = []
        async def mock_callback(data):
            processed_data.append(data)
            
        # We need a way to stop the infinite loop
        # The consumer runs until self.running is False
        
        task = asyncio.create_task(consumer.consume_messages(mock_callback, batch_size=2))
        
        # Wait for processing
        for _ in range(10):
            if len(processed_data) >= 2:
                break
            await asyncio.sleep(0.1)
            
        consumer.stop()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        assert len(processed_data) == 2
        assert processed_data[0] == {"symbol": "A"}
        assert processed_data[1] == {"symbol": "B"}

@pytest.mark.asyncio
async def test_consumer_error_handling(mock_schema_file):
    with patch('streaming.kafka_consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.kafka_consumer.SchemaRegistryClient'), \
         patch('streaming.kafka_consumer.AvroDeserializer') as mock_avro_deserializer:
        
        from streaming.kafka_consumer import MarketDataConsumer
        from confluent_kafka import KafkaError
        
        mock_instance = MagicMock()
        mock_kafka_consumer.return_value = mock_instance
        
        # Partition EOF message
        mock_eof_msg = MagicMock()
        mock_eof_err = MagicMock()
        mock_eof_err.code.return_value = KafkaError._PARTITION_EOF
        mock_eof_msg.error.return_value = mock_eof_err
        
        # Valid message that fails deserialization
        mock_bad_msg = MagicMock()
        mock_bad_msg.error.return_value = None
        mock_bad_msg.value.return_value = b"bad"
        
        import itertools
        mock_instance.consume.side_effect = itertools.chain([[mock_eof_msg, mock_bad_msg]], itertools.repeat([]))
        
        mock_avro_deserializer.return_value.side_effect = Exception("Deserialization failed")
        
        consumer = MarketDataConsumer()
        
        async def mock_callback(data): pass
        
        with patch('streaming.kafka_consumer.logger') as mock_logger:
            task = asyncio.create_task(consumer.consume_messages(mock_callback))
            await asyncio.sleep(0.2)
            consumer.stop()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
            # Check logs
            # EOF should not log error (it might log nothing or debug/info depending on implementation)
            # Deserialization error should log error
            # Check for 'message_processing_error' which is what kafka_consumer.py logs on exception inside the loop
            
            # The code catches Exception inside the loop:
            # try: deserializer... except Exception as e: logger.error("message_processing_error"...)
            
            error_calls = [str(c) for c in mock_logger.error.call_args_list]
            assert any("message_processing_error" in c for c in error_calls)

@pytest.mark.asyncio
async def test_batch_processing_error(mock_schema_file):
    with patch('streaming.kafka_consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.kafka_consumer.SchemaRegistryClient'), \
         patch('streaming.kafka_consumer.AvroDeserializer') as mock_avro_deserializer:
        
        from streaming.kafka_consumer import MarketDataConsumer
        
        mock_instance = MagicMock()
        mock_kafka_consumer.return_value = mock_instance
        
        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.value.return_value = b"data"
        
        import itertools
        mock_instance.consume.side_effect = itertools.chain([[mock_msg]], itertools.repeat([]))
        
        # deserializer instance should be callable and return data
        mock_deserializer_instance = MagicMock()
        mock_deserializer_instance.return_value = {"data": "test"}
        mock_avro_deserializer.return_value = mock_deserializer_instance
        
        consumer = MarketDataConsumer()
        
        async def error_callback(data):
            raise Exception("Processing failed")
            
        with patch('streaming.kafka_consumer.logger') as mock_logger:
            task = asyncio.create_task(consumer.consume_messages(error_callback, batch_size=1))
            await asyncio.sleep(0.2)
            consumer.stop()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
            # _process_batch catches exceptions and logs 'batch_processing_error'
            error_calls = [str(c) for c in mock_logger.error.call_args_list]
            assert any("batch_processing_error" in c for c in error_calls)
