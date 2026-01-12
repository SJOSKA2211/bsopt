import pytest
from unittest.mock import MagicMock, patch
import os
import sys
import time
import itertools
import asyncio

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Note: We will implement MarketDataConsumer in src/streaming/consumer.py

def test_consumer_initialization():
    with patch('streaming.consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.consumer.SchemaRegistryClient'), \
         patch('streaming.consumer.AvroDeserializer'):
        
        from streaming.consumer import MarketDataConsumer
        
        consumer = MarketDataConsumer(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
            topics=["test-topic"]
        )
        
        assert consumer.config['bootstrap.servers'] == "localhost:9092"
        assert consumer.config['group.id'] == "test-group"
        assert consumer.config['auto.offset.reset'] == 'latest'
        
        mock_kafka_consumer.assert_called_once()
        mock_kafka_consumer.return_value.subscribe.assert_called_once_with(["test-topic"])

@pytest.mark.asyncio
async def test_consume_messages_batch():
    with patch('streaming.consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.consumer.SchemaRegistryClient'), \
         patch('streaming.consumer.AvroDeserializer') as mock_avro_deserializer:
        
        from streaming.consumer import MarketDataConsumer
        
        mock_instance = MagicMock()
        mock_kafka_consumer.return_value = mock_instance
        
        # Mock messages
        mock_msg1 = MagicMock()
        mock_msg1.value.return_value = b"data1"
        mock_msg1.error.return_value = None
        
        mock_msg2 = MagicMock()
        mock_msg2.value.return_value = b"data2"
        mock_msg2.error.return_value = None
        
        import itertools
        # Polling returns msg1, msg2, then None indefinitely
        mock_instance.poll.side_effect = itertools.chain([mock_msg1, mock_msg2], itertools.repeat(None))
        
        # Deserializer returns dicts
        mock_deserializer_instance = MagicMock()
        mock_deserializer_instance.side_effect = [{"symbol": "A"}, {"symbol": "B"}]
        mock_avro_deserializer.return_value = mock_deserializer_instance
        
        consumer = MarketDataConsumer(batch_size=2)
        
        processed_data = []
        async def mock_callback(data):
            processed_data.append(data)
            
        # We need a way to stop the infinite loop in consume_messages
        # We'll patch 'running' attribute
        consumer.running = True
        
        # Run consume_messages in a task so we can stop it
        import asyncio
        consume_task = asyncio.create_task(consumer.consume_messages(mock_callback))
        
        # Wait until we have enough data or timeout
        start_wait = time.time()
        while len(processed_data) < 2 and time.time() - start_wait < 2:
            await asyncio.sleep(0.05)
            
        consumer.stop()
        try:
            await asyncio.wait_for(consume_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        assert len(processed_data) == 2
        assert processed_data[0] == {"symbol": "A"}
        assert processed_data[1] == {"symbol": "B"}

@pytest.mark.asyncio
async def test_consumer_error_handling():
    with patch('streaming.consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.consumer.SchemaRegistryClient'), \
         patch('streaming.consumer.AvroDeserializer') as mock_avro_deserializer:
        
        from streaming.consumer import MarketDataConsumer
        from confluent_kafka import KafkaError
        import itertools
        
        mock_instance = MagicMock()
        mock_kafka_consumer.return_value = mock_instance
        
        # Scenario 1: Partition EOF (should log info)
        mock_eof_msg = MagicMock()
        mock_eof_err = MagicMock()
        mock_eof_err.code.return_value = KafkaError._PARTITION_EOF
        mock_eof_msg.error.return_value = mock_eof_err
        mock_eof_msg.value.return_value = None
        mock_eof_msg.partition.return_value = 0
        
        # Scenario 2: Other Kafka Error (should log error)
        mock_err_msg = MagicMock()
        mock_err = MagicMock()
        mock_err.code.return_value = KafkaError.UNKNOWN
        mock_err_msg.error.return_value = mock_err
        mock_err_msg.value.return_value = None        
        # Scenario 3: Deserialization Error
        mock_valid_msg = MagicMock()
        mock_valid_msg.error.return_value = None
        mock_valid_msg.value.return_value = b"invalid_avro"
        
        # Setup poll sequence
        mock_instance.poll.side_effect = itertools.chain(
            [mock_eof_msg, mock_err_msg, mock_valid_msg], 
            itertools.repeat(None)
        )
        
        # Setup deserializer to raise exception for the valid_msg (which has invalid data)
        mock_avro_deserializer.return_value.side_effect = Exception("Deserialization failed")
        
        consumer = MarketDataConsumer()
        consumer.running = True
        
        # Mock callback
        async def mock_callback(data): pass
        
        with patch('streaming.consumer.logger') as mock_logger:
            # Run for a short time
            consume_task = asyncio.create_task(consumer.consume_messages(mock_callback))
            await asyncio.sleep(0.1)
            consumer.stop()
            try:
                await asyncio.wait_for(consume_task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
            # Verify logging calls
            # 1. EOF
            mock_logger.info.assert_any_call("kafka_partition_eof", partition=0)
            
            # 2. Kafka Error - Need to check if error log was called with correct args
            # Using assert_any_call might be tricky with kwarg matching if other calls happened
            # Let's check call_args_list
            error_calls = [call for call in mock_logger.error.call_args_list]
            assert any("kafka_consumer_error" in str(call) for call in error_calls)
            
            # 3. Deserialization Error
            assert any("deserialization_error" in str(call) for call in error_calls)

@pytest.mark.asyncio
async def test_batch_processing_error():
    with patch('streaming.consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.consumer.SchemaRegistryClient'), \
         patch('streaming.consumer.AvroDeserializer') as mock_avro_deserializer:
        
        from streaming.consumer import MarketDataConsumer
        
        mock_instance = MagicMock()
        mock_kafka_consumer.return_value = mock_instance
        
        # One valid message
        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.value.return_value = b"data"
        mock_instance.poll.side_effect = itertools.chain([mock_msg], itertools.repeat(None))
        
        mock_avro_deserializer.return_value.return_value = {"data": "test"}
        
        consumer = MarketDataConsumer(batch_size=1)
        consumer.running = True
        
        # Callback raises exception
        async def error_callback(data):
            raise Exception("Processing failed")
            
        with patch('streaming.consumer.logger') as mock_logger:
            consume_task = asyncio.create_task(consumer.consume_messages(error_callback))
            await asyncio.sleep(0.1)
            consumer.stop()
            try:
                await asyncio.wait_for(consume_task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
                
            mock_logger.error.assert_any_call("batch_processing_error", error="Processing failed")

def test_consumer_stop_error():
    with patch('streaming.consumer.Consumer') as mock_kafka_consumer, \
         patch('streaming.consumer.SchemaRegistryClient'), \
         patch('streaming.consumer.AvroDeserializer'):
        
        from streaming.consumer import MarketDataConsumer
        
        mock_instance = MagicMock()
        mock_kafka_consumer.return_value = mock_instance
        mock_instance.close.side_effect = Exception("Close failed")
        
        consumer = MarketDataConsumer()
        consumer.stop()
        
        # Should not raise exception
        mock_instance.close.assert_called_once()
