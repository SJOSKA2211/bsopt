import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Note: We will implement MarketDataProducer in src/streaming/producer.py
# For TDD, we write the tests first and expect them to fail until implementation

def test_producer_initialization():
    with patch('streaming.producer.Producer') as mock_kafka_producer, \
         patch('streaming.producer.SchemaRegistryClient') as mock_sr_client, \
         patch('streaming.producer.AvroSerializer') as mock_avro_serializer:
        
        from streaming.producer import MarketDataProducer
        
        producer = MarketDataProducer(
            bootstrap_servers="localhost:9092",
            schema_registry_url="http://localhost:8081"
        )
        
        assert producer.config['bootstrap.servers'] == "localhost:9092"
        assert producer.config['compression.type'] == 'lz4'
        assert producer.config['enable.idempotence'] is True
        
        mock_kafka_producer.assert_called_once()
        mock_sr_client.assert_called_once()
        mock_avro_serializer.assert_called_once()

@pytest.mark.asyncio
async def test_produce_market_data_success():
    with patch('streaming.producer.Producer') as mock_kafka_producer, \
         patch('streaming.producer.SchemaRegistryClient'), \
         patch('streaming.producer.AvroSerializer') as mock_avro_serializer:
        
        from streaming.producer import MarketDataProducer
        
        mock_instance = MagicMock()
        mock_kafka_producer.return_value = mock_instance
        
        mock_serializer_instance = MagicMock()
        mock_serializer_instance.return_value = b"serialized_data"
        mock_avro_serializer.return_value = mock_serializer_instance
        
        producer = MarketDataProducer()
        
        market_data = {
            "symbol": "AAPL",
            "timestamp": 1700000000000000,
            "bid": 150.0,
            "ask": 151.0,
            "last": 150.5,
            "volume": 1000,
            "open_interest": 500,
            "implied_volatility": 0.25,
            "delta": 0.5,
            "gamma": 0.05,
            "vega": 0.1,
            "theta": -0.02
        }
        
        await producer.produce_market_data("market-data", market_data, key="AAPL")
        
        # Verify serialization was called
        mock_serializer_instance.assert_called_once()
        
        # Verify kafka produce was called
        mock_instance.produce.assert_called_once()
        args, kwargs = mock_instance.produce.call_args
        assert kwargs['topic'] == "market-data"
        assert kwargs['key'] == b"AAPL"
        assert kwargs['value'] == b"serialized_data"

def test_producer_delivery_callback():
    with patch('streaming.producer.Producer'), \
         patch('streaming.producer.SchemaRegistryClient'), \
         patch('streaming.producer.AvroSerializer'):
        
        from streaming.producer import MarketDataProducer
        
        producer = MarketDataProducer()
        
        # Test success callback
        mock_msg = MagicMock()
        mock_msg.topic.return_value = "test-topic"
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 100
        
        with patch('streaming.producer.logger') as mock_logger:
            producer._delivery_callback(None, mock_msg)
            mock_logger.debug.assert_called()
            
        # Test error callback
        with patch('streaming.producer.logger') as mock_logger:
            producer._delivery_callback("Kafka Error", mock_msg)
            mock_logger.error.assert_called_with("kafka_delivery_failed", error="Kafka Error")

def test_producer_flush():
    with patch('streaming.producer.Producer') as mock_kafka_producer, \
         patch('streaming.producer.SchemaRegistryClient'), \
         patch('streaming.producer.AvroSerializer'):
        
        from streaming.producer import MarketDataProducer
        
        mock_instance = MagicMock()
        mock_kafka_producer.return_value = mock_instance
        
        producer = MarketDataProducer()
        producer.flush(timeout=5.0)
        
        mock_instance.flush.assert_called_once_with(5.0)

@pytest.mark.asyncio
async def test_produce_exception():
    with patch('streaming.producer.Producer') as mock_kafka_producer, \
         patch('streaming.producer.SchemaRegistryClient'), \
         patch('streaming.producer.AvroSerializer') as mock_avro_serializer:
        
        from streaming.producer import MarketDataProducer
        
        mock_instance = MagicMock()
        mock_instance.produce.side_effect = Exception("Kafka connection failed")
        mock_kafka_producer.return_value = mock_instance
        
        mock_avro_serializer.return_value = lambda x, y: b"data"
        
        producer = MarketDataProducer()
        
        with patch('streaming.producer.logger') as mock_logger:
            with pytest.raises(Exception, match="Kafka connection failed"):
                await producer.produce_market_data("topic", {"data": "test"})
            
            mock_logger.error.assert_called()
