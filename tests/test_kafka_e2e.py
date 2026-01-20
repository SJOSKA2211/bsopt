import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.streaming.producer import MarketDataProducer
from src.streaming.consumer import MarketDataConsumer
from src.streaming.analytics import VolatilityAggregationStream

@pytest.mark.asyncio
async def test_streaming_e2e_flow(mocker):
    """
    Test the full flow from Producer to Analytics to Consumer.
    We mock the underlying Kafka library to simulate message passing.
    """
    # Patch components where they are used
    mocker.patch('src.streaming.producer.Producer')
    mocker.patch('src.streaming.producer.SchemaRegistryClient')
    mocker.patch('src.streaming.producer.AvroSerializer')
    
    mock_kafka_consumer_cls = mocker.patch('src.streaming.consumer.Consumer')
    mock_consumer_instance = MagicMock()
    mock_kafka_consumer_cls.return_value = mock_consumer_instance
    
    mocker.patch('src.streaming.consumer.SchemaRegistryClient')
    mocker.patch('src.streaming.consumer.AvroDeserializer')
    mocker.patch('src.streaming.analytics.App')

    # 1. Initialize components
    MarketDataProducer()
    consumer = MarketDataConsumer(batch_size=1)
    analytics = VolatilityAggregationStream()
    
    # 2. Setup mock data
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
    
    # 3. Simulate Kafka -> Analytics
    analytics.price_history = {"AAPL": 150.0}
    analytics.volatility_table = {"AAPL": 0.2}
    
    import numpy as np
    price = market_data['last']
    prev_price = analytics.price_history["AAPL"]
    log_return = np.log(price / prev_price)
    
    new_vol = analytics._update_volatility("AAPL", log_return, market_data['timestamp'])
    analytics.volatility_table["AAPL"] = new_vol
    
    assert new_vol > 0
    
    # 4. Simulate Kafka -> Consumer
    mock_msg = MagicMock()
    # IMPORTANT: Ensure error() returns None (falsy)
    mock_msg.error.return_value = None
    mock_msg.value.return_value = b"serialized_data"

    # Use a side effect that returns the message then None
    msgs = [mock_msg, None]
    def poll_side_effect(timeout):
        if msgs:
            return msgs.pop(0)
        return None
    
    mock_consumer_instance.poll.side_effect = poll_side_effect

    processed_messages = []
    async def consumer_callback(data):
        processed_messages.append(data)

    mock_callback = AsyncMock(side_effect=consumer_callback)

    # Mock the deserializer to return our data
    mock_deser_instance = MagicMock()
    mock_deser_instance.return_value = market_data
    consumer.avro_deserializer = mock_deser_instance

    # Start consumer
    consumer.running = True
    consume_task = asyncio.create_task(consumer.consume_messages(mock_callback))

    # Wait for processing
    # We need enough time for the loop to run at least once
    await asyncio.sleep(0.1)
    consumer.stop()
    await consume_task

    # 5. Verify final consumption
    assert len(processed_messages) == 1
    assert processed_messages[0]["symbol"] == "AAPL"