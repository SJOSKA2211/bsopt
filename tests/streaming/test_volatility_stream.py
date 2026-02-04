import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np 

# Assuming VolatilityAggregationStream will be importable from src.streaming.analytics
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from streaming.analytics import VolatilityAggregationStream
except ImportError:
    VolatilityAggregationStream = None


VOLATILITY_STREAM_PATH = "src/streaming/analytics.py"
TEST_BOOTSTRAP_SERVERS = "kafka://localhost:9092"
TEST_TOPIC = "market-data"
TEST_TABLE_NAME = "volatility-1min"


def test_volatility_stream_file_exists():
    """
    Test that the analytics.py file exists for VolatilityAggregationStream.
    """
    assert os.path.exists(VOLATILITY_STREAM_PATH), f"VolatilityAggregationStream file not found at {VOLATILITY_STREAM_PATH}"


def test_volatility_aggregation_stream_class_exists():
    """
    Test that the VolatilityAggregationStream class can be imported.
    This test will fail if the class is not yet defined or importable.
    """
    assert VolatilityAggregationStream is not None, "VolatilityAggregationStream class is not defined or importable."


@patch('streaming.analytics.App') 
def test_volatility_aggregation_stream_init(mock_app):
    """
    Test that VolatilityAggregationStream constructor initializes Faust App, Topic, and Table.
    """
    mock_app_instance = mock_app.return_value
    mock_app_instance.topic.return_value = MagicMock()
    mock_table_instance = MagicMock()
    mock_app_instance.Table.return_value = mock_table_instance

    stream = VolatilityAggregationStream(bootstrap_servers=TEST_BOOTSTRAP_SERVERS)

    # Simplified assertions: check if attributes are set
    assert hasattr(stream, 'app')
    assert hasattr(stream, 'market_data_topic')
    assert hasattr(stream, 'volatility_table')
    assert hasattr(stream, 'price_history')
    assert mock_app.called
    assert mock_app_instance.topic.called
    assert mock_app_instance.Table.called
    assert mock_app_instance.agent.called # Verify agent registration

@pytest.mark.asyncio
@patch('streaming.analytics.np') # Correct patch target
async def test_update_volatility_calculation(mock_np):
    """
    Test the _update_volatility method directly.
    """
    # Use real numpy functions but keep the module patched
    mock_np.log.side_effect = np.log
    mock_np.sqrt.side_effect = np.sqrt

    stream = VolatilityAggregationStream(bootstrap_servers=TEST_BOOTSTRAP_SERVERS)
    stream.price_history = {'AAPL': 100.0}
    stream.volatility_table = {} # Mock as a dict for direct interaction

    # Simulate first event for log_return
    symbol = 'AAPL'
    price_t1 = 101.0
    timestamp_t1 = 1060
    log_return_t1 = np.log(price_t1 / stream.price_history[symbol])
    stream.price_history[symbol] = price_t1

    # Call _update_volatility directly
    volatility = stream._update_volatility(symbol, log_return_t1, timestamp_t1)
    
    assert volatility > 0 # Expect some positive volatility
    mock_np.sqrt.assert_called()

    # Simulate a second event to ensure EMA works
    price_t2 = 102.0
    timestamp_t2 = 1120
    log_return_t2 = np.log(price_t2 / stream.price_history[symbol])
    stream.price_history[symbol] = price_t2
    
    volatility_t2 = stream._update_volatility(symbol, log_return_t2, timestamp_t2)
    assert volatility_t2 != volatility # Should change with new data
