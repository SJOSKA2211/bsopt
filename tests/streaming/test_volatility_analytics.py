import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
import numpy as np

# Assuming src.streaming.analytics.py is correctly in the path
from src.streaming.analytics import VolatilityAggregationStream

@pytest.fixture
def mock_faust_app():
    """Mocks the faust.App for isolated testing."""
    with patch('src.streaming.analytics.App', autospec=True) as MockApp:
        mock_app_instance = MockApp.return_value
        mock_app_instance.topic = MagicMock(return_value=MagicMock())
        mock_app_instance.Table = MagicMock(return_value=MagicMock())
        # Mock agent decorator to return the function it decorates
        mock_app_instance.agent = MagicMock(side_effect=lambda *args, **kwargs: lambda func: func)
        yield mock_app_instance

@pytest.fixture
def volatility_stream_instance(mock_faust_app):
    """Provides an instance of VolatilityAggregationStream with mocked Faust app."""
    return VolatilityAggregationStream(bootstrap_servers='kafka://mock-kafka:9092')

def test_volatility_stream_init(volatility_stream_instance, mock_faust_app):
    """Test the initialization of VolatilityAggregationStream."""
    assert volatility_stream_instance.app == mock_faust_app
    mock_faust_app.topic.assert_called_with('market-data', partitions=16)
    mock_faust_app.Table.assert_any_call(
        'volatility-1min-v2',
        default=float,
        partitions=16
    )
    mock_faust_app.Table.assert_any_call(
        'price-history-v2',
        default=float,
        partitions=16
    )
    # The agent decorator is called with the topic
    mock_faust_app.agent.assert_called_once_with(volatility_stream_instance.market_data_topic)

@pytest.mark.asyncio
async def test_calculate_realized_volatility_updates_table(volatility_stream_instance):
    """Test that calculate_realized_volatility processes events and updates the volatility table."""
    symbol = 'AAPL'
    
    # 1. First event - sets initial price history
    event1 = {'symbol': symbol, 'last': 100.0, 'timestamp': 1672531200}
    
    # Mock tables
    volatility_stream_instance.price_history = MagicMock()
    volatility_stream_instance.price_history.__getitem__.return_value = 0.0 # Initial state
    volatility_stream_instance.volatility_table = MagicMock()
    
    await volatility_stream_instance.calculate_realized_volatility(event1)
    
    # Verify price history updated
    volatility_stream_instance.price_history.__setitem__.assert_called_with(symbol, 100.0)
    # Volatility should NOT be updated on first tick (no prev price)
    volatility_stream_instance.volatility_table.__setitem__.assert_not_called()

    # 2. Second event - calculates volatility
    event2 = {'symbol': symbol, 'last': 101.0, 'timestamp': 1672531260}
    
    # Setup previous price
    volatility_stream_instance.price_history.__getitem__.return_value = 100.0
    # Setup current variance window
    volatility_stream_instance.volatility_table.__getitem__.return_value.now.return_value = 0.0
    
    await volatility_stream_instance.calculate_realized_volatility(event2)
    
    # Calculate expected
    log_return = np.log(101.0 / 100.0)
    expected_variance = 0.0 + log_return**2
    
    # Verify volatility table update
    volatility_stream_instance.volatility_table.__setitem__.assert_called_with(symbol, expected_variance)
    # Verify price history updated
    volatility_stream_instance.price_history.__setitem__.assert_called_with(symbol, 101.0)
