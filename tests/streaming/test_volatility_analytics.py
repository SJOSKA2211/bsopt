import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

# Removed top-level import to allow proper mocking
# from src.streaming.analytics import VolatilityAggregationStream

@pytest.fixture
def mock_faust_app(monkeypatch):
    """Mocks the faust.App for isolated testing."""
    import src.streaming.analytics
    mock_app_instance = MagicMock()
    mock_app_instance.topic.return_value = MagicMock()
    mock_app_instance.Table.return_value = MagicMock()
    mock_app_instance.agent.return_value = lambda x: x
    
    monkeypatch.setattr(src.streaming.analytics, "App", MagicMock(return_value=mock_app_instance))
    yield mock_app_instance

@pytest.fixture
def volatility_stream_instance(mock_faust_app):
    """Provides an instance of VolatilityAggregationStream with mocked Faust app."""
    from src.streaming.analytics import VolatilityAggregationStream
    return VolatilityAggregationStream(bootstrap_servers='kafka://mock-kafka:9092')

def test_volatility_stream_init(volatility_stream_instance, mock_faust_app):
    """Test the initialization of VolatilityAggregationStream."""
    assert volatility_stream_instance.app == mock_faust_app
    mock_faust_app.topic.assert_called_with('market-data')
    mock_faust_app.Table.assert_called_with(
        'volatility-1min',
        default=float,
        partitions=8
    )
    mock_faust_app.agent.assert_called_once_with(volatility_stream_instance.market_data_topic)

@pytest.mark.asyncio
async def test_calculate_realized_volatility_updates_table(volatility_stream_instance):
    """Test that calculate_realized_volatility processes events and updates the volatility table."""
    # Mock stream to yield events
    mock_stream = MagicMock()
    
    event1 = {'symbol': 'AAPL', 'last': 100.0, 'timestamp': 1672531200}
    event2 = {'symbol': 'AAPL', 'last': 101.0, 'timestamp': 1672531260}

    # Mock the aiter method of the object returned by group_by
    mock_group_by_result = AsyncMock()
    mock_group_by_result.__aiter__.return_value = [event1, event2]
    mock_stream.group_by.return_value = mock_group_by_result
    
    # Mock volatility_table
    volatility_stream_instance.volatility_table = MagicMock()
    # Ensure initial volatility is 0 for the symbol
    volatility_stream_instance.volatility_table.get.return_value = 0.0

    with patch.object(volatility_stream_instance, '_update_volatility') as mock_update_volatility:
        mock_update_volatility.return_value = 0.0123 # Dummy volatility value
        
        # Run the agent
        await volatility_stream_instance.calculate_realized_volatility(mock_stream)

        # Assert update_volatility was called for the second event
        mock_update_volatility.assert_called_once()
        # Assert that the volatility table was updated for the symbol
        volatility_stream_instance.volatility_table.__setitem__.assert_called_once_with('AAPL', 0.0123)
        assert volatility_stream_instance.price_history['AAPL'] == event2['last']

def test_update_volatility_calculation(volatility_stream_instance):
    """Test the _update_volatility method with a known log return."""
    symbol = 'TEST'
    log_return = np.log(101.0 / 100.0) # ~0.00995
    timestamp = 1672531260

    # Mock volatility_table
    volatility_stream_instance.volatility_table = MagicMock()
    # Simulate initial volatility in the table for the symbol
    volatility_stream_instance.volatility_table.get.return_value = 0.0 # Starting from zero

    # Perform calculation
    vol = volatility_stream_instance._update_volatility(symbol, log_return, timestamp)

    # Expected value (simplified calculation for testing)
    # variance = 0.94 * 0**2 + (1 - 0.94) * log_return**2
    # variance = 0.06 * (0.009950330853168251)**2 = 0.06 * 0.000099009 = 0.00000594054
    # annualized_vol = sqrt(0.00000594054 * 252 * 6.5 * 60) = sqrt(0.00000594054 * 98280) = sqrt(0.5839) ~= 0.764
    
    # This is a basic assertion to ensure it returns a float and is non-zero
    # A more precise assertion would require replicating the exact calculation, which might be brittle
    assert isinstance(vol, float)
    assert vol > 0
    
    # Test with a non-zero initial volatility
    volatility_stream_instance.volatility_table.get.return_value = 0.1 # Some initial annualized vol
    # Mocking current_vol for _update_volatility is more effective to isolate logic
    # but for a failing test, we let it use the mocked table.get.
    
    # Let's verify by setting return value of table.get to a specific variance directly.
    with patch.object(volatility_stream_instance.volatility_table, 'get') as mock_table_get:
        mock_table_get.return_value = 0.1 # Annualized volatility

        # Calculate expected variance from annualized volatility
        initial_annualized_vol = mock_table_get.return_value
        annualization_factor_sqrt = np.sqrt(252 * 6.5 * 60)
        initial_variance_unannualized = (initial_annualized_vol / annualization_factor_sqrt)**2 if initial_annualized_vol > 0 else 0.0

        log_return_sq = log_return**2
        alpha = 0.94
        expected_variance = alpha * initial_variance_unannualized + (1 - alpha) * log_return_sq
        expected_annualized_vol = annualization_factor_sqrt * np.sqrt(expected_variance)
        
        calculated_vol = volatility_stream_instance._update_volatility(symbol, log_return, timestamp)
        assert np.isclose(calculated_vol, expected_annualized_vol)