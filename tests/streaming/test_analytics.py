import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.streaming.analytics import VolatilityAggregationStream
import numpy as np

@pytest.fixture
def mock_faust():
    with patch("src.streaming.analytics.App") as mock_app:
        # mock_app is the App class, mock_app.return_value is the app instance
        instance = mock_app.return_value
        instance.topic.return_value = MagicMock()
        instance.Table.return_value = {} # Mock table as a dict
        instance.agent = MagicMock(return_value=lambda f: f)
        yield instance

def test_volatility_aggregation_init(mock_faust):
    stream = VolatilityAggregationStream()
    assert stream.app == mock_faust
    mock_faust.topic.assert_called_with('market-data')
    mock_faust.Table.assert_called()

def test_update_volatility():
    with patch("src.streaming.analytics.App"):
        stream = VolatilityAggregationStream()
        stream.volatility_table = {"AAPL": 0.2}
        
        # Test calculation
        new_vol = stream._update_volatility("AAPL", 0.01, 123456789)
        assert new_vol > 0
        assert isinstance(new_vol, float)

@pytest.mark.asyncio
async def test_calculate_realized_volatility(mock_faust):
    stream = VolatilityAggregationStream()
    stream.volatility_table = mock_faust.Table.return_value
    
    # Mock stream: group_by returns an async iterator
    mock_event = {"symbol": "AAPL", "last": 150.0, "timestamp": 1000}
    mock_event2 = {"symbol": "AAPL", "last": 151.5, "timestamp": 1001}
    
    mock_grouped_stream = AsyncMock()
    mock_grouped_stream.__aiter__.return_value = [mock_event, mock_event2]
    
    mock_input_stream = MagicMock()
    mock_input_stream.group_by.return_value = mock_grouped_stream
    
    await stream.calculate_realized_volatility(mock_input_stream)
    
    assert "AAPL" in stream.price_history
    assert stream.price_history["AAPL"] == 151.5
    assert "AAPL" in stream.volatility_table
