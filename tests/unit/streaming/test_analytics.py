import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.streaming.analytics import VolatilityAggregationStream
import numpy as np

@pytest.fixture
def volatility_stream():
    with patch('src.streaming.analytics.App'):
        stream = VolatilityAggregationStream(bootstrap_servers='kafka://localhost:9092')
        return stream

def test_init(volatility_stream):
    assert volatility_stream.app is not None
    assert volatility_stream.market_data_topic is not None
    assert volatility_stream.volatility_table is not None

def test_update_volatility(volatility_stream):
    symbol = "AAPL"
    log_return = 0.01
    timestamp = 1600000000
    
    volatility_stream.volatility_table = {symbol: 0.2}
    
    new_vol = volatility_stream._update_volatility(symbol, log_return, timestamp)
    assert new_vol > 0
    assert isinstance(new_vol, float)

def test_update_volatility_zero_current(volatility_stream):
    symbol = "GOOG"
    log_return = 0.02
    timestamp = 1600000000
    
    volatility_stream.volatility_table = {}
    
    new_vol = volatility_stream._update_volatility(symbol, log_return, timestamp)
    assert new_vol > 0

@pytest.mark.asyncio
async def test_calculate_realized_volatility(volatility_stream):
    mock_event = {'symbol': 'TSLA', 'last': 100.0, 'timestamp': 1600000000}
    mock_event2 = {'symbol': 'TSLA', 'last': 101.0, 'timestamp': 1600000060}
    
    mock_stream = MagicMock()
    mock_stream.group_by.return_value = AsyncMock()
    mock_stream.group_by.return_value.__aiter__.return_value = [mock_event, mock_event2]
    
    volatility_stream.volatility_table = MagicMock()
    volatility_stream.volatility_table.get.return_value = 0.0
    
    await volatility_stream.calculate_realized_volatility(mock_stream)
    
    assert volatility_stream.price_history['TSLA'] == 101.0
    # The table should have been updated after the second event
    assert volatility_stream.volatility_table.__setitem__.called
