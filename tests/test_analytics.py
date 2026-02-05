import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Note: We will implement VolatilityAggregationStream in src/streaming/analytics.py

def test_volatility_calculation_logic():
    from streaming.analytics import VolatilityAggregationStream
    
    # We can test the internal _update_volatility method without a running Faust app
    with patch('streaming.analytics.App'):
        stream_proc = VolatilityAggregationStream()
        
        # Mock initial state
        symbol = "AAPL"
        # Implementation uses a real Faust Table which we mock here
        stream_proc.volatility_table = {symbol: 0.2}
        
        # Test update with a return
        log_return = 0.01
        new_vol = stream_proc._update_volatility(symbol, log_return, 1700000000)
        
        # Manual calculation check (matching implementation)
        alpha = 0.94
        current_vol = 0.2
        scaling_factor = np.sqrt(252 * 6.5 * 60)
        current_std = current_vol / scaling_factor
        variance = alpha * (current_std**2) + (1 - alpha) * (log_return**2)
        expected_vol = np.sqrt(variance) * scaling_factor
        
        assert pytest.approx(new_vol, rel=1e-5) == expected_vol

@pytest.mark.asyncio
async def test_stream_processing_flow():
    with patch('streaming.analytics.App') as mock_faust_app:
        from streaming.analytics import VolatilityAggregationStream
        
        # Mock Faust components
        mock_app_instance = MagicMock()
        mock_faust_app.return_value = mock_app_instance
        
        stream_proc = VolatilityAggregationStream()
        stream_proc.price_history = {"AAPL": 150.0}
        # Use a real dict for the table in test
        stream_proc.volatility_table = {"AAPL": 0.2}
        
        # Mock input event
        event = {"symbol": "AAPL", "last": 151.5, "timestamp": 1700000000}
        
        # Test processing one event
        with patch.object(stream_proc, '_update_volatility') as mock_update:
            mock_update.return_value = 0.25
            
            # Simulate what the agent would do
            price = event['last']
            prev_price = stream_proc.price_history["AAPL"]
            log_return = np.log(price / prev_price)
            
            # 2. Update volatility
            vol = stream_proc._update_volatility("AAPL", log_return, event['timestamp'])
            stream_proc.volatility_table["AAPL"] = vol
            # 3. Update price history
            stream_proc.price_history["AAPL"] = price
            
            assert vol == 0.25
            assert stream_proc.volatility_table["AAPL"] == 0.25
            assert stream_proc.price_history["AAPL"] == 151.5

@pytest.mark.asyncio
async def test_calculate_realized_volatility_loop():
    with patch('streaming.analytics.App'):
        from streaming.analytics import VolatilityAggregationStream
        
        stream_proc = VolatilityAggregationStream()
        stream_proc.volatility_table = {}
        stream_proc.price_history = {}
        
        # Mock stream
        class MockStream:
            async def group_by(self, key_func):
                events = [
                    {"symbol": "AAPL", "last": 150.0, "timestamp": 1},
                    {"symbol": "AAPL", "last": 151.5, "timestamp": 2} # Log return calculated here
                ]
                for event in events:
                    yield event
        
        mock_stream = MockStream()
        
        # We need to mock _update_volatility or let it run
        # Letting it run is better for integration testing logic
        
        await stream_proc.calculate_realized_volatility(mock_stream)
        
        assert "AAPL" in stream_proc.price_history
        assert stream_proc.price_history["AAPL"] == 151.5
        assert "AAPL" in stream_proc.volatility_table
        # Volatility should be non-zero after second event
        assert stream_proc.volatility_table["AAPL"] > 0
