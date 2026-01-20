import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import asyncio
import time

# We need to mock settings before importing the worker because it initializes redis_client at module level
with patch('src.config.get_settings') as mock_settings_getter:
    mock_settings = MagicMock()
    mock_settings.REDIS_URL = "redis://localhost:6379/0"
    mock_settings.RABBITMQ_URL = "amqp://guest@localhost:5672//"
    mock_settings_getter.return_value = mock_settings
    
    # Now we can import the worker
    from src.workers.math_worker import recalibrate_symbol

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

class TestMathWorkerIntegration:
    @patch('src.workers.math_worker.HestonCalibrator')
    @patch('src.workers.math_worker.MarketDataRouter')
    @patch('src.workers.math_worker.redis_client')
    @patch('src.workers.math_worker.push_metrics')
    def test_recalibrate_symbol_flow(self, mock_push, mock_redis, mock_router, mock_calibrator, event_loop):
        """Verify the recalibrate_symbol task stores data in Redis."""
        # 1. Setup mock market data
        mock_chain = [
            {
                "T": 0.5, "strike": 100.0, "spot": 100.0, "price": 5.0,
                "bid": 4.9, "ask": 5.1, "volume": 100, "open_interest": 500,
                "option_type": "call"
            } for _ in range(10)
        ]
        
        # Mock async response correctly
        async def mock_get_snapshot(symbol):
            return mock_chain
            
        mock_router_instance = mock_router.return_value
        mock_router_instance.get_option_chain_snapshot.side_effect = mock_get_snapshot
        
        # Mock calibrator
        mock_calibrator_instance = mock_calibrator.return_value
        from src.pricing.models.heston_fft import HestonParams
        mock_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
        mock_metrics = {'rmse': 0.01, 'r_squared': 0.99, 'num_options': 10, 'success': True}
        mock_calibrator_instance.calibrate.return_value = (mock_params, mock_metrics)
        mock_calibrator_instance.calibrate_surface.return_value = {0.5: (0.1, 0.1, -0.3, 0.0, 0.1)}
        
        # Mock DB context
        with patch('src.workers.math_worker.get_db_context') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # 2. Run task
            mock_task_self = MagicMock()
            mock_task_self.request.id = "test-id"
            
            # Get the original function from the task object
            func = recalibrate_symbol.run
            if hasattr(func, '__func__'):
                func = func.__func__
                
            result = func(mock_task_self, 'SPY')
            
            # 3. Verify
            assert result['status'] == 'success'
            assert result['symbol'] == 'SPY'
            assert mock_redis.setex.called
            
            # Verify database persistence
            assert mock_session.add.called
            db_res = mock_session.add.call_args[0][0]
            assert db_res.symbol == 'SPY'
            assert db_res.v0 == 0.04