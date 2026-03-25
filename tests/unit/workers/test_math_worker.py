import json
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Mock dependencies that are initialized at module level
mock_redis_client = MagicMock()
mock_redis_module = MagicMock()
mock_redis_module.from_url.return_value = mock_redis_client
sys.modules["redis"] = mock_redis_module

# Mock settings before import
mock_settings = MagicMock()
mock_settings.REDIS_URL = "redis://localhost:6379/0"
mock_settings.RABBITMQ_URL = "amqp://guest@localhost//"
with patch("src.workers.math_worker.get_settings", return_value=mock_settings):
    # Now we can safely import
    from src.workers.math_worker import health_check, recalibrate_symbol

import pytest  # noqa: E402

from src.math_kernel.models.heston_fft import HestonParams  # noqa: E402


@pytest.fixture
def mock_market_data():
    return [
        {
            "T": 0.5,
            "strike": 90.0 + i * 2,
            "spot": 100.0,
            "price": 10.0 - i,
            "bid": 9.9 - i,
            "ask": 10.1 - i,
            "volume": 100,
            "open_interest": 500,
            "option_type": "call",
        }
        for i in range(10)
    ]

@patch("src.workers.math_worker.MarketDataRouter")
@patch("src.workers.math_worker.get_pool")
@patch("src.shared.celery.BaseAsyncTask.run_async")
def test_recalibrate_symbol_success(mock_run_async, mock_get_pool, mock_router, mock_market_data):
    # 1. Mock Router
    router_instance = mock_router.return_value
    router_instance.get_option_chain_snapshot = AsyncMock(return_value=mock_market_data)

    # 2. Mock Ray Pool & Actor
    mock_actor = AsyncMock()
    mock_result = {
        "status": "success",
        "symbol": "SPY",
        "params": {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04},
    }
    mock_actor.run_calibration.remote = AsyncMock(return_value=mock_result)

    mock_pool = mock_get_pool.return_value
    mock_pool.get_actor.return_value = mock_actor

    # 3. Mock run_async to return the expected dict
    mock_run_async.return_value = mock_result

    # 4. Run Task
    result = recalibrate_symbol("SPY")

    assert result["status"] == "success"
    assert result["symbol"] == "SPY"
    assert "params" in result

def test_health_check_success():
    assert health_check() is True
