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

import pytest

from src.pricing.models.heston_fft import HestonParams


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
@patch("src.workers.math_worker.HestonCalibrator")
@patch("src.workers.math_worker.get_db_context")
@patch("src.workers.math_worker.push_metrics")
def test_recalibrate_symbol_success(
    mock_push, mock_db, mock_calibrator, mock_router, mock_market_data
):
    # Mock router (async)
    router_instance = mock_router.return_value
    router_instance.get_option_chain_snapshot = AsyncMock(return_value=mock_market_data)

    # Mock Calibrator
    calib_instance = mock_calibrator.return_value
    mock_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    mock_metrics = {"rmse": 0.01, "r_squared": 0.99, "num_options": 10, "success": True}
    calib_instance.calibrate.return_value = (mock_params, mock_metrics)
    calib_instance.calibrate_surface.return_value = {0.5: (0.1, 0.1, -0.3, 0.0, 0.1)}

    # Mock DB context
    mock_session = MagicMock()
    mock_db.return_value.__enter__.return_value = mock_session

    with patch("src.workers.math_worker.redis_client", mock_redis_client):
        # Run task
        result = recalibrate_symbol("SPY")

        assert result["status"] == "success"
        assert result["symbol"] == "SPY"

        # Verify Redis interaction
        assert mock_redis_client.setex.called

    # Verify DB interaction
    assert mock_session.add.called
    db_res = mock_session.add.call_args[0][0]
    assert db_res.symbol == "SPY"
    assert db_res.v0 == 0.04


def test_health_check_success():
    # Mock redis response for SPY (healthy) and QQQ (missing)
    mock_redis_client.get.side_effect = [
        json.dumps({"timestamp": time.time(), "params": {}}),
        None,
    ]

    with patch("src.workers.math_worker.redis_client", mock_redis_client):
        result = health_check()

    assert result["SPY"] == "healthy"
    assert result["QQQ"] == "missing"
