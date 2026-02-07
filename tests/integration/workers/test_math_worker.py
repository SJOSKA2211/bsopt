import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We need to mock settings before importing the worker because it initializes redis_client at module level
with patch("src.config.get_settings") as mock_settings_getter:
    mock_settings = MagicMock()
    mock_settings.REDIS_URL = "redis://localhost:6379/0"
    mock_settings.RABBITMQ_URL = "amqp://guest@localhost:5672//"
    mock_settings_getter.return_value = mock_settings

    # Now we can import the worker


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestMathWorkerIntegration:
    @patch("src.workers.math_worker.HestonCalibrator")
    @patch("src.workers.math_worker.MarketDataRouter")
    @patch("src.workers.math_worker.redis_client", new_callable=AsyncMock)
    @patch("src.workers.math_worker.push_metrics")
    def test_recalibrate_symbol_flow(
        self, mock_push, mock_redis, mock_router, mock_calibrator, event_loop
    ):
        """Verify the recalibrate_symbol task stores data in Redis."""
        # 1. Setup mock market data
        mock_chain = [
            {
                "T": 0.5,
                "strike": 100.0,
                "spot": 100.0,
                "price": 5.0,
                "bid": 4.9,
                "ask": 5.1,
                "volume": 100,
                "open_interest": 500,
                "option_type": "call",
            }
            for _ in range(10)
        ]

        mock_router_instance = mock_router.return_value
        mock_router_instance.get_option_chain_snapshot = AsyncMock(
            return_value=mock_chain
        )

        # Mock redis setex
        mock_redis.setex = AsyncMock(return_value=True)
