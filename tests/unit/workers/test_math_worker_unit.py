from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workers.math_worker import (
    _calibration_worker,
    _recalibrate_symbol_async,
    recalibrate_symbol,
)


@pytest.mark.asyncio
async def test_recalibrate_symbol_async_no_data():
    # Mock MarketDataRouter to return no data
    mock_router = MagicMock()
    mock_router.get_option_chain_snapshot = AsyncMock(return_value=None)

    with patch("src.workers.math_worker.MarketDataRouter", return_value=mock_router):
        result = await _recalibrate_symbol_async(None, "AAPL")
        assert result["status"] == "failed"
        assert result["reason"] == "no_data"


@pytest.mark.asyncio
async def test_recalibrate_symbol_async_success():
    # Mock dependencies
    mock_router = MagicMock()
    mock_router.get_option_chain_snapshot = AsyncMock(return_value={"data": "fake"})

    # Use a regular mock for params
    mock_params = MagicMock()
    mock_params.v0 = 0.1
    mock_params.kappa = 1.0
    mock_params.theta = 0.1
    mock_params.sigma = 0.1
    mock_params.rho = -0.5
    # Manually mock __dict__ since we use it in the code
    mock_params.__dict__ = {
        "v0": 0.1,
        "kappa": 1.0,
        "theta": 0.1,
        "sigma": 0.1,
        "rho": -0.5,
    }

    mock_metrics = {"rmse": 0.01, "r_squared": 0.99, "num_options": 100}
    mock_surface = {"1.0": [0.1, 0.2]}

    mock_redis = AsyncMock()
    mock_db = MagicMock()
    mock_db_context = AsyncMock()
    mock_db.__aenter__.return_value = mock_db_context

    with (
        patch("src.workers.math_worker.MarketDataRouter", return_value=mock_router),
        patch("src.workers.math_worker.executor"),
        patch("src.workers.math_worker.async_redis_client", mock_redis),
        patch("src.workers.math_worker.get_async_db_context", return_value=mock_db),
        patch("asyncio.get_event_loop") as mock_loop,
    ):

        # Mock executor result
        mock_loop.return_value.run_in_executor = AsyncMock(
            return_value=(mock_params, mock_metrics, mock_surface)
        )

        # Use a mock for self that has a retry method
        mock_self = MagicMock()
        result = await _recalibrate_symbol_async(mock_self, "AAPL")

        assert result["status"] == "success"
        mock_redis.setex.assert_called_once()


def test_calibration_worker_integration():
    # Test the internal worker function
    mock_calibrator = MagicMock()
    mock_calibrator.calibrate.return_value = (MagicMock(), {"rmse": 0.01})
    mock_calibrator.calibrate_surface.return_value = {"1.0": [0.1]}

    with patch(
        "src.workers.math_worker.HestonCalibrator", return_value=mock_calibrator
    ):
        params, metrics, surface = _calibration_worker({"data": "test"})
        assert metrics["rmse"] == 0.01
        assert "1.0" in surface


def test_recalibrate_symbol_task_failure():
    # Test the Celery task wrapper failure path
    mock_self = MagicMock()
    mock_self.retry = MagicMock(side_effect=Exception("Retry Triggered"))

    with (
        patch("src.workers.math_worker.math_swarm", [MagicMock()]),
        patch("ray.get", side_effect=Exception("Ray Exploded")),
        patch("asyncio.run") as mock_run,
    ):

        # Access the raw function directly from the task object's __wrapped__ attribute if it exists,
        # otherwise use the function itself. Celery tasks wrap the function.
        if hasattr(recalibrate_symbol, "__wrapped__"):
            # If it's a bound method, we need to call it without the first argument (self)
            # or pass the mock_self as the first argument if it's not bound.
            raw_func = recalibrate_symbol.__wrapped__

            # Use inspection to see if it's bound
            if hasattr(raw_func, "__self__"):
                # It's bound, so we only pass the symbol
                raw_func("AAPL")
            else:
                # It's not bound, pass both
                raw_func(mock_self, "AAPL")
        else:
            recalibrate_symbol(mock_self, "AAPL")

        mock_run.assert_called_once()
