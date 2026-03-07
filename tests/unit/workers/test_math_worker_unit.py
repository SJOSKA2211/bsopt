from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workers.math_worker import (
    _calibration_worker,
    _recalibrate_symbol_impl,
    recalibrate_symbol,
)


@pytest.mark.asyncio
async def test_recalibrate_symbol_async_no_data():
    # Mock MarketDataRouter to return no data
    mock_router = MagicMock()
    mock_router.get_option_chain_snapshot = AsyncMock(return_value=None)

    with patch("src.workers.math_worker.MarketDataRouter", return_value=mock_router):
        result = await _recalibrate_symbol_impl("AAPL")
        assert result["status"] == "failed"
        assert result["reason"] == "no_data"


@pytest.mark.asyncio
async def test_recalibrate_symbol_async_success():
    # Mock dependencies
    mock_router = MagicMock()
    mock_router.get_option_chain_snapshot = AsyncMock(return_value={"data": "fake"})

    # Use a regular mock for params
    mock_params = MagicMock()
    mock_params.kappa = 1.0
    mock_params.theta = 0.1
    mock_params.sigma = 0.1
    mock_params.rho = -0.5
    mock_params.v0 = 0.1

    mock_pool = MagicMock()
    mock_actor = AsyncMock()
    mock_actor.run_calibration.remote.return_value = {"status": "success", "params": {}}
    mock_pool.get_actor.return_value = mock_actor

    with (
        patch("src.workers.math_worker.MarketDataRouter", return_value=mock_router),
        patch("src.workers.math_worker.get_pool", return_value=mock_pool),
    ):
        result = await _recalibrate_symbol_impl("AAPL")

        assert result["status"] == "success"


def test_calibration_worker_integration():
    # Test the internal worker function
    mock_calibrator = MagicMock()
    mock_calibrator.calibrate.return_value = (MagicMock(), {"rmse": 0.01})
    mock_calibrator.calibrate_surface.return_value = {"1.0": [0.1]}

    with patch("src.workers.math_worker.HestonCalibrator", return_value=mock_calibrator):
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
