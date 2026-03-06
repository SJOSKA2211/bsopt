from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.trading.risk_kernels import IncrementalDeltaTracker
from src.workers.math_worker import _recalibrate_symbol_impl


class TestRevampPhase1:
    def test_incremental_delta_tracker(self):
        tracker = IncrementalDeltaTracker(initial_delta=100.0, max_net_delta=1000.0)

        # Valid trade (100 + 500 = 600 <= 1000)
        assert tracker.validate_and_update(500.0) is True
        assert tracker.current_net_delta == 600.0

        # Invalid trade (600 + 500 = 1100 > 1000)
        assert tracker.validate_and_update(500.0) is False
        assert tracker.current_net_delta == 600.0

        # Valid negative trade (600 - 1500 = -900, abs(-900) <= 1000)
        assert tracker.validate_and_update(-1500.0) is True
        assert tracker.current_net_delta == -900.0

        # Reset
        tracker.reset(0.0)
        assert tracker.current_net_delta == 0.0

    @pytest.mark.asyncio
    async def test_math_worker_async_delegation(self):
        # Mocking external dependencies
        with (
            patch("src.workers.math_worker.MarketDataRouter") as mock_router_cls,
            patch("src.workers.math_worker.get_actors") as mock_get_actors,
        ):
            mock_router = mock_router_cls.return_value
            mock_router.get_option_chain_snapshot = AsyncMock(return_value=[{"price": 10.0}])

            mock_actor = MagicMock()
            # Ray remote calls: await actor.method.remote(...)
            # So remote must be an AsyncMock
            mock_actor.run_calibration.remote = AsyncMock(return_value={"status": "success"})

            mock_get_actors.return_value = [mock_actor]

            result = await _recalibrate_symbol_impl(None, None, "AAPL")

            assert result["status"] == "success"
            mock_router.get_option_chain_snapshot.assert_called_once_with("AAPL")
            mock_actor.run_calibration.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_math_worker_local_fallback(self):
        with (
            patch("src.workers.math_worker.MarketDataRouter") as mock_router_cls,
            patch("src.workers.math_worker.get_actors") as mock_get_actors,
            patch("src.workers.math_worker.HestonCalibrator") as mock_calibrator_cls,
        ):
            mock_router = mock_router_cls.return_value
            mock_router.get_option_chain_snapshot = AsyncMock(return_value=[{"price": 10.0}])

            # No actors available -> should fallback
            mock_get_actors.return_value = []

            mock_calibrator = mock_calibrator_cls.return_value
            mock_params = MagicMock()
            mock_params.kappa = 1.0
            mock_params.theta = 0.1
            mock_params.sigma = 0.2
            mock_params.rho = -0.5
            mock_params.v0 = 0.05
            mock_calibrator.calibrate.return_value = (mock_params, {"rmse": 0.01})

            result = await _recalibrate_symbol_impl(None, None, "MSFT")

            assert result["status"] == "success"
            assert result["params"]["kappa"] == 1.0
            mock_calibrator.calibrate.assert_called_once()
