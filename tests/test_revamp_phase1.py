from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.math_kernel.risk_kernels import IncrementalDeltaTracker
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
            patch("src.workers.math_worker.get_pool") as mock_get_pool,
        ):
            mock_router = mock_router_cls.return_value
            mock_router.get_option_chain_snapshot = AsyncMock(return_value=[{"price": 10.0}])

            mock_pool = MagicMock()
            mock_actor = MagicMock()
            # Ray remote calls: await actor.method.remote(...)
            mock_actor.run_calibration.remote = AsyncMock(return_value={"status": "success"})
            mock_pool.get_actor.return_value = mock_actor
            mock_get_pool.return_value = mock_pool

            # Actual signature: _recalibrate_symbol_impl(symbol: str)
            result = await _recalibrate_symbol_impl("AAPL")

            assert result["status"] == "success"
            mock_router.get_option_chain_snapshot.assert_called_once_with("AAPL")
            mock_actor.run_calibration.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_math_worker_local_fallback(self):
        with (
            patch("src.workers.math_worker.MarketDataRouter") as mock_router_cls,
            patch("src.workers.math_worker.get_pool") as mock_get_pool,
            patch("src.workers.math_worker.HestonCalibrator") as mock_calibrator_cls,
        ):
            mock_router = mock_router_cls.return_value
            mock_router.get_option_chain_snapshot = AsyncMock(return_value=[{"price": 10.0}])

            # No pool/actor available -> should fallback
            mock_get_pool.return_value.get_actor.side_effect = Exception("Ray down")

            mock_calibrator = mock_calibrator_cls.return_value
            mock_params = MagicMock()
            mock_params.kappa = 1.0
            mock_params.theta = 0.1
            mock_params.sigma = 0.2
            mock_params.rho = -0.5
            mock_params.v0 = 0.05
            mock_calibrator.calibrate.return_value = (mock_params, {"rmse": 0.01})

            result = await _recalibrate_symbol_impl("MSFT")

            assert result["status"] == "success"
            assert result["params"]["kappa"] == 1.0
            mock_calibrator.calibrate.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_drift_correction(self):
        """Verify that check_risk_limits detects drift and resets tracker/Redis/SHM."""
        import src.workers.tasks.trading_tasks as tt
        from src.math_kernel.risk_kernels import IncrementalDeltaTracker

        # 1. Setup Tracker with DRIFT (Local = 500, DB = 1000)
        tracker = IncrementalDeltaTracker(initial_delta=500.0, max_net_delta=5000.0)

        # Save originals
        orig_get_tracker = tt.get_persistent_delta_tracker
        orig_get_actual = tt.get_actual_portfolio_delta

        # Manually override in module
        tt.get_persistent_delta_tracker = lambda: tracker
        tt.get_actual_portfolio_delta = lambda pid: 1000.0

        try:
            with (
                patch("src.shared.utils.cache.get_redis") as mock_get_redis,
                patch("src.shared.shm_mesh.RiskStateBuffer") as mock_shm_cls,
            ):
                mock_redis = AsyncMock()
                mock_get_redis.return_value = mock_redis
                mock_shm = mock_shm_cls.return_value

                # 2. Run Task
                # Patch run_async on the task instance directly
                with patch.object(tt.check_risk_limits, "run_async", side_effect=lambda x: x):
                    result = tt.check_risk_limits.run("portfolio_123")

                # 3. Verify Drift Correction
                assert result["status"] == "success"
                assert tracker.current_net_delta == 1000.0
                assert result["net_delta"] == 1000.0

                mock_redis.set.assert_called_with("portfolio_net_delta", "1000.0")
                mock_shm.update.assert_called_with(1000.0, 5000.0)
        finally:
            # Restore originals
            tt.get_persistent_delta_tracker = orig_get_tracker
            tt.get_actual_portfolio_delta = orig_get_actual
