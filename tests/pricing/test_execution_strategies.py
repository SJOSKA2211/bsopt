from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from services.quant.pricing.execution_strategies import (
    RayStrategy,
    SequentialStrategy,
    SHMStrategy,
    StrategyFactory,
    WASMStrategy,
)


@pytest.fixture
def inputs():
    n = 10
    return {
        "spots": np.ones(n) * 100.0,
        "strikes": np.ones(n) * 100.0,
        "maturities": np.ones(n) * 1.0,
        "vols": np.ones(n) * 0.2,
        "rates": np.ones(n) * 0.05,
        "dividends": np.zeros(n),
        "is_call": np.ones(n, dtype=bool),
    }


@pytest.mark.asyncio
async def test_sequential_strategy(inputs):
    strategy = SequentialStrategy()
    with patch(
        "services.quant.pricing.black_scholes.BlackScholesEngine.price_options"
    ) as mock_price:
        mock_price.side_effect = lambda *args, **kwargs: kwargs["out"].fill(10.5)

        prices = await strategy.execute(inputs)
        assert len(prices) == 10
        assert np.all(prices == 10.5)
        mock_price.assert_called_once()


def test_strategy_factory_sequential():
    with patch("services.quant.pricing.wasm_engine.WASM_AVAILABLE", False):
        strategy = StrategyFactory.get_strategy(count=10, ray_active=False)
        assert isinstance(strategy, SequentialStrategy)


def test_strategy_factory_wasm():
    with patch("services.quant.pricing.wasm_engine.WASM_AVAILABLE", True):
        with patch("core.shared.config.settings.PRICING_LARGE_BATCH_THRESHOLD", 100):
            strategy = StrategyFactory.get_strategy(count=10, ray_active=False)
            assert isinstance(strategy, WASMStrategy)


def test_strategy_factory_shm():
    with patch("core.shared.config.settings.PRICING_LARGE_BATCH_THRESHOLD", 5):
        strategy = StrategyFactory.get_strategy(count=10, ray_active=False)
        assert isinstance(strategy, SHMStrategy)


def test_strategy_factory_ray():
    with patch("core.shared.config.settings.PRICING_LARGE_BATCH_THRESHOLD", 5):
        strategy = StrategyFactory.get_strategy(count=10, ray_active=True)
        assert isinstance(strategy, RayStrategy)


@pytest.mark.asyncio
async def test_shm_strategy_fallback(inputs):
    # Mock the service module to avoid ImportError
    mock_service = MagicMock()
    mock_service._worker_shared_memory_pricing = MagicMock()
    with patch.dict("sys.modules", {"services.pricing_service": mock_service}):
        # Test fallback when SHM acquire fails
        strategy = SHMStrategy()
        with patch("core.shared.shared_memory.shm_manager.acquire", return_value=None):
            with patch.object(SequentialStrategy, "execute", new_callable=AsyncMock) as mock_seq:
                mock_seq.return_value = np.zeros(10)
                await strategy.execute(inputs)
                mock_seq.assert_called_once()


@pytest.mark.asyncio
async def test_shm_strategy_success(inputs):
    # Mock the service module
    mock_service = MagicMock()
    # Ensure worker returns True
    mock_service._worker_shared_memory_pricing = MagicMock(return_value=True)

    with patch.dict("sys.modules", {"services.pricing_service": mock_service}):
        strategy = SHMStrategy()
        # Mock acquire to return valid names
        with patch(
            "core.shared.shared_memory.shm_manager.acquire",
            side_effect=["shm_in", "shm_out"],
        ):
            with patch("core.shared.shared_memory.shm_manager.release") as mock_release:
                # Mock SHMContextManager
                mock_cm = MagicMock()
                mock_cm.__enter__.return_value = [MagicMock(buf=bytearray(10000))]
                mock_cm.__exit__.return_value = None

                with patch("core.shared.shm_worker.SHMContextManager", return_value=mock_cm):
                    prices = await strategy.execute(inputs)
                    # Result comes from the bytearray buffer (zeros)
                    assert len(prices) == 10
                    assert mock_release.call_count == 2


@pytest.mark.asyncio
async def test_ray_strategy_success(inputs):
    # Mock the service module
    mock_service = MagicMock()
    mock_remote = MagicMock()
    mock_future = MagicMock()  # Not a real future, just a handle
    mock_remote.remote.return_value = mock_future
    mock_service._ray_worker_pricing = mock_remote

    with patch.dict("sys.modules", {"services.pricing_service": mock_service}):
        with patch("services.quant.pricing.execution_strategies.ray") as mock_ray:
            mock_ray.put.side_effect = lambda x: x
            mock_ray.get.return_value = np.ones(10)

            strategy = RayStrategy()
            prices = await strategy.execute(inputs)
            assert len(prices) == 10
            assert np.all(prices == 1.0)
