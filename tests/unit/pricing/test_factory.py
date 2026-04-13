from unittest.mock import patch

import pytest

from src.math_kernel.black_scholes import BlackScholesEngine
from src.math_kernel.factory import PricingEngineFactory, PricingEngineNotFound
from src.math_kernel.monte_carlo import MonteCarloEngine


def test_get_engine_lazy_load():
    # BS and MC should be pre-loaded or lazy-loaded
    bs = PricingEngineFactory.get_engine("black_scholes")
    assert isinstance(bs, BlackScholesEngine)

    mc = PricingEngineFactory.get_engine("monte_carlo")
    assert isinstance(mc, MonteCarloEngine)


def test_register_engine():
    class MockEngine:
        pass

    PricingEngineFactory.register("mock", MockEngine)
    engine = PricingEngineFactory.get_engine("mock")
    assert isinstance(engine, MockEngine)


def test_engine_not_found():
    with pytest.raises(PricingEngineNotFound, match="Unknown pricing engine"):
        PricingEngineFactory.get_engine("non_existent_engine")


def test_wasm_override():
    # If we force wasm, it should try to load wasm even if name is different
    # (Assuming WASMPricingEngine can be lazy-loaded)
    with patch("src.math_kernel.wasm_engine.WASM_AVAILABLE", True):
        # We need to mock the import or ensure it doesn't fail
        with patch("src.math_kernel.wasm_engine.WASMPricingEngine") as mock_wasm:
            PricingEngineFactory.register("wasm", mock_wasm)
            engine = PricingEngineFactory.get_engine("heston", execution_strategy="wasm")
            assert engine == PricingEngineFactory._instances["wasm"]