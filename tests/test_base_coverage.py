import numpy as np
import pytest

from src.pricing.base import (
    PricingEngine,
    PricingStrategy,
    VectorizedPricingStrategy,
)
from src.pricing.models import BSParameters, OptionGreeks


class MockPricingStrategy(PricingStrategy):
    def price(self, params: BSParameters, option_type: str = "call") -> float:
        return 10.0

    def calculate_greeks(
        self, params: BSParameters, option_type: str = "call"
    ) -> OptionGreeks:
        return OptionGreeks(0, 0, 0, 0, 0)


class MockVectorizedStrategy(VectorizedPricingStrategy):
    def price_batch(self, S, K, T, sigma, r, q, is_call) -> np.ndarray:
        return np.array([10.0])

    def price_single(self, params: BSParameters, option_type: str = "call") -> float:
        return 10.0


def test_pricing_engine_strategy():
    strategy = MockPricingStrategy()
    engine = PricingEngine(strategy)
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    assert engine.get_price(params) == 10.0


def test_pricing_engine_vectorized():
    strategy = MockVectorizedStrategy()
    engine = PricingEngine(strategy)
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    assert engine.get_price(params) == 10.0


def test_pricing_engine_invalid():
    engine = PricingEngine("not a strategy")
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    with pytest.raises(TypeError, match="Unsupported pricing strategy type"):
        engine.get_price(params)
