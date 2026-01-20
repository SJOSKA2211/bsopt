import pytest
from src.pricing.base import PricingStrategy, VectorizedPricingStrategy, PricingEngine
from src.pricing.models import BSParameters, OptionGreeks

class MockPricingStrategy(PricingStrategy):
    def price(self, params: BSParameters, option_type: str = "call") -> float:
        super().price(params, option_type)
        return 10.0
    
    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        super().calculate_greeks(params, option_type)
        return OptionGreeks(delta=0.5, gamma=0.1, vega=0.2, theta=-0.05, rho=0.1)

class MockVectorizedStrategy(VectorizedPricingStrategy):
    def price_batch(self, S, K, T, sigma, r, q, is_call):
        super().price_batch(S, K, T, sigma, r, q, is_call)
        import numpy as np
        return np.array([10.0])
    
    def price_single(self, params: BSParameters, option_type: str = "call") -> float:
        super().price_single(params, option_type)
        return 15.0

def test_pricing_engine_with_strategy():
    strategy = MockPricingStrategy()
    engine = PricingEngine(strategy)
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    assert engine.get_price(params) == 10.0

def test_pricing_engine_with_vectorized_strategy():
    strategy = MockVectorizedStrategy()
    engine = PricingEngine(strategy)
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    assert engine.get_price(params) == 15.0

def test_pricing_engine_invalid_strategy():
    with pytest.raises(TypeError, match="Unsupported pricing strategy type"):
        engine = PricingEngine("not a strategy")
        params = BSParameters(100, 100, 1.0, 0.2, 0.05)
        engine.get_price(params)

def test_mock_strategy_greeks():
    strategy = MockPricingStrategy()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    greeks = strategy.calculate_greeks(params)
    assert greeks.delta == 0.5

def test_mock_vectorized_batch():
    strategy = MockVectorizedStrategy()
    import numpy as np
    res = strategy.price_batch(None, None, None, None, None, None, None)
    assert res[0] == 10.0
