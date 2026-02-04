import pytest
from src.pricing.base import PricingEngine, PricingStrategy, VectorizedPricingStrategy
from src.pricing.models import BSParameters

class MockPricingStrategy(PricingStrategy):
    def price(self, params, option_type="call"):
        return 10.0
    def calculate_greeks(self, params, option_type="call"):
        return None

class MockVectorizedStrategy(VectorizedPricingStrategy):
    def price_batch(self, S, K, T, sigma, r, q, is_call):
        return None
    def price_single(self, params, option_type="call"):
        return 20.0

def test_pricing_engine_with_strategy():
    strategy = MockPricingStrategy()
    engine = PricingEngine(strategy)
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    assert engine.get_price(params) == 10.0

def test_pricing_engine_with_vectorized():
    strategy = MockVectorizedStrategy()
    engine = PricingEngine(strategy)
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    assert engine.get_price(params) == 20.0

def test_pricing_engine_unsupported():
    engine = PricingEngine("not a strategy")
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    with pytest.raises(TypeError, match="Unsupported pricing strategy type"):
        engine.get_price(params)
