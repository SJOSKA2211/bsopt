import numpy as np

from src.pricing.black_scholes import BlackScholesEngine as VectorizedBlackScholesEngine


def test_deprecated_wrapper_price():
    prices = VectorizedBlackScholesEngine.price_options(100.0, 100.0, 1.0, 0.2, 0.05)
    assert isinstance(prices, np.ndarray)
    assert prices[0] > 0


def test_deprecated_wrapper_greeks():
    greeks = VectorizedBlackScholesEngine.calculate_greeks(100.0, 100.0, 1.0, 0.2, 0.05)
    assert isinstance(greeks, dict)
    assert "delta" in greeks
    assert isinstance(greeks["delta"], np.ndarray)
