import pytest
import numpy as np
from src.pricing.vectorized_black_scholes import VectorizedBlackScholesEngine

def test_deprecated_vectorized_pricing():
    with pytest.warns(DeprecationWarning):
        prices = VectorizedBlackScholesEngine.price_options(100, 100, 1, 0.2, 0.05)
        assert isinstance(prices, np.ndarray)
        assert prices[0] > 0

def test_deprecated_vectorized_greeks():
    with pytest.warns(DeprecationWarning):
        # Case where it returns a dict
        greeks_dict = VectorizedBlackScholesEngine.calculate_greeks(
            np.array([100, 110]), 100, 1, 0.2, 0.05
        )
        assert isinstance(greeks_dict, dict)
        assert "delta" in greeks_dict
        
        # Case where it returns OptionGreeks and we convert it
        greeks_single = VectorizedBlackScholesEngine.calculate_greeks(
            100, 100, 1, 0.2, 0.05
        )
        assert isinstance(greeks_single, dict)
        assert len(greeks_single["delta"]) == 1
