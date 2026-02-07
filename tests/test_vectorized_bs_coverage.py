import numpy as np

from src.pricing.black_scholes import BlackScholesEngine as VectorizedBlackScholesEngine
from src.pricing.models import OptionGreeks


def test_vectorized_bs_scalar_greeks():
    # Pass scalar inputs to hit line 71 branch (return from OptionGreeks)
    res = VectorizedBlackScholesEngine.calculate_greeks(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    assert isinstance(res, OptionGreeks)
    assert isinstance(res.delta, float)

def test_vectorized_bs_array_greeks():
    # Pass array inputs to hit line 69 branch (return dict directly)
    S = np.array([100.0, 110.0])
    res = VectorizedBlackScholesEngine.calculate_greeks(
        spot=S, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    assert isinstance(res, dict)
    assert len(res["delta"]) == 2

def test_vectorized_bs_price_options_scalar():
    res = VectorizedBlackScholesEngine.price_options(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    assert isinstance(res, float)
