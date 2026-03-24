import numpy as np

from src.math_kernel.black_scholes import BlackScholesEngine

def test_price_options_scalar():
    engine = BlackScholesEngine()
    # Call
    price = engine.price_options(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        option_type="call",
    )
    assert 10.4 < price < 10.5

    # Put
    price_put = engine.price_options(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        option_type="put",
    )
    assert 5.5 < price_put < 5.6

def test_price_options_vectorized():
    engine = BlackScholesEngine()
    spots = np.array([100.0, 100.0])
    strikes = np.array([100.0, 110.0])
    res = engine.price_options(
        spot=spots,
        strike=strikes,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        option_type="call",
    )
    assert len(res) == 2
    assert res[0] > res[1]

def test_calculate_greeks_scalar():
    engine = BlackScholesEngine()
    greeks = engine.calculate_greeks(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        option_type="call",
    )
    assert 0.5 < greeks.delta < 0.7
    assert greeks.vega > 0

def test_calculate_greeks_vectorized():
    engine = BlackScholesEngine()
    spots = np.array([100.0, 105.0])
    greeks = engine.calculate_greeks(
        spot=spots,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        option_type="call",
    )
    # Check if delta is an array
    assert hasattr(greeks.delta, "__len__")
    assert len(greeks.delta) == 2
    assert greeks.delta[1] > greeks.delta[0]
