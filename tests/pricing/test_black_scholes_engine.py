import pytest
import numpy as np
from src.pricing.black_scholes import BlackScholesEngine, BSParameters, verify_put_call_parity, black_scholes

def test_price_options_call_scalar():
    engine = BlackScholesEngine()
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    price = engine.price(params, option_type="call")
    assert isinstance(price, float)
    assert price > 0

def test_price_options_put_scalar():
    engine = BlackScholesEngine()
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    price = engine.price(params, option_type="put")
    assert isinstance(price, float)
    assert price > 0

def test_price_options_vectorized():
    engine = BlackScholesEngine()
    spots = np.array([100, 110])
    params = BSParameters(spot=spots, strike=100, maturity=1, volatility=0.2, rate=0.05)
    prices = engine.price(params, option_type="call")
    assert len(prices) == 2
    assert all(prices > 0)

def test_price_options_kwargs():
    engine = BlackScholesEngine()
    price = engine.price(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05, option_type="call")
    assert price > 0

def test_calculate_greeks():
    engine = BlackScholesEngine()
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    greeks = engine.calculate_greeks(params, option_type="call")
    assert "delta" in greeks
    assert "gamma" in greeks
    assert "vega" in greeks
    assert "theta" in greeks
    assert "rho" in greeks

def test_edge_cases():
    engine = BlackScholesEngine()
    # Zero maturity
    params = BSParameters(spot=100, strike=100, maturity=0, volatility=0.2, rate=0.05)
    price = engine.price(params, option_type="call")
    assert price == 0.0
    
    # Very high volatility
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=10.0, rate=0.05)
    price = engine.price(params, option_type="call")
    assert price > 0

def test_static_price_methods():
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    assert BlackScholesEngine.price_call(params) > 0
    assert BlackScholesEngine.price_put(params) > 0

def test_price_options_static():
    price = BlackScholesEngine.price_options(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05, option_type="call")
    assert price > 0
    
    prices = BlackScholesEngine.price_options(spot=[100, 110], strike=100, maturity=1, volatility=0.2, rate=0.05, option_type="call")
    assert len(prices) == 2

def test_calculate_greeks_batch():
    greeks = BlackScholesEngine.calculate_greeks_batch(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05, option_type="call")
    assert greeks.delta > 0
    
    greeks_dict = BlackScholesEngine.calculate_greeks_batch(spot=[100, 110], strike=100, maturity=1, volatility=0.2, rate=0.05, option_type=["call", "put"])
    assert "delta" in greeks_dict
    assert len(greeks_dict["delta"]) == 2

def test_put_call_parity():
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    assert verify_put_call_parity(params)

def test_black_scholes_wrapper():
    res = black_scholes(100, 100, 1, 0.2, 0.05, type='call')
    assert res['price'] > 0

def test_greeks_put():
    engine = BlackScholesEngine()
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    greeks = engine.calculate_greeks(params, option_type="put")
    assert greeks.delta < 0

def test_greeks_zero_maturity():
    engine = BlackScholesEngine()
    params = BSParameters(spot=100, strike=100, maturity=0, volatility=0.2, rate=0.05)
    greeks = engine.calculate_greeks(params, option_type="call")
    assert greeks.vega == 0.0

def test_price_options_with_params_static():
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    assert BlackScholesEngine.price_options(params=params) > 0

def test_calculate_greeks_batch_with_params():
    params = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    res = BlackScholesEngine.calculate_greeks_batch(params=params)
    assert res.delta > 0

def test_price_options_missing_params():
    with pytest.raises(TypeError, match="Missing required pricing parameters"):
        BlackScholesEngine.price_options(spot=100)

def test_calculate_greeks_batch_missing_params():
    with pytest.raises(TypeError, match="Missing required pricing parameters"):
        BlackScholesEngine.calculate_greeks_batch(spot=100)

def test_price_options_vectorized_types():
    # Trigger line 164
    prices = BlackScholesEngine.price_options(spot=[100], strike=100, maturity=1, volatility=0.2, rate=0.05, option_type=["call"])
    assert len(prices) == 1