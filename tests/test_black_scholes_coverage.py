import numpy as np
import pytest
from src.pricing.black_scholes import (
    BlackScholesEngine,
    BSParameters,
    black_scholes,
)
from src.pricing.models import OptionGreeks

def test_zero_maturity_d1_d2():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=0.0, volatility=0.20, rate=0.05, dividend=0.0
    )
    d1, d2 = BlackScholesEngine.calculate_d1_d2(params)
    assert d1 == 0.0
    assert d2 == 0.0

def test_zero_maturity_pricing_call():
    params = BSParameters(
        spot=100.0, strike=90.0, maturity=0.0, volatility=0.20, rate=0.05, dividend=0.0
    )
    # ITM call at expiry, price should be S - K
    price = BlackScholesEngine.price_call(params)
    assert price == 10.0

    params_otm = BSParameters(
        spot=80.0, strike=90.0, maturity=0.0, volatility=0.20, rate=0.05, dividend=0.0
    )
    price = BlackScholesEngine.price_call(params_otm)
    assert price == 0.0

def test_zero_maturity_pricing_put():
    params = BSParameters(
        spot=80.0, strike=90.0, maturity=0.0, volatility=0.20, rate=0.05, dividend=0.0
    )
    # ITM put at expiry, price should be K - S
    price = BlackScholesEngine.price_put(params)
    assert price == 10.0

    params_otm = BSParameters(
        spot=100.0, strike=90.0, maturity=0.0, volatility=0.20, rate=0.05, dividend=0.0
    )
    price = BlackScholesEngine.price_put(params_otm)
    assert price == 0.0

def test_zero_maturity_greeks():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=0.0, volatility=0.20, rate=0.05, dividend=0.0
    )
    greeks = BlackScholesEngine().calculate_greeks(params, "call")
    assert greeks.vega == 0.0
    assert greeks.theta == 0.0
    assert greeks.rho == 0.0

def test_black_scholes_wrapper():
    res = black_scholes(100.0, 100.0, 1.0, 0.20, 0.05, type='call', q=0.0)
    assert res['price'] > 0

def test_price_options_batch_scalar():
    price = BlackScholesEngine.price_options(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05
    )
    assert isinstance(price, float)
    assert price > 0

def test_price_options_batch_vector():
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.20, 0.20])
    r = np.array([0.05, 0.05])
    
    prices = BlackScholesEngine.price_options(
        spot=S, strike=K, maturity=T, volatility=sigma, rate=r
    )
    assert isinstance(prices, np.ndarray)
    assert len(prices) == 2
    assert prices[0] == prices[1]

def test_price_options_batch_mixed_types():
    # Test with list of types
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.20, 0.20])
    r = np.array([0.05, 0.05])
    
    prices = BlackScholesEngine.price_options(
        spot=S, strike=K, maturity=T, volatility=sigma, rate=r, option_type=["call", "put"]
    )
    assert isinstance(prices, np.ndarray)
    assert len(prices) == 2
    assert prices[0] != prices[1] # Call vs Put

def test_calculate_greeks_batch_scalar():
    greeks = BlackScholesEngine.calculate_greeks_batch(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05
    )
    assert isinstance(greeks, OptionGreeks)

def test_calculate_greeks_batch_vector():
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.20, 0.20])
    r = np.array([0.05, 0.05])
    
    greeks = BlackScholesEngine.calculate_greeks_batch(
        spot=S, strike=K, maturity=T, volatility=sigma, rate=r
    )
    assert isinstance(greeks, dict)
    assert "delta" in greeks
    assert len(greeks["delta"]) == 2

def test_price_instance_method():
    engine = BlackScholesEngine()
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.0
    )
    price = engine.price(params, "call")
    assert price > 0
    price_put = engine.price(params, "put")
    assert price_put > 0

def test_calculate_greeks_batch_mixed_types():
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.20, 0.20])
    r = np.array([0.05, 0.05])
    
    greeks = BlackScholesEngine.calculate_greeks_batch(
        spot=S, strike=K, maturity=T, volatility=sigma, rate=r, option_type=["call", "put"]
    )
    assert isinstance(greeks, dict)
    # Check that we got different deltas for call and put
    assert greeks["delta"][0] != greeks["delta"][1]

def test_calculate_greeks_call_coverage():
    # Explicitly test call greeks with non-zero dividend to hit all parts of the formula
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02
    )
    greeks = BlackScholesEngine().calculate_greeks(params, "call")
    assert greeks.theta != 0

def test_calculate_greeks_static():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02
    )
    # Call the static method directly
    greeks = BlackScholesEngine.calculate_greeks_static(params, "call")
    assert greeks.delta > 0

def test_price_options_with_params():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05
    )
    price = BlackScholesEngine.price_options(params=params)
    assert price > 0

def test_price_options_missing_args():
    with pytest.raises(TypeError, match="Missing required pricing parameters"):
        BlackScholesEngine.price_options(spot=100.0)

def test_calculate_greeks_batch_with_params():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05
    )
    greeks = BlackScholesEngine.calculate_greeks_batch(params=params)
    assert isinstance(greeks, OptionGreeks)

def test_calculate_greeks_batch_missing_args():
    with pytest.raises(TypeError, match="Missing required pricing parameters"):
        BlackScholesEngine.calculate_greeks_batch(spot=100.0)

def test_verify_put_call_parity():
    from src.pricing.black_scholes import verify_put_call_parity
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.0
    )
    assert verify_put_call_parity(params)

    # Broken parity test (high dividend vs none in parity check context? No, just modify price manually if possible, 
    # but here we test the function logic. If parameters are consistent, it should be true.)
    
def test_calculate_vector_expired():
    # Test calculate with vector of expired options
    S = np.array([100.0, 90.0])
    K = np.array([90.0, 100.0])
    T = np.array([0.0, 0.0]) # All expired
    sigma = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    
    params = BSParameters(spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=q)
    engine = BlackScholesEngine()
    
    # Call
    res_call = engine.calculate(params, "call")
    # 100-90 = 10 (ITM), 90-100 = 0 (OTM)
    np.testing.assert_array_almost_equal(res_call["price"], np.array([10.0, 0.0]))
    
    # Put
    res_put = engine.calculate(params, "put")
    # 90-100 = 0 (OTM), 100-90 = 10 (ITM)
    np.testing.assert_array_almost_equal(res_put["price"], np.array([0.0, 10.0]))

def test_price_instance_method_kwargs():
    engine = BlackScholesEngine()
    # Test passing kwargs instead of params object
    price = engine.price(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.0,
        option_type="call"
    )
    assert price > 0

def test_calculate_greeks_put_coverage():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02
    )
    # Use instance method or static wrapper that calls _calculate_greeks_internal
    greeks = BlackScholesEngine().calculate_greeks(params, "put")
    assert greeks.delta < 0 # Put delta is negative
    assert greeks.theta != 0

    