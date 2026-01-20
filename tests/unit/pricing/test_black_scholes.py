import pytest
import numpy as np
from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.pricing.models import OptionGreeks

def test_bs_engine_single_pricing():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02)
    
    call_price = BlackScholesEngine.price_options(params=params, option_type="call")
    put_price = BlackScholesEngine.price_options(params=params, option_type="put")
    
    assert isinstance(call_price, float)
    assert call_price > 0
    assert put_price > 0
    # With r=0.05, q=0.02, S=K, call should be more than put
    assert call_price > put_price

def test_bs_engine_greeks():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0)
    engine = BlackScholesEngine()
    greeks = engine.calculate_greeks(params=params, option_type="call")
    
    assert 0 < greeks.delta < 1
    assert greeks.gamma > 0
    assert greeks.vega > 0
    assert greeks.theta < 0
    assert greeks.rho > 0

def test_bs_engine_batch_pricing():
    spots = np.array([100.0, 110.0])
    params = BSParameters(spot=spots, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0)
    
    prices = BlackScholesEngine.price_options(params=params, option_type="call")
    assert isinstance(prices, np.ndarray)
    assert len(prices) == 2
    assert prices[1] > prices[0] # S is higher

def test_bs_engine_edge_cases():
    # Zero maturity
    engine = BlackScholesEngine()
    params = BSParameters(spot=100.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    res = engine.calculate(params, type="call")
    assert res["price"] == 0.0 # ATM at maturity
    assert res["delta"] == 0.0
    
    params_itm = BSParameters(spot=110.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    res_itm = engine.calculate(params_itm, type="call")
    assert res_itm["price"] == 10.0 # Intrinsic value

    res_put = engine.calculate(params, type="put")
    assert res_put["price"] == 0.0

def test_calculate_d1_d2():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    d1, d2 = BlackScholesEngine.calculate_d1_d2(params)
    assert isinstance(d1, float)
    assert isinstance(d2, float)
    assert d1 > d2

    # Zero maturity
    params_zero = BSParameters(spot=100.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    d1_z, d2_z = BlackScholesEngine.calculate_d1_d2(params_zero)
    assert d1_z == 0.0
    assert d2_z == 0.0

def test_bs_engine_calculate_put():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0)
    engine = BlackScholesEngine()
    res = engine.calculate(params, type='put')
    assert res['price'] > 0
    assert res['delta'] < 0

def test_bs_engine_static_price_methods():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    call_price = BlackScholesEngine.price_call(params)
    put_price = BlackScholesEngine.price_put(params)
    assert call_price > 0
    assert put_price > 0

def test_bs_engine_greeks_put():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    greeks = BlackScholesEngine.calculate_greeks_static(params, option_type="put")
    assert greeks.delta < 0
    assert greeks.rho < 0

def test_bs_engine_greeks_zero_maturity():
    params = BSParameters(spot=100.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    greeks = BlackScholesEngine.calculate_greeks_static(params, option_type="call")
    assert greeks.vega == 0.0
    assert greeks.theta == 0.0
    assert greeks.rho == 0.0

def test_price_options_missing_params():
    with pytest.raises(TypeError):
        BlackScholesEngine.price_options(spot=100.0)

def test_calculate_greeks_batch_missing_params():
    with pytest.raises(TypeError):
        BlackScholesEngine.calculate_greeks_batch(spot=100.0)

def test_calculate_greeks_batch_single():
    res = BlackScholesEngine.calculate_greeks_batch(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    assert isinstance(res, OptionGreeks)
    assert res.delta > 0

def test_verify_put_call_parity():
    from src.pricing.black_scholes import verify_put_call_parity
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02)
    assert verify_put_call_parity(params)

def test_bs_engine_price_instance():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    engine = BlackScholesEngine()
    price = engine.price(params, option_type="call")
    assert price > 0

def test_bs_engine_calculate_greeks_instance():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    engine = BlackScholesEngine()
    greeks_call = engine.calculate_greeks(params, option_type="call")
    assert greeks_call.delta > 0
    greeks_put = engine.calculate_greeks(params, option_type="put")
    assert greeks_put.delta < 0

def test_price_options_array_type():
    spots = np.array([100.0, 100.0])
    option_types = np.array(["call", "put"])
    prices = BlackScholesEngine.price_options(
        spot=spots, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, option_type=option_types
    )
    assert len(prices) == 2
    assert prices[0] > prices[1]

def test_calculate_greeks_batch_with_params():
    params = BSParameters(spot=np.array([100.0, 110.0]), strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    res = BlackScholesEngine.calculate_greeks_batch(params=params)
    assert isinstance(res, dict)
    assert "delta" in res
    assert len(res["delta"]) == 2

def test_calculate_greeks_batch_array_type():
    spots = np.array([100.0, 100.0])
    option_types = np.array(["call", "put"])
    res = BlackScholesEngine.calculate_greeks_batch(
        spot=spots, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, option_type=option_types
    )
    assert isinstance(res, dict)
    assert len(res["delta"]) == 2
    assert res["delta"][0] > 0
    assert res["delta"][1] < 0

def test_black_scholes_wrapper():
    from src.pricing.black_scholes import black_scholes
    res = black_scholes(S=100, K=100, T=1.0, sigma=0.2, r=0.05)
    assert "price" in res
    assert res["price"] > 0
