import pytest
import numpy as np
from src.pricing.black_scholes import BlackScholesEngine, BSParameters

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
    greeks = BlackScholesEngine.calculate_greeks(params=params, option_type="call")
    
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
    params = BSParameters(spot=100.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    call_price = BlackScholesEngine.price_options(params=params, option_type="call")
    assert call_price == 0.0 # ATM at maturity
    
    params_itm = BSParameters(spot=110.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    call_itm = BlackScholesEngine.price_options(params=params_itm, option_type="call")
    assert call_itm == 10.0 # Intrinsic value

def test_calculate_d1_d2():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    d1, d2 = BlackScholesEngine.calculate_d1_d2(params)
    assert isinstance(d1, float)
    assert isinstance(d2, float)
    assert d1 > d2
