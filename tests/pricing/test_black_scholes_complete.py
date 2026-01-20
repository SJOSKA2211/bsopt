import pytest
import numpy as np
from src.pricing.black_scholes import BlackScholesEngine, verify_put_call_parity, black_scholes
from src.pricing.models import BSParameters, OptionGreeks

# Constants for testing
SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
VOLATILITY = 0.2
RATE = 0.05
DIVIDEND = 0.0

@pytest.fixture
def bs_params():
    return BSParameters(
        spot=SPOT,
        strike=STRIKE,
        maturity=MATURITY,
        volatility=VOLATILITY,
        rate=RATE,
        dividend=DIVIDEND
    )

def test_bs_parameters_validation():
    # Test valid parameters
    BSParameters(100, 100, 1, 0.2, 0.05, 0.0)
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        BSParameters(-100, 100, 1, 0.2, 0.05, 0.0)
    with pytest.raises(ValueError):
        BSParameters(100, -100, 1, 0.2, 0.05, 0.0)
    with pytest.raises(ValueError):
        BSParameters(100, 100, -1, 0.2, 0.05, 0.0)
    with pytest.raises(ValueError):
        BSParameters(100, 100, 1, -0.2, 0.05, 0.0)

def test_calculate_d1_d2(bs_params):
    d1, d2 = BlackScholesEngine.calculate_d1_d2(bs_params)
    
    # Manual calculation
    expected_d1 = (np.log(SPOT / STRIKE) + (RATE - DIVIDEND + 0.5 * VOLATILITY ** 2) * MATURITY) / (VOLATILITY * np.sqrt(MATURITY))
    expected_d2 = expected_d1 - VOLATILITY * np.sqrt(MATURITY)
    
    assert np.isclose(d1, expected_d1)
    assert np.isclose(d2, expected_d2)

def test_price_call(bs_params):
    price = BlackScholesEngine.price_call(bs_params)
    # Approx value: S=100, K=100, T=1, r=0.05, vol=0.2 -> Call ~ 10.4506
    assert 10.4 < price < 10.5
    
    # Check instance method
    engine = BlackScholesEngine()
    assert np.isclose(engine.price(bs_params, "call"), price)

def test_price_put(bs_params):
    price = BlackScholesEngine.price_put(bs_params)
    # Approx value: Put ~ 5.5735
    assert 5.5 < price < 5.6
    
    # Check instance method
    engine = BlackScholesEngine()
    assert np.isclose(engine.price(bs_params, "put"), price)

def test_calculate_greeks_call(bs_params):
    greeks = BlackScholesEngine.calculate_greeks_static(bs_params, "call")
    
    assert isinstance(greeks, OptionGreeks)
    assert 0.0 <= greeks.delta <= 1.0  # Call delta between 0 and 1
    assert greeks.gamma > 0.0          # Long gamma
    assert greeks.vega > 0.0           # Long vega
    assert greeks.theta < 0.0          # Time decay

def test_calculate_greeks_put(bs_params):
    greeks = BlackScholesEngine.calculate_greeks_static(bs_params, "put")
    
    assert isinstance(greeks, OptionGreeks)
    assert -1.0 <= greeks.delta <= 0.0 # Put delta between -1 and 0
    assert greeks.gamma > 0.0
    assert greeks.vega > 0.0

def test_put_call_parity(bs_params):
    assert verify_put_call_parity(bs_params)

def test_edge_cases(bs_params):
    # Maturity = 0
    params_zero_t = BSParameters(SPOT, STRIKE, 0.0, VOLATILITY, RATE, DIVIDEND)
    
    # At money
    price = BlackScholesEngine.price_call(params_zero_t)
    assert price == 0.0 # max(100-100, 0)
    
    # In money call
    params_itm = BSParameters(110, 100, 0.0, 0.2, 0.05, 0.0)
    price_itm = BlackScholesEngine.price_call(params_itm)
    assert price_itm == 10.0
    
    # Out money call
    params_otm = BSParameters(90, 100, 0.0, 0.2, 0.05, 0.0)
    price_otm = BlackScholesEngine.price_call(params_otm)
    assert price_otm == 0.0

def test_vectorized_pricing():
    # Try with small arrays
    spots = np.array([100.0, 110.0])
    strikes = np.array([100.0, 100.0])
    maturities = np.array([1.0, 1.0])
    vols = np.array([0.2, 0.2])
    rates = np.array([0.05, 0.05])
    
    prices = BlackScholesEngine.price_options(
        spot=spots, strike=strikes, maturity=maturities, 
        volatility=vols, rate=rates, option_type="call"
    )
    
    assert isinstance(prices, np.ndarray)
    assert len(prices) == 2
    assert prices[1] > prices[0] # Higher spot -> Higher call price

def test_vectorized_greeks():
    spots = np.array([100.0, 110.0])
    greeks = BlackScholesEngine.calculate_greeks_batch(
        spot=spots, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05
    )
    
    assert "delta" in greeks
    assert len(greeks["delta"]) == 2
    assert greeks["delta"][1] > greeks["delta"][0] # Higher spot -> Higher delta

def test_wrapper_function():
    res = black_scholes(SPOT, STRIKE, MATURITY, VOLATILITY, RATE, type='call')
    assert isinstance(res, dict)
    assert "price" in res

def test_price_options_with_params(bs_params):
    # Test passing params object
    price = BlackScholesEngine.price_options(params=bs_params)
    assert isinstance(price, float)
    assert price > 0

def test_calculate_greeks_batch_with_params(bs_params):
    # Test passing params object
    greeks = BlackScholesEngine.calculate_greeks_batch(params=bs_params)
    assert isinstance(greeks, OptionGreeks)

def test_missing_parameters_error():
    with pytest.raises(TypeError):
        BlackScholesEngine.price_options(spot=100.0) # Missing others

    with pytest.raises(TypeError):
        BlackScholesEngine.calculate_greeks_batch(spot=100.0) # Missing others

def test_vectorized_mixed_option_types():
    spots = np.array([100.0, 100.0])
    strikes = np.array([100.0, 100.0])
    # One call, one put
    types = np.array(["call", "put"])
    
    prices = BlackScholesEngine.price_options(
        spot=spots, strike=strikes, maturity=1.0, volatility=0.2, rate=0.05, option_type=types
    )
    
    assert len(prices) == 2
    # Call price should be different from Put price (unless parity makes them equal, but usually different with r/q)
    # S=100, K=100, r=0.05 -> Call > Put
    assert prices[0] != prices[1]
    
    greeks = BlackScholesEngine.calculate_greeks_batch(
        spot=spots, strike=strikes, maturity=1.0, volatility=0.2, rate=0.05, option_type=types
    )
    assert greeks["delta"][0] > 0 # Call delta
    assert greeks["delta"][1] < 0 # Put delta
