import pytest
import numpy as np
from src.pricing.lattice import BinomialTreePricer, TrinomialTreePricer, validate_convergence
from src.pricing.models import BSParameters, OptionGreeks

# Constants
SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
VOLATILITY = 0.2
RATE = 0.05
DIVIDEND = 0.0

@pytest.fixture
def bs_params():
    return BSParameters(SPOT, STRIKE, MATURITY, VOLATILITY, RATE, DIVIDEND)

def test_binomial_european_call(bs_params):
    pricer = BinomialTreePricer(n_steps=100, exercise_type="european")
    price = pricer.price(bs_params, "call")
    # BS Price ~ 10.45
    assert 10.3 < price < 10.6

def test_binomial_american_put(bs_params):
    # American Put should be >= European Put
    pricer = BinomialTreePricer(n_steps=100, exercise_type="american")
    price = pricer.price(bs_params, "put")
    # European Put ~ 5.57. American Put with r=0.05 is usually slightly higher or same.
    assert price > 5.5

def test_trinomial_european_call(bs_params):
    pricer = TrinomialTreePricer(n_steps=100, exercise_type="european")
    price = pricer.price(bs_params, "call")
    assert 10.3 < price < 10.6

def test_trinomial_american_put(bs_params):
    pricer = TrinomialTreePricer(n_steps=100, exercise_type="american")
    price = pricer.price(bs_params, "put")
    assert price > 5.5

def test_convergence_validation(bs_params):
    res = validate_convergence(
        SPOT, STRIKE, MATURITY, VOLATILITY, RATE, DIVIDEND, "call", step_sizes=[10, 50]
    )
    assert "binomial_errors" in res
    assert "trinomial_errors" in res
    # Error should decrease with more steps usually, or be small
    assert res["binomial_errors"][1] < res["binomial_errors"][0] or res["binomial_errors"][1] < 0.5

def test_greeks_binomial(bs_params):
    pricer = BinomialTreePricer(n_steps=50)
    greeks = pricer.calculate_greeks(bs_params, "call")
    assert isinstance(greeks, OptionGreeks)
    assert 0.5 < greeks.delta < 0.7
    assert greeks.gamma > 0
    assert greeks.theta < 0 # Time decay

def test_greeks_trinomial(bs_params):
    pricer = TrinomialTreePricer(n_steps=50)
    greeks = pricer.calculate_greeks(bs_params, "call")
    assert isinstance(greeks, OptionGreeks)
    assert 0.5 < greeks.delta < 0.7

def test_build_tree(bs_params):
    pricer = BinomialTreePricer(n_steps=3)
    tree = pricer.build_tree(bs_params)
    assert tree.shape == (4, 4)
    assert tree[0, 0] == SPOT

def test_zero_maturity(bs_params):
    bs_params.maturity = 0.0
    
    bin_pricer = BinomialTreePricer()
    assert bin_pricer.price(bs_params, "call") == 0.0
    
    tri_pricer = TrinomialTreePricer()
    assert tri_pricer.price(bs_params, "call") == 0.0
    
    # ITM
    bs_params.spot = 110.0
    assert bin_pricer.price(bs_params, "call") == 10.0
    assert tri_pricer.price(bs_params, "call") == 10.0

def test_binomial_american_call_no_div(bs_params):
    # American Call on non-dividend stock = European Call
    pricer_am = BinomialTreePricer(n_steps=100, exercise_type="american")
    price_am = pricer_am.price(bs_params, "call")
    
    pricer_eu = BinomialTreePricer(n_steps=100, exercise_type="european")
    price_eu = pricer_eu.price(bs_params, "call")
    
    assert np.isclose(price_am, price_eu, atol=0.01)
