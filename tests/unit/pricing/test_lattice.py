import pytest
import numpy as np
from src.pricing.lattice import BinomialTreePricer, TrinomialTreePricer, validate_convergence
from src.pricing.models import BSParameters

def test_binomial_pricer_european():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    pricer = BinomialTreePricer(n_steps=100, exercise_type="european")
    
    price_call = pricer.price(params, "call")
    assert price_call > 0
    
    price_put = pricer.price(params, "put")
    assert price_put > 0

def test_binomial_pricer_american():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    pricer = BinomialTreePricer(n_steps=100, exercise_type="american")
    
    price_call = pricer.price(params, "call")
    assert price_call > 0
    
    price_put = pricer.price(params, "put")
    assert price_put > 0

def test_binomial_pricer_greeks():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    pricer = BinomialTreePricer(n_steps=10) # Small steps for speed
    greeks = pricer.calculate_greeks(params, "call")
    assert greeks.delta > 0
    assert greeks.gamma > 0
    assert greeks.vega > 0
    assert greeks.theta <= 0
    assert greeks.rho > 0

def test_binomial_pricer_zero_maturity():
    params = BSParameters(100, 100, 0.0, 0.2, 0.05)
    pricer = BinomialTreePricer(n_steps=10)
    assert pricer.price(params, "call") == 0.0
    assert pricer.price(params, "put") == 0.0
    
    greeks = pricer.calculate_greeks(params, "call")
    assert greeks.theta == 0.0

def test_binomial_pricer_build_tree():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    pricer = BinomialTreePricer(n_steps=5)
    tree = pricer.build_tree(params)
    assert tree.shape == (6, 6)
    assert tree[0, 0] == 100.0

def test_trinomial_pricer_european():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    pricer = TrinomialTreePricer(n_steps=100, exercise_type="european")
    
    price_call = pricer.price(params, "call")
    assert price_call > 0
    
    price_put = pricer.price(params, "put")
    assert price_put > 0

def test_trinomial_pricer_american():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    pricer = TrinomialTreePricer(n_steps=100, exercise_type="american")
    
    price_call = pricer.price(params, "call")
    assert price_call > 0
    
    price_put = pricer.price(params, "put")
    assert price_put > 0

def test_trinomial_pricer_greeks():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    pricer = TrinomialTreePricer(n_steps=10)
    greeks = pricer.calculate_greeks(params, "call")
    assert greeks.delta > 0

def test_trinomial_pricer_zero_maturity():
    pricer = TrinomialTreePricer(n_steps=10)
    params = BSParameters(110, 100, 0.0, 0.2, 0.05)
    assert pricer.price(params, "call") == 10.0
    assert pricer.price(params, "put") == 0.0

def test_binomial_american_put():
    # American put should be worth more than European if dividends are high or rates low
    params = BSParameters(100, 100, 1.0, 0.2, 0.01, 0.1) # High dividend
    euro_pricer = BinomialTreePricer(n_steps=100, exercise_type="european")
    amer_pricer = BinomialTreePricer(n_steps=100, exercise_type="american")
    
    euro_price = euro_pricer.price(params, "put")
    amer_price = amer_pricer.price(params, "put")
    assert amer_price >= euro_price

def test_trinomial_american_call_dividend():
    # American call can be exercised early if there are high dividends
    params = BSParameters(100, 100, 1.0, 0.2, 0.01, 0.1)
    euro_pricer = TrinomialTreePricer(n_steps=100, exercise_type="european")
    amer_pricer = TrinomialTreePricer(n_steps=100, exercise_type="american")
    
    euro_price = euro_pricer.price(params, "call")
    amer_price = amer_pricer.price(params, "call")
    assert amer_price >= euro_price

def test_validate_convergence():
    res = validate_convergence(100, 100, 1.0, 0.2, 0.05, 0.0, "call", [10, 20, 50])
    assert len(res["binomial_errors"]) == 3
    assert len(res["trinomial_errors"]) == 3

def test_binomial_no_arbitrage_violation():
    # p = (a-d)/(u-d). If u < a, p > 1.
    # This happens if vol is very low and rate high.
    params = BSParameters(100, 100, 1.0, 0.001, 0.5)
    pricer = BinomialTreePricer(n_steps=10)
    price = pricer.price(params, "call")
    assert price >= 0