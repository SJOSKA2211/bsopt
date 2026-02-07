import pytest
import numpy as np
from src.pricing import black_scholes

def test_bs_price_call():
    # S=100, K=100, T=1, r=0.05, sigma=0.2
    price = black_scholes.bs_price(100, 100, 1, 0.05, 0.2, "call")
    assert 10.0 < price < 11.0

def test_bs_price_put():
    price = black_scholes.bs_price(100, 100, 1, 0.05, 0.2, "put")
    assert 5.0 < price < 6.0

def test_bs_greeks():
    delta, gamma, theta, vega, rho = black_scholes.bs_greeks(100, 100, 1, 0.05, 0.2, "call")
    assert 0.0 <= delta <= 1.0
    assert gamma > 0
    assert vega > 0
