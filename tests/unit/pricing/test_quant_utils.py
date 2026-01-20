import pytest
import numpy as np
import math
from src.pricing.quant_utils import (
    fast_normal_cdf,
    fast_normal_pdf,
    corrado_miller_initial_guess,
    calculate_d1_d2_jit,
    batch_bs_price_jit,
    batch_greeks_jit
)

def test_fast_normal_cdf():
    assert math.isclose(fast_normal_cdf(0), 0.5, rel_tol=1e-7)
    assert math.isclose(fast_normal_cdf(1.96), 0.9750021048517795, rel_tol=1e-5)

def test_fast_normal_pdf():
    assert math.isclose(fast_normal_pdf(0), 1.0 / math.sqrt(2 * math.pi), rel_tol=1e-7)

def test_corrado_miller_initial_guess():
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    option_type = np.array([0, 1]) # 0 for call, 1 for put
    market_price = np.array([10.0, 5.0])
    
    sigma = corrado_miller_initial_guess(market_price, S, K, T, r, q, option_type)
    assert len(sigma) == 2
    assert np.all(sigma > 0)

def test_calculate_d1_d2_jit():
    d1, d2 = calculate_d1_d2_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0)
    assert d1 > d2

def test_batch_bs_price_jit():
    S = np.array([100.0, 100.0, 100.0, 100.0])
    K = np.array([100.0, 100.0, 100.0, 110.0])
    T = np.array([1.0, 0.0, 1.0, 0.0]) # Zero maturity cases included
    sigma = np.array([0.2, 0.2, 0.2, 0.2])
    r = np.array([0.05, 0.05, 0.05, 0.05])
    q = np.array([0.0, 0.0, 0.0, 0.0])
    is_call = np.array([True, True, False, False])
    
    prices = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
    assert len(prices) == 4
    assert prices[0] > 0
    assert prices[1] == 0.0 # ATM at maturity call
    assert prices[2] > 0
    assert prices[3] == 10.0 # ITM at maturity put

def test_batch_bs_price_jit_stability():
    S = np.array([100.0, 100.0, 100.0, 100.0])
    K = np.array([100.0, 100.0, 100.0, 100.0])
    T = np.array([1.0, 1.0, 1.0, 1.0])
    sigma = np.array([-0.1, 6.0, 0.2, 0.2])
    r = np.array([0.05, 0.05, -0.1, 0.6])
    q = np.array([0.0, 0.0, 0.0, 0.0])
    is_call = np.array([True, True, True, True])
    
    prices = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
    assert np.all(np.isnan(prices))

def test_batch_greeks_jit():
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    is_call = np.array([True, False])
    
    delta, gamma, vega, theta, rho = batch_greeks_jit(S, K, T, sigma, r, q, is_call)
    assert len(delta) == 2
    assert delta[0] > 0
    assert delta[1] < 0
    assert np.all(gamma > 0)
    assert np.all(vega > 0)
    assert theta[0] < 0
    assert rho[0] > 0
    assert rho[1] < 0