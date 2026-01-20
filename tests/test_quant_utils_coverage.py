import numpy as np
import pytest
from src.pricing.quant_utils import (
    fast_normal_cdf,
    fast_normal_pdf,
    corrado_miller_initial_guess,
    calculate_d1_d2_jit,
    batch_bs_price_jit,
    batch_greeks_jit,
)

def test_fast_normal_cdf():
    assert fast_normal_cdf(0.0) == pytest.approx(0.5)
    assert fast_normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)

def test_fast_normal_pdf():
    assert fast_normal_pdf(0.0) == pytest.approx(1.0 / np.sqrt(2 * np.pi))

def test_corrado_miller_guess():
    market_price = np.array([10.45, 5.50])
    spot = np.array([100.0, 100.0])
    strike = np.array([100.0, 100.0])
    maturity = np.array([1.0, 1.0])
    rate = np.array([0.05, 0.05])
    dividend = np.array([0.0, 0.0])
    option_type = np.array([0, 1]) # call, put
    
    guess = corrado_miller_initial_guess(
        market_price, spot, strike, maturity, rate, dividend, option_type
    )
    assert len(guess) == 2
    assert np.all(guess > 0)

def test_calculate_d1_d2_jit():
    d1, d2 = calculate_d1_d2_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0)
    assert d1 > d2

def test_batch_bs_price_jit():
    S = np.array([100.0, 100.0, 100.0, 80.0])
    K = np.array([100.0, 100.0, 100.0, 90.0])
    T = np.array([1.0, 1.0, 0.0, 0.0])
    sigma = np.array([0.2, 0.2, 0.2, 0.2])
    r = np.array([0.05, 0.05, 0.05, 0.05])
    q = np.array([0.0, 0.0, 0.0, 0.0])
    is_call = np.array([True, False, True, False])
    
    prices = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
    assert len(prices) == 4
    assert prices[0] > 0
    assert prices[1] > 0
    assert prices[2] == 0.0 # ATM call at expiry
    assert prices[3] == 10.0 # ITM put at expiry

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
