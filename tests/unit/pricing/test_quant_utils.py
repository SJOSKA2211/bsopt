import math

import numpy as np
import pytest

from src.pricing.quant_utils import (
    batch_bs_price_jit,
    batch_greeks_jit,
    calculate_d1_d2_jit,
    corrado_miller_initial_guess,
    fast_normal_cdf,
    fast_normal_pdf,
)


def test_fast_normal_cdf():
    assert pytest.approx(fast_normal_cdf(0), abs=1e-7) == 0.5
    assert pytest.approx(fast_normal_cdf(1.96), abs=1e-2) == 0.975

def test_fast_normal_pdf():
    assert pytest.approx(fast_normal_pdf(0), abs=1e-7) == 1.0 / math.sqrt(2 * math.pi)

def test_corrado_miller_guess():
    # S=100, K=100, T=1, r=0.05, q=0, sigma=0.2 -> price ~ 10.45
    market_prices = np.array([10.45])
    spots = np.array([100.0])
    strikes = np.array([100.0])
    maturities = np.array([1.0])
    rates = np.array([0.05])
    dividends = np.array([0.0])
    option_types = np.array([0]) # 0 for call
    
    sigma_guess = corrado_miller_initial_guess(
        market_prices, spots, strikes, maturities, rates, dividends, option_types
    )
    assert sigma_guess[0] > 0
    assert pytest.approx(sigma_guess[0], abs=0.05) == 0.2

def test_calculate_d1_d2_jit():
    d1, d2 = calculate_d1_d2_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0)
    # d1 = (ln(1) + (0.05 + 0.5*0.04)*1) / (0.2 * 1) = 0.07 / 0.2 = 0.35
    # d2 = 0.35 - 0.2 = 0.15
    assert pytest.approx(d1, abs=1e-7) == 0.35
    assert pytest.approx(d2, abs=1e-7) == 0.15

def test_batch_bs_price_jit():
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    is_call = np.array([True, False])
    
    prices = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
    assert len(prices) == 2
    assert prices[0] > 0
    assert prices[1] > 0
    # Put-Call Parity: C - P = S - K*exp(-rT) = 100 - 100*exp(-0.05) = 100 * (1 - 0.951229) = 4.877
    assert pytest.approx(prices[0] - prices[1], abs=1e-7) == 100 * (1 - math.exp(-0.05))

def test_batch_greeks_jit():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    sigma = np.array([0.2])
    r = np.array([0.05])
    q = np.array([0.0])
    is_call = np.array([True])
    
    delta, gamma, vega, theta, rho = batch_greeks_jit(S, K, T, sigma, r, q, is_call)
    assert delta[0] > 0
    assert gamma[0] > 0
    assert vega[0] > 0
    assert theta[0] < 0
    assert rho[0] > 0
