import pytest
import numpy as np
from src.shared.math_utils import (
    calculate_price, 
    calculate_greeks, 
    _fast_normal_cdf, 
    _fast_normal_pdf,
    rk4_gbm_step,
    run_gbm_simulation
)

def test_fast_normal_cdf():
    # Test key values of the normal distribution
    assert np.isclose(_fast_normal_cdf(0.0, np), 0.5, atol=1e-5)
    assert _fast_normal_cdf(1.96, np) > 0.975  # 97.5th percentile
    assert _fast_normal_cdf(-1.96, np) < 0.025

def test_fast_normal_pdf():
    # Value at mean for standard normal is 1/sqrt(2pi) approx 0.3989
    assert np.isclose(_fast_normal_pdf(0.0, np), 0.39894228, atol=1e-7)

def test_calculate_price_scalar():
    # BS Price for S=100, K=100, T=1, Sigma=0.2, R=0.05, Q=0
    # Expected approx 10.45 for call, 5.57 for put
    c_price = calculate_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
    p_price = calculate_price(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, False)
    
    assert isinstance(c_price, float)
    assert np.isclose(c_price, 10.45058, atol=1e-4)
    assert np.isclose(p_price, 5.57352, atol=1e-4)

def test_calculate_price_vectorized():
    s = np.array([100.0, 110.0])
    k = np.array([100.0, 100.0])
    t = np.array([1.0, 1.0])
    sig = np.array([0.2, 0.2])
    v = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    flags = np.array([True, False])
    
    prices = calculate_price(s, k, t, sig, v, q, flags)
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (2,)

def test_calculate_greeks_values():
    # Test Greeks for standard ATM option
    delta, gamma, theta, vega, rho = calculate_greeks(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
    
    # Call Delta should be approx 0.637
    assert 0.6 < delta < 0.7
    assert gamma > 0
    assert vega > 0
    assert rho > 0
    # Gamma should be approx 0.0187
    assert np.isclose(gamma, 0.0187, atol=1e-3)

def test_gbm_simulation():
    s0 = 100.0
    mu = 0.05
    sigma = 0.2
    t = 1.0
    steps = 10
    
    prices = run_gbm_simulation(s0, mu, sigma, t, steps)
    assert len(prices) == steps + 1
    assert prices[0] == s0
    # Each step is RK4
    step = rk4_gbm_step(100.0, 0.05, 0.2, 0.1, 0.01)
    assert isinstance(step, float)