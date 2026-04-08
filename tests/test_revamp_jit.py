import numpy as np
import pytest

from src.math_kernel.quant_utils import (
    batch_bs_price_jit_v2,
    fast_normal_cdf_v2,
    fast_normal_pdf_v2,
    scalar_bs_price_jit,
    scalar_greeks_jit_v2,
)


def test_fast_normal_kernels():
    # Test CDF
    assert pytest.approx(fast_normal_cdf_v2(0.0), 1e-5) == 0.5
    assert fast_normal_cdf_v2(10.0) > 0.999
    assert fast_normal_cdf_v2(-10.0) < 0.001

    # Test PDF
    assert pytest.approx(fast_normal_pdf_v2(0.0), 1e-5) == 0.39894228
    assert fast_normal_pdf_v2(10.0) < 1e-10


def test_scalar_bs_price_jit():
    S, K, T, sigma, r, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.0
    price_call = scalar_bs_price_jit(S, K, T, sigma, r, q, True)
    price_put = scalar_bs_price_jit(S, K, T, sigma, r, q, False)

    # Approx Black-Scholes price for ATM call is ~10.45
    assert price_call > 0
    assert price_put > 0
    assert price_call > price_put  # Call-Put parity: C - P = S - K*exp(-rT) = 100 - 100*0.95 = 4.87
    assert pytest.approx(price_call - price_put, 1e-5) == S - K * np.exp(-r * T)


def test_batch_bs_price_jit():
    S = np.array([100.0, 100.0], dtype=np.float64)
    K = np.array([100.0, 110.0], dtype=np.float64)
    T = np.array([1.0, 1.0], dtype=np.float64)
    sigma = np.array([0.2, 0.2], dtype=np.float64)
    r = np.array([0.05, 0.05], dtype=np.float64)
    q = np.array([0.0, 0.0], dtype=np.float64)
    is_call = np.array([True, True], dtype=bool)

    prices = batch_bs_price_jit_v2(S, K, T, sigma, r, q, is_call)
    assert len(prices) == 2
    assert prices[0] > prices[1]  # ATM call > OTM call


def test_scalar_greeks_jit():
    S, K, T, sigma, r, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.0
    d, g, th, v, rh = scalar_greeks_jit_v2(S, K, T, sigma, r, q, True)

    assert 0 < d < 1.0
    assert g > 0
    assert v > 0
    assert rh > 0
    # Theta is usually negative for long options
    assert th < 0
