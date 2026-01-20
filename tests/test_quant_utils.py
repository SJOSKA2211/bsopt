import numpy as np

from src.pricing.quant_utils import (
    batch_bs_price_jit,
    calculate_d1_d2_jit,
    corrado_miller_initial_guess,
    fast_normal_cdf,
    fast_normal_pdf,
)
from tests.test_utils import assert_equal


def test_fast_normal_functions():
    assert_equal(fast_normal_cdf(0.0), 0.5)
    assert_equal(fast_normal_pdf(0.0), 1.0 / np.sqrt(2 * np.pi))


def test_calculate_d1_d2_jit():
    d1, d2 = calculate_d1_d2_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.02)
    assert isinstance(d1, float)
    assert isinstance(d2, float)
    assert d1 > d2


def test_batch_bs_price_jit():
    S = np.array([100.0, 110.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    v = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.02, 0.02])
    is_call = np.array([True, False])

    prices = batch_bs_price_jit(S, K, T, v, r, q, is_call)
    assert_equal(len(prices), 2)
    assert prices[0] > 0
    assert prices[1] > 0


def test_batch_bs_price_jit_extreme_values():
    # Test zero vol, negative rate, etc.
    S = np.array([100.0, 100.0, 100.0])
    K = np.array([100.0, 100.0, 100.0])
    T = np.array([1.0, 1.0, 1.0])
    v = np.array([1e-9, 0.2, 0.2]) # Near zero vol
    r = np.array([0.05, -0.05, 0.05]) # Negative rate
    q = np.array([0.0, 0.0, 0.0])
    is_call = np.array([True, True, True])

    prices = batch_bs_price_jit(S, K, T, v, r, q, is_call)
    assert np.isnan(prices[0])
    assert np.isnan(prices[1])
    assert not np.isnan(prices[2])

def test_batch_bs_price_jit_zero_maturity():
    S = np.array([110.0, 90.0])
    K = np.array([100.0, 100.0])
    T = np.array([0.0, 0.0])
    v = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    is_call = np.array([True, True])

    prices = batch_bs_price_jit(S, K, T, v, r, q, is_call)
    assert prices[0] == 10.0
    assert prices[1] == 0.0
