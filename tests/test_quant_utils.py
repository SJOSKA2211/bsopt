import numpy as np

from src.pricing.quant_utils import (
    batch_bs_price_jit,
    calculate_d1_d2_jit,
    corrado_miller_initial_guess,
    fast_normal_cdf,
    fast_normal_pdf,
    jit_lsm_american,
    jit_mc_european_with_control_variate,
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


def test_corrado_miller():
    # Just check if it runs and returns reasonable values
    market_prices = np.array([10.0, 5.0])
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    is_call = np.array([0, 1])  # 0 for call, 1 for put

    vols = corrado_miller_initial_guess(market_prices, S, K, T, r, q, is_call)
    assert_equal(len(vols), 2)
    assert np.all(vols > 0)


def test_jit_mc_european_with_control_variate():
    price, std_err = jit_mc_european_with_control_variate(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        q=0.02,
        n_paths=1000,
        is_call=True,
        antithetic=True,
    )
    assert price > 0
    assert std_err >= 0
    # Standard error with control variate should be very small
    assert std_err < 1.0


def test_jit_lsm_american():
    price = jit_lsm_american(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        q=0.02,
        n_paths=1000,
        n_steps=50,
        is_call=True,
    )
    assert price > 0
    # American call with dividends should be >= European call
    # (Simplified check)
    assert price > 5.0
