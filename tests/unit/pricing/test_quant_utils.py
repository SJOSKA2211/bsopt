import numpy as np
import pytest

from src.math_kernel.quant_utils import (
    batch_bs_price_jit,
    batch_greeks_jit,
    corrado_miller_initial_guess,
    heston_char_func_jit,
    jit_lsm_american,
    jit_mc_european_price,
    jit_mc_european_price_and_greeks,
    jit_mc_european_with_control_variate,
    scalar_bs_price_jit,
    scalar_greeks_jit,
    thomas_algorithm,
    vectorized_newton_raphson_iv_jit,
)


#  ENGINEER: Use small arrays for faster JIT warmup in tests
@pytest.fixture
def sample_data():
    S = np.array([100.0, 110.0], dtype=np.float64)
    K = np.array([100.0, 100.0], dtype=np.float64)
    T = np.array([1.0, 1.0], dtype=np.float64)
    sigma = np.array([0.2, 0.2], dtype=np.float64)
    r = np.array([0.05, 0.05], dtype=np.float64)
    q = np.array([0.0, 0.0], dtype=np.float64)
    is_call = np.array([True, False], dtype=np.bool_)
    return S, K, T, sigma, r, q, is_call


def test_corrado_miller(sample_data):
    S, K, T, sigma, r, q, is_call = sample_data
    # 0 for call, 1 for put in this kernel's logic
    option_type = np.array([0, 1])
    # Dummy market price
    market_price = np.array([10.0, 10.0])
    guess = corrado_miller_initial_guess(market_price, S, K, T, r, q, option_type)
    assert len(guess) == 2
    assert np.all(guess > 0)


def test_batch_bs_price(sample_data):
    S, K, T, sigma, r, q, is_call = sample_data
    prices = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
    assert len(prices) == 2
    assert prices[0] > 0  # Call
    assert prices[1] > 0  # Put


def test_batch_greeks(sample_data):
    S, K, T, sigma, r, q, is_call = sample_data
    delta, gamma, vega, theta, rho = batch_greeks_jit(S, K, T, sigma, r, q, is_call)
    assert len(delta) == 2
    assert np.all(gamma > 0)


def test_thomas_algorithm():
    # Simple diagonally dominant tridiagonal system Ax = b
    # A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    # x = [1, 1, 1] => b = [3, 4, 3]
    lower = np.array([1.0, 1.0])
    diag = np.array([2.0, 2.0, 2.0])
    upper = np.array([1.0, 1.0])
    rhs = np.array([3.0, 4.0, 3.0])

    x = thomas_algorithm(lower, diag, upper, rhs)
    assert np.allclose(x, np.array([1.0, 1.0, 1.0]))


def test_newton_raphson_iv(sample_data):
    S, K, T, sigma, r, q, is_call = sample_data
    # Calculate prices first using scalar BS
    p1 = scalar_bs_price_jit(S[0], K[0], T[0], sigma[0], r[0], q[0], is_call[0])
    p2 = scalar_bs_price_jit(S[1], K[1], T[1], sigma[1], r[1], q[1], is_call[1])
    market_prices = np.array([p1, p2])

    initial_sigma = np.array([0.5, 0.5])
    iv = vectorized_newton_raphson_iv_jit(market_prices, S, K, T, r, q, is_call, initial_sigma)
    assert np.allclose(iv, 0.2, atol=1e-4)


def test_heston_char_func():
    val = heston_char_func_jit(1.0 + 0.5j, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
    assert isinstance(val, complex)


def test_mc_methods():
    # Test MC European
    p, se = jit_mc_european_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 1000, True, True)
    assert p > 0
    assert se > 0

    # Test MC with Control Variate
    p_cv, se_cv = jit_mc_european_with_control_variate(
        100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 1000, True, True
    )
    assert p_cv > 0
    assert se_cv < se  # Error reduction verification


def test_mc_greeks():
    p, d, g, v, r = jit_mc_european_price_and_greeks(
        100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 1000, True, True
    )
    assert p > 0
    assert d > 0
    assert v > 0


def test_lsm_american():
    price = jit_lsm_american(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 1000, 10, True)
    assert price > 0


def test_scalar_kernels():
    price = scalar_bs_price_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
    assert price > 0

    d, g, v, t, r = scalar_greeks_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
    assert d > 0
    assert g > 0