import math

import numpy as np

from src.shared import math_utils


def test_fast_normal_cdf_scalar():
    assert math.isclose(math_utils.scalar_normal_cdf(0.0), 0.5, rel_tol=1e-5)
    assert math.isclose(math_utils.scalar_normal_cdf(100.0), 1.0, rel_tol=1e-5)
    assert math.isclose(math_utils.scalar_normal_cdf(-100.0), 0.0, rel_tol=1e-5)


def test_fast_normal_pdf_scalar():
    assert math.isclose(math_utils.scalar_normal_pdf(0.0), 0.39894228, rel_tol=1e-5)


def test_vectorized_cdf():
    x = np.array([0.0, 100.0, -100.0])
    y = math_utils.fast_normal_cdf(x)
    assert np.allclose(y, [0.5, 1.0, 0.0], atol=1e-5)


def test_calculate_price_call():
    # S=100, K=100, T=1, sigma=0.2, r=0.05, q=0.0
    # Expected call ~ 10.45
    price = math_utils.calculate_price_scalar(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
    assert 10.0 < price < 11.0


def test_calculate_price_put():
    # Put-Call Parity: C - P = S - K*e(-rT)
    # 10.45 - P = 100 - 100*e(-0.05) = 100 - 95.12 = 4.88
    # P ~ 10.45 - 4.88 = 5.57
    price = math_utils.calculate_price_scalar(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, False)
    assert 5.0 < price < 6.0


def test_vectorized_pricing():
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 0.5])
    sigma = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    is_call = True

    prices = math_utils.calculate_price(S, K, T, sigma, r, q, is_call)
    assert len(prices) == 2
    assert prices[0] > prices[1]  # Longer expiry -> higher price


def test_greeks_scalar():
    delta, gamma, theta, vega, rho = math_utils.calculate_greeks_scalar(
        100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True
    )
    assert 0.0 <= delta <= 1.0
    assert gamma > 0
    assert vega > 0


def test_expiry_boundary():
    # T=0
    price = math_utils.calculate_price_scalar(100.0, 90.0, 0.0, 0.2, 0.05, 0.0, True)
    assert price == 10.0  # Intrinsic value
