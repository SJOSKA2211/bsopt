import numpy as np
import pytest

from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.implied_vol import (
    ImpliedVolatilityError,
    implied_volatility,
    vectorized_implied_volatility,
)


@pytest.fixture
def sample_data():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.2
    # Calculate price first
    price = BlackScholesEngine.price_options(
        spot=S,
        strike=K,
        maturity=T,
        volatility=sigma,
        rate=r,
        dividend=q,
        option_type="call",
    )
    return float(price), S, K, T, r, q


def test_implied_volatility_newton(sample_data):
    price, S, K, T, r, q = sample_data
    iv = implied_volatility(price, S, K, T, r, q, option_type="call", method="newton")
    assert np.allclose(iv, 0.2, atol=1e-4)


def test_implied_volatility_brent(sample_data):
    price, S, K, T, r, q = sample_data
    iv = implied_volatility(price, S, K, T, r, q, option_type="call", method="brent")
    assert np.allclose(iv, 0.2, atol=1e-4)


def test_implied_volatility_validation():
    with pytest.raises(ValueError, match="negative"):
        implied_volatility(-1.0, 100, 100, 1, 0.05)

    with pytest.raises(ValueError, match="Arbitrage violation"):
        # Price below intrinsic (Intrinsic = 100 - 100*exp(-0.05) = 4.87)
        implied_volatility(1.0, 100, 100, 1, 0.05)


def test_vectorized_implied_volatility():
    spots = np.array([100.0, 110.0])
    strikes = np.array([100.0, 100.0])
    maturities = np.array([1.0, 1.0])
    rates = np.array([0.05, 0.05])
    dividends = np.array([0.0, 0.0])
    sigmas = np.array([0.2, 0.25])

    prices = BlackScholesEngine.price_options(
        spot=spots,
        strike=strikes,
        maturity=maturities,
        volatility=sigmas,
        rate=rates,
        dividend=dividends,
        option_type="call",
    )

    ivs = vectorized_implied_volatility(
        prices, spots, strikes, maturities, rates, dividends, np.array(["call", "call"])
    )
    assert np.allclose(ivs, sigmas, atol=1e-4)


def test_newton_raphson_failure():
    # Force failure by using a price that's hard to hit or absurd max_iterations
    with pytest.raises(ImpliedVolatilityError, match="converge"):
        implied_volatility(10.0, 100, 100, 1, 0.05, method="newton", max_iterations=1)
