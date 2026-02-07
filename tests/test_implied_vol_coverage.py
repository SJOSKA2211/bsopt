import numpy as np
import pytest

from src.pricing.implied_vol import (
    ImpliedVolatilityError,
    _brent_iv,
    _newton_raphson_iv,
    implied_volatility,
    vectorized_implied_volatility,
)


def test_iv_method_newton_fail():
    # Test method='newton' failing to converge to hit the raise line
    with pytest.raises(ImpliedVolatilityError, match="failed to converge"):
        implied_volatility(
            market_price=20.0,
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            method="newton",
            max_iterations=1,
            initial_guess=0.01,
        )


def test_iv_method_auto_fallback():
    # Test method='auto' falling back to Brent when Newton fails
    # Set max_iterations=1 to force Newton failure
    # and provide a case Brent can solve
    iv = implied_volatility(
        market_price=10.45,
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        method="auto",
        max_iterations=1,
        initial_guess=0.01,  # Far away
    )
    assert iv > 0


def test_brent_iv_fail():
    with pytest.raises(ImpliedVolatilityError, match="failed to converge"):
        _brent_iv(
            market_price=1000.0,
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            dividend=0.0,
            option_type="call",
        )


def test_newton_iv_low_vega_break():
    with pytest.raises(ImpliedVolatilityError, match="failed to converge"):
        _newton_raphson_iv(
            market_price=10.0,
            spot=100.0,
            strike=10000.0,  # deep OTM
            maturity=1.0,
            rate=0.05,
            dividend=0.0,
            option_type="call",
            initial_guess=0.00001,
        )


def test_vectorized_iv_basic():
    market_prices = np.array([10.45, 5.50])
    spots = np.array([100.0, 100.0])
    strikes = np.array([100.0, 100.0])
    maturities = np.array([1.0, 1.0])
    rates = np.array([0.05, 0.05])
    dividends = np.array([0.0, 0.0])
    option_types = np.array(["call", "put"])

    ivs = vectorized_implied_volatility(
        market_prices, spots, strikes, maturities, rates, dividends, option_types
    )
    assert len(ivs) == 2
    assert ivs[0] > 0
    assert ivs[1] > 0


def test_vectorized_iv_empty():
    # Test with empty arrays to hit the break at top of loop (line 230)
    empty = np.array([])
    ivs = vectorized_implied_volatility(empty, empty, empty, empty, empty, empty, empty)
    assert len(ivs) == 0


def test_vectorized_iv_low_vega():
    market_prices = np.array([0.0001])
    spots = np.array([100.0])
    strikes = np.array([1000.0])
    maturities = np.array([0.00001])
    rates = np.array([0.05])
    dividends = np.array([0.0])
    option_types = np.array(["call"])

    ivs = vectorized_implied_volatility(
        market_prices,
        spots,
        strikes,
        maturities,
        rates,
        dividends,
        option_types,
        max_iterations=2,
    )
    assert len(ivs) == 1
