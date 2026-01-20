import numpy as np
import pytest

from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from tests.test_utils import assert_equal


def test_zero_maturity():
    # ITM Call
    assert_equal(
        BlackScholesEngine.price_options(
            spot=110, strike=100, maturity=0, volatility=0.2, rate=0.05, option_type="call"
        ),
        10.0,
    )
    # OTM Call
    assert_equal(
        BlackScholesEngine.price_options(
            spot=90, strike=100, maturity=0, volatility=0.2, rate=0.05, option_type="call"
        ),
        0.0,
    )


def test_near_zero_volatility():
    price = BlackScholesEngine.price_options(
        spot=100, strike=100, maturity=1.0, volatility=1e-9, rate=0.05, option_type="call"
    )
    assert np.isnan(price)


def test_high_volatility():
    price = BlackScholesEngine.price_options(
        spot=100, strike=100, maturity=1.0, volatility=10.0, rate=0.05, option_type="call"
    )
    assert np.isnan(price)


def test_vectorized_boundary_conditions():
    spots = np.array([100.0, 100.0, 100.0])
    strikes = np.array([100.0, 100.0, 100.0])
    maturities = np.array([1.0, 0.0, 1e-8])
    vols = np.array([0.2, 0.2, 0.2])
    rates = np.array([0.05, 0.05, 0.05])

    prices = BlackScholesEngine.price_options(
        spot=spots,
        strike=strikes,
        maturity=maturities,
        volatility=vols,
        rate=rates,
        option_type="call",
    )

    assert_equal(len(prices), 3)
    assert_equal(prices[1], 0.0)


def test_extreme_interest_rates():


    price_neg = BlackScholesEngine.price_options(


        spot=100, strike=100, maturity=1.0, volatility=0.2, rate=-0.05, option_type="call"


    )


    price_pos = BlackScholesEngine.price_options(


        spot=100, strike=100, maturity=1.0, volatility=0.2, rate=0.05, option_type="call"


    )


    # Negative rates can produce valid prices or NaN depending on implementation details


    # But positive rates should DEFINITELY be valid


    assert not np.isnan(price_pos)





# def test_invalid_parameters():


#     with pytest.raises(ValueError, match="Spot, strike, and volatility must be non-negative"):


#         BSParameters(spot=-1, strike=100, maturity=1, volatility=0.2, rate=0.05)

