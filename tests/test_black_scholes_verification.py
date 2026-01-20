import numpy as np

from src.pricing.black_scholes import (
    BlackScholesEngine,
    BSParameters,
    verify_put_call_parity,
)
from tests.test_utils import assert_equal


def test_atm_option_pricing():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02
    )

    d1, d2 = BlackScholesEngine.calculate_d1_d2(params)
    expected_d2_diff = params.volatility * np.sqrt(params.maturity)
    assert_equal(d1 - d2, expected_d2_diff)

    call_price = BlackScholesEngine.price_call(params)
    put_price = BlackScholesEngine.price_put(params)

    assert 9.20 < call_price < 9.25
    assert 6.30 < put_price < 6.35
    assert_equal(verify_put_call_parity(params), True)


def test_deep_itm_call():
    params = BSParameters(
        spot=150.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.00
    )

    call_price = BlackScholesEngine.price_call(params)
    greeks = BlackScholesEngine().calculate_greeks(params, "call")

    intrinsic = params.spot * np.exp(-params.dividend * params.maturity) - params.strike * np.exp(
        -params.rate * params.maturity
    )

    assert greeks.delta > 0.95
    assert greeks.gamma > 0
    assert call_price > intrinsic


def test_greeks_consistency():
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=0.5, volatility=0.25, rate=0.05, dividend=0.02
    )

    call_greeks = BlackScholesEngine().calculate_greeks(params, "call")
    put_greeks = BlackScholesEngine().calculate_greeks(params, "put")

    assert_equal(call_greeks.gamma, put_greeks.gamma)
    assert_equal(call_greeks.vega, put_greeks.vega)

    expected_delta_diff = np.exp(-params.dividend * params.maturity)
    assert_equal(call_greeks.delta - put_greeks.delta, expected_delta_diff)

    assert call_greeks.rho > 0
    assert put_greeks.rho < 0


def test_zero_volatility_limit():
    params_itm = BSParameters(
        spot=110.0, strike=100.0, maturity=1.0, volatility=0.01, rate=0.05, dividend=0.02
    )

    call_price = BlackScholesEngine.price_call(params_itm)
    intrinsic = max(
        params_itm.spot * np.exp(-params_itm.dividend * params_itm.maturity)
        - params_itm.strike * np.exp(-params_itm.rate * params_itm.maturity),
        0,
    )

    assert_equal(call_price, intrinsic, tolerance=0.5)


def test_put_call_parity_comprehensive():
    test_cases = [
        ("ATM, 1Y", 100, 100, 1.0, 0.20, 0.05, 0.02),
        ("ITM Call, 6M", 110, 100, 0.5, 0.25, 0.04, 0.01),
    ]

    for name, S, K, T, sigma, r, q in test_cases:
        params = BSParameters(spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=q)
        assert_equal(verify_put_call_parity(params), True)


def test_numerical_stability():
    params_short = BSParameters(
        spot=100.0, strike=100.0, maturity=1 / 365, volatility=0.20, rate=0.05, dividend=0.02
    )
    assert BlackScholesEngine.price_call(params_short) > 0
    assert_equal(verify_put_call_parity(params_short), True)


def test_monotonicity():
    spots = [90, 100, 110]
    call_prices = []
    for S in spots:
        params = BSParameters(
            spot=S, strike=100, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02
        )
        call_prices.append(BlackScholesEngine.price_call(params))

    for i in range(len(call_prices) - 1):
        assert call_prices[i] < call_prices[i + 1]
