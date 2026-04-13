import numpy as np
import pytest

from src.math_kernel.black_scholes import BlackScholesEngine, BSParameters


@pytest.fixture
def params():
    return BSParameters(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        dividend=0.01,
    )


def test_price_options_scalar(params):
    engine = BlackScholesEngine()
    price = engine.price_options(params=params, option_type="call")
    assert isinstance(price, float)
    assert price > 0


def test_price_options_vectorized():
    spots = np.array([100.0, 110.0])
    strikes = np.array([100.0, 100.0])
    engine = BlackScholesEngine()
    prices = engine.price_options(
        spot=spots, strike=strikes, maturity=1.0, volatility=0.2, rate=0.05
    )
    assert len(prices) == 2
    assert prices[1] > prices[0]


def test_calculate_greeks(params):
    engine = BlackScholesEngine()
    greeks = engine.calculate_greeks(params=params, option_type="call")
    # Handle both scalar and array results (Numba might return 0-dim array)
    delta = float(greeks.delta) if isinstance(greeks.delta, np.ndarray) else greeks.delta
    assert delta > 0
    assert delta < 1.0


def test_put_call_parity(params):
    engine = BlackScholesEngine()
    cp = engine.price_options(params=params, option_type="call")
    pp = engine.price_options(params=params, option_type="put")
    parity = engine.verify_put_call_parity(
        params.spot,
        params.strike,
        params.maturity,
        params.rate,
        cp,
        pp,
        params.dividend,
    )
    assert parity


def test_price_call_put(params):
    engine = BlackScholesEngine()
    call_p = engine.price_call(params)
    put_p = engine.price_put(params)
    assert call_p > 0
    assert put_p > 0


def test_price_batch():
    engine = BlackScholesEngine()
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 110.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    option_types = np.array(["call", "call"])
    prices = engine.price_batch(S, K, T, sigma, r, q, option_types)
    assert len(prices) == 2


def test_calculate_greeks_batch():
    engine = BlackScholesEngine()
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 110.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    greeks = engine.calculate_greeks_batch(
        spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=q
    )
    assert "delta" in greeks
    assert len(greeks["delta"]) == 2


def test_instance_price(params):
    engine = BlackScholesEngine()
    price = engine.price(params, option_type="call")
    assert price > 0


def test_module_level_funcs(params):
    from src.math_kernel.black_scholes import black_scholes
    from src.math_kernel.black_scholes import verify_put_call_parity as vpcp

    # When called with kwargs, returns float directly
    res = black_scholes(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    assert isinstance(res, float)

    # Test parity func
    parity = vpcp(params)
    assert parity