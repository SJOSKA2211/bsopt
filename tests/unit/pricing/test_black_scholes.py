import numpy as np
import pytest

from src.pricing.black_scholes import (
    BlackScholesEngine,
    black_scholes,
    verify_put_call_parity,
)
from src.pricing.models import BSParameters


@pytest.fixture
def sample_params():
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0
    )


def test_price_options_scalar(sample_params):
    engine = BlackScholesEngine()
    # Call price
    call_price = engine.price_options(params=sample_params, option_type="call")
    assert isinstance(call_price, float)
    assert call_price > 0

    # Put price
    put_price = engine.price_options(params=sample_params, option_type="put")
    assert isinstance(put_price, float)
    assert put_price > 0


def test_price_options_vectorized():
    spots = np.array([90.0, 100.0, 110.0])
    strikes = np.array([100.0, 100.0, 100.0])
    maturities = np.array([1.0, 1.0, 1.0])
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
    assert len(prices) == 3
    assert np.all(prices > 0)


def test_calculate_greeks_scalar(sample_params):
    greeks = BlackScholesEngine.calculate_greeks(
        params=sample_params, option_type="call"
    )
    assert hasattr(greeks, "delta")
    assert hasattr(greeks, "gamma")
    assert isinstance(greeks.delta, float)


def test_put_call_parity(sample_params):
    call_price = BlackScholesEngine.price_call(sample_params)
    put_price = BlackScholesEngine.price_put(sample_params)

    is_valid = BlackScholesEngine.verify_put_call_parity(
        sample_params.spot,
        sample_params.strike,
        sample_params.maturity,
        sample_params.rate,
        call_price,
        put_price,
    )
    assert is_valid is True


def test_module_level_helpers(sample_params):
    res = black_scholes(params=sample_params)
    assert "price" in res

    parity = verify_put_call_parity(sample_params)
    assert parity is True


def test_extract_params_missing():
    with pytest.raises(ValueError, match="Missing required parameters"):
        BlackScholesEngine._extract_params(spot=100.0)
