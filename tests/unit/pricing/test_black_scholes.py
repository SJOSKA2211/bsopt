import numpy as np
import pytest

from src.pricing.black_scholes import BlackScholesEngine, BSParameters, verify_put_call_parity
from src.pricing.models import OptionGreeks


def test_bs_engine_single_pricing():
    """Test basic call/put pricing and Greeks."""
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02)

    call_price = BlackScholesEngine.price_options(params=params, option_type="call")
    put_price = BlackScholesEngine.price_options(params=params, option_type="put")

    assert 9.20 < call_price < 9.25
    assert 6.30 < put_price < 6.35
    assert call_price > put_price

    engine = BlackScholesEngine()
    greeks = engine.calculate_greeks(params=params, option_type="call")
    assert isinstance(greeks, OptionGreeks)
    assert 0 < greeks.delta < 1
    assert greeks.gamma > 0
    assert greeks.vega > 0
    assert greeks.theta < 0
    assert greeks.rho > 0

def test_bs_engine_batch_pricing():
    """Test vectorized pricing performance."""
    spots = np.array([100.0, 110.0, 120.0])
    params = BSParameters(spot=spots, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0)
    
    prices = BlackScholesEngine.price_options(params=params, option_type="call")
    assert isinstance(prices, np.ndarray)
    assert len(prices) == 3
    assert prices[2] > prices[1] > prices[0]

def test_bs_engine_edge_cases():
    """Test boundaries and stability."""
    # Zero maturity
    params = BSParameters(spot=100.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    assert BlackScholesEngine.price_options(params=params, option_type="call") == 0.0
    
    # ITM at maturity
    params_itm = BSParameters(spot=110.0, strike=100.0, maturity=0.0, volatility=0.2, rate=0.05)
    assert BlackScholesEngine.price_options(params=params_itm, option_type="call") == 10.0

    # Very short maturity
    params_short = BSParameters(spot=100.0, strike=100.0, maturity=1/365, volatility=0.2, rate=0.05)
    assert BlackScholesEngine.price_options(params=params_short, option_type="call") > 0

def test_greeks_consistency_and_parity():
    """Verify analytical consistency and put-call parity."""
    params = BSParameters(spot=100.0, strike=100.0, maturity=0.5, volatility=0.25, rate=0.05, dividend=0.02)

    engine = BlackScholesEngine()
    call_greeks = engine.calculate_greeks(params, "call")
    put_greeks = engine.calculate_greeks(params, "put")

    # Gamma and Vega are identical for European calls/puts
    assert pytest.approx(call_greeks.gamma) == put_greeks.gamma
    assert pytest.approx(call_greeks.vega) == put_greeks.vega

    # Delta diff should be e^(-qT)
    expected_delta_diff = np.exp(-params.dividend * params.maturity)
    assert pytest.approx(call_greeks.delta - put_greeks.delta) == expected_delta_diff

    # Put-Call Parity check
    assert verify_put_call_parity(params)

def test_monotonicity():
    """Verify price increases with spot price."""
    spots = np.linspace(80, 120, 10)
    params = BSParameters(spot=spots, strike=100, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02)
    prices = BlackScholesEngine.price_options(params=params, option_type="call")
    assert np.all(np.diff(prices) > 0)
