import dataclasses

import pytest

from src.math_kernel.models import BSParameters
from src.math_kernel.monte_carlo import MCConfig, MonteCarloEngine, geometric_asian_price

@pytest.fixture
def sample_params():
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0
    )

def test_mc_config_validation():
    with pytest.raises(ValueError, match="positive"):
        MCConfig(n_paths=0)

    config = MCConfig(method="sobol", n_paths=100)
    assert config.n_paths == 128  # Next power of 2

def test_price_european(sample_params):
    engine = MonteCarloEngine(MCConfig(n_paths=1000, control_variate=False))
    price, ci = engine.price_european(sample_params, option_type="call")
    assert price > 0
    assert ci > 0

def test_price_european_with_control_variate(sample_params):
    engine = MonteCarloEngine(MCConfig(n_paths=1000, control_variate=True))
    price, ci = engine.price_european(sample_params, option_type="call")
    assert price > 0

def test_calculate_greeks_pwm(sample_params):
    # PWM requires standard MC (no CV, no Sobol)
    engine = MonteCarloEngine(MCConfig(n_paths=1000, control_variate=False))
    greeks = engine.calculate_greeks(sample_params, option_type="call")
    assert greeks.delta > 0
    assert greeks.gamma > 0
    assert greeks.theta != 0

def test_calculate_greeks_fd(sample_params):
    # CV or Sobol triggers FD fallback
    engine = MonteCarloEngine(MCConfig(n_paths=1000, control_variate=True))
    greeks = engine.calculate_greeks(sample_params, option_type="call")
    assert greeks.delta > 0

def test_price_american_lsm(sample_params):
    engine = MonteCarloEngine(MCConfig(n_paths=1000, n_steps=10))
    price = engine.price_american_lsm(sample_params, option_type="call")
    assert price > 0

def test_geometric_asian_price(sample_params):
    price = geometric_asian_price(sample_params, option_type="call", n_obs=10)
    assert price > 0
    with pytest.raises(ValueError, match="positive"):
        geometric_asian_price(sample_params, option_type="call", n_obs=0)

def test_price_european_at_maturity(sample_params):
    params = dataclasses.replace(sample_params, maturity=0)
    engine = MonteCarloEngine()
    price, _ = engine.price_european(params, option_type="call")
    assert price == 0.0  # OTM or ATM call at maturity

    params.spot = 110.0
    price, _ = engine.price_european(params, option_type="call")
    assert price == 10.0
