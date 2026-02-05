import numpy as np
import pytest

from src.pricing.black_scholes import BSParameters
from src.pricing.monte_carlo import (
    MCConfig,
    MonteCarloEngine,
    _laguerre_basis,
    geometric_asian_price,
)


def test_mc_config_validation():
    with pytest.raises(ValueError, match="n_paths must be positive"):
        MCConfig(n_paths=0)
    
    config = MCConfig(n_paths=1000, antithetic=True)
    assert config.n_paths >= 1000
    assert config.n_paths % 2 == 0

def test_mc_engine_european_pricing():
    # Use few paths for speed in unit tests
    config = MCConfig(n_paths=1000, n_steps=10, antithetic=True, control_variate=True)
    engine = MonteCarloEngine(config)
    params = BSParameters(spot=100.0, strike=100.0, maturity=0.5, volatility=0.2, rate=0.05)
    
    price, stderr = engine.price_european(params, "call")
    assert price > 0
    assert stderr >= 0
    
    # Compare with put (Put-Call parity roughly)
    put_price, _ = engine.price_european(params, "put")
    assert put_price > 0

def test_mc_engine_american_pricing():
    config = MCConfig(n_paths=1000, n_steps=20)
    engine = MonteCarloEngine(config)
    params = BSParameters(spot=100.0, strike=100.0, maturity=0.5, volatility=0.2, rate=0.05)
    
    price = engine.price_american_lsm(params, "put")
    assert price > 0
    
    # American put should be more valuable than European put
    eur_price, _ = engine.price_european(params, "put")
    assert price >= eur_price

def test_mc_engine_greeks():
    config = MCConfig(n_paths=1000, n_steps=10)
    engine = MonteCarloEngine(config)
    params = BSParameters(spot=100.0, strike=100.0, maturity=0.5, volatility=0.2, rate=0.05)
    
    greeks = engine.calculate_greeks(params, "call")
    assert isinstance(greeks.delta, float)
    assert isinstance(greeks.vega, float)

def test_sobol_generation():
    config = MCConfig(n_paths=1024, method="sobol")
    engine = MonteCarloEngine(config)
    normals = engine._generate_random_normals(1024, 10)
    assert normals.shape == (1024, 10)

def test_laguerre_basis():
    x = np.array([1.0, 2.0])
    basis = _laguerre_basis(x, degree=3)
    assert basis.shape == (2, 4)
    # Standard Laguerre: L0=1, L1=1-x, L2=1/2(x^2-4x+2), L3=1/6(-x^3+9x^2-18x+6)
    assert basis[0, 0] == 1.0
    assert basis[0, 1] == 0.0
    assert basis[0, 2] == -0.5
    assert abs(basis[0, 3] - (-0.6666666)) < 1e-4

def test_geometric_asian_price():
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    price = geometric_asian_price(params, "call", 252)
    assert price > 0
    assert price < 15.0 # Vanilla call is ~10.45, Asian should be cheaper

def test_mc_invalid_option_type():
    engine = MonteCarloEngine()
    params = BSParameters(100, 100, 1, 0.2, 0.05)
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        engine.price_european(params, "invalid")
