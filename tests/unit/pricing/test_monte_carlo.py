import pytest
import numpy as np
from src.pricing.monte_carlo import MonteCarloEngine, MCConfig
from src.pricing.black_scholes import BSParameters

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
