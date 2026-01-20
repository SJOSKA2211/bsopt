import pytest
import numpy as np
from src.pricing.monte_carlo import MonteCarloEngine, MCConfig, geometric_asian_price
from src.pricing.models import BSParameters

def test_mc_config_validation():
    with pytest.raises(ValueError, match="n_paths must be positive"):
        MCConfig(n_paths=0)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        MCConfig(n_steps=0)
    
    config = MCConfig(n_paths=1001, antithetic=True)
    assert config.n_paths == 1002
    
    config_sobol = MCConfig(n_paths=1000, method="sobol")
    assert config_sobol.n_paths == 1024

def test_mc_engine_basic_pricing():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    engine = MonteCarloEngine(MCConfig(n_paths=1000))
    price = engine.price(params, "call")
    assert price > 0
    
    greeks = engine.calculate_greeks(params)
    assert greeks.delta == 0.0

def test_mc_engine_price_european_variants():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    
    # Put option
    engine = MonteCarloEngine(MCConfig(n_paths=1000))
    price_put, _ = engine.price_european(params, "put")
    assert price_put > 0
    
    # No variance reduction
    engine_basic = MonteCarloEngine(MCConfig(n_paths=1000, antithetic=False, control_variate=False))
    price_basic, ci = engine_basic.price_european(params, "call")
    assert price_basic > 0
    assert ci > 0
    
    # Zero maturity
    params_zero = BSParameters(100, 100, 0.0, 0.2, 0.05)
    price_zero, ci_zero = engine.price_european(params_zero, "call")
    assert price_zero == 0.0
    assert ci_zero == 0.0

def test_mc_engine_invalid_option_type():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    engine = MonteCarloEngine()
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        engine.price_european(params, "invalid")
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        engine.price_american_lsm(params, "invalid")

def test_mc_engine_sobol():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    engine = MonteCarloEngine(MCConfig(n_paths=1024, method="sobol", n_steps=2))
    normals = engine._generate_random_normals(1024, 2)
    assert normals.shape == (1024, 2)

def test_mc_engine_generate_normals_standard():
    engine = MonteCarloEngine(MCConfig(n_paths=100, method="monte_carlo", n_steps=2))
    normals = engine._generate_random_normals(100, 2)
    assert normals.shape == (100, 2)

def test_mc_engine_american_lsm():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    engine = MonteCarloEngine(MCConfig(n_paths=1000, n_steps=50))
    
    price_call = engine.price_american_lsm(params, "call")
    assert price_call > 0
    
    price_put = engine.price_american_lsm(params, "put")
    assert price_put > 0
    
    # Zero maturity
    params_zero = BSParameters(100, 100, 0.0, 0.2, 0.05)
    assert engine.price_american_lsm(params_zero, "call") == 0.0

def test_mc_engine_american_lsm_no_itm():
    # OTM call at maturity, very low spot
    params = BSParameters(10, 100, 1.0, 0.2, 0.05)
    engine = MonteCarloEngine(MCConfig(n_paths=100, n_steps=10))
    price = engine.price_american_lsm(params, "call")
    # Should be close to 0
    assert price >= 0

def test_geometric_asian_price():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    
    price_call = geometric_asian_price(params, "call", 10)
    assert price_call > 0
    
    price_put = geometric_asian_price(params, "put", 10)
    assert price_put > 0
    
    with pytest.raises(ValueError, match="Number of observations must be positive"):
        geometric_asian_price(params, "call", 0)
        
    with pytest.raises(ValueError, match="Option type must be 'call' or 'put'"):
        geometric_asian_price(params, "invalid", 10)