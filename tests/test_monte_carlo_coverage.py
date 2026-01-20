import numpy as np
import pytest
from src.pricing.monte_carlo import (
    MonteCarloEngine,
    MCConfig,
    BSParameters,
    geometric_asian_price,
    _laguerre_basis,
)
from src.pricing.models import OptionGreeks

def test_mc_calculate_greeks():
    engine = MonteCarloEngine()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    greeks = engine.calculate_greeks(params, "call")
    assert isinstance(greeks, OptionGreeks)
    assert greeks.delta == 0.0

def test_price_european_invalid_type():
    engine = MonteCarloEngine()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        engine.price_european(params, "binary")

def test_price_european_zero_maturity():
    engine = MonteCarloEngine()
    params = BSParameters(110, 100, 0.0, 0.2, 0.05)
    price, ci = engine.price_european(params, "call")
    assert price == 10.0
    assert ci == 0.0
    
    price_put, _ = engine.price_european(params, "put")
    assert price_put == 0.0

def test_price_american_zero_maturity():
    engine = MonteCarloEngine()
    params = BSParameters(110, 100, 0.0, 0.2, 0.05)
    price = engine.price_american_lsm(params, "call")
    assert price == 10.0
    
    price_put = engine.price_american_lsm(params, "put")
    assert price_put == 0.0

def test_price_american_lsm_no_itm_paths():
    # Set strike very high so no paths are ITM for a call
    engine = MonteCarloEngine(MCConfig(n_paths=100, n_steps=2))
    params = BSParameters(10, 1000, 1.0, 0.01, 0.01)
    price = engine.price_american_lsm(params, "call")
    assert price == 0.0

def test_geometric_asian_invalid_type():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    with pytest.raises(ValueError, match="Option type must be 'call' or 'put'"):
        geometric_asian_price(params, "binary", 10)

def test_geometric_asian_put():
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    price = geometric_asian_price(params, "put", 10)
    assert price > 0

def test_mc_config_sobol():
    config = MCConfig(n_paths=100, method="sobol")
    # 100 rounded up to next power of 2 is 128
    assert config.n_paths == 128

def test_price_american_invalid_type():
    engine = MonteCarloEngine()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
        engine.price_american_lsm(params, "binary")

def test_price_european_zero_variance():
    # Test control variate with zero variance to hit cov_matrix[1,1] <= 0 branch
    engine = MonteCarloEngine(MCConfig(control_variate=True))
    params = BSParameters(100, 100, 1.0, 0.0, 0.05) # zero vol
    price, _ = engine.price_european(params, "call")
    # Call price should be max(S0 - K*exp(-rT), 0) = 100 - 100*exp(-0.05)
    expected = 100 - 100 * np.exp(-0.05)
    assert price == pytest.approx(expected, rel=1e-5)
    assert price > 0

def test_laguerre_basis_branches():
    x = np.array([1.0, 2.0])
    # Test n=0 to hit n>=1 False branch
    basis = _laguerre_basis(x, n=0)
    assert basis.shape == (2, 1)
    
    # Test n=1 to hit n>=2 False branch
    basis = _laguerre_basis(x, n=1)
    assert basis.shape == (2, 2)
    
    # Test n=2 to hit n>=3 False branch
    basis = _laguerre_basis(x, n=2)
    assert basis.shape == (2, 3)

def test_mc_price_instance_method():
    engine = MonteCarloEngine()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    price = engine.price(params, "call")
    assert price > 0