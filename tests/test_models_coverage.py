import numpy as np
import pytest
from src.pricing.models import BSParameters, OptionGreeks

def test_bs_parameters_validation():
    # Valid
    p = BSParameters(100, 100, 1.0, 0.2, 0.05)
    assert p.spot == 100
    
    # Invalid spot
    with pytest.raises(ValueError, match="Spot, strike, and volatility must be positive"):
        BSParameters(0, 100, 1.0, 0.2, 0.05)
        
    # Invalid maturity
    with pytest.raises(ValueError, match="Maturity cannot be negative"):
        BSParameters(100, 100, -1.0, 0.2, 0.05)
        
    # Invalid rate
    with pytest.raises(ValueError, match="Rate and dividend cannot be negative"):
        BSParameters(100, 100, 1.0, 0.2, -0.05)

def test_bs_parameters_vectorized_validation():
    # Vectorized valid
    BSParameters(np.array([100, 110]), 100, 1.0, 0.2, 0.05)
    
    # Vectorized invalid
    with pytest.raises(ValueError, match="Spot, strike, and volatility must be positive"):
        BSParameters(np.array([100, -1]), 100, 1.0, 0.2, 0.05)

def test_option_greeks_getitem():
    g = OptionGreeks(delta=0.5, gamma=0.01, theta=-5.0, vega=10.0, rho=20.0)
    assert g['delta'] == 0.5
    assert g['gamma'] == 0.01
    assert g.phi is None
