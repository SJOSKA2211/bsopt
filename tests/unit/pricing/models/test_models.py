import pytest
import numpy as np
from src.pricing.models import BSParameters, OptionGreeks

def test_bs_parameters_validation():
    # Valid parameters
    params = BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    assert params.spot == 100.0

def test_bs_parameters_invalid_spot():
    with pytest.raises(ValueError, match="Spot, strike, and volatility must be non-negative"):
        BSParameters(spot=-100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)

def test_bs_parameters_invalid_maturity():
    with pytest.raises(ValueError, match="Maturity must be non-negative"):
        BSParameters(spot=100.0, strike=100.0, maturity=-1.0, volatility=0.2, rate=0.05)

def test_bs_parameters_invalid_rate():
    with pytest.raises(ValueError, match="Rate and dividend must be non-negative"):
        BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=-0.05)

def test_option_greeks_getitem():
    greeks = OptionGreeks(delta=0.5, gamma=0.1, theta=-0.05, vega=0.2, rho=0.1)
    assert greeks["delta"] == 0.5
    assert greeks["gamma"] == 0.1

def test_bs_parameters_array_validation():
    # Valid array parameters
    BSParameters(spot=np.array([100.0, 110.0]), strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
    
    # Invalid array parameter
    with pytest.raises(ValueError, match="Spot, strike, and volatility must be non-negative"):
        BSParameters(spot=np.array([100.0, -10.0]), strike=100.0, maturity=1.0, volatility=0.2, rate=0.05)
