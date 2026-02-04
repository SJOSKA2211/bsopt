import pytest
from src.pricing.models import BSParameters, OptionGreeks

def test_bs_parameters_validation():
    # Valid
    BSParameters(100, 100, 1, 0.2, 0.05)
    
    # Negative spot
    with pytest.raises(ValueError, match="Spot, strike, and volatility must be non-negative"):
        BSParameters(-100, 100, 1, 0.2, 0.05)
        
    # Negative maturity
    with pytest.raises(ValueError, match="Maturity cannot be negative"):
        BSParameters(100, 100, -1, 0.2, 0.05)
        
    # Negative rate
    with pytest.raises(ValueError, match="Rate and dividend cannot be negative"):
        BSParameters(100, 100, 1, 0.2, -0.05)

def test_option_greeks_getitem():
    greeks = OptionGreeks(delta=0.5, gamma=0.1, theta=-0.05, vega=0.2, rho=0.1)
    assert greeks["delta"] == 0.5
    
    with pytest.raises(TypeError, match="OptionGreeks indices must be strings"):
        _ = greeks[0]

def test_option_greeks_contains():
    greeks = OptionGreeks(delta=0.5, gamma=0.1, theta=-0.05, vega=0.2, rho=0.1)
    assert "delta" in greeks
    assert "invalid" not in greeks
