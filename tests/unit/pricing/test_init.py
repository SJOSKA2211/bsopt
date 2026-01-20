import pytest
import os
import src.pricing as pricing

def test_pricing_init_dir():
    d = dir(pricing)
    assert "BlackScholesEngine" in d
    assert "QuantumOptionPricer" in d

def test_pricing_init_getattr():
    # Trigger lazy import
    engine = pricing.BlackScholesEngine
    assert engine is not None
    
    with pytest.raises(AttributeError):
        _ = pricing.InvalidAttribute

def test_preload_classical_pricers():
    pricing.preload_classical_pricers()
    # If it doesn't crash, it works

def test_auto_preload_logic(mocker):
    # Mock ENVIRONMENT and PRELOAD_PRICING
    mocker.patch.dict(os.environ, {"ENVIRONMENT": "production", "PRELOAD_PRICING": "true"})
    
    # We need to reload the module to trigger the top-level code
    import importlib
    importlib.reload(pricing)
    # The check is at module level, so reloading should trigger it
