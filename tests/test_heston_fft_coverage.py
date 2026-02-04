import numpy as np
import pytest
from src.pricing.models.heston_fft import HestonParams, HestonModelFFT

def test_heston_params_validation():
    # Correlation OOB
    with pytest.raises(ValueError, match="Correlation must be in"):
        HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=1.5)
        
    # Non-positive params
    with pytest.raises(ValueError, match="All parameters except rho must be positive"):
        HestonParams(v0=0.0, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)

def test_heston_model_validation():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    # Unusual rate (warning)
    HestonModelFFT(params, r=0.6, T=1.0)
    
    # Invalid T
    with pytest.raises(ValueError, match="Time to maturity must be in"):
        HestonModelFFT(params, r=0.05, T=11.0)

def test_heston_price_call_validation():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    with pytest.raises(ValueError, match="Prices must be positive"):
        model.price_call(0, 100)

def test_heston_price_call_itm():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    # S0=110, K=100 -> ITM
    price = model.price_call(110, 100)
    assert price > 10.0

def test_heston_char_func_singularity():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    # u=0 makes xi=d=kappa for rho=0. xi-d=0.
    res = model.characteristic_func(0.0)
    assert res == 0.0 + 0.0j

def test_heston_price_call_min_price():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    # Deep OTM call to hit MIN_PRICE
    price = model.price_call(10, 1000)
    assert price == model.MIN_PRICE

def test_heston_char_func_overflow():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    from unittest.mock import patch
    with patch('numpy.exp', return_value=complex(1e20, 0)):
        res = model.characteristic_func(1.0)
        assert res == 0.0 + 0.0j

def test_heston_char_func_runtime_error():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    from unittest.mock import patch
    with patch('numpy.exp', side_effect=OverflowError("overflow")):
        res = model.characteristic_func(1.0)
        assert res == 0.0 + 0.0j

def test_heston_price_call_high_error():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    from unittest.mock import patch
    with patch('src.pricing.models.heston_fft.quad', return_value=(1.0, 0.1)):
        model.price_call(100, 100)

def test_heston_price_call_exceeds_spot():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    from unittest.mock import patch
    with patch('src.pricing.models.heston_fft.quad', return_value=(1e10, 0.0)):
        price = model.price_call(100, 100)
        assert price <= 100.0

def test_heston_price_put_error_fallback():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    from unittest.mock import patch
    # Patch the actual method on the instance to bypass cache and hit exception
    with patch.object(model, 'characteristic_func', side_effect=Exception("mock error")):
        price = model.price_put(100, 100)
        assert price >= model.MIN_PRICE

def test_heston_price_call_error_fallback():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    from unittest.mock import patch
    with patch.object(model, 'characteristic_func', side_effect=Exception("mock error")):
        price = model.price_call(100, 100)
        assert price >= model.MIN_PRICE

def test_heston_integrand_den_zero():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=0.0)
    model = HestonModelFFT(params, r=0.05, T=1.0)
    # Use alpha=0.0. At v=0, den=0.
    price = model.price_call(100, 100, alpha=0.0)
    assert price >= 0