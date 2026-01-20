import pytest
import numpy as np
import math
from src.pricing.models.heston_fft import HestonParams, HestonModelFFT

@pytest.fixture
def valid_params():
    # v0, kappa, theta, sigma, rho
    # 2*kappa*theta > sigma^2 -> 2*2*0.04 = 0.16 > 0.1^2 = 0.01 (True)
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=-0.7)

def test_heston_params_validation():
    # Valid
    HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=-0.7)
    
    # Feller condition violation
    with pytest.raises(ValueError, match="Feller condition violated"):
        HestonParams(v0=0.04, kappa=0.1, theta=0.04, sigma=0.5, rho=-0.7)
        
    # Correlation out of bounds
    with pytest.raises(ValueError, match="Correlation must be in"):
        HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=1.1)
        
    # Positive parameters
    with pytest.raises(ValueError, match="must be positive"):
        HestonParams(v0=-0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=-0.7)

def test_heston_model_init(valid_params):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    assert model.r == 0.05
    assert model.T == 1.0
    
    # Unusual rate (warning logged)
    _ = HestonModelFFT(valid_params, r=0.6, T=1.0)
    
    # Invalid maturity
    with pytest.raises(ValueError, match="Time to maturity must be in"):
        HestonModelFFT(valid_params, r=0.05, T=0.0)

def test_characteristic_func(valid_params):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    phi = model.characteristic_func(1.0)
    assert isinstance(phi, complex)
    assert abs(phi) <= 1.0 # Characteristic function of distribution
    
    # Trigger log-space computation
    phi_large = model.characteristic_func(10.0)
    assert isinstance(phi_large, complex)

def test_characteristic_func_errors(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    
    # Mock overflow
    mocker.patch("numpy.exp", side_effect=OverflowError("Overflow"))
    res = model.characteristic_func(1.0)
    assert res == 0j

def test_price_call(valid_params):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # OTM call
    price = model.price_call(S0=100, K=110)
    assert price > 0
    assert price < 100
    
    # ITM call (uses parity internally)
    price_itm = model.price_call(S0=110, K=100)
    assert price_itm > 10.0 # Intrinsic is ~10 + time value
    
    with pytest.raises(ValueError, match="Prices must be positive"):
        model.price_call(S0=0, K=100)

def test_price_put(valid_params):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # OTM put
    price = model.price_put(S0=110, K=100)
    assert price > 0
    
    # ITM put (uses parity internally)
    price_itm = model.price_put(S0=100, K=110)
    assert price_itm > 0

def test_pricing_integration_error_high(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock quad to return high error estimate
    mocker.patch("src.pricing.models.heston_fft.quad", return_value=(0.1, 0.5))
    p = model.price_call(100, 100)
    assert p > 0

def test_price_exceeds_spot(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock result to exceed spot (S0=100)
    # The integral result is multiplied by (exp(-alpha*k)/pi) * exp(-r*T) * S0
    # To exceed S0, integral needs to be large.
    mocker.patch("src.pricing.models.heston_fft.quad", return_value=(1e10, 0.0))
    p = model.price_call(100, 100)
    assert p <= 100.0

def test_pricing_failed_exception(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock quad to fail at top level of price_call
    mocker.patch("src.pricing.models.heston_fft.quad", side_effect=RuntimeError("Extreme fail"))
    p = model.price_call(100, 100)
    assert p >= 0 # Returns fallback

def test_price_floor(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock result to be very small
    mocker.patch("src.pricing.models.heston_fft.quad", return_value=(1e-20, 0.0))
    p = model.price_call(100, 100)
    assert p == 1e-10

def test_price_put_integrand_error(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock characteristic_func to raise error inside put integrand
    mocker.patch.object(HestonModelFFT, "characteristic_func", side_effect=Exception("CF error"))
    # We need to trigger the integrand call. 
    # Since we can't easily mock just the integrand function inside price_put,
    # we rely on the fact that quad will call it.
    p = model.price_put(100, 90)
    assert p == 1e-10 # Returns MIN_PRICE because integrand always returns 0.0

def test_pricing_integration_errors(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock quad to fail inside price_call
    mocker.patch("src.pricing.models.heston_fft.quad", side_effect=Exception("Integration failed"))
    
    # price_call will catch this and hit lines 143-145
    p = model.price_call(100, 90)
    assert p >= 10.0

def test_price_put_failure(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock quad to fail at top level of price_put
    mocker.patch("src.pricing.models.heston_fft.quad", side_effect=Exception("Top level fail"))
    p = model.price_put(100, 90)
    assert p > 0

def test_integrand_zero_denominator(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Trigger line 115 by mocking den calculation or forcing it
    # Since it's local, we mock 'abs' to return 0 for the denominator check
    mocker.patch("src.pricing.models.heston_fft.abs", side_effect=lambda x: 0.0 if isinstance(x, complex) else abs(x))
    p = model.price_call(100, 110)
    assert p > 0

def test_integrand_exception(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # The integrand function is nested, but it calls self.characteristic_func
    # We can mock that to raise an exception when called from within the integrand
    mocker.patch.object(HestonModelFFT, "characteristic_func", side_effect=Exception("integrand error"))
    
    # We need to trigger the quad call which will execute the integrand
    # price_call will catch the exception from quad and return fallback
    p = model.price_call(100, 110)
    assert p >= 0

def test_price_put_integrand_exception(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock den calculation or complex conversion to raise exception in put integrand
    mocker.patch("numpy.real", side_effect=Exception("put integrand error"))
    p = model.price_put(100, 90)
    assert p >= 0




def test_characteristic_func_overflow_guard(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Mock result of np.exp(A+B) to be very large
    mocker.patch("numpy.exp", return_value=complex(1e20, 0))
    res = model.characteristic_func(1.0)
    assert res == 0j

def test_characteristic_func_singularity(valid_params, mocker):
    model = HestonModelFFT(valid_params, r=0.05, T=1.0)
    # Force xi - d to be small
    # xi = p.kappa - p.sigma * p.rho * u * 1j
    # d = np.sqrt(xi**2 + p.sigma**2 * (u**2 + 1j * u))
    # We can mock np.sqrt or xi
    mocker.patch("numpy.sqrt", return_value=model.params.kappa - model.params.sigma * model.params.rho * 1.0 * 1j)
    res = model.characteristic_func(1.0)
    assert res == 0j
