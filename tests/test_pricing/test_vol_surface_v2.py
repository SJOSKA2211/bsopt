import pytest
import numpy as np
from src.pricing.vol_surface import SVIModel, SABRModel, VolatilitySurface, SVIParameters

def test_svi_model():
    params = SVIParameters(a=0.1, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
    model = SVIModel(params)
    
    # Check total variance
    k = np.array([-0.1, 0.0, 0.1])
    var = model.total_variance(k)
    assert len(var) == 3
    assert np.all(var > 0)

def test_sabr_model():
    # SABRModel expects (alpha, beta, rho, nu)
    model = SABRModel(0.2, 0.5, -0.2, 0.3)
    # Check ATM vol
    vol = model.implied_volatility(100.0, 100.0, 1.0)
    assert vol > 0

def test_vol_surface_interpolation():
    surface = VolatilitySurface()
    model1 = SVIModel(SVIParameters(0.1, 0.1, -0.5, 0.0, 0.1))
    model2 = SVIModel(SVIParameters(0.2, 0.1, -0.5, 0.0, 0.1))
    
    # add_slice(expiry, model, forward)
    surface.add_slice(0.5, model1, 100.0)
    surface.add_slice(1.0, model2, 100.0)
    
    # Interpolated vol
    vol = surface.get_volatility(100.0, 100.0, 0.75)
    assert vol > 0
