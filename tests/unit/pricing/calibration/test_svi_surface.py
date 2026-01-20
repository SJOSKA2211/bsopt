import pytest
import numpy as np
from src.pricing.calibration.svi_surface import SVISurface

def test_raw_svi():
    # a, b, rho, m, sigma
    val = SVISurface.raw_svi(0.0, 0.04, 0.1, -0.5, 0.0, 0.1)
    assert val > 0

def test_validate_no_arbitrage():
    # Valid
    assert SVISurface.validate_no_arbitrage((0.04, 0.1, -0.5, 0.0, 0.1)) == True
    
    # b < 0
    assert SVISurface.validate_no_arbitrage((0.04, -0.1, -0.5, 0.0, 0.1)) == False
    
    # |rho| >= 1
    assert SVISurface.validate_no_arbitrage((0.04, 0.1, 1.0, 0.0, 0.1)) == False
    
    # non-negative variance violation
    assert SVISurface.validate_no_arbitrage((-1.0, 0.1, 0.0, 0.0, 0.1)) == False
    
    # b(1+|rho|) >= 4
    assert SVISurface.validate_no_arbitrage((0.04, 3.0, 0.5, 0.0, 0.1)) == False

def test_fit_svi_slice():
    log_strikes = np.linspace(-0.5, 0.5, 10)
    # Generate some synthetic total variances
    true_params = (0.04, 0.1, -0.5, 0.0, 0.1)
    total_variances = np.array([SVISurface.raw_svi(k, *true_params) for k in log_strikes])
    
    fitted = SVISurface.fit_svi_slice(log_strikes, total_variances, 1.0)
    assert len(fitted) == 5
    assert all(f >= 0 for f in fitted[:2]) # a, b >= 0

def test_fit_svi_slice_arbitrage_penalty(mocker):
    log_strikes = np.array([0.0])
    total_variances = np.array([0.04])
    
    # Mock validate_no_arbitrage to return False once
    mock_val = mocker.patch("src.pricing.calibration.svi_surface.SVISurface.validate_no_arbitrage", side_effect=[False, True, True, True, True, True, True, True, True, True, True])
    
    fitted = SVISurface.fit_svi_slice(log_strikes, total_variances, 1.0)
    assert len(fitted) == 5
    assert mock_val.called

def test_get_implied_vol():
    params = (0.04, 0.1, -0.5, 0.0, 0.1)
    iv = SVISurface.get_implied_vol(100, 100, 1.0, params)
    assert iv > 0
    assert math.isclose(iv, np.sqrt(SVISurface.raw_svi(0.0, *params)), rel_tol=1e-7)

import math
