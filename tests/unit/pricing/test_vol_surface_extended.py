import pytest
import numpy as np
from src.pricing.vol_surface import (
    SVIParameters, SVINaturalParameters, SABRParameters, 
    SVIModel, SABRModel, MarketQuote, CalibrationEngine, 
    VolatilitySurface, ArbitrageDetector
)
from decimal import Decimal

def test_svi_parameters_validation():
    with pytest.raises(ValueError, match="b must be non-negative"):
        SVIParameters(a=0.1, b=-0.1, rho=0.5, m=0.0, sigma=0.1)
    
    with pytest.raises(ValueError, match="rho must be in"):
        SVIParameters(a=0.1, b=0.1, rho=1.1, m=0.0, sigma=0.1)
        
    with pytest.raises(ValueError, match="sigma must be positive"):
        SVIParameters(a=0.1, b=0.1, rho=0.5, m=0.0, sigma=0.0)

def test_svi_natural_to_raw():
    nat = SVINaturalParameters(delta=0.1, mu=0.0, rho=0.5, omega=0.2, zeta=0.1)
    raw = nat.to_raw()
    assert isinstance(raw, SVIParameters)
    assert raw.a == nat.delta
    assert raw.m == nat.mu
    assert raw.rho == nat.rho

def test_sabr_parameters_validation():
    with pytest.raises(ValueError, match="alpha must be positive"):
        SABRParameters(alpha=0.0, beta=0.5, rho=0.0, nu=0.1)
    
    with pytest.raises(ValueError, match="beta must be in"):
        SABRParameters(alpha=0.1, beta=1.5, rho=0.0, nu=0.1)
        
    with pytest.raises(ValueError, match="rho must be in"):
        SABRParameters(alpha=0.1, beta=0.5, rho=1.5, nu=0.1)
        
    with pytest.raises(ValueError, match="nu must be non-negative"):
        SABRParameters(alpha=0.1, beta=0.5, rho=0.0, nu=-0.1)

def test_svi_model_iv_and_derivatives():
    params = SVIParameters(a=0.04, b=0.4, rho=-0.5, m=0.1, sigma=0.1)
    model = SVIModel(params)
    iv = model.implied_volatility(100, 100, 1.0)
    assert iv > 0
    
    # Derivatives
    d1 = model.variance_derivative(0.0)
    d2 = model.variance_second_derivative(0.0)
    assert isinstance(d1, float)
    assert isinstance(d2, float)
    
    # Durrleman check
    cond = model.check_durrleman_condition(np.array([0.0]))
    assert cond.all()

def test_sabr_model_iv():
    params = SABRParameters(alpha=0.3, beta=1.0, rho=-0.5, nu=0.4)
    model = SABRModel(params)
    iv = model.implied_volatility(100, 100, 1.0)
    assert iv > 0
    
    # Near ATM
    iv_atm = model.implied_volatility(100.000000001, 100, 1.0)
    assert iv_atm > 0

def test_calibration_engine():
    engine = CalibrationEngine()
    quotes = [
        MarketQuote(strike=90, maturity=1.0, implied_vol=0.25, forward=100),
        MarketQuote(strike=100, maturity=1.0, implied_vol=0.20, forward=100),
        MarketQuote(strike=110, maturity=1.0, implied_vol=0.18, forward=100),
    ]
    # SVI
    p_svi, diag_svi = engine.calibrate_svi(quotes)
    assert isinstance(p_svi, SVIParameters)
    
    # SABR
    p_sabr, diag_sabr = engine.calibrate_sabr(quotes, fix_beta=0.7)
    assert isinstance(p_sabr, SABRParameters)
    assert p_sabr.beta == 0.7

def test_volatility_surface_extended():
    surface = VolatilitySurface()
    params = SVIParameters(a=0.04, b=0.4, rho=-0.5, m=0.1, sigma=0.1)
    model1 = SVIModel(params)
    surface.add_slice(1.0, model1, 100)
    
    # Interpolation
    params2 = SVIParameters(a=0.05, b=0.4, rho=-0.5, m=0.1, sigma=0.1)
    model2 = SVIModel(params2)
    surface.add_slice(2.0, model2, 100)
    
    # Exact match
    assert surface.implied_volatility(100, 1.0) > 0
    
    # Interp
    assert surface.implied_volatility(100, 1.5) > 0
    
    # Extrap short
    with pytest.warns(UserWarning, match="Extrapolating short"):
        assert surface.implied_volatility(100, 0.5) > 0
        
    # Extrap long
    with pytest.warns(UserWarning, match="Extrapolating long"):
        assert surface.implied_volatility(100, 3.0) > 0
        
    # Dataframe
    df = surface.to_dataframe()
    assert not df.empty
    
    # Smile
    smile = surface.get_smile(1.0, (80, 120))
    assert not smile.empty

def test_arbitrage_detector_butterfly():
    detector = ArbitrageDetector()
    strikes = np.array([90, 100, 110])
    
    # Free
    prices_free = np.array([15.0, 10.0, 6.0]) 
    free, _ = detector.check_butterfly_arbitrage(strikes, prices_free)
    assert free
    
    # Violation
    prices_bad = np.array([15.0, 10.0, 2.0]) # Concave: 10-15=-5, 2-10=-8. -8-(-5)=-3
    free_bad, _ = detector.check_butterfly_arbitrage(strikes, prices_bad)
    assert not free_bad

def test_arbitrage_detector_calendar_and_svi():
    detector = ArbitrageDetector()
    
    # Calendar
    maturities = np.array([1.0, 2.0])
    vars_free = np.array([0.04, 0.05])
    assert detector.check_calendar_arbitrage(maturities, vars_free)[0]
    
    vars_bad = np.array([0.04, 0.03])
    assert not detector.check_calendar_arbitrage(maturities, vars_bad)[0]
    
    # SVI check
    params = SVIParameters(a=0.04, b=0.4, rho=-0.5, m=0.1, sigma=0.1)
    model = SVIModel(params)
    res = detector.check_svi_arbitrage(model)
    assert res["is_arbitrage_free"]
    
    params_bad = SVIParameters(a=-0.01, b=0.4, rho=-0.5, m=0.1, sigma=0.1)
    model_bad = SVIModel(params_bad)
    res_bad = detector.check_svi_arbitrage(model_bad)
    assert not res_bad["is_arbitrage_free"]