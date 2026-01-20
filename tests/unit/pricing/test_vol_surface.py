import pytest
import numpy as np
import math
import warnings
from src.pricing.vol_surface import (
    SVIParameters,
    SVINaturalParameters,
    SABRParameters,
    MarketQuote,
    SVIModel,
    SABRModel,
    CalibrationEngine,
    CalibrationConfig,
    ArbitrageDetector,
    VolatilitySurface,
    InterpolationMethod
)

def test_svi_parameters_validation():
    # Valid
    SVIParameters(a=0.1, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    
    with pytest.raises(ValueError, match="b must be non-negative"):
        SVIParameters(a=0.1, b=-0.1, rho=0.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="rho must be in"):
        SVIParameters(a=0.1, b=0.1, rho=1.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="sigma must be positive"):
        SVIParameters(a=0.1, b=0.1, rho=0.0, m=0.0, sigma=0.0)
        
    # Non-negative variance violation warning
    with pytest.warns(UserWarning, match="non-negative variance violation"):
        SVIParameters(a=-1.0, b=0.1, rho=0.0, m=0.0, sigma=0.1)

def test_svi_natural_parameters():
    nat = SVINaturalParameters(delta=0.1, mu=0.0, rho=0.0, omega=0.1, zeta=0.1)
    raw = nat.to_raw()
    assert isinstance(raw, SVIParameters)
    assert raw.a == 0.1

def test_sabr_parameters_validation():
    # Valid
    SABRParameters(alpha=0.2, beta=0.5, rho=0.0, nu=0.1)
    
    with pytest.raises(ValueError, match="alpha must be positive"):
        SABRParameters(alpha=0.0, beta=0.5, rho=0.0, nu=0.1)
    with pytest.raises(ValueError, match="beta must be in"):
        SABRParameters(alpha=0.2, beta=1.1, rho=0.0, nu=0.1)
    with pytest.raises(ValueError, match="rho must be in"):
        SABRParameters(alpha=0.2, beta=0.5, rho=-1.0, nu=0.1)
    with pytest.raises(ValueError, match="nu must be non-negative"):
        SABRParameters(alpha=0.2, beta=0.5, rho=0.0, nu=-0.1)

def test_svi_model():
    params = SVIParameters(a=0.04, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
    model = SVIModel(params)
    
    # total_variance
    assert model.total_variance(0.0) > 0
    assert isinstance(model.total_variance(np.array([0.0, 0.1])), np.ndarray)
    
    # implied_volatility
    iv = model.implied_volatility(100, 100, 1.0)
    assert iv > 0
    iv_arr = model.implied_volatility([90, 100, 110], 100, 1.0)
    assert len(iv_arr) == 3
    
    with pytest.raises(ValueError, match="Maturity must be positive"):
        model.implied_volatility(100, 100, 0.0)
        
    # Derivatives
    assert model.variance_derivative(0.0) is not None
    assert model.variance_second_derivative(0.0) > 0
    
    # Durrleman
    assert np.all(model.check_durrleman_condition(np.array([0.0, 0.1])))

def test_sabr_model():
    params = SABRParameters(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    model = SABRModel(params)
    
    iv = model.implied_volatility(100, 100, 1.0)
    assert iv > 0
    
    # Array strike
    iv_arr = model.implied_volatility(np.array([90, 100, 110]), 100, 1.0)
    assert len(iv_arr) == 3
    
    with pytest.raises(ValueError, match="Maturity must be positive"):
        model.implied_volatility(100, 100, -1.0)

def test_calibration_engine_svi():
    quotes = [
        MarketQuote(strike=90, maturity=1.0, implied_vol=0.22, forward=100),
        MarketQuote(strike=100, maturity=1.0, implied_vol=0.20, forward=100),
        MarketQuote(strike=110, maturity=1.0, implied_vol=0.19, forward=100)
    ]
    engine = CalibrationEngine()
    params, diag = engine.calibrate_svi(quotes)
    assert isinstance(params, SVIParameters)
    assert diag["rmse"] >= 0
    
    # Weighted by vega
    quotes_vega = [
        MarketQuote(strike=90, maturity=1.0, implied_vol=0.22, forward=100, vega=0.1),
        MarketQuote(strike=100, maturity=1.0, implied_vol=0.20, forward=100, vega=0.2),
        MarketQuote(strike=110, maturity=1.0, implied_vol=0.19, forward=100, vega=0.1)
    ]
    engine_weighted = CalibrationEngine(CalibrationConfig(weighted_by_vega=True))
    params_w, _ = engine_weighted.calibrate_svi(quotes_vega)
    assert isinstance(params_w, SVIParameters)

def test_calibration_engine_errors():
    engine = CalibrationEngine()
    with pytest.raises(ValueError, match="No market quotes"):
        engine.calibrate_svi([])
        
    with pytest.raises(ValueError, match="All quotes must have the same maturity"):
        engine.calibrate_svi([
            MarketQuote(100, 1.0, 0.2, 100),
            MarketQuote(100, 2.0, 0.2, 100)
        ])

def test_calibrate_sabr():
    engine = CalibrationEngine()
    quotes = [MarketQuote(100, 1.0, 0.2, 100)]
    params, diag = engine.calibrate_sabr(quotes, fix_beta=0.7)
    assert params.beta == 0.7

def test_arbitrage_detector():
    detector = ArbitrageDetector()
    strikes = np.array([90, 100, 110])
    prices = np.array([15.0, 10.0, 6.0]) # Convex: (10-15)=-5, (6-10)=-4. Increments: 5, 4. Diff: -1? No.
    # Prices: 15, 10, 6. diff: -5, -4. diff2: 1.
    is_free, violations = detector.check_butterfly_arbitrage(strikes, prices)
    assert is_free == True
    
    # Calendar arbitrage
    maturities = np.array([1.0, 2.0])
    vars = np.array([0.04, 0.05])
    is_cal_free, _ = detector.check_calendar_arbitrage(maturities, vars)
    assert is_cal_free == True
    
    # SVI arbitrage
    params = SVIParameters(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    res = detector.check_svi_arbitrage(SVIModel(params))
    assert res["is_arbitrage_free"] == True

def test_volatility_surface():
    surface = VolatilitySurface()
    params = SVIParameters(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    model = SVIModel(params)
    surface.add_slice(1.0, model, 100.0)
    
    # Exact maturity
    assert surface.implied_volatility(100, 1.0) == model.implied_volatility(100, 100, 1.0)
    
    # Mix models error
    with pytest.raises(ValueError, match="Cannot mix model types"):
        surface.add_slice(2.0, SABRModel(SABRParameters(0.2, 0.5, 0, 0.1)), 100.0)
        
    # Interpolation
    params2 = SVIParameters(a=0.06, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    surface.add_slice(2.0, SVIModel(params2), 100.0)
    iv_interp = surface.implied_volatility(100, 1.5)
    assert iv_interp > 0
    
    # Extrapolation
    with pytest.warns(UserWarning, match="Extrapolating short maturity"):
        assert surface.implied_volatility(100, 0.5) > 0
    with pytest.warns(UserWarning, match="Extrapolating long maturity"):
        assert surface.implied_volatility(100, 2.5) > 0

def test_volatility_surface_empty():
    surface = VolatilitySurface()
    with pytest.raises(ValueError, match="No models in surface"):
        surface.implied_volatility(100, 1.0)

def test_volatility_surface_dataframe_and_smile():
    surface = VolatilitySurface()
    params = SVIParameters(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    surface.add_slice(1.0, SVIModel(params), 100.0)
    
    df = surface.to_dataframe()
    assert len(df) == 1
    
    smile = surface.get_smile(1.0, (90, 110), num_points=10)
    assert len(smile) == 10