import pytest
import numpy as np
from src.pricing.vol_surface import SVIParameters, SABRParameters, SVIModel, SABRModel, VolatilitySurface

def test_svi_parameters_validation():
    with pytest.raises(ValueError, match="b must be non-negative"):
        SVIParameters(a=0.1, b=-0.1, rho=0.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="rho must be in"):
        SVIParameters(a=0.1, b=0.1, rho=1.1, m=0.0, sigma=0.1)

def test_svi_model_vol():
    params = SVIParameters(a=0.04, b=0.4, rho=-0.7, m=0.0, sigma=0.1)
    model = SVIModel(params)
    vol = model.implied_volatility(100.0, 100.0, 1.0)
    assert vol > 0
    
    # Skew: OTM put (K=90) should have higher vol than OTM call (K=110) due to negative rho
    vol_put = model.implied_volatility(90.0, 100.0, 1.0)
    vol_call = model.implied_volatility(110.0, 100.0, 1.0)
    assert vol_put > vol_call

def test_sabr_model_vol():
    params = SABRParameters(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    model = SABRModel(params)
    vol = model.implied_volatility(100.0, 100.0, 1.0)
    assert vol > 0

def test_vol_surface_interpolation():
    surface = VolatilitySurface()
    p1 = SVIParameters(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    p2 = SVIParameters(a=0.06, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    
    surface.add_slice(0.5, SVIModel(p1), 100.0)
    surface.add_slice(1.0, SVIModel(p2), 100.0)
    
    vol = surface.implied_volatility(100.0, 0.75)
    assert vol > 0
    # Interpolated variance should be between slices
    vol05 = surface.implied_volatility(100.0, 0.5)
    vol10 = surface.implied_volatility(100.0, 1.0)
    assert vol05 <= vol <= vol10 or vol10 <= vol <= vol05

def test_calibration_engine_svi():
    from src.pricing.vol_surface import MarketQuote, CalibrationEngine
    quotes = [
        MarketQuote(strike=90, maturity=1.0, implied_vol=0.25, forward=100),
        MarketQuote(strike=100, maturity=1.0, implied_vol=0.20, forward=100),
        MarketQuote(strike=110, maturity=1.0, implied_vol=0.18, forward=100),
    ]
    engine = CalibrationEngine()
    params, diag = engine.calibrate_svi(quotes)
    assert isinstance(params, SVIParameters)
    assert "rmse" in diag

def test_arbitrage_detector():
    from src.pricing.vol_surface import ArbitrageDetector
    detector = ArbitrageDetector()
    strikes = np.array([90, 100, 110])
    # Butterfly arbitrage: prices must be convex in K
    # 5, 10, 15 -> linear (free)
    # 5, 12, 15 -> not convex (arbitrage)
    is_free, _ = detector.check_butterfly_arbitrage(strikes, np.array([5.0, 10.0, 15.0]))
    assert is_free is True
    
    is_free_2, _ = detector.check_butterfly_arbitrage(strikes, np.array([5.0, 12.0, 15.0]))
    assert is_free_2 is False
