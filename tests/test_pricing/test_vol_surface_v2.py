import numpy as np

from src.quant.pricing.vol_surface import (
    CalibrationEngine,
    MarketQuote,
    SABRModel,
    SABRParameters,
    SVIModel,
    SVIParameters,
    VolatilitySurface,
)


def test_svi_model():
    params = SVIParameters(a=0.1, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
    model = SVIModel(params)

    k = np.array([-0.1, 0.0, 0.1])
    var = model.total_variance(k)
    assert len(var) == 3
    assert np.all(var > 0)


def test_sabr_model():
    # SABRModel expects SABRParameters
    params = SABRParameters(alpha=0.2, beta=0.5, rho=-0.2, nu=0.3)
    model = SABRModel(params)
    # Check ATM vol
    vol = model.implied_volatility(100.0, 100.0, 1.0)
    assert vol > 0


def test_vol_surface_interpolation():
    surface = VolatilitySurface()
    model1 = SVIModel(SVIParameters(0.1, 0.1, -0.5, 0.0, 0.1))
    model2 = SVIModel(SVIParameters(0.2, 0.1, -0.5, 0.0, 0.1))

    surface.add_slice(0.5, model1, 100.0)
    surface.add_slice(1.0, model2, 100.0)

    # Method is implied_volatility
    vol = surface.implied_volatility(100.0, 0.75)
    assert vol > 0


def test_calibration_engine_svi():
    engine = CalibrationEngine()
    quotes = [
        MarketQuote(strike=90, maturity=1.0, implied_vol=0.25, forward=100),
        MarketQuote(strike=100, maturity=1.0, implied_vol=0.20, forward=100),
        MarketQuote(strike=110, maturity=1.0, implied_vol=0.22, forward=100),
    ]
    params, diag = engine.calibrate_svi(quotes)
    assert isinstance(params, SVIParameters)
    assert diag["rmse"] >= 0
