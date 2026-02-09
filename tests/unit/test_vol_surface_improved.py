import unittest

import numpy as np

from src.pricing.vol_surface import (
    CalibrationEngine,
    MarketQuote,
    SABRModel,
    SABRParameters,
    SVIModel,
    SVIParameters,
    VolatilitySurface,
)


class TestVolSurface(unittest.TestCase):
    def setUp(self):
        self.svi_params = SVIParameters(a=0.04, b=0.1, rho=-0.4, m=0.0, sigma=0.2)
        self.sabr_params = SABRParameters(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)

    def test_svi_model(self):
        model = SVIModel(self.svi_params)
        vol = model.implied_volatility(100.0, 100.0, 1.0)
        self.assertGreater(vol, 0)
        
        # Test vectorized
        strikes = np.array([90.0, 100.0, 110.0])
        vols = model.implied_volatility(strikes, 100.0, 1.0)
        self.assertEqual(len(vols), 3)

    def test_sabr_model(self):
        model = SABRModel(self.sabr_params)
        vol = model.implied_volatility(100.0, 100.0, 1.0)
        self.assertGreater(vol, 0)

    def test_vol_surface(self):
        surface = VolatilitySurface()
        model1 = SVIModel(self.svi_params)
        model2 = SVIModel(self.svi_params) # Same for simplicity
        surface.add_slice(0.5, model1, 100.0)
        surface.add_slice(1.0, model2, 100.0)
        
        # Test interpolation
        vol = surface.implied_volatility(100.0, 0.75)
        self.assertGreater(vol, 0)

    def test_calibration_engine(self):
        engine = CalibrationEngine()
        quotes = [
            MarketQuote(strike=90, maturity=1.0, implied_vol=0.25, forward=100),
            MarketQuote(strike=100, maturity=1.0, implied_vol=0.20, forward=100),
            MarketQuote(strike=110, maturity=1.0, implied_vol=0.18, forward=100),
        ]
        params, diag = engine.calibrate_svi(quotes)
        self.assertIsInstance(params, SVIParameters)
        self.assertIn("rmse", diag)

if __name__ == '__main__':
    unittest.main()
