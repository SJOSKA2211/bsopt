import numpy as np
import pytest

from src.pricing.calibration.engine import HestonCalibrator, MarketOption
from src.pricing.models.heston_fft import HestonParams


class TestHestonCalibrator:
    def test_calibration_accuracy_synthetic(self):
        """
        Verify that calibrator can recover known parameters from synthetic data.
        """
        true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
        calibrator = HestonCalibrator(risk_free_rate=0.03)

        # Generate synthetic data
        strikes = np.linspace(80, 120, 10)
        maturities = [0.25, 0.5, 1.0]
        market_data = []

        from src.pricing.models.heston_fft import HestonModelFFT

        for T in maturities:
            model = HestonModelFFT(true_params, r=0.03, T=T)
            for K in strikes:
                price = model.price_call(100.0, K)
                market_data.append(
                    MarketOption(
                        T=T,
                        strike=K,
                        spot=100.0,
                        price=price,
                        bid=price - 0.01,
                        ask=price + 0.01,
                        volume=100,
                        open_interest=500,
                        option_type="call",
                    )
                )

        # Calibrate
        # Using popsize=5 and maxiter=10 for fast test, normally much higher
        calibrated_params, metrics = calibrator.calibrate(
            market_data, maxiter=10, popsize=5
        )

        assert metrics["success"]
        assert metrics["rmse"] < 0.1  # Loose threshold for fast/coarse optimization
        assert metrics["r_squared"] > 0.9

    def test_insufficient_data(self):
        calibrator = HestonCalibrator()
        with pytest.raises(ValueError, match="Insufficient liquid options"):
            calibrator.calibrate([])
