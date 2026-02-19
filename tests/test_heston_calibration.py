from src.pricing.calibration.engine import HestonCalibrator, MarketOption
from src.pricing.models.heston_fft import HestonModelFFT, HestonParams


class TestHestonCalibration:
    """Test suite for Heston calibration engine."""

    def test_calibration_simulated_data(self):
        """Calibrate against data generated from known Heston parameters."""
        true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
        r = 0.03
        T = 0.5
        spot = 100.0

        # Generate mock market data
        strikes = [90, 95, 100, 105, 110]
        market_data = []
        model = HestonModelFFT(true_params, r, T)

        for k in strikes:
            price = model.price_call(spot, k)
            market_data.append(
                MarketOption(
                    T=T,
                    strike=float(k),
                    spot=spot,
                    price=price,
                    bid=price - 0.01,
                    ask=price + 0.01,
                    volume=100,
                    open_interest=1000,
                    option_type="call",
                )
            )

        calibrator = HestonCalibrator(risk_free_rate=r)
        # Use low iteration count for fast test
        calibrated_params, metrics = calibrator.calibrate(
            market_data, maxiter=5, popsize=5
        )

        assert metrics["success"]
        assert metrics["rmse"] < 0.1
        assert calibrated_params.v0 > 0
        assert calibrated_params.kappa > 0

    def test_calibrate_surface(self):
        """Test SVI surface fitting integration."""
        r = 0.03
        spot = 100.0
        T = 0.5

        # Mock data with a smile
        strikes = [80, 90, 100, 110, 120]
        # IVs forming a smile
        ivs = [0.35, 0.25, 0.2, 0.22, 0.3]

        market_data = []
        from src.pricing.black_scholes import BlackScholesEngine, BSParameters

        engine = BlackScholesEngine()
        for k, v in zip(strikes, ivs):
            params = BSParameters(
                spot=spot,
                strike=float(k),
                maturity=T,
                volatility=v,
                rate=r,
                dividend=0.0,
            )
            price = engine.price(params, option_type="call")
            market_data.append(
                MarketOption(
                    T=T,
                    strike=float(k),
                    spot=spot,
                    price=price,
                    bid=price - 0.01,
                    ask=price + 0.01,
                    volume=100,
                    open_interest=1000,
                    option_type="call",
                )
            )

        calibrator = HestonCalibrator(risk_free_rate=r)
        surface_params = calibrator.calibrate_surface(market_data)

        assert T in surface_params
        assert len(surface_params[T]) == 5  # a, b, rho, m, sigma
