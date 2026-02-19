import numpy as np

from src.pricing.calibration.svi_surface import SVISurface


class TestSVISurface:
    def test_raw_svi_calculation(self):
        # a=0.04, b=0.1, rho=-0.5, m=0, sigma=0.1
        params = (0.04, 0.1, -0.5, 0.0, 0.1)
        k = 0.0  # ATM
        # w = 0.04 + 0.1 * (-0.5*0 + sqrt(0 + 0.01)) = 0.04 + 0.1 * 0.1 = 0.05
        assert abs(SVISurface.raw_svi(k, *params) - 0.05) < 1e-10

    def test_arbitrage_validation(self):
        # Valid params
        assert SVISurface.validate_no_arbitrage((0.04, 0.1, -0.5, 0.0, 0.1))
        # Invalid b < 0
        assert not SVISurface.validate_no_arbitrage((0.04, -0.1, -0.5, 0.0, 0.1))
        # Invalid rho
        assert not SVISurface.validate_no_arbitrage((0.04, 0.1, -1.5, 0.0, 0.1))

    def test_svi_fitting(self):
        # Generate synthetic slice
        log_strikes = np.linspace(-0.5, 0.5, 20)
        true_params = (0.04, 0.1, -0.5, 0.0, 0.1)
        total_variances = np.array(
            [SVISurface.raw_svi(k, *true_params) for k in log_strikes]
        )

        fitted_params = SVISurface.fit_svi_slice(log_strikes, total_variances, T=1.0)

        # Verify fitted params result in low error
        fitted_vars = np.array(
            [SVISurface.raw_svi(k, *fitted_params) for k in log_strikes]
        )
        rmse = np.sqrt(np.mean((total_variances - fitted_vars) ** 2))
        assert rmse < 1e-2
