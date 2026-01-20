import pytest
import numpy as np
from src.pricing.models.heston_fft import HestonModelFFT, HestonParams

class TestHestonFFT:
    """Test suite for Heston pricing engine."""

    @pytest.fixture
    def standard_params(self):
        """Standard Heston parameters from literature."""
        return HestonParams(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma=0.3,
            rho=-0.7
        )

    def test_feller_condition_validation(self):
        """Verify Feller condition is enforced."""
        with pytest.raises(ValueError, match="Feller condition"):
            HestonParams(v0=0.04, kappa=0.1, theta=0.04, sigma=1.0, rho=-0.5)

    def test_atm_call_price_reasonable(self, standard_params):
        """ATM call should be roughly reasonable."""
        model = HestonModelFFT(standard_params, r=0.03, T=1.0)
        S0, K = 100.0, 100.0
        price = model.price_call(S0, K)
        rough_estimate = 0.4 * S0 * np.sqrt(1.0) * np.sqrt(standard_params.v0)
        assert 0.5 * rough_estimate < price < 2.0 * rough_estimate
        assert 0 < price < S0

    def test_put_call_parity(self, standard_params):
        """Verify C - P = S - K*exp(-rT)."""
        model = HestonModelFFT(standard_params, r=0.05, T=0.5)
        S0, K = 100.0, 105.0
        call_price = model.price_call(S0, K)
        put_price = model.price_put(S0, K)
        lhs = call_price - put_price
        rhs = S0 - K * np.exp(-0.05 * 0.5)
        assert abs(lhs - rhs) < 0.01

    def test_otm_options_nonzero(self, standard_params):
        """OTM options should have nonzero value (capture tail risk)."""
        model = HestonModelFFT(standard_params, r=0.03, T=1.0)
        # Deep OTM call
        otm_call = model.price_call(100.0, 150.0)
        assert otm_call > 0.001
        # Deep OTM put
        otm_put = model.price_put(100.0, 70.0)
        assert otm_put > 0.001
