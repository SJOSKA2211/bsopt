import numpy as np
import pytest

from src.pricing.models.heston_fft import HestonModelFFT, HestonParams


class TestHestonParams:
    def test_valid_params(self):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
        assert params.v0 == 0.04
        assert params.kappa == 2.0

    def test_feller_condition_violation(self):
        # 2κθ > σ² => 2 * 0.1 * 0.04 = 0.008. σ² = 1.0. 0.008 <= 1.0 (Violated)
        with pytest.raises(ValueError, match="Feller condition violated"):
            HestonParams(v0=0.04, kappa=0.1, theta=0.04, sigma=1.0, rho=-0.5)

    def test_invalid_rho(self):
        with pytest.raises(ValueError, match=r"Correlation must be in \[-1,1\]"):
            HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=1.5)

    def test_non_positive_params(self):
        with pytest.raises(ValueError, match="All parameters except rho must be positive"):
            HestonParams(v0=-0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)

class TestHestonModelFFT:
    @pytest.fixture
    def standard_params(self):
        return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)

    def test_atm_call_price_reasonable(self, standard_params):
        model = HestonModelFFT(standard_params, r=0.03, T=1.0)
        S0, K = 100.0, 100.0
        price = model.price_call(S0, K)
        
        # Rough estimate: ATM call ~= 0.4 * S * sqrt(T) * sqrt(v0)
        # 0.4 * 100 * 1 * 0.2 = 8.0
        assert 4.0 < price < 16.0
        assert 0 < price < S0

    def test_put_call_parity(self, standard_params):
        model = HestonModelFFT(standard_params, r=0.05, T=0.5)
        S0, K = 100.0, 105.0
        call_price = model.price_call(S0, K)
        put_price = model.price_put(S0, K)
        
        lhs = call_price - put_price
        rhs = S0 - K * np.exp(-0.05 * 0.5)
        assert abs(lhs - rhs) < 0.01

    def test_otm_options_nonzero(self, standard_params):
        model = HestonModelFFT(standard_params, r=0.03, T=1.0)
        # Deep OTM call
        otm_call = model.price_call(100.0, 150.0)
        assert otm_call > 0.0001
        # Deep OTM put
        otm_put = model.price_put(100.0, 70.0)
        assert otm_put >= 1e-10

    def test_invalid_ttm(self, standard_params):
        with pytest.raises(ValueError, match="Time to maturity must be in"):
            HestonModelFFT(standard_params, r=0.03, T=-1.0)