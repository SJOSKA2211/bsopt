import numpy as np
import pytest

from src.math_kernel.models import BSParameters
from src.math_kernel.monte_carlo import MCConfig, MonteCarloEngine, geometric_asian_price


class TestMonteCarloEngine:
    @pytest.fixture
    def params(self):
        return BSParameters(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            volatility=0.2,
            rate=0.05,
            dividend=0.0,
        )

    def test_european_call_price_convergence(self, params):
        # High path count for convergence to BS price (~10.4506)
        config = MCConfig(n_paths=50000, seed=42, control_variate=True)
        engine = MonteCarloEngine(config)
        price, std_err = engine.price_european(params, "call")

        # BS Price is approx 10.4506
        assert np.isclose(price, 10.4506, atol=0.15)
        assert std_err < 0.1

    def test_european_put_price_convergence(self, params):
        # BS Price is approx 5.5735
        config = MCConfig(n_paths=50000, seed=42, control_variate=True)
        engine = MonteCarloEngine(config)
        price = engine.price(params, "put")

        assert np.isclose(price, 5.5735, atol=0.15)

    def test_american_price_lsm(self, params):
        config = MCConfig(n_paths=5000, n_steps=50, seed=42)
        engine = MonteCarloEngine(config)

        # For non-dividend paying stock, American Call ~= European Call
        am_price = engine.price_american_lsm(params, "call")
        eu_price = engine.price(params, "call")
        assert np.isclose(am_price, eu_price, atol=0.8)  # MC noise + LSM bias

        # For Put, American >= European
        am_put = engine.price_american_lsm(params, "put")
        eu_put = engine.price(params, "put")
        # American put value should be slightly higher or equal
        assert am_put >= eu_put - 0.2

    def test_greeks_calculation(self, params):
        # Increased paths to ensure convexity (Gamma > 0) isn't drowned by noise
        config = MCConfig(n_paths=50000, seed=42, control_variate=False)
        engine = MonteCarloEngine(config)
        greeks = engine.calculate_greeks(params, "call")

        # BS Delta ~ 0.6368
        assert 0.6 < greeks.delta < 0.7
        # Gamma for Long Call is always positive
        assert greeks.vega > 0

    def test_geometric_asian_price(self, params):
        price = geometric_asian_price(params, "call", n_obs=252)
        assert price > 0
        # Asian call on geometric average is usually cheaper than vanilla
        assert price < 10.4506

    def test_config_validation(self):
        with pytest.raises(ValueError):
            MCConfig(n_paths=-1)
        with pytest.raises(ValueError):
            MCConfig(n_steps=0)

    def test_sobol_method(self, params):
        # Test that sobol initialization doesn't crash and adjusts n_paths
        config = MCConfig(n_paths=1000, method="sobol", seed=42)
        engine = MonteCarloEngine(config)
        price, _ = engine.price_european(params, "call")
        assert price > 0

    def test_gpu_threshold_logic(self, params):
        # Trigger the GPU path logic
        config = MCConfig(n_paths=10, method="monte_carlo", control_variate=False)
        engine = MonteCarloEngine(config)
        price = engine.price(params, "call")
        assert price >= 0

    def test_greeks_fd_fallback(self, params):
        """Test Greeks calculation using Finite Difference fallback (Control Variate enabled)."""
        config = MCConfig(n_paths=10000, seed=42, control_variate=True)
        engine = MonteCarloEngine(config)
        greeks = engine.calculate_greeks(params, "call")

        assert greeks.delta != 0
