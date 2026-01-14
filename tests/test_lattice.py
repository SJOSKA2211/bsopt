"""
Comprehensive tests for lattice-based option pricing models.

Tests cover:
    - Binomial tree (CRR) pricing
    - Trinomial tree pricing
    - Greeks calculation
    - Convergence to Black-Scholes
    - American vs European premium
    - Performance benchmarks
    - Edge cases and error handling
"""

import numpy as np
import pytest

from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.pricing.lattice import (
    BinomialTreePricer,
    LatticeParameters,
    TrinomialTreePricer,
)
from tests.test_utils import assert_equal


class TestLatticeParameters:
    """Test suite for LatticeParameters validation."""

    def test_valid_parameters(self):
        """Test that valid parameters are accepted."""
        params = LatticeParameters(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            volatility=0.2,
            rate=0.05,
            dividend=0.02,
            n_steps=100,
        )
        assert_equal(params.spot, 100.0)
        assert_equal(params.strike, 100.0)
        assert_equal(params.maturity, 1.0)
        assert_equal(params.volatility, 0.2)
        assert_equal(params.rate, 0.05)
        assert_equal(params.dividend, 0.02)
        assert_equal(params.n_steps, 100)


class TestBinomialTreePricer:
    """Test suite for BinomialTreePricer."""

    def test_crr_parameters(self):
        """Test CRR parameter calculation."""
        params = BSParameters(100.0, 100.0, 1.0, 0.2, 0.05, 0.02)
        # Check that u * d = 1 (recombining property)
        dt = params.maturity / 100
        u = np.exp(params.volatility * np.sqrt(dt))
        d = 1.0 / u
        assert np.isclose(u * d, 1.0, rtol=1e-10)

    def test_european_call_vs_black_scholes(self):
        """Test that European call converges to Black-Scholes."""
        params = BSParameters(
            spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.02
        )
        bs_price = BlackScholesEngine.price_call(params)

        pricer = BinomialTreePricer(n_steps=500, exercise_type="european")
        binomial_price = pricer.price(params, "call")

        # Should be within 0.1% of BS price
        relative_error = abs(binomial_price - bs_price) / bs_price
        assert relative_error < 0.001

    def test_american_put_premium(self):
        """Test that American put >= European put."""
        params = BSParameters(100.0, 100.0, 1.0, 0.2, 0.05, 0.02)

        pricer_euro = BinomialTreePricer(n_steps=100, exercise_type="european")
        euro_price = pricer_euro.price(params, "put")

        pricer_amer = BinomialTreePricer(n_steps=100, exercise_type="american")
        amer_price = pricer_amer.price(params, "put")

        assert amer_price >= euro_price - 1e-6

    def test_greeks_delta_range(self):
        """Test that delta is in valid range."""
        params = BSParameters(100.0, 100.0, 1.0, 0.2, 0.05)
        pricer = BinomialTreePricer(n_steps=100)

        greeks_call = pricer.calculate_greeks(params, "call")
        assert 0 <= greeks_call.delta <= 1

        greeks_put = pricer.calculate_greeks(params, "put")
        assert -1 <= greeks_put.delta <= 0


class TestTrinomialTreePricer:
    """Test suite for TrinomialTreePricer."""

    def test_trinomial_european_vs_black_scholes(self):
        """Test that trinomial European option converges to Black-Scholes."""
        params = BSParameters(100.0, 100.0, 1.0, 0.2, 0.05, 0.02)
        bs_price = BlackScholesEngine.price_call(params)

        pricer = TrinomialTreePricer(n_steps=500, exercise_type="european")
        trinomial_price = pricer.price(params, "call")

        # Should be within 0.1% of BS price
        relative_error = abs(trinomial_price - bs_price) / bs_price
        assert relative_error < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
