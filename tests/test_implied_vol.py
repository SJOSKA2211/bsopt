"""
Comprehensive Test Suite for Implied Volatility Calculator

This module provides extensive testing for edge cases, accuracy, performance,
and integration with the Black-Scholes pricing engine.

Test Categories:
    1. Round-trip tests (recover known volatility)
    2. Edge cases (deep ITM/OTM, extreme maturities, extreme volatilities)
    3. Method comparison (Newton vs Brent)
    4. Performance validation (convergence speed)
    5. Accuracy validation (precision requirements)
    6. Error handling (invalid inputs, arbitrage violations)
    7. Integration tests (with BSParameters and pricing engines)
"""

import numpy as np
import pytest

from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.pricing.implied_vol import (
    ImpliedVolatilityError,
    _calculate_intrinsic_value,
    implied_volatility,
)
from tests.test_utils import assert_equal


class TestRoundTrip:
    """Test that we can recover the original volatility from calculated prices."""

    @pytest.mark.parametrize("vol_true", [0.05, 0.15, 0.25, 0.40, 0.80])
    def test_round_trip_atm_call(self, vol_true):
        """Test round-trip for ATM calls with various volatilities."""
        params = BSParameters(100, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.02, "call")

        assert_equal(iv, vol_true, tolerance=1e-6, message=f"Failed to recover vol={vol_true}")

    @pytest.mark.parametrize("vol_true", [0.05, 0.15, 0.25, 0.40, 0.80])
    def test_round_trip_atm_put(self, vol_true):
        """Test round-trip for ATM puts with various volatilities."""
        params = BSParameters(100, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_put(params)

        iv = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.02, "put")

        assert_equal(iv, vol_true, tolerance=1e-6, message=f"Failed to recover vol={vol_true}")

    @pytest.mark.parametrize("moneyness", [0.8, 0.9, 1.0, 1.1, 1.2])
    def test_round_trip_various_strikes(self, moneyness):
        """Test round-trip across different moneyness levels."""
        spot = 100.0
        strike = spot / moneyness
        vol_true = 0.25

        params = BSParameters(spot, strike, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, spot, strike, 1.0, 0.05, 0.02, "call")

        assert_equal(iv, vol_true, tolerance=1e-5, message=f"Failed for moneyness={moneyness}")


class TestEdgeCases:
    """Test edge cases: deep ITM/OTM, extreme maturities, extreme volatilities."""

    def test_deep_itm_call(self):
        """Deep ITM call: S=200, K=100 (100% ITM)."""
        vol_true = 0.20
        params = BSParameters(200, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 200, 100, 1.0, 0.05, 0.02, "call")

        # Deep ITM options have low volatility sensitivity
        assert_equal(iv, vol_true, tolerance=1e-4)
        assert 0.001 < iv < 5.0

    def test_deep_otm_call(self):
        """Deep OTM call: S=50, K=100 (50% OTM)."""
        vol_true = 0.30
        params = BSParameters(50, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        # Only test if price is above minimum threshold
        if market_price > 1e-6:
            iv = implied_volatility(market_price, 50, 100, 1.0, 0.05, 0.02, "call")
            assert_equal(iv, vol_true, tolerance=1e-4)
            assert 0.001 < iv < 5.0

    def test_deep_itm_put(self):
        """Deep ITM put: S=50, K=100 (50% ITM for put)."""
        vol_true = 0.20
        params = BSParameters(50, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_put(params)

        iv = implied_volatility(market_price, 50, 100, 1.0, 0.05, 0.02, "put")

        assert_equal(iv, vol_true, tolerance=1e-4)
        assert 0.001 < iv < 5.0

    def test_deep_otm_put(self):
        """Deep OTM put: S=200, K=100 (put is far OTM)."""
        vol_true = 0.30
        params = BSParameters(200, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_put(params)

        if market_price > 1e-6:
            iv = implied_volatility(market_price, 200, 100, 1.0, 0.05, 0.02, "put")
            assert_equal(iv, vol_true, tolerance=1e-4)
            assert 0.001 < iv < 5.0

    def test_very_short_maturity(self):
        """Very short maturity: 1 hour = 1/8760 years."""
        maturity = 1.0 / 8760.0  # 1 hour
        vol_true = 0.50
        params = BSParameters(100, 100, maturity, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        if market_price > 1e-8:
            iv = implied_volatility(market_price, 100, 100, maturity, 0.05, 0.02, "call")
            # Relaxed tolerance for very short maturities
            assert_equal(iv, vol_true, tolerance=1e-3)

    def test_short_maturity_1_day(self):
        """Short maturity: 1 day = 1/365 years."""
        maturity = 1.0 / 365.0
        vol_true = 0.40
        params = BSParameters(100, 100, maturity, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, maturity, 0.05, 0.02, "call")
        assert_equal(iv, vol_true, tolerance=1e-5)

    def test_short_maturity_1_week(self):
        """Short maturity: 1 week = 1/52 years."""
        maturity = 1.0 / 52.0
        vol_true = 0.35
        params = BSParameters(100, 100, maturity, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, maturity, 0.05, 0.02, "call")
        assert_equal(iv, vol_true, tolerance=1e-6)

    def test_long_maturity_5_years(self):
        """Long maturity: 5 years."""
        maturity = 5.0
        vol_true = 0.25
        params = BSParameters(100, 100, maturity, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, maturity, 0.05, 0.02, "call")
        assert_equal(iv, vol_true, tolerance=1e-6)

    def test_long_maturity_10_years(self):
        """Long maturity: 10 years (LEAPS)."""
        maturity = 10.0
        vol_true = 0.30
        params = BSParameters(100, 100, maturity, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, maturity, 0.05, 0.02, "call")
        assert_equal(iv, vol_true, tolerance=1e-6)

    def test_low_volatility(self):
        """Low volatility: 5% annualized."""
        vol_true = 0.05
        params = BSParameters(100, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.02, "call")
        assert_equal(iv, vol_true, tolerance=1e-6)

    def test_high_volatility(self):
        """High volatility: 100% annualized."""
        vol_true = 1.00
        params = BSParameters(100, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.02, "call")
        assert_equal(iv, vol_true, tolerance=1e-5)

    def test_very_high_volatility(self):
        """Very high volatility: 200% annualized."""
        vol_true = 2.00
        params = BSParameters(100, 100, 1.0, vol_true, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.02, "call")
        assert_equal(iv, vol_true, tolerance=1e-4)

    def test_zero_dividend(self):
        """Option with zero dividend yield."""
        vol_true = 0.25
        params = BSParameters(100, 100, 1.0, vol_true, 0.05, 0.0)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.0, "call")
        assert_equal(iv, vol_true, tolerance=1e-6)

    def test_high_dividend(self):
        """Option with high dividend yield (10%)."""
        vol_true = 0.25
        params = BSParameters(100, 100, 1.0, vol_true, 0.05, 0.10)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.10, "call")
        assert_equal(iv, vol_true, tolerance=1e-6)


class TestMethodComparison:
    """Compare Newton-Raphson and Brent's methods."""

    def test_newton_vs_brent_atm(self):
        """Newton and Brent should agree for ATM options."""
        params = BSParameters(100, 100, 1.0, 0.25, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv_newton = implied_volatility(
            market_price, 100, 100, 1.0, 0.05, 0.02, "call", method="newton"
        )
        iv_brent = implied_volatility(
            market_price, 100, 100, 1.0, 0.05, 0.02, "call", method="brent"
        )

        assert_equal(iv_newton, iv_brent, tolerance=1e-6)

    def test_newton_vs_brent_itm(self):
        """Newton and Brent should agree for ITM options."""
        params = BSParameters(120, 100, 1.0, 0.30, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv_newton = implied_volatility(
            market_price, 120, 100, 1.0, 0.05, 0.02, "call", method="newton"
        )
        iv_brent = implied_volatility(
            market_price, 120, 100, 1.0, 0.05, 0.02, "call", method="brent"
        )

        assert_equal(iv_newton, iv_brent, tolerance=1e-5)

    def test_newton_vs_brent_otm(self):
        """Newton and Brent should agree for OTM options."""
        params = BSParameters(90, 100, 1.0, 0.30, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv_newton = implied_volatility(
            market_price, 90, 100, 1.0, 0.05, 0.02, "call", method="newton"
        )
        iv_brent = implied_volatility(
            market_price, 90, 100, 1.0, 0.05, 0.02, "call", method="brent"
        )

        assert_equal(iv_newton, iv_brent, tolerance=1e-5)

    def test_auto_method_uses_newton_for_good_cases(self):
        """Auto method should use Newton for well-behaved cases."""
        params = BSParameters(100, 100, 1.0, 0.25, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        # Auto should give same result as Newton for good cases
        iv_auto = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.02, "call", method="auto")
        iv_newton = implied_volatility(
            market_price, 100, 100, 1.0, 0.05, 0.02, "call", method="newton"
        )

        assert_equal(iv_auto, iv_newton, tolerance=1e-10)


class TestPerformance:
    """Test convergence speed and efficiency."""

    def test_newton_converges_quickly_atm(self):
        """Newton-Raphson should converge in < 6 iterations for ATM."""
        params = BSParameters(100, 100, 1.0, 0.25, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        # Track iterations by modifying max_iterations
        iv = implied_volatility(
            market_price, 100, 100, 1.0, 0.05, 0.02, "call", method="newton", max_iterations=6
        )

        # Should succeed within 6 iterations
        assert_equal(iv, 0.25, tolerance=1e-6)

    def test_newton_converges_quickly_near_atm(self):
        """Newton should converge quickly for near-ATM options."""
        for moneyness in [0.95, 1.0, 1.05]:
            strike = 100 / moneyness
            params = BSParameters(100, strike, 1.0, 0.25, 0.05, 0.02)
            market_price = BlackScholesEngine.price_call(params)

            iv = implied_volatility(
                market_price,
                100,
                strike,
                1.0,
                0.05,
                0.02,
                "call",
                method="newton",
                max_iterations=6,
            )

            assert_equal(iv, 0.25, tolerance=1e-6)


class TestAccuracy:
    """Test numerical accuracy requirements."""

    def test_accuracy_within_0_0001(self):
        """Recovered volatility should be within ±0.0001."""
        vol_true = 0.2567  # Non-round number
        params = BSParameters(103.47, 98.23, 0.752, vol_true, 0.0423, 0.0187)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 103.47, 98.23, 0.752, 0.0423, 0.0187, "call")

        assert_equal(iv, vol_true, tolerance=0.0001)

    def test_price_accuracy_after_recovery(self):
        """Repricing with recovered IV should match market price."""
        params = BSParameters(100, 100, 1.0, 0.25, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        iv = implied_volatility(market_price, 100, 100, 1.0, 0.05, 0.02, "call")

        # Reprice with recovered volatility
        params_recovered = BSParameters(100, 100, 1.0, iv, 0.05, 0.02)
        repriced = BlackScholesEngine.price_call(params_recovered)

        assert_equal(repriced, market_price, tolerance=1e-8)


class TestErrorHandling:
    """Test error handling and validation."""

    def test_negative_market_price(self):
        """Negative market price should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            implied_volatility(-1.0, 100, 100, 1.0, 0.05, 0.02, "call")

    def test_zero_spot_price(self):
        """Zero spot price should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            implied_volatility(10.0, 0, 100, 1.0, 0.05, 0.02, "call")

    def test_negative_spot_price(self):
        """Negative spot price should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            implied_volatility(10.0, -100, 100, 1.0, 0.05, 0.02, "call")

    def test_zero_strike_price(self):
        """Zero strike price should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            implied_volatility(10.0, 100, 0, 1.0, 0.05, 0.02, "call")

    def test_negative_maturity(self):
        """Negative maturity should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            implied_volatility(10.0, 100, 100, -1.0, 0.05, 0.02, "call")

    def test_zero_maturity(self):
        """Zero maturity should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            implied_volatility(10.0, 100, 100, 0, 0.05, 0.02, "call")

    def test_invalid_option_type(self):
        """Invalid option type should raise ValueError."""
        with pytest.raises(ValueError, match="must be 'call' or 'put'"):
            implied_volatility(10.0, 100, 100, 1.0, 0.05, 0.02, "invalid")

    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            implied_volatility(10.0, 100, 100, 1.0, 0.05, 0.02, "call", method="invalid")

    def test_negative_initial_guess(self):
        """Negative initial guess should raise ValueError."""
        with pytest.raises(ValueError, match="initial_guess must be positive"):
            implied_volatility(10.0, 100, 100, 1.0, 0.05, 0.02, "call", initial_guess=-0.5)

    def test_zero_initial_guess(self):
        """Zero initial guess should raise ValueError."""
        with pytest.raises(ValueError, match="initial_guess must be positive"):
            implied_volatility(10.0, 100, 100, 1.0, 0.05, 0.02, "call", initial_guess=0)

    def test_negative_tolerance(self):
        """Negative tolerance should raise ValueError."""
        with pytest.raises(ValueError, match="tolerance must be positive"):
            implied_volatility(10.0, 100, 100, 1.0, 0.05, 0.02, "call", tolerance=-1e-6)

    def test_zero_max_iterations(self):
        """Zero max_iterations should raise ValueError."""
        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            implied_volatility(10.0, 100, 100, 1.0, 0.05, 0.02, "call", max_iterations=0)

    def test_price_below_intrinsic_call(self):
        """Price below intrinsic value should raise arbitrage error."""
        # Intrinsic value for ITM call: S - K (simplified)
        # For S=150, K=100, intrinsic ≈ 50 (adjusted for PV)
        with pytest.raises(ValueError, match="Arbitrage violation"):
            implied_volatility(40.0, 150, 100, 1.0, 0.05, 0.02, "call")

    def test_price_below_intrinsic_put(self):
        """Price below intrinsic value should raise arbitrage error."""
        # Intrinsic value for ITM put: K - S (simplified)
        # For S=50, K=100, intrinsic ≈ 50 (adjusted for PV)
        with pytest.raises(ValueError, match="Arbitrage violation"):
            implied_volatility(40.0, 50, 100, 1.0, 0.05, 0.02, "put")

    def test_market_price_too_small(self):
        """Market price too close to zero should raise error."""
        # Use deep OTM option where intrinsic is zero
        with pytest.raises(ImpliedVolatilityError, match="too close to zero"):
            implied_volatility(1e-15, 50, 100, 1.0, 0.05, 0.02, "call")

    def test_max_iterations_exceeded(self):
        """Exceeding max iterations should raise error."""
        params = BSParameters(100, 100, 1.0, 0.25, 0.05, 0.02)
        market_price = BlackScholesEngine.price_call(params)

        with pytest.raises(ImpliedVolatilityError, match="failed to converge"):
            implied_volatility(
                market_price,
                100,
                100,
                1.0,
                0.05,
                0.02,
                "call",
                method="newton",
                max_iterations=1,
                initial_guess=5.0,
            )


class TestIntrinsicValue:
    """Test intrinsic value calculation."""

    def test_itm_call_intrinsic(self):
        """ITM call should have positive intrinsic value."""
        intrinsic = _calculate_intrinsic_value(110, 100, 0.05, 0.02, 1.0, "call")
        assert intrinsic > 0
        # Should be close to (S*e^(-qT) - K*e^(-rT))
        expected = 110 * np.exp(-0.02) - 100 * np.exp(-0.05)
        assert_equal(intrinsic, expected, tolerance=1e-10)

    def test_otm_call_intrinsic(self):
        """OTM call should have zero intrinsic value."""
        intrinsic = _calculate_intrinsic_value(90, 100, 0.05, 0.02, 1.0, "call")
        assert_equal(intrinsic, 0)

    def test_itm_put_intrinsic(self):
        """ITM put should have positive intrinsic value."""
        intrinsic = _calculate_intrinsic_value(90, 100, 0.05, 0.02, 1.0, "put")
        assert intrinsic > 0
        expected = 100 * np.exp(-0.05) - 90 * np.exp(-0.02)
        assert_equal(intrinsic, expected, tolerance=1e-10)

    def test_otm_put_intrinsic(self):
        """OTM put should have zero intrinsic value."""
        intrinsic = _calculate_intrinsic_value(110, 100, 0.05, 0.02, 1.0, "put")
        assert_equal(intrinsic, 0)


class TestIntegration:
    """Integration tests with Black-Scholes engine."""

    def test_integration_with_bs_parameters(self):
        """Test integration with BSParameters dataclass."""
        params = BSParameters(
            spot=105.50, strike=100.00, maturity=0.5, volatility=0.28, rate=0.045, dividend=0.015
        )

        call_price = BlackScholesEngine.price_call(params)
        put_price = BlackScholesEngine.price_put(params)

        iv_call = implied_volatility(
            call_price,
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            params.dividend,
            "call",
        )
        iv_put = implied_volatility(
            put_price,
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            params.dividend,
            "put",
        )

        # Both should recover the same volatility
        assert_equal(iv_call, params.volatility, tolerance=1e-6)
        assert_equal(iv_put, params.volatility, tolerance=1e-6)

    def test_integration_with_greeks(self):
        """Test that vega is consistent with IV calculation."""
        params = BSParameters(100, 100, 1.0, 0.25, 0.05, 0.02)
        greeks = BlackScholesEngine().calculate_greeks(params, "call")

        # Vega is sensitivity to volatility
        # If we perturb volatility slightly, price should change by ~vega
        vol_perturb = 0.01  # 1% volatility change
        params_perturbed = BSParameters(100, 100, 1.0, 0.26, 0.05, 0.02)

        price_original = BlackScholesEngine.price_call(params)
        price_perturbed = BlackScholesEngine.price_call(params_perturbed)

        price_change = price_perturbed - price_original
        expected_change = greeks.vega * vol_perturb

        # Should be close (linear approximation)
        assert_equal(price_change, expected_change, tolerance=0.01)

    def test_put_call_parity_preserved(self):
        """Test that put-call parity is preserved through IV calculation."""
        params = BSParameters(100, 95, 1.0, 0.30, 0.05, 0.02)

        call_price = BlackScholesEngine.price_call(params)
        put_price = BlackScholesEngine.price_put(params)

        iv_call = implied_volatility(call_price, 100, 95, 1.0, 0.05, 0.02, "call")
        iv_put = implied_volatility(put_price, 100, 95, 1.0, 0.05, 0.02, "put")

        # IVs should be the same (put-call parity)
        assert_equal(iv_call, iv_put, tolerance=1e-6)


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
