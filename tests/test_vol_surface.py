"""
Comprehensive Test Suite for Volatility Surface Calibration

Tests cover:
1. SVI Model: parameter validation, variance calculation, arbitrage conditions
2. SABR Model: Hagan formula, ATM/OTM cases, numerical stability
3. Calibration Engine: convergence, quality metrics, weighted calibration
4. Arbitrage Detection: butterfly, calendar, Durrleman conditions
5. Volatility Surface: interpolation, extrapolation, multi-slice management

Numerical Validation:
- Known analytical results for special cases
- Benchmarks against reference implementations
"""

import time
from decimal import Decimal

import numpy as np
import pytest

from src.pricing.vol_surface import (
    ArbitrageDetector,
    CalibrationConfig,
    CalibrationEngine,
    InterpolationMethod,
    MarketQuote,
    OptimizationMethod,
    SABRModel,
    SABRParameters,
    SVIModel,
    SVINaturalParameters,
    SVIParameters,
    VolatilitySurface,
)
from tests.test_utils import assert_equal

# ============================================================================
# Test Fixtures and Utilities
# ============================================================================


@pytest.fixture
def typical_svi_params():
    """Realistic SVI parameters for equity index options"""
    return SVIParameters(a=0.04, b=0.1, rho=-0.4, m=0.0, sigma=0.2)


@pytest.fixture
def typical_sabr_params():
    """Realistic SABR parameters for FX options"""
    return SABRParameters(alpha=0.25, beta=0.5, rho=-0.3, nu=0.4)


@pytest.fixture
def sample_market_quotes():
    """Generate sample market quotes for testing"""
    forward = Decimal("100.0")
    maturity = 0.25  # 3 months

    strikes = [Decimal(str(k)) for k in [80, 85, 90, 95, 100, 105, 110, 115, 120]]

    # Realistic smile: higher vol for OTM puts
    implied_vols = [0.30, 0.27, 0.24, 0.22, 0.20, 0.21, 0.22, 0.24, 0.26]

    quotes = []
    for K, vol in zip(strikes, implied_vols):
        quotes.append(
            MarketQuote(
                strike=K,
                maturity=maturity,
                implied_vol=vol,
                forward=forward,
                option_type="call",
                vega=Decimal("10.0"),  # Simplified vega
            )
        )

    return quotes


def assert_close(
    actual: float, expected: float, rtol: float = 1e-5, atol: float = 1e-8
):
    """Assert two floats are close with relative and absolute tolerance"""
    assert np.isclose(
        actual, expected, rtol=rtol, atol=atol
    ), f"Expected {expected}, got {actual} (diff: {abs(actual - expected)})"


# ============================================================================
# SVI Model Tests
# ============================================================================


class TestSVIParameters:
    """Test SVI parameter validation and conversion"""

    def test_valid_parameters(self, typical_svi_params):
        """Valid SVI parameters should be accepted"""
        params = typical_svi_params
        assert_equal(params.a, 0.04)
        assert_equal(params.b, 0.1)
        assert_equal(params.rho, -0.4)
        assert_equal(params.m, 0.0)
        assert_equal(params.sigma, 0.2)

    def test_invalid_b_negative(self):
        """Negative b parameter should raise error"""
        with pytest.raises(ValueError, match="b must be non-negative"):
            SVIParameters(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.2)

    def test_invalid_rho_out_of_bounds(self):
        """ρ outside (-1, 1) should raise error"""
        with pytest.raises(ValueError, match="rho must be in"):
            SVIParameters(a=0.04, b=0.1, rho=1.0, m=0.0, sigma=0.2)

        with pytest.raises(ValueError, match="rho must be in"):
            SVIParameters(a=0.04, b=0.1, rho=-1.0, m=0.0, sigma=0.2)

    def test_invalid_sigma_non_positive(self):
        """Non-positive σ should raise error"""
        with pytest.raises(ValueError, match="sigma must be positive"):
            SVIParameters(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            SVIParameters(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=-0.1)

    def test_non_negative_variance_warning(self):
        """Parameters violating non-negative variance should warn"""
        # a + b*σ*sqrt(1-ρ²) < 0 should warn
        with pytest.warns(UserWarning, match="non-negative variance"):
            SVIParameters(a=-1.0, b=0.1, rho=0.0, m=0.0, sigma=0.2)

    def test_natural_to_raw_conversion(self):
        """Test natural SVI parameterization conversion"""
        natural = SVINaturalParameters(
            delta=0.04, mu=0.0, rho=-0.4, omega=0.05, zeta=0.3
        )

        raw = natural.to_raw()

        # Verify conversion formulas
        denominator = np.sqrt(1 + natural.zeta**2 - 2 * natural.rho * natural.zeta)

        assert_close(raw.a, natural.delta)
        assert_close(raw.b, natural.omega / denominator)
        assert_close(raw.m, natural.mu)
        assert_close(raw.rho, natural.rho)
        assert_close(raw.sigma, natural.zeta * denominator)


class TestSVIModel:
    """Test SVI model calculations"""

    def test_total_variance_scalar(self, typical_svi_params):
        """Test total variance for scalar log-moneyness"""
        model = SVIModel(typical_svi_params)
        k = 0.0  # ATM

        w = model.total_variance(k)

        # Manual calculation
        p = typical_svi_params
        k_shifted = k - p.m
        disc = np.sqrt(k_shifted**2 + p.sigma**2)
        expected_w = p.a + p.b * (p.rho * k_shifted + disc)

        assert_close(w, expected_w)

    def test_total_variance_array(self, typical_svi_params):
        """Test total variance for array of log-moneyness"""
        model = SVIModel(typical_svi_params)
        k = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])

        w = model.total_variance(k)

        assert isinstance(w, np.ndarray)
        assert_equal(w.shape, k.shape)
        assert np.all(w > 0)  # Valid parameters should give positive variance

    def test_implied_volatility_atm(self, typical_svi_params):
        """Test implied volatility at-the-money"""
        model = SVIModel(typical_svi_params)

        strike = Decimal("100.0")
        forward = Decimal("100.0")
        maturity = 1.0

        vol = model.implied_volatility(strike, forward, maturity)

        # ATM: k = 0
        w = model.total_variance(0.0)
        expected_vol = np.sqrt(w / maturity)

        assert_close(vol, expected_vol)

    def test_implied_volatility_smile_shape(self):
        """Test that SVI produces realistic smile shape"""
        svi_params = SVIParameters(a=0.04, b=0.4, rho=-0.7, m=0.0, sigma=0.2)
        model = SVIModel(svi_params)

        forward = Decimal("100.0")
        maturity = 0.25

        # OTM put, ATM, OTM call
        strikes = [Decimal("80.0"), Decimal("100.0"), Decimal("120.0")]

        vols = [model.implied_volatility(K, forward, maturity) for K in strikes]

        # With negative rho, should see put skew (higher vol for low strikes)
        assert vols[0] > vols[1]  # OTM put > ATM
        assert vols[2] < vols[0]  # OTM call < OTM put

    def test_variance_derivatives(self, typical_svi_params):
        """Test first and second derivatives of total variance"""
        model = SVIModel(typical_svi_params)
        k = 0.1

        # First derivative
        dw = model.variance_derivative(k)

        # Numerical derivative for validation
        h = 1e-6
        numerical_dw = (model.total_variance(k + h) - model.total_variance(k - h)) / (
            2 * h
        )

        assert_close(dw, numerical_dw, rtol=1e-4)

        # Second derivative
        d2w = model.variance_second_derivative(k)

        # Numerical second derivative
        numerical_d2w = (
            model.total_variance(k + h)
            - 2 * model.total_variance(k)
            + model.total_variance(k - h)
        ) / h**2

        assert_close(d2w, numerical_d2w, rtol=1e-3)

    def test_durrleman_condition_valid_params(self, typical_svi_params):
        """Valid SVI parameters should satisfy Durrleman condition"""
        model = SVIModel(typical_svi_params)
        k_grid = np.linspace(-1.0, 1.0, 50)

        valid = model.check_durrleman_condition(k_grid)

        # Should be valid for all k
        assert isinstance(valid, np.ndarray)
        assert np.all(valid)

    def test_zero_maturity_error(self, typical_svi_params):
        """Zero maturity should raise error"""
        model = SVIModel(typical_svi_params)

        with pytest.raises(ValueError, match="Maturity must be positive"):
            model.implied_volatility(Decimal("100.0"), Decimal("100.0"), 0.0)


# ============================================================================
# SABR Model Tests
# ============================================================================


class TestSABRParameters:
    """Test SABR parameter validation"""

    def test_valid_parameters(self, typical_sabr_params):
        """Valid SABR parameters should be accepted"""
        params = typical_sabr_params
        assert_equal(params.alpha, 0.25)
        assert_equal(params.beta, 0.5)
        assert_equal(params.rho, -0.3)
        assert_equal(params.nu, 0.4)

    def test_invalid_alpha_non_positive(self):
        """Non-positive α should raise error"""
        with pytest.raises(ValueError, match="alpha must be positive"):
            SABRParameters(alpha=0.0, beta=0.5, rho=0.0, nu=0.3)

    def test_invalid_beta_out_of_bounds(self):
        """β outside [0, 1] should raise error"""
        with pytest.raises(ValueError, match="beta must be in"):
            SABRParameters(alpha=0.2, beta=1.5, rho=0.0, nu=0.3)

    def test_invalid_rho_out_of_bounds(self):
        """ρ outside (-1, 1) should raise error"""
        with pytest.raises(ValueError, match="rho must be in"):
            SABRParameters(alpha=0.2, beta=0.5, rho=1.0, nu=0.3)

    def test_invalid_nu_negative(self):
        """Negative ν should raise error"""
        with pytest.raises(ValueError, match="nu must be non-negative"):
            SABRParameters(alpha=0.2, beta=0.5, rho=0.0, nu=-0.1)


class TestSABRModel:
    """Test SABR model calculations"""

    def test_atm_volatility(self, typical_sabr_params):
        """Test ATM volatility calculation"""
        model = SABRModel(typical_sabr_params)

        forward = Decimal("100.0")
        maturity = 1.0

        # ATM strike
        vol_atm = model.implied_volatility(forward, forward, maturity)

        # For ATM, should be close to alpha (with correction)
        assert vol_atm > 0
        assert 0.02 < vol_atm < 0.03  # Reasonable range

    def test_otm_volatility(self, typical_sabr_params):
        """Test OTM volatility calculation"""
        model = SABRModel(typical_sabr_params)

        forward = Decimal("100.0")
        maturity = 1.0

        strike_otm_put = Decimal("80.0")
        strike_otm_call = Decimal("120.0")

        vol_put = model.implied_volatility(strike_otm_put, forward, maturity)
        vol_call = model.implied_volatility(strike_otm_call, forward, maturity)

        # With negative rho, put wing should be higher
        assert vol_put > vol_call

    def test_volatility_smile_array(self, typical_sabr_params):
        """Test SABR volatility for array of strikes"""
        model = SABRModel(typical_sabr_params)

        forward = Decimal("100.0")
        maturity = 0.5

        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

        vols = model.implied_volatility(strikes, forward, maturity)

        assert isinstance(vols, np.ndarray)
        assert_equal(vols.shape, strikes.shape)
        assert np.all(vols > 0)

    def test_beta_one_lognormal(self):
        """β=1 should behave like lognormal model"""
        params = SABRParameters(alpha=0.2, beta=1.0, rho=-0.3, nu=0.4)
        model = SABRModel(params)

        forward = Decimal("100.0")
        maturity = 1.0
        strike = Decimal("105.0")

        vol = model.implied_volatility(strike, forward, maturity)

        # Should be well-behaved
        assert 0.1 < vol < 0.5

    def test_beta_zero_normal(self):
        """β=0 should behave like normal model"""
        params = SABRParameters(alpha=20.0, beta=0.0, rho=-0.3, nu=0.4)
        model = SABRModel(params)

        forward = Decimal("100.0")
        maturity = 1.0
        strike = Decimal("105.0")

        vol = model.implied_volatility(strike, forward, maturity)

        # Should be well-behaved for normal model
        assert vol > 0

    def test_small_moneyness_numerical_stability(self, typical_sabr_params):
        """Test numerical stability for nearly ATM strikes"""
        model = SABRModel(typical_sabr_params)

        forward = Decimal("100.0")
        maturity = 1.0

        # Very close to ATM
        strikes = [
            Decimal(str(100.0 + eps)) for eps in [-0.01, -0.001, 0.0, 0.001, 0.01]
        ]

        vols = [model.implied_volatility(K, forward, maturity) for K in strikes]

        # Should be smooth and close to each other
        vol_range = max(vols) - min(vols)
        assert vol_range < 0.01  # Small variation for small moneyness

    def test_zero_maturity_error(self, typical_sabr_params):
        """Zero maturity should raise error"""
        model = SABRModel(typical_sabr_params)

        with pytest.raises(ValueError, match="Maturity must be positive"):
            model.implied_volatility(Decimal("100.0"), Decimal("100.0"), 0.0)


# ============================================================================
# Calibration Engine Tests
# ============================================================================


class TestCalibrationEngine:
    """Test volatility model calibration"""

    def test_svi_calibration_convergence(self, sample_market_quotes):
        """SVI calibration should converge on realistic data"""
        engine = CalibrationEngine(
            CalibrationConfig(
                method=OptimizationMethod.LBFGSB, max_iterations=500, multi_start=1
            )
        )

        params, diagnostics = engine.calibrate_svi(sample_market_quotes)

        # Check convergence
        assert diagnostics["rmse"] < 0.05  # 5% RMSE threshold
        assert diagnostics["r_squared"] > 0.9  # Good fit

        # Validate parameters
        assert params.a > 0
        assert params.b > 0
        assert -1 < params.rho < 1
        assert params.sigma > 0

    def test_svi_calibration_multi_start(self, sample_market_quotes):
        """Multi-start should improve calibration quality"""
        engine_single = CalibrationEngine(CalibrationConfig(multi_start=1))
        engine_multi = CalibrationEngine(CalibrationConfig(multi_start=5))

        _, diag_single = engine_single.calibrate_svi(sample_market_quotes)
        _, diag_multi = engine_multi.calibrate_svi(sample_market_quotes)

        # Multi-start should be at least as good (potentially better)
        assert diag_multi["rmse"] <= diag_single["rmse"] * 1.1  # Allow 10% tolerance

    def test_sabr_calibration_convergence(self, sample_market_quotes):
        """SABR calibration should converge on realistic data"""
        engine = CalibrationEngine(
            CalibrationConfig(
                method=OptimizationMethod.LBFGSB, max_iterations=500, multi_start=1
            )
        )

        params, diagnostics = engine.calibrate_sabr(sample_market_quotes)

        # Check convergence
        assert diagnostics["rmse"] < 0.05  # 5% RMSE threshold
        assert diagnostics["r_squared"] > 0.85  # Good fit

        # Validate parameters
        assert params.alpha > 0
        assert 0 <= params.beta <= 1
        assert -1 < params.rho < 1
        assert params.nu >= 0

    def test_sabr_calibration_fixed_beta(self, sample_market_quotes):
        """SABR calibration with fixed β should respect constraint"""
        engine = CalibrationEngine()

        fixed_beta = 0.7
        params, diagnostics = engine.calibrate_sabr(
            sample_market_quotes, fix_beta=fixed_beta
        )

        # Beta should be exactly the fixed value
        assert_equal(params.beta, fixed_beta)

        # Should still converge reasonably
        assert diagnostics["rmse"] < 0.10

    def test_weighted_calibration_vega(self):
        """Vega-weighted calibration should prioritize high-vega points"""
        forward = Decimal("100.0")
        maturity = 0.25

        # Create quotes with varying vegas
        quotes = []
        for K, vol, vega in [
            (Decimal("90"), 0.25, Decimal("5.0")),
            (Decimal("100"), 0.20, Decimal("20.0")),  # High vega
            (Decimal("110"), 0.23, Decimal("5.0")),
        ]:
            quotes.append(
                MarketQuote(
                    strike=K,
                    maturity=maturity,
                    implied_vol=vol,
                    forward=forward,
                    vega=vega,
                )
            )

        engine = CalibrationEngine(CalibrationConfig(weighted_by_vega=True))
        params, _ = engine.calibrate_svi(quotes)

        # Model should fit ATM (high vega) best
        model = SVIModel(params)
        vol_atm = model.implied_volatility(Decimal("100"), forward, maturity)

        # Should be very close to market ATM vol
        assert_equal(vol_atm, 0.20, tolerance=0.02)

    def test_calibration_performance(self, sample_market_quotes):
        """Calibration should meet performance targets"""
        engine = CalibrationEngine()

        start = time.time()
        params, diagnostics = engine.calibrate_svi(sample_market_quotes)
        elapsed = time.time() - start

        # Performance target: <5s for single expiry
        assert elapsed < 5.0, f"Calibration took {elapsed:.2f}s, target is <5s"

        # Also check reported time
        assert diagnostics["calibration_time_seconds"] < 5.0

    def test_inconsistent_maturity_error(self):
        """Calibration with mixed maturities should raise error"""
        quotes = [
            MarketQuote(Decimal("100"), 0.25, 0.20, Decimal("100")),
            MarketQuote(
                Decimal("100"), 0.50, 0.20, Decimal("100")
            ),  # Different maturity
        ]

        engine = CalibrationEngine()

        with pytest.raises(ValueError, match="same maturity"):
            engine.calibrate_svi(quotes)

    def test_empty_quotes_error(self):
        """Calibration with no quotes should raise error"""
        engine = CalibrationEngine()

        with pytest.raises(ValueError, match="No market quotes"):
            engine.calibrate_svi([])


# ============================================================================
# Arbitrage Detection Tests
# ============================================================================


class TestArbitrageDetector:
    """Test arbitrage detection algorithms"""

    def test_butterfly_no_arbitrage(self):
        """Convex call prices should pass butterfly check"""
        detector = ArbitrageDetector()

        # Convex call prices (d²C/dK² > 0)
        strikes = np.array([90.0, 100.0, 110.0])
        call_prices = np.array([15.0, 10.0, 6.0])  # Convex

        is_free, violations = detector.check_butterfly_arbitrage(strikes, call_prices)

        assert is_free
        assert np.all(violations >= 0)

    def test_butterfly_with_arbitrage(self):
        """Non-convex call prices should fail butterfly check"""
        detector = ArbitrageDetector()

        # Non-convex prices
        strikes = np.array([90.0, 100.0, 110.0])
        call_prices = np.array([15.0, 11.0, 6.0])  # Not convex enough

        is_free, violations = detector.check_butterfly_arbitrage(strikes, call_prices)

        assert not is_free

    def test_calendar_no_arbitrage(self):
        """Increasing total variance should pass calendar check"""
        detector = ArbitrageDetector()

        maturities = np.array([0.25, 0.5, 1.0])
        total_vars = np.array([0.04, 0.09, 0.20])  # Increasing

        is_free, increments = detector.check_calendar_arbitrage(maturities, total_vars)

        assert is_free
        assert np.all(increments >= 0)

    def test_calendar_with_arbitrage(self):
        """Decreasing total variance should fail calendar check"""
        detector = ArbitrageDetector()

        maturities = np.array([0.25, 0.5, 1.0])
        total_vars = np.array([0.04, 0.09, 0.08])  # Decreasing at end

        is_free, increments = detector.check_calendar_arbitrage(maturities, total_vars)

        assert not is_free
        assert increments[-1] < 0

    def test_svi_arbitrage_check_valid_params(self, typical_svi_params):
        """Valid SVI parameters should pass arbitrage checks"""
        detector = ArbitrageDetector()
        model = SVIModel(typical_svi_params)

        results = detector.check_svi_arbitrage(model)

        assert results["is_arbitrage_free"]
        assert_equal(results["num_violations"], 0)
        assert_equal(len(results["violations"]), 0)

    def test_svi_arbitrage_check_invalid_params(self):
        """Invalid SVI parameters should fail arbitrage checks"""
        detector = ArbitrageDetector()

        # Parameters that violate non-negative variance
        params = SVIParameters(a=-0.5, b=0.1, rho=0.0, m=0.0, sigma=0.1)
        model = SVIModel(params)

        results = detector.check_svi_arbitrage(model)

        assert not results["is_arbitrage_free"]
        assert results["num_violations"] > 0


# ============================================================================
# Volatility Surface Tests
# ============================================================================


class TestVolatilitySurface:
    """Test multi-maturity volatility surface"""

    def test_add_slice(self, typical_svi_params):
        """Adding slices should store models correctly"""
        surface = VolatilitySurface()

        model1 = SVIModel(typical_svi_params)
        model2 = SVIModel(typical_svi_params)

        surface.add_slice(0.25, model1, Decimal("100.0"))
        surface.add_slice(0.5, model2, Decimal("101.0"))

        assert_equal(len(surface.models), 2)
        assert 0.25 in surface.models
        assert 0.5 in surface.models

    def test_mixed_model_types_error(self, typical_svi_params, typical_sabr_params):
        """Mixing SVI and SABR models should raise error"""
        surface = VolatilitySurface()

        svi_model = SVIModel(typical_svi_params)
        sabr_model = SABRModel(typical_sabr_params)

        surface.add_slice(0.25, svi_model, Decimal("100.0"))

        with pytest.raises(ValueError, match="Cannot mix model types"):
            surface.add_slice(0.5, sabr_model, Decimal("100.0"))

    def test_exact_maturity_lookup(self, typical_svi_params):
        """Lookup at exact maturity should use model directly"""
        surface = VolatilitySurface()
        model = SVIModel(typical_svi_params)
        forward = Decimal("100.0")
        maturity = 0.25

        surface.add_slice(maturity, model, forward)

        strike = Decimal("105.0")
        vol = surface.implied_volatility(strike, maturity)

        # Should match model directly
        expected_vol = model.implied_volatility(strike, forward, maturity)
        assert_close(vol, expected_vol)

    def test_interpolation_between_maturities(self, typical_svi_params):
        """Interpolation between maturities should be smooth"""
        surface = VolatilitySurface(InterpolationMethod.LINEAR)

        model1 = SVIModel(typical_svi_params)
        model2 = SVIModel(typical_svi_params)

        surface.add_slice(0.25, model1, Decimal("100.0"))
        surface.add_slice(0.75, model2, Decimal("100.0"))

        # Interpolate at midpoint
        vol_mid = surface.implied_volatility(Decimal("100.0"), 0.5)

        # For total variance interpolation, mid vol is between
        assert vol_mid > 0
        assert 0.1 < vol_mid < 0.4  # Broad reasonable range for interpolated vol

    def test_extrapolation_warning(self, typical_svi_params):
        """Extrapolation beyond maturity range should warn"""
        surface = VolatilitySurface()
        model = SVIModel(typical_svi_params)

        surface.add_slice(0.5, model, Decimal("100.0"))

        # Extrapolate to shorter maturity
        with pytest.warns(UserWarning, match="Extrapolating short maturity"):
            surface.implied_volatility(Decimal("100.0"), 0.1)

        # Extrapolate to longer maturity
        with pytest.warns(UserWarning, match="Extrapolating long maturity"):
            surface.implied_volatility(Decimal("100.0"), 1.0)

    def test_get_smile(self, typical_svi_params):
        """Get smile should return DataFrame with correct structure"""
        surface = VolatilitySurface()
        model = SVIModel(typical_svi_params)

        surface.add_slice(0.25, model, Decimal("100.0"))

        smile = surface.get_smile(0.25, strike_range=(80.0, 120.0), num_points=10)

        assert "strike" in smile.columns
        assert "log_moneyness" in smile.columns
        assert "implied_vol" in smile.columns
        assert_equal(len(smile), 10)

    def test_to_dataframe(self, typical_svi_params):
        """Export to DataFrame should include all models"""
        surface = VolatilitySurface()

        model1 = SVIModel(typical_svi_params)
        model2 = SVIModel(typical_svi_params)

        surface.add_slice(0.25, model1, Decimal("100.0"))
        surface.add_slice(0.5, model2, Decimal("101.0"))

        df = surface.to_dataframe()

        assert_equal(len(df), 2)
        assert "maturity" in df.columns
        assert "model_type" in df.columns
        assert (df["model_type"] == "SVIModel").all()

    def test_empty_surface_error(self):
        """Querying empty surface should raise error"""
        surface = VolatilitySurface()

        with pytest.raises(ValueError, match="No models in surface"):
            surface.implied_volatility(Decimal("100.0"), 0.5)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests"""

    def test_full_calibration_workflow(self):
        """Test complete workflow: quotes -> calibration -> surface -> lookup"""
        # Step 1: Generate market quotes for multiple maturities
        forward = Decimal("100.0")
        maturities = [0.25, 0.5, 1.0]

        all_quotes = {}
        for T in maturities:
            strikes = [Decimal(str(k)) for k in [80, 90, 100, 110, 120]]
            # Smile with skew
            vols = [
                0.30 - 0.05 * T,
                0.25 - 0.03 * T,
                0.22,
                0.24 + 0.02 * T,
                0.28 + 0.04 * T,
            ]

            quotes = []
            for K, vol in zip(strikes, vols):
                quotes.append(
                    MarketQuote(
                        strike=K,
                        maturity=T,
                        implied_vol=vol,
                        forward=forward,
                        vega=Decimal("10.0"),
                    )
                )
            all_quotes[T] = quotes

        # Step 2: Calibrate models
        engine = CalibrationEngine(CalibrationConfig(multi_start=3))
        surface = VolatilitySurface()

        for T, quotes in all_quotes.items():
            params, diagnostics = engine.calibrate_svi(quotes)

            # Verify good fit
            assert diagnostics["rmse"] < 0.05

            model = SVIModel(params)
            surface.add_slice(T, model, forward)

        # Step 3: Query surface at intermediate maturity
        vol_interp = surface.implied_volatility(Decimal("95.0"), 0.375)

        # Should be reasonable
        assert 0.15 < vol_interp < 0.35

        # Step 4: Check arbitrage-free
        detector = ArbitrageDetector()

        for T in maturities:
            model = surface.models[T]
            results = detector.check_svi_arbitrage(model)

            # Calibrated models should be arbitrage-free
            assert results[
                "is_arbitrage_free"
            ], f"Maturity {T} has arbitrage: {results['violations']}"

    def test_performance_full_surface(self):
        """Test performance for full surface construction"""
        forward = Decimal("100.0")
        maturities = [0.08, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # 7 slices

        all_quotes = {}
        for T in maturities:
            strikes = [Decimal(str(k)) for k in np.linspace(70, 130, 15)]
            vols = 0.20 + 0.1 * np.exp(
                -((np.log(np.array([float(K) / 100.0 for K in strikes]))) ** 2) / 0.1
            )

            quotes = []
            for K, vol in zip(strikes, vols):
                quotes.append(
                    MarketQuote(
                        strike=K,
                        maturity=T,
                        implied_vol=float(vol),
                        forward=forward,
                    )
                )
            all_quotes[T] = quotes

        # Calibrate full surface
        engine = CalibrationEngine(CalibrationConfig(multi_start=2))
        surface = VolatilitySurface()

        start = time.time()
        for T, quotes in all_quotes.items():
            params, _ = engine.calibrate_svi(quotes)
            model = SVIModel(params)
            surface.add_slice(T, model, forward)

        total_time = time.time() - start

        # Performance target: <30s for full surface
        assert (
            total_time < 30.0
        ), f"Surface construction took {total_time:.2f}s, target is <30s"

    def test_sabr_vs_svi_comparison(self, sample_market_quotes):
        """Compare SABR and SVI calibration quality"""
        engine = CalibrationEngine()

        # Calibrate both models
        svi_params, svi_diag = engine.calibrate_svi(sample_market_quotes)
        sabr_params, sabr_diag = engine.calibrate_sabr(
            sample_market_quotes, fix_beta=0.5
        )

        # Both should converge reasonably
        assert svi_diag["rmse"] < 0.10
        assert sabr_diag["rmse"] < 0.10

        # For this data, SVI may fit slightly better (more parameters)
        # But SABR provides more financial interpretation


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and numerical stability"""

    def test_extreme_strike_svi(self, typical_svi_params):
        """SVI should handle extreme strikes gracefully"""
        model = SVIModel(typical_svi_params)
        forward = Decimal("100.0")
        maturity = 1.0

        # Very deep OTM
        strike_otm = Decimal("10.0")
        vol_otm = model.implied_volatility(strike_otm, forward, maturity)

        assert vol_otm > 0
        assert vol_otm < 2.0  # Should not explode

        # Very deep ITM
        strike_itm = Decimal("1000.0")
        vol_itm = model.implied_volatility(strike_itm, forward, maturity)

        assert vol_itm > 0
        assert vol_itm < 2.0

    def test_short_maturity_sabr(self, typical_sabr_params):
        """SABR should handle short maturities"""
        model = SABRModel(typical_sabr_params)
        forward = Decimal("100.0")

        # Very short maturity
        maturity = 0.01  # ~3.65 days

        vol = model.implied_volatility(Decimal("105.0"), forward, maturity)

        assert vol > 0
        assert vol < 1.0

    def test_long_maturity_extrapolation(self, typical_svi_params):
        """Surface should extrapolate to long maturities"""
        surface = VolatilitySurface()
        model = SVIModel(typical_svi_params)

        surface.add_slice(1.0, model, Decimal("100.0"))

        # Extrapolate to 5 years
        with pytest.warns(UserWarning):
            vol_5y = surface.implied_volatility(Decimal("100.0"), 5.0)

        assert vol_5y > 0

    def test_zero_vega_quotes(self):
        """Quotes with zero vega should be handled"""
        quotes = [
            MarketQuote(
                Decimal("100"), 0.25, 0.20, Decimal("100"), vega=Decimal("0.0")
            ),
            MarketQuote(
                Decimal("105"), 0.25, 0.22, Decimal("100"), vega=Decimal("10.0")
            ),
        ]

        engine = CalibrationEngine(CalibrationConfig(weighted_by_vega=True))

        # Should not error, fall back to uniform weighting for zero vega
        params, diag = engine.calibrate_svi(quotes)

        assert diag["rmse"] is not None


# ============================================================================
# Performance Benchmarks
# ============================================================================


class TestPerformance:
    """Performance benchmark tests"""

    def test_arbitrage_check_performance(self, typical_svi_params):
        """Arbitrage check should be fast (< 100 ms)"""
        detector = ArbitrageDetector()
        model = SVIModel(typical_svi_params)

        start = time.time()
        for _ in range(10):  # 10 runs
            detector.check_svi_arbitrage(model)
        elapsed = time.time() - start

        avg_time = elapsed / 10.0

        # Target: <100ms per check
        assert (
            avg_time < 0.1
        ), f"Arbitrage check took {avg_time*1000:.1f}ms, target is <100ms"

    def test_vol_lookup_performance(self, typical_svi_params):
        """Volatility lookup should be very fast"""
        model = SVIModel(typical_svi_params)
        forward = Decimal("100.0")
        maturity = 0.25

        strikes = [Decimal(str(k)) for k in np.linspace(50, 150, 1000)]

        start = time.time()
        _ = model.implied_volatility(
            np.array([float(K) for K in strikes]), forward, maturity
        )
        elapsed = time.time() - start

        # Should compute 1000 vols very fast (<10ms)
        assert elapsed < 0.01, f"1000 vol calculations took {elapsed*1000:.1f}ms"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
