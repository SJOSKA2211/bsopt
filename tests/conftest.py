"""
Pytest configuration and shared fixtures for the Black-Scholes Option Pricing Platform.

This module provides:
- Reusable fixtures for option parameters
- Mock data factories
- Test configuration
- Common test utilities
"""

import uuid
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.pricing.monte_carlo import MCConfig, MonteCarloEngine

# ============================================================================
# Black-Scholes Parameter Fixtures
# ============================================================================


@pytest.fixture
def atm_params() -> BSParameters:
    """
    At-the-money (ATM) option parameters.

    Standard parameters for testing where spot == strike.
    """
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02
    )


@pytest.fixture
def itm_call_params() -> BSParameters:
    """
    In-the-money (ITM) call option parameters.

    Spot > Strike, so call has intrinsic value.
    """
    return BSParameters(
        spot=110.0, strike=100.0, maturity=1.0, volatility=0.25, rate=0.05, dividend=0.02
    )


@pytest.fixture
def otm_call_params() -> BSParameters:
    """
    Out-of-the-money (OTM) call option parameters.

    Spot < Strike, so call has no intrinsic value.
    """
    return BSParameters(
        spot=90.0, strike=100.0, maturity=1.0, volatility=0.25, rate=0.05, dividend=0.02
    )


@pytest.fixture
def deep_itm_call_params() -> BSParameters:
    """
    Deep in-the-money call option parameters.

    Spot >> Strike, delta should be close to 1.
    """
    return BSParameters(
        spot=150.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02
    )


@pytest.fixture
def deep_otm_call_params() -> BSParameters:
    """
    Deep out-of-the-money call option parameters.

    Spot << Strike, delta should be close to 0.
    """
    return BSParameters(
        spot=60.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02
    )


@pytest.fixture
def short_maturity_params() -> BSParameters:
    """
    Short time to maturity (30 days).
    """
    return BSParameters(
        spot=100.0,
        strike=100.0,
        maturity=30.0 / 365.0,  # 30 days
        volatility=0.20,
        rate=0.05,
        dividend=0.02,
    )


@pytest.fixture
def long_maturity_params() -> BSParameters:
    """
    Long time to maturity (5 years).
    """
    return BSParameters(
        spot=100.0, strike=100.0, maturity=5.0, volatility=0.20, rate=0.05, dividend=0.02
    )


@pytest.fixture
def high_volatility_params() -> BSParameters:
    """
    High volatility (50%).
    """
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.50, rate=0.05, dividend=0.02
    )


@pytest.fixture
def low_volatility_params() -> BSParameters:
    """
    Low volatility (10%).
    """
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.10, rate=0.05, dividend=0.02
    )


@pytest.fixture
def zero_dividend_params() -> BSParameters:
    """
    Parameters with no dividend yield.
    """
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.0
    )


@pytest.fixture
def high_dividend_params() -> BSParameters:
    """
    Parameters with high dividend yield.
    """
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.08
    )


@pytest.fixture
def negative_rate_params() -> BSParameters:
    """
    Parameters with negative interest rate (realistic in some markets).
    """
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=-0.01, dividend=0.0
    )


# ============================================================================
# Known Values for Validation (from literature)
# ============================================================================


@pytest.fixture
def known_values() -> Dict[str, Any]:
    """
    Known option prices from financial literature for validation.

    These values come from Hull's "Options, Futures, and Other Derivatives"
    and can be used to verify implementation accuracy.
    """
    return {
        # Example 13.6 from Hull (9th edition)
        "hull_example_13_6": {
            "params": BSParameters(
                spot=42.0, strike=40.0, maturity=0.5, volatility=0.20, rate=0.10, dividend=0.0
            ),
            "call_price": 4.76,  # Approximate
            "put_price": 0.81,  # Approximate
        },
        # Example 17.1 from Hull
        "hull_example_17_1": {
            "params": BSParameters(
                spot=50.0, strike=50.0, maturity=1.0, volatility=0.30, rate=0.05, dividend=0.0
            ),
            "call_price": 7.04,  # Approximate
            "delta": 0.6368,
            "gamma": 0.0651,
            "vega": 0.1945,  # Per 1% vol change
        },
    }


# ============================================================================
# Monte Carlo Configuration Fixtures
# ============================================================================


@pytest.fixture
def mc_config_fast() -> MCConfig:
    """
    Fast Monte Carlo configuration for unit testing.

    Uses fewer paths for speed in tests.
    """
    return MCConfig(n_paths=10000, n_steps=100, antithetic=True, control_variate=False, seed=42)


@pytest.fixture
def mc_config_accurate() -> MCConfig:
    """
    Accurate Monte Carlo configuration for validation tests.

    Uses more paths for better convergence.
    """
    return MCConfig(n_paths=100000, n_steps=252, antithetic=True, control_variate=True, seed=42)


@pytest.fixture
def mc_config_minimal() -> MCConfig:
    """
    Minimal Monte Carlo configuration for edge case testing.
    """
    return MCConfig(n_paths=1000, n_steps=50, antithetic=False, control_variate=False, seed=42)


# ============================================================================
# Engine Fixtures
# ============================================================================


@pytest.fixture
def bs_engine() -> BlackScholesEngine:
    """Black-Scholes analytical engine instance."""
    return BlackScholesEngine()


@pytest.fixture
def mc_engine_fast(mc_config_fast: MCConfig) -> MonteCarloEngine:
    """Monte Carlo engine with fast configuration."""
    return MonteCarloEngine(mc_config_fast)


@pytest.fixture
def mc_engine_accurate(mc_config_accurate: MCConfig) -> MonteCarloEngine:
    """Monte Carlo engine with accurate configuration."""
    return MonteCarloEngine(mc_config_accurate)


# ============================================================================
# API Testing Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def mock_redis_and_celery(monkeypatch):
    """Mock Redis and Celery to avoid connection errors in tests."""
    from unittest.mock import MagicMock, AsyncMock

    # Mock redis.asyncio
    mock_redis = MagicMock()
    # Set up async methods using AsyncMock
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.publish = AsyncMock(return_value=1)
    mock_redis.ping = AsyncMock(return_value=True)
    
    mock_pipeline = MagicMock()
    mock_pipeline.incr = MagicMock()
    mock_pipeline.expire = MagicMock()
    # Mock execute to return [1, True] representing [count, success]
    mock_pipeline.execute = AsyncMock(return_value=[1, True])
    mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
    
    # Mock pipeline as a context manager
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=None)
    
    # Mock pipeline as a context manager if needed (though not used in rate_limit.py currently)
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock ConnectionPool
    mock_pool = MagicMock()

    monkeypatch.setattr("redis.asyncio.Redis", MagicMock(return_value=mock_redis))
    monkeypatch.setattr("redis.asyncio.ConnectionPool", mock_pool)
    monkeypatch.setattr("redis.asyncio.ConnectionPool.from_url", MagicMock(return_value=mock_pool))

    # Mock src.utils.cache functions
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)

    # Force RS256 for tests to match PEM keys
    from src.config import settings

    monkeypatch.setattr(settings, "JWT_ALGORITHM", "RS256")

    # Mock Celery delay method
    from celery.app.task import Task

    monkeypatch.setattr(Task, "delay", MagicMock())
    monkeypatch.setattr(Task, "apply_async", MagicMock())


@pytest.fixture
def api_client(mock_db_session):
    """
    FastAPI test client for integration testing.
    """
    from src.api.main import app
    from src.database import get_db

    # Override get_db dependency
    app.dependency_overrides[get_db] = lambda: mock_db_session

    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """
    Mock database session for testing with simple in-memory user store.
    """
    from unittest.mock import MagicMock

    from src.database.models import User, APIKey

    mock_session = MagicMock()
    users_by_email = {}
    users_by_id = {}
    api_keys_by_hash = {}

    class MockQuery:
        def __init__(self, model):
            self.model = model
            self.filter_val = None

        def filter(self, *args, **kwargs):
            for arg in args:
                # Recursively look for values in the expression tree
                self._extract_filter_val(arg)
            return self

        def _extract_filter_val(self, expr):
            # Handle standard BinaryExpression (model.attr == value)
            if hasattr(expr, 'right') and hasattr(expr.right, 'value'):
                val = expr.right.value
                # Prioritize 64-char strings (SHA256 hashes) for API keys
                if isinstance(val, str) and len(val) == 64:
                    self.filter_val = val
                elif self.filter_val is None:
                    self.filter_val = val
            # Handle and_ / or_ clauses
            elif hasattr(expr, 'clauses'):
                for clause in expr.clauses:
                    self._extract_filter_val(clause)

        def first(self):
            if self.model == User:
                user = users_by_email.get(self.filter_val)
                if not user:
                    user = users_by_id.get(self.filter_val)
                return user
            if self.model == APIKey:
                return api_keys_by_hash.get(self.filter_val)
            return None

    def mock_query(model):
        return MockQuery(model)

    mock_session.query = MagicMock(side_effect=mock_query)

    def mock_add(obj):
        if isinstance(obj, User):
            if not obj.id:
                obj.id = uuid.uuid4()
            obj.is_verified = True
            if obj.is_mfa_enabled is None:
                obj.is_mfa_enabled = False
            if obj.created_at is None:
                from datetime import datetime, timezone
                obj.created_at = datetime.now(timezone.utc)
            users_by_email[obj.email] = obj
            users_by_id[str(obj.id)] = obj
            users_by_id[obj.id] = obj
        elif isinstance(obj, APIKey):
            if not obj.id:
                obj.id = uuid.uuid4()
            api_keys_by_hash[obj.key_hash] = obj

    mock_session.add = MagicMock(side_effect=mock_add)
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()

    return mock_session


# ============================================================================
# Numerical Tolerance Fixtures
# ============================================================================


@pytest.fixture
def tight_tolerance() -> float:
    """Very tight numerical tolerance for exact calculations."""
    return 1e-10


@pytest.fixture
def normal_tolerance() -> float:
    """Normal tolerance for numerical calculations."""
    return 1e-6


@pytest.fixture
def loose_tolerance() -> float:
    """Loose tolerance for Monte Carlo and numerical methods."""
    return 1e-3


@pytest.fixture
def percentage_tolerance() -> float:
    """Percentage tolerance for validating against known values."""
    return 0.01  # 1%


# ============================================================================
# Parametrized Test Data
# ============================================================================


@pytest.fixture
def option_types():
    """List of option types for parametrized tests."""
    return ["call", "put"]


@pytest.fixture
def spot_strike_combinations():
    """
    Various spot/strike combinations for testing moneyness.

    Returns list of tuples: (spot, strike, description)
    """
    return [
        (80.0, 100.0, "deep_otm_call_itm_put"),
        (90.0, 100.0, "otm_call_itm_put"),
        (100.0, 100.0, "atm"),
        (110.0, 100.0, "itm_call_otm_put"),
        (120.0, 100.0, "deep_itm_call_otm_put"),
    ]


@pytest.fixture
def volatility_scenarios():
    """
    Different volatility scenarios for testing.

    Returns list of tuples: (volatility, description)
    """
    return [
        (0.05, "very_low"),
        (0.10, "low"),
        (0.20, "normal"),
        (0.40, "high"),
        (0.60, "very_high"),
    ]


@pytest.fixture
def maturity_scenarios():
    """
    Different maturity scenarios for testing.

    Returns list of tuples: (maturity_years, description)
    """
    return [
        (7.0 / 365.0, "one_week"),
        (30.0 / 365.0, "one_month"),
        (90.0 / 365.0, "three_months"),
        (0.5, "six_months"),
        (1.0, "one_year"),
        (2.0, "two_years"),
        (5.0, "five_years"),
    ]


# ============================================================================
# Helper Functions
# ============================================================================


def assert_close(actual: float, expected: float, tolerance: float, message: str = "") -> None:
    """
    Assert that two values are close within tolerance.

    Args:
        actual: Actual value from computation
        expected: Expected value
        tolerance: Absolute tolerance
        message: Custom error message
    """
    diff = abs(actual - expected)
    if message:
        assert diff < tolerance, f"{message}: {actual} != {expected} (diff: {diff})"
    else:
        assert diff < tolerance, f"{actual} != {expected} (diff: {diff})"


def assert_relative_close(
    actual: float, expected: float, rel_tolerance: float, message: str = ""
) -> None:
    """
    Assert that two values are relatively close.

    Args:
        actual: Actual value from computation
        expected: Expected value
        rel_tolerance: Relative tolerance (e.g., 0.01 for 1%)
        message: Custom error message
    """
    if expected == 0:
        assert abs(actual) < 1e-10, f"Expected 0, got {actual}"
    else:
        rel_diff = abs((actual - expected) / expected)
        if message:
            assert (
                rel_diff < rel_tolerance
            ), f"{message}: {actual} != {expected} (rel diff: {rel_diff:.2%})"
        else:
            assert rel_diff < rel_tolerance, f"{actual} != {expected} (rel diff: {rel_diff:.2%})"


def is_close(actual: float, expected: float, tolerance: float) -> bool:
    """
    Check if two values are close within tolerance.

    Args:
        actual: Actual value
        expected: Expected value
        tolerance: Absolute tolerance

    Returns:
        bool: True if values are within tolerance
    """
    return abs(actual - expected) < tolerance


# Make helper functions available as fixtures
@pytest.fixture
def assert_close_helper():
    """Fixture providing the assert_close helper function."""
    return assert_close


@pytest.fixture
def assert_relative_close_helper():
    """Fixture providing the assert_relative_close helper function."""
    return assert_relative_close


@pytest.fixture
def is_close_helper():
    """Fixture providing the is_close helper function."""
    return is_close


# ============================================================================
# Performance Benchmarking Fixtures
# ============================================================================


@pytest.fixture
def benchmark_params() -> BSParameters:
    """Standard parameters for performance benchmarking."""
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02
    )


# ============================================================================
# Random Seed Control
# ============================================================================


@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Reset random seed before each test for reproducibility.

    This fixture runs automatically before every test.
    """
    np.random.seed(42)
    yield
    # Cleanup if needed
