from jose import JWTError

_mock_jwt_payload_store = {} # Added globally

"""
Pytest configuration and shared fixtures for the Black-Scholes Option Pricing Platform.

This module provides:
- Reusable fixtures for option parameters
- Mock data factories
- Test configuration
- Common test utilities
"""

import uuid
from typing import Any, Dict, Optional, List

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
    from unittest.mock import MagicMock, AsyncMock, patch
    # Import actual settings to get paths BEFORE src.config is mocked
    from src.config import settings as app_settings_real 
    from datetime import datetime, timedelta, timezone # Added for JWT mock

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



    def mock_jose_jwt_encode(payload, key, algorithm):
        jti = payload.get("jti", str(uuid.uuid4()))
        token_type = payload.get("type", "access")
        fake_token_string = f"fake_token_for_{token_type}_{jti}"
        _mock_jwt_payload_store[fake_token_string] = payload
        return fake_token_string

    def mock_jose_jwt_decode(token, key, algorithms):
        payload = _mock_jwt_payload_store.get(token)
        if payload:
            return payload
        raise JWTError("Invalid token or not found in mock store")

    monkeypatch.setattr("jose.jwt.encode", MagicMock(side_effect=mock_jose_jwt_encode))
    monkeypatch.setattr("jose.jwt.decode", MagicMock(side_effect=mock_jose_jwt_decode))

    # Mock file reading for JWT keys (keep existing functionality)
    original_open = open
    def mock_open_jwt_keys(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if file in [app_settings_real.JWT_PRIVATE_KEY_PATH, app_settings_real.JWT_PUBLIC_KEY_PATH, "dummy_path"]:
            mock_file = MagicMock()
            if "private" in str(file):
                 mock_file.read.return_value = app_settings_real.JWT_PRIVATE_KEY
            else:
                 mock_file.read.return_value = app_settings_real.JWT_PUBLIC_KEY
            
            # Setup context manager support
            mock_file.__enter__.return_value = mock_file
            return mock_file
        return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr("builtins.open", mock_open_jwt_keys)

    # Mock TokenBlacklist methods (moved from test_auth.py to conftest.py)
    from src.security.auth import token_blacklist
    monkeypatch.setattr(token_blacklist, "contains", AsyncMock(return_value=False))
    monkeypatch.setattr(token_blacklist, "add", AsyncMock(return_value=None)) # Mock add to do nothing

# ============================================================================ 
# Pytest Configuration Hook
# ============================================================================ 


import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from jose import JWTError

def pytest_configure(config):
    """
    Hook to configure pytest before any tests are run.
    This is where we can inject mocks for module-level imports.
    """
    import os
    import sys
    import sys
    os.environ["TESTING"] = "true"
    
    from unittest.mock import MagicMock
    from _pytest.monkeypatch import MonkeyPatch # Import MonkeyPatch class

    # Create a MonkeyPatch instance for early patching
    mpatch = MonkeyPatch()

    # Create a mock settings object early and patch src.config.Settings
    mock_settings = MagicMock()
    # Populate mock_settings with all required values
    mock_settings.DATABASE_URL = "sqlite:///:memory:"
    mock_settings.DEBUG = False
    mock_settings.SLOW_QUERY_THRESHOLD_MS = 100
    mock_settings.ENVIRONMENT = "test"
    mock_settings.ML_TRAINING_OPTUNA_TRIALS = 1
    mock_settings.ML_TRAINING_TEST_SIZE = 0.2
    mock_settings.ML_TRAINING_RANDOM_STATE = 42
    mock_settings.ML_TRAINING_KFOLD_SPLITS = 3
    mock_settings.ML_TRAINING_PROMOTE_THRESHOLD_R2 = 0.98
    mock_settings.ML_TRAINING_NN_EPOCHS = 10
    mock_settings.ML_TRAINING_NN_LR = 0.001
    mock_settings.DASK_LOCAL_CLUSTER_THREADS_PER_WORKER = 2
    mock_settings.DASK_ARRAY_DEFAULT_CHUNKS_FRACTION = 4
    mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    mock_settings.REFRESH_TOKEN_EXPIRE_DAYS = 7
    mock_settings.JWT_PRIVATE_KEY_PATH = None # Set to None to skip file reading
    mock_settings.JWT_PUBLIC_KEY_PATH = None # Set to None to skip file reading
    mock_settings.JWT_PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQC5L/Ra/v7tnihy\n2ISWqOW6qDW//uSZGAG46PgmNU3QKZOANFbopT8aGpZsjzbleM+dh5YyXefTN69h\nHyo+oO8ndZ7vHmEPzsvV2YvVkzp7Z0DhMXIolYDSvGQmZh+mFdtn2wUb6jSiHXaq\nEHr4BMs3W1wWg1jEAaTZIYyjyQg6ngkmZpEuMnHdgJ4RTow8M0uisYveZ3e3rPtY\n6i8V+Jm13NtoYw0DyXdVilrHpoaDIbJDXvMaSiDzuREOU2DPDSTo0hb2LnwG5i1Z\nc82Ked+1CIVuuRy0k1td6B5d9pWyuy+uU8httwj9ohd/u34nNZjBKwK13SOU4eSr\n7gYKFVWdAgMBAAECggEALOHB88HkcBuKSxdNvyPtFZmJ+V/h8HbAiBuydLQriT1i\nWcqVm6ndCssR1Pq6v8/RZakETm2y716KG1xNxUME40fe/q4asRWaEli4ysj1fg4O\ny+VLeSmOwMRW83WByyN1+ww6h0Lyga3pYhs/alfjFqYGQhCQVWTemp9GcjWZUHct\nyuUWw+D25nBH36VfvFMLpBh6YWXA0Jgukoy5lFfZRhpJfsv6U9P6GFaPf9rZwx/e\nwSzLtx5dftXLDM2vRSexbCZ3RtFS1uqACi3gx3LdixrkC/7u6MNmTf8hQQ9neNz/\nbseV7wmVV/w+WhT7z7S74mvfcUp856pCpgxDcyAW0wKBgQDwWvYv3cNG+ANiHkPQ\nDs4hSsjjY8seLaxb16Px4eO2f7iC8NpkUg4rK9ATXj1iMlSgRs3hl1cc4HyJUIUA\nqI/B/mfr+V8LP3RrZf3oVV1M1PSEeFGsDOkBj7qp4UOhhxhFZKV312UsA3zFgmeh\n6Mi15DBJA9istN7tO6297+qLkwKBgQDFPbqjZVfCu0ZhvNn1MoQWTJevVsFqj7tx\n1z1S+AV8S38ZzzJQ+HMd9zBuCdIzcTHewcDuP4ViQwDuG/fP5xGPXq/mnPeDiStj\n1LGebk9NoKhMDhreN7F1HB9EiPAHrRuKbvXGlJId7+CryDtKiCdILKSxcM7ISAWz\nkka4d1U4DwKBgFSRfLucyH+o9e/I/E0xWF+oY4R2cVo98b4i4oFR3IAy8iXIPt87\na9bOxJh7+Rca5GG8Z+YvDKyRY+Agn5t6s5IrRD9n74QSoJDFQxBDMN++sgYGG55w\nsG8nNiNdqiEkWmrYHcxL0kIpO/Z2iqYSNMSi9EZMfylNPn60F86Toq/XAoGAaW+T\nCm9NF4p2mB6UwG7tjuIIn7EXmXgV8kflvylX/VZ6LfjFY+U73EPb/qZ/WAJlbzdR\ng8ou6hT5NU98DAXwLByomPVFlmnH6Qi1ngVHnS6JaIaNtEY/NWQADunIrcHKnBiu\nhyW6aghlqE5T64aWwQ0kSLoNv6BIc/J4nXkk898CgYAfhWlkH+Sf/BKImOy/1TNF\nYCN2AK07UgiCB9t0JNT0+ir7cK9X3FjlyFLFqdD2QG7ARWhobesZTMsVhyHwIFQF\nwToSX83WKyftQ5jUvyPXilfBtg+glMXecvcaiNHZ257jtL7jgK+iEWWoso9XoCyy\nfoff69nqRl7Q7Z+BX4OAzA==\n-----END PRIVATE KEY-----"
    mock_settings.JWT_PUBLIC_KEY = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuS/0Wv7+7Z4octiElqjl\nuqg1v/7kmRgBuOj4JjVN0CmTgDRU6KU/GhqWbI825XjPnYeWMl3n0zevYR8qPqDv\nJ3We7x5hD87L1dmL1ZM6e2dA4TFyKJWA0rxkJmYfphXbZ9sFG+o0oh12qhB6+ATL\nN1tcFoNYxAGk2SGMo8kIOp4JJmaRLjJx3YCeEU6MPDNLorGL3md3t6z7WOovFfiZ\ntdzbaGMNA8l3VYpax6aGgyGyQ17zGkog87kRDlNgzw0k6NIW9i58BuYtWXPNinnf\ntQiFbrkctJNbXegeXfaVsrsvrlPIbbcI/aIXf7t+JzWYwSsCtd0jlOHkq+4GChVV\nnQIDAQAB\n-----END PUBLIC KEY-----"
    mock_settings.JWT_ALGORITHM = "RS256"
    mock_settings.REDIS_HOST = "localhost"
    mock_settings.REDIS_PORT = 6379
    mock_settings.REDIS_DB = 0
    mock_settings.REDIS_PASSWORD = None
    mock_settings.CORS_ORIGINS = ["http://localhost"]
    mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    mock_settings.REFRESH_TOKEN_EXPIRE_DAYS = 7
    mock_settings.BCRYPT_ROUNDS = 12 # Explicitly set to an integer
    mock_settings.PASSWORD_MIN_LENGTH = 8
    # Add valid base64-encoded key for Fernet
    mock_settings.MFA_ENCRYPTION_KEY = "cUMkImRgwyuUNS_WDJPWOnJhlZlB_1cTOEMjtR2TMhU="
    mock_settings.ML_SERVICE_URL = "http://ml-service"
    mock_settings.rate_limit_tiers = {"free": 100, "pro": 1000, "enterprise": 0}

    # Patch src.config.Settings to return our mock_settings
    mpatch.setattr("src.config.Settings", MagicMock(return_value=mock_settings))
    mpatch.setattr("src.config.get_settings", MagicMock(return_value=mock_settings))

    # Import src.config AFTER patching to ensure our mock is used
    import src.config
    src.config.settings = mock_settings
    src.config._settings = mock_settings

    # Mock src.database internals
    mock_db_engine = MagicMock()
    mpatch.setattr("src.database._engine", mock_db_engine)
    mpatch.setattr("src.database._SessionLocal", MagicMock())
    mpatch.setattr("src.database._initialize_db_components", MagicMock())

    # Mock Celery delay method globally
    from celery.app.task import Task
    
    Task.delay = MagicMock()
    Task.apply_async = MagicMock()

    # Mock psycopg2
    mock_psycopg2 = MagicMock()
    sys.modules["psycopg2"] = mock_psycopg2
    
    # Mock confluent_kafka
    mock_confluent_kafka = MagicMock()
    sys.modules["confluent_kafka"] = mock_confluent_kafka
    sys.modules["confluent_kafka.admin"] = MagicMock()
    sys.modules["confluent_kafka.schema_registry"] = MagicMock()
    sys.modules["confluent_kafka.schema_registry.avro"] = MagicMock()
    
    # Mock redis (sync and async)
    class MockRedisError(Exception): pass
    
    mock_redis_mod = MagicMock()
    class MockRedis:
        def __init__(self, *args, **kwargs): pass
        def ping(self): return True
        def pipeline(self): return MagicMock()
    mock_redis_mod.Redis = MockRedis
    mock_redis_mod.StrictRedis = MockRedis
    mock_redis_mod.RedisError = MockRedisError
    sys.modules["redis"] = mock_redis_mod
    
    mock_async_redis_mod = MagicMock()
    mock_async_redis_mod.Redis = MagicMock() # Ensure it's a MagicMock
    mock_async_redis_mod.RedisError = MockRedisError
    sys.modules["redis.asyncio"] = mock_async_redis_mod

    # Mock sqlalchemy.create_engine
    mock_engine = MagicMock()
    mpatch.setattr("sqlalchemy.create_engine", MagicMock(return_value=mock_engine))


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
            self._results = []

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

        def count(self):
            if self.model == User:
                return len(users_by_email)
            return 0

        def offset(self, n):
            return self

        def limit(self, n):
            return self

        def all(self):
            if self.model == User:
                return list(users_by_email.values())
            return []

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
    
    def mock_refresh(obj):
        if isinstance(obj, User):
            # Simulate refresh by populating object from our mock store
            # This assumes obj.id is already set by mock_add
            if str(obj.id) in users_by_id: # Use str(obj.id) for dictionary key lookup
                refreshed_user = users_by_id[str(obj.id)]
                # Copy attributes from the stored mock user to the passed object
                obj.email = refreshed_user.email
                obj.hashed_password = refreshed_user.hashed_password
                obj.full_name = refreshed_user.full_name
                obj.tier = refreshed_user.tier
                obj.is_active = refreshed_user.is_active
                obj.is_verified = refreshed_user.is_verified
                obj.verification_token = refreshed_user.verification_token
                obj.last_login = refreshed_user.last_login
                obj.is_mfa_enabled = refreshed_user.is_mfa_enabled
                obj.mfa_secret = refreshed_user.mfa_secret
            elif obj.email in users_by_email: # Fallback to email if ID not found (e.g. initial add)
                refreshed_user = users_by_email[obj.email]
                obj.id = refreshed_user.id
                obj.hashed_password = refreshed_user.hashed_password # Ensure hashed password is set
                # ... copy other relevant attributes ...
    mock_session.refresh = MagicMock(side_effect=mock_refresh)

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
