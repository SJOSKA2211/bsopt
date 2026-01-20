import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Iterable

import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os

# Add the project root to sys.path to ensure src is discoverable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Set environment variables BEFORE importing src modules
os.environ["TESTING"] = "true"
os.environ["JWT_ALGORITHM"] = "RS256"
os.environ["JWT_PRIVATE_KEY"] = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCxnuu6P25C2gbg
K4Q6QKgFcjHCJ/+M3raqsvz1kiZCpIgP/qKGKu70j4KcwsJrWPxop10x/KhbHpSv
8OCnF3kMWCNxHw4QLJ/u5Aufbe++Irayic9j9lIn1cyWcWk1E9qtCWJQ/zBdgucC
2stZdLMpMWmAzIsVcqTgSU2OosGwf6Uf77dqCLSXFKf2+khn8BiH2mXAK/bLFGyh
4biPlzaTSIcNmdnJPH4iWzC9WJARqjr9ZRGkWKbgn+7lGJeHRxf6axRv8t+p9isH
xuQzALa8iO6HhxiF3H5cduDPghU6DkmKl6V/bSB5qFsP2FHjWfMnPipTvbeuEKCk
nzIEptDJAgMBAAECggEABr5mLE7auJqPDsVQMqcASh6lEX5Tx00wacg8bvVy0u5s
xRCxqn7oTixBtQJ2/7zj7nRGR07UtIrzcb+nQ+jR/Xw+Mj4P2mDbXKZXY6D4rIMk
ZSBy2ZSBV4ZYS3D4Yd3EXHQCAdnChBZjf3n/pQCXic2YuB1r/W86H9LgqTT4PiN2
V/WOf5eiBZsQVj4bsOlMPdO+6hmwnx86raUbKE+FJYAhvqWlesjUeNm4r6Yfotct
Ig7IltZnUvBcYgxKr2tvzzlUtsiiCx4qIXphqO9zVzg9U4B5Wc4X72xHVP0kZG7O
afxFZSkLzwPfSqsVJXXkW6xF0GHXw30H8raO648/9QKBgQDWObYgEtq58GOD5peN
p0yHgwZVpRoko55Aj0LmlWzpE2Ny5yEvV/EZNSZi+aaCN+a6Wqgyfgoofg2hIVZe
nvdUvXtCeLvw2Fe6u/hbEzhIXKOclAQ6cwtisyfYwSfi8j6oMXuhuukYthUdhRAp
uddaUy+jVzDiiN6EPUlHWCqm5QKBgQDUQeD2G1BHbd2eQhWFsDRVaSQiBfe+KaM0
SmfMQAFBsQtnQGIsNifzP1bAZyVHs1shPRJaUIPs0i5e53eOaF0C1EovHW0EFwSn
UvtNMAb0uRZqpOcrGsUzRX1asgrv1fMxOffWdFmK6JGjzd/3wEGedFyQ2O+wTfwG
xeKLd/igFQKBgA1Tp8XVBnBcyQQSm0j/qF4hw4oebELtPtILV4EauJzDTQN/52uX
j/MegFXV7ArbyWm8bAxAFQex1803UrUuNHq8EufutNplywdd3DRmPLEbuj3qY1zz
fTjVplvwoDeZFFbIRUWpaAjWgvfEKF5AJmqDFEqYCP1+wED/wwhCLt0VAoGBAJ/K
LIn5y/jKC9HNHBi1quA1s97tMTF2dQezj+qisI98sgH75Sw1ZOPpZeyYeec9bbhb
GorlHDvXitMlW8rYZFTx7hsEAwLWNUml3cuhAUuQXwDPvbukfpp3kMQLTtJ49Yi0
hBBtLM+2/5UaMqZ3lK6uGNVuixrlynpq1H58Ra51AoGALlRt4xPMOMk+4F9mVk2R
3aZczsnsfi3o6En0hHSZnoftJZHxAGEFPC4p8zavMJrtUM/EE1fKnYR8+JJyYFCA
pbZ7HoWVhxY/QluS6GdDs74A90BDLFTI9SNfBcYf7Gn981i5+ofysSpoII3SEj14
/5M82ANrBoH+QMxDbY2rnLs=
-----END PRIVATE KEY-----"""
os.environ["JWT_PUBLIC_KEY"] = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsZ7ruj9uQtoG4CuEOkCo
BXIxwif/jN62qrL89ZImQqSID/6ihiru9I+CnMLCa1j8aKddMfyoWx6Ur/Dgpxd5
DFgjcR8OECyf7uQLn23vviK2sonPY/ZSJ9XMlnFpNRParQliUP8wXYLnAtrLWXSz
KTFpgMyLFXKk4ElNjqLBsH+lH++3agi0lxSn9vpIZ/AYh9plwCv2yxRsoeG4j5c2
k0iHDZnZyTx+IlswvViQEao6/WURpFim4J/u5RiXh0cX+msUb/LfqfYrB8bkMwC2
vIjuh4cYhdx+XHbgz4IVOg5Jipelf20geahbD9hR41nzJz4qU723rhCgpJ8yBKbQ
yQIDAQAB
-----END PUBLIC KEY-----"""

from src.pricing.black_scholes import BlackScholesEngine, BSParameters
# from src.pricing.monte_carlo import MCConfig, MonteCarloEngine

# ============================================================================
# Black-Scholes Parameter Fixtures
# ============================================================================

@pytest.fixture
def atm_params() -> BSParameters:
    return BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02)

@pytest.fixture
def itm_call_params() -> BSParameters:
    return BSParameters(spot=110.0, strike=100.0, maturity=1.0, volatility=0.25, rate=0.05, dividend=0.02)

@pytest.fixture
def otm_call_params() -> BSParameters:
    return BSParameters(spot=90.0, strike=100.0, maturity=1.0, volatility=0.25, rate=0.05, dividend=0.02)

@pytest.fixture
def deep_itm_call_params() -> BSParameters:
    return BSParameters(spot=150.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02)

@pytest.fixture
def deep_otm_call_params() -> BSParameters:
    return BSParameters(spot=60.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.02)

@pytest.fixture
def short_maturity_params() -> BSParameters:
    return BSParameters(spot=100.0, strike=100.0, maturity=30.0 / 365.0, volatility=0.20, rate=0.05, dividend=0.02)

@pytest.fixture
def long_maturity_params() -> BSParameters:
    return BSParameters(spot=100.0, strike=100.0, maturity=5.0, volatility=0.20, rate=0.05, dividend=0.02)

@pytest.fixture
def high_volatility_params() -> BSParameters:
    return BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.50, rate=0.05, dividend=0.02)

@pytest.fixture
def low_volatility_params() -> BSParameters:
    return BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.10, rate=0.05, dividend=0.02)

@pytest.fixture
def zero_dividend_params() -> BSParameters:
    return BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.0)

@pytest.fixture
def high_dividend_params() -> BSParameters:
    return BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=0.05, dividend=0.08)

@pytest.fixture
def negative_rate_params() -> BSParameters:
    return BSParameters(spot=100.0, strike=100.0, maturity=1.0, volatility=0.20, rate=-0.01, dividend=0.0)

# @pytest.fixture
# def mc_config_fast() -> MCConfig:
#     return MCConfig(n_paths=10000, n_steps=100, antithetic=True, control_variate=False, seed=42)

# @pytest.fixture
# def mc_config_accurate() -> MCConfig:
#     return MCConfig(n_paths=100000, n_steps=252, antithetic=True, control_variate=True, seed=42)

# @pytest.fixture
# def mc_config_minimal() -> MCConfig:
#     return MCConfig(n_paths=1000, n_steps=50, antithetic=False, control_variate=False, seed=42)

@pytest.fixture
def bs_engine() -> BlackScholesEngine:
    return BlackScholesEngine()

# @pytest.fixture
# def mc_engine_fast(mc_config_fast: MCConfig) -> MonteCarloEngine:
#     return MonteCarloEngine(mc_config_fast)

# @pytest.fixture
# def mc_engine_accurate(mc_config_accurate: MCConfig) -> MonteCarloEngine:
#     return MonteCarloEngine(mc_config_accurate)

# ============================================================================
# Global Isolation Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def isolate_dependency_overrides():
    """Ensure that dependency overrides are isolated per test."""
    from src.api.main import app
    original_overrides = app.dependency_overrides.copy()
    yield
    app.dependency_overrides = original_overrides

@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset global circuit breakers before each test to prevent state leakage."""
    from src.utils.circuit_breaker import pricing_circuit, db_circuit, CircuitState
    if hasattr(pricing_circuit, 'reset'):
        pricing_circuit.reset()
    else:
        pricing_circuit.state = CircuitState.CLOSED
        pricing_circuit.failure_count = 0
        pricing_circuit.last_failure_time = 0
    if hasattr(db_circuit, 'reset'):
        db_circuit.reset()
    else:
        db_circuit.state = CircuitState.CLOSED
        db_circuit.failure_count = 0
        db_circuit.last_failure_time = 0
    yield

@pytest.fixture(autouse=True)
def cleanup_lazy_import():
    """Clean up lazy import state to prevent cross-test leakage."""
    from src.utils.lazy_import import reset_import_stats
    reset_import_stats()
    yield

# ============================================================================
# API & Auth Fixtures
# ============================================================================

@pytest.fixture
def mock_auth_dependency():
    """Fixture to override authentication dependency for functional tests."""
    from src.api.main import app
    from src.security.auth import get_current_active_user, get_api_key, get_current_user_flexible, get_current_user
    from src.utils.cache import get_redis, get_redis_client
    from src.security.rate_limit import rate_limit
    from src.database.models import User
    
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.publish = AsyncMock(return_value=1)
    mock_redis.exists = AsyncMock(return_value=False)
    
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=(1, True))
    mock_redis.pipeline.return_value = mock_pipe

    async def mock_user():
        return User(id=uuid.uuid4(), email="test@example.com", full_name="Test User",
                    is_active=True, tier="pro", is_verified=True, is_mfa_enabled=False,
                    created_at=datetime.now(timezone.utc), last_login=datetime.now(timezone.utc))
    
    async def mock_api_key():
        return User(id=uuid.uuid4(), email="apikey@example.com", full_name="API Key User",
                    is_active=True, tier="enterprise", is_verified=True, is_mfa_enabled=False,
                    created_at=datetime.now(timezone.utc), last_login=datetime.now(timezone.utc))
    
    app.dependency_overrides[get_current_active_user] = mock_user
    app.dependency_overrides[get_api_key] = mock_api_key
    app.dependency_overrides[get_current_user_flexible] = mock_user
    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[get_redis] = lambda: mock_redis
    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    app.dependency_overrides[rate_limit] = lambda: None
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def api_client(mock_db_session):
    from src.api.main import app
    from src.database import get_db
    app.dependency_overrides[get_db] = lambda: mock_db_session
    return TestClient(app)

@pytest.fixture
def mock_db_session():
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
            for arg in args: self._extract_filter_val(arg)
            return self
        def _extract_filter_val(self, expr):
            if hasattr(expr, 'right') and hasattr(expr.right, 'value'):
                val = expr.right.value
                if isinstance(val, str) and len(val) == 64: self.filter_val = val
                elif self.filter_val is None: self.filter_val = val
            elif hasattr(expr, 'clauses'):
                for clause in expr.clauses: self._extract_filter_val(clause)
        def first(self):
            if self.model == User:
                return users_by_email.get(self.filter_val) or users_by_id.get(str(self.filter_val))
            if self.model == APIKey: return api_keys_by_hash.get(self.filter_val)
            return None
        def all(self): return list(users_by_email.values()) if self.model == User else []
        def count(self): return len(users_by_email) if self.model == User else 0
        def offset(self, n): return self
        def limit(self, n): return self

    mock_session.query = MagicMock(side_effect=lambda model: MockQuery(model))
    def mock_add(obj):
        if isinstance(obj, User):
            if not obj.id: obj.id = uuid.uuid4()
            obj.is_verified = True
            obj.is_active = True
            obj.tier = getattr(obj, "tier", None) or "free"
            obj.is_mfa_enabled = getattr(obj, "is_mfa_enabled", None) or False
            obj.created_at = getattr(obj, "created_at", None) or datetime.now(timezone.utc)
            obj.last_login = getattr(obj, "last_login", None) or datetime.now(timezone.utc)
            obj.full_name = getattr(obj, "full_name", None) or "Mock User"
            users_by_email[obj.email] = obj
            users_by_id[str(obj.id)] = obj
        elif isinstance(obj, APIKey):
            if not obj.id: obj.id = uuid.uuid4()
            api_keys_by_hash[obj.key_hash] = obj
    mock_session.add.side_effect = mock_add
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    return mock_session

# ============================================================================
# Pytest Config Hooks
# ============================================================================

def pytest_configure(config):
    import os
    os.environ["TESTING"] = "true"
    mock_settings = MagicMock()
    mock_settings.ENVIRONMENT = os.environ["TESTING"]
    mock_settings.JWT_PRIVATE_KEY = os.environ["JWT_PRIVATE_KEY"]
    mock_settings.JWT_PUBLIC_KEY = os.environ["JWT_PUBLIC_KEY"]
    mock_settings.JWT_ALGORITHM = os.environ["JWT_ALGORITHM"]
    mock_settings.DATABASE_URL = "sqlite:///:memory:"
    mock_settings.DEBUG = False
    mock_settings.SLOW_QUERY_THRESHOLD_MS = 100
    mock_settings.BCRYPT_ROUNDS = 12
    mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    mock_settings.REFRESH_TOKEN_EXPIRE_DAYS = 7
    mock_settings.PASSWORD_MIN_LENGTH = 8
    mock_settings.rate_limit_tiers = {"free": 100, "pro": 1000, "enterprise": 0}
    mock_settings.JWT_SECRET = "test-secret-key-that-is-long-enough-32-chars"
    mock_settings.ML_SERVICE_URL = "http://ml-service"
    mock_settings.DASK_LOCAL_CLUSTER_THREADS_PER_WORKER = 2
    mock_settings.DASK_ARRAY_DEFAULT_CHUNKS_FRACTION = 4
    mock_settings.ML_TRAINING_PROMOTE_THRESHOLD_R2 = 0.98
    mock_settings.ML_TRAINING_NN_EPOCHS = 1
    mock_settings.ML_TRAINING_OPTUNA_TRIALS = 1
    mock_settings.ML_TRAINING_KFOLD_SPLITS = 2
    
    # Patch src.config.settings directly in sys.modules
    # This ensures that any module importing src.config gets this mocked object
    if "src.config" in sys.modules:
        config_mod = sys.modules["src.config"]
        # Update existing settings object if it exists
        if hasattr(config_mod, 'settings'):
            for key, val in mock_settings.__dict__['_mock_children'].items():
                try:
                    setattr(config_mod.settings, key, val)
                except:
                    pass
            # Manually set critical ones
            config_mod.settings.BCRYPT_ROUNDS = 12
            config_mod.settings.JWT_ALGORITHM = "RS256"
            config_mod.settings.ENVIRONMENT = "test"
            config_mod.settings.is_production = False
            config_mod.settings.PASSWORD_MIN_LENGTH = 8
            config_mod.settings.rate_limit_tiers = {"free": 100, "pro": 1000, "enterprise": 0}
        
        config_mod.settings = mock_settings
        config_mod._settings = mock_settings
        config_mod.get_settings = lambda: mock_settings


    # Global Mocks
    from celery.app.task import Task
    Task.delay = MagicMock()
    Task.apply_async = MagicMock()

    sys.modules["confluent_kafka"] = MagicMock()
    sys.modules["confluent_kafka.admin"] = MagicMock()
    sys.modules["confluent_kafka.schema_registry"] = MagicMock()
    sys.modules["confluent_kafka.schema_registry.avro"] = MagicMock()
    
    mock_redis = MagicMock()
    sys.modules["redis"] = mock_redis
    sys.modules["redis.asyncio"] = mock_redis

# ============================================================================
# Helper Functions
# ============================================================================

def assert_close(actual: float, expected: float, tolerance: float, message: str = "") -> None:
    diff = abs(actual - expected)
    assert diff < tolerance, f"{message}: {actual} != {expected} (diff: {diff})"

@pytest.fixture
def percentage_tolerance() -> float: return 0.01
@pytest.fixture
def normal_tolerance() -> float: return 1e-6
@pytest.fixture
def tight_tolerance() -> float: return 1e-10
@pytest.fixture
def loose_tolerance() -> float: return 1e-3
@pytest.fixture
def known_values() -> Dict[str, Any]:
    return {
        "hull_example_13_6": {"params": BSParameters(spot=42.0, strike=40.0, maturity=0.5, volatility=0.20, rate=0.10, dividend=0.0), "call_price": 4.76},
        "hull_example_17_1": {"params": BSParameters(spot=50.0, strike=50.0, maturity=1.0, volatility=0.30, rate=0.05, dividend=0.0), "call_price": 7.04}
    }