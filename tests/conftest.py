import sys
from unittest.mock import MagicMock, AsyncMock, patch

# ============================================================================
# EARLY MOCKS (Immediate Deception for Python 3.14 Compatibility)
# ============================================================================

# Version Info for Compatibility
version_tuple = (4, 6, 0)
version_str = "4.6.0"

# Create a base MagicMock that handles version comparisons
class VersionedMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.VERSION = version_tuple
        self.__version__ = version_str
    def __gt__(self, other):
        if isinstance(other, int): return self.VERSION[0] > other
        if isinstance(other, tuple): return self.VERSION > other
        return super().__gt__(other)
    def __ge__(self, other):
        if isinstance(other, int): return self.VERSION[0] >= other
        if isinstance(other, tuple): return self.VERSION >= other
        return super().__ge__(other)
    def __lt__(self, other):
        if isinstance(other, int): return self.VERSION[0] < other
        if isinstance(other, tuple): return self.VERSION < other
        return super().__lt__(other)
    def __le__(self, other):
        if isinstance(other, int): return self.VERSION[0] <= other
        if isinstance(other, tuple): return self.VERSION <= other
        return super().__le__(other)

# Mock Ray
mock_ray = VersionedMock()
mock_ray.remote = lambda x: x
mock_ray.init = MagicMock()
mock_ray.shutdown = MagicMock()
sys.modules["ray"] = mock_ray
sys.modules["ray.util"] = VersionedMock()
sys.modules["ray.util.iter"] = VersionedMock()

# Mock Numba
mock_numba = VersionedMock()
mock_numba.jit = lambda *args, **kwargs: lambda func: func
mock_numba.njit = lambda *args, **kwargs: lambda func: func
mock_numba.float64 = MagicMock()
mock_numba.int64 = MagicMock()
sys.modules["numba"] = mock_numba
sys.modules["numba.core"] = VersionedMock()
sys.modules["numba.core.decorators"] = VersionedMock()
sys.modules["llvmlite"] = VersionedMock()
sys.modules["llvmlite.binding"] = VersionedMock()

# Mock pandas_ta
sys.modules["pandas_ta"] = VersionedMock()

# Mock torch
mock_torch = VersionedMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.device = MagicMock()
class MockTensor: pass
mock_torch.Tensor = MockTensor
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = VersionedMock()
sys.modules["torch.optim"] = VersionedMock()
sys.modules["torch.utils"] = VersionedMock()
sys.modules["torch.utils.data"] = VersionedMock()

# Mock scikit-learn
sys.modules["sklearn"] = VersionedMock()
sys.modules["sklearn.ensemble"] = VersionedMock()
sys.modules["sklearn.metrics"] = VersionedMock()
sys.modules["sklearn.model_selection"] = VersionedMock()
sys.modules["sklearn.preprocessing"] = VersionedMock()

# Mock mlflow
sys.modules["mlflow"] = VersionedMock()
sys.modules["mlflow.pyfunc"] = VersionedMock()
sys.modules["mlflow.models"] = VersionedMock()
sys.modules["mlflow.pytorch"] = VersionedMock()
sys.modules["mlflow.xgboost"] = VersionedMock()
sys.modules["mlflow.data"] = VersionedMock()

# Mock qiskit
sys.modules["qiskit"] = VersionedMock()
sys.modules["qiskit_aer"] = VersionedMock()

# Mock cvxpy
sys.modules["cvxpy"] = VersionedMock()

# Mock web3
sys.modules["web3"] = VersionedMock()
sys.modules["eth_account"] = VersionedMock()

# Mock prometheus_api_client
sys.modules["prometheus_api_client"] = VersionedMock()

# Mock prefect
sys.modules["prefect"] = VersionedMock()

# Mock pytorch_forecasting
sys.modules["pytorch_forecasting"] = VersionedMock()

# Mock selectolax
sys.modules["selectolax"] = VersionedMock()
sys.modules["selectolax.lexbor"] = VersionedMock()

# Mock stable_baselines3
sys.modules["stable_baselines3"] = VersionedMock()
sys.modules["stable_baselines3.common"] = VersionedMock()
sys.modules["stable_baselines3.common.on_policy_algorithm"] = VersionedMock()
sys.modules["stable_baselines3.common.base_class"] = VersionedMock()
sys.modules["stable_baselines3.common.env_util"] = VersionedMock()
sys.modules["stable_baselines3.common.monitor"] = VersionedMock()

# Mock gymnasium.core
sys.modules["gymnasium.core"] = VersionedMock()

# Mock optuna
sys.modules["optuna"] = VersionedMock()

# Mock authlib
sys.modules["authlib"] = VersionedMock()
sys.modules["authlib.jose"] = VersionedMock()

# Mock onnxruntime
sys.modules["onnxruntime"] = VersionedMock()

# Mock gymnasium
sys.modules["gymnasium"] = VersionedMock()
sys.modules["gym"] = VersionedMock()

# Mock redis (sync and async)
class MockRedisError(Exception): pass

mock_redis_mod = VersionedMock()
mock_redis_mod.client = VersionedMock()
mock_redis_mod.connection = VersionedMock()

class MockRedis(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.VERSION = version_tuple
        self.__version__ = version_str
    def ping(self): return True
    async def get(self, *args, **kwargs): return None
    async def set(self, *args, **kwargs): return True
    async def setex(self, *args, **kwargs): return True
    def pipeline(self):
        m = VersionedMock()
        m.execute = AsyncMock(return_value=[None, 0])
        m.__aenter__ = AsyncMock(return_value=m)
        m.__aexit__ = AsyncMock(return_value=None)
        return m
    @classmethod
    def from_url(cls, *args, **kwargs): return MockRedis()

mock_redis_mod.Redis = MockRedis
mock_redis_mod.StrictRedis = MockRedis
mock_redis_mod.RedisError = MockRedisError
sys.modules["redis"] = mock_redis_mod

mock_async_redis_mod = VersionedMock()
mock_async_redis_mod.Redis = MockRedis
mock_async_redis_mod.RedisError = MockRedisError
sys.modules["redis.asyncio"] = mock_async_redis_mod

import pytest
import os
from _pytest.monkeypatch import MonkeyPatch # Import MonkeyPatch class
import uuid
import re
import jwt
from jwt.exceptions import PyJWTError as JWTError
from src.database.models import User, APIKey
from datetime import datetime, timezone
import numpy as np
import importlib
from fastapi.testclient import TestClient

# Shared stores for mocked database state
_users_by_email = {}
_users_by_id = {}
_api_keys_by_hash = {}

# A global store for mocked JWT payloads
_mock_jwt_payload_store = {}


# ============================================================================
# Pytest Configuration Hook
# ============================================================================


def pytest_configure(config):
    """
    Hook to configure pytest before any tests are run.
    This is where we can inject mocks for module-level imports.
    """
    import os
    import sys
    from unittest.mock import MagicMock
    from _pytest.monkeypatch import MonkeyPatch # Import MonkeyPatch class
    import src.config # Ensure src.config is imported early

    # Create a MonkeyPatch instance for early patching
    mpatch = MonkeyPatch()
    os.environ["TESTING"] = "true"

    # Create a mock settings object early and patch src.config.Settings
    mock_settings = MagicMock()
    # Populate mock_settings with all required values
    mock_settings.DATABASE_URL = "sqlite:///:memory:"
    mock_settings.REDIS_URL = "redis://localhost:6379/0"
    mock_settings.DEBUG = False
    mock_settings.SLOW_QUERY_THRESHOLD_MS = 100
    mock_settings.ENVIRONMENT = "dev"
    mock_settings.LOG_LEVEL = "INFO"
    mock_settings.PROJECT_NAME = "BS-Opt-Test"
    mock_settings.JWT_SECRET = "test-secret-for-hmac"
    mock_settings.JWT_ALGORITHM = "HS256"
    mock_settings.MFA_ENCRYPTION_KEY = "cUMkImRgwyuUNS_WDJPWOnJhlZlB_1cTOEMjtR2TMhU="
    mock_settings.ML_SERVICE_GRPC_URLS = "localhost:50051" # Fix unpacking error
    mock_settings.ML_GRPC_POOL_SIZE = 1
    mock_settings.rate_limit_tiers = {"free": 100, "pro": 1000, "enterprise": 0}

    mpatch.setattr("src.config.Settings", MagicMock(return_value=mock_settings))
    
    import src.config
    src.config.settings = mock_settings
    src.config._settings = mock_settings

    # Mock src.database internals
    mock_db_engine = MagicMock()
    mpatch.setattr("src.database.engine", mock_db_engine)
    mpatch.setattr("src.database.async_engine", AsyncMock())
    mpatch.setattr("src.database.SessionLocal", MagicMock())
    mpatch.setattr("src.database.AsyncSessionLocal", MagicMock())
    
    # Mock async db
    async def mock_get_async_db():
        mock_session = AsyncMock()
        
        # Helper to simulate SQLAlchemy result
        class MockResult:
            def __init__(self, val): self.val = val
            def scalar_one_or_none(self): return self.val
            def scalar(self): return self.val
            def scalars(self):
                m = MagicMock()
                m.unique.return_value.all.return_value = [self.val] if self.val else []
                m.all.return_value = [self.val] if self.val else []
                m.first.return_value = self.val
                return m

        async def mock_execute(query, *args, **kwargs):
            query_str = str(query)
            
            # Simple extraction for common test cases
            if "FROM \"user\"" in query_str or "FROM users" in query_str:
                # Try to find by looking at _users_by_email
                # Very loose matching for tests
                user = None
                if "_users_by_email" in globals() and _users_by_email:
                    # In tests like test_login, it's usually the only user
                    user = list(_users_by_email.values())[-1]
                return MockResult(user)
            return MockResult(None)

        mock_session.execute = AsyncMock(side_effect=mock_execute)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        
        def sync_add(obj):
            if isinstance(obj, User):
                if not obj.id: obj.id = uuid.uuid4()
                obj.is_verified = True # Auto-verify for tests
                if obj.is_mfa_enabled is None: obj.is_mfa_enabled = False
                if not hasattr(obj, 'created_at') or obj.created_at is None:
                    obj.created_at = datetime.now(timezone.utc)
                _users_by_email[obj.email] = obj
                _users_by_id[str(obj.id)] = obj
            elif isinstance(obj, APIKey):
                if not obj.id: obj.id = uuid.uuid4()
                _api_keys_by_hash[obj.key_hash] = obj
        
        mock_session.add = MagicMock(side_effect=sync_add)
        
        yield mock_session

    mpatch.setattr("src.database.get_async_db", mock_get_async_db)


    # Mock Celery delay method globally
    from celery.app.task import Task
    
    Task.delay = MagicMock()
    Task.apply_async = MagicMock()

    # Mock psycopg2
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # Configure mock_cursor to return expected values for common test queries
    def mock_fetchone():
        # If it's the EXISTS query, return True
        return (True,)
    
    def mock_fetchall():
        # If it's the columns query, return expected column names
        return [("symbol",), ("market",), ("source_type",), ("close",)]
        
    mock_cursor.fetchone.side_effect = mock_fetchone
    mock_cursor.fetchall.side_effect = mock_fetchall
    mock_conn.cursor.return_value = mock_cursor
    mock_psycopg2.connect.return_value = mock_conn
    sys.modules["psycopg2"] = mock_psycopg2
    
    # Mock confluent_kafka
    mock_confluent_kafka = MagicMock()
    sys.modules["confluent_kafka"] = mock_confluent_kafka
    sys.modules["confluent_kafka.admin"] = MagicMock()
    sys.modules["confluent_kafka.schema_registry"] = MagicMock()
    sys.modules["confluent_kafka.schema_registry.avro"] = MagicMock()


@pytest.fixture(autouse=True)
def reset_shared_stores():
    """Clear the shared database stores before each test."""
    _users_by_email.clear()
    _users_by_id.clear()
    _api_keys_by_hash.clear()
    _mock_jwt_payload_store.clear()


@pytest.fixture
def mock_redis_and_celery(monkeypatch):
    """Mock Redis and Celery to avoid connection errors in tests."""
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
    monkeypatch.setattr("redis.asyncio.Redis.from_url", MagicMock(return_value=mock_redis))
    monkeypatch.setattr("redis.asyncio.ConnectionPool", mock_pool)
    monkeypatch.setattr("redis.asyncio.ConnectionPool.from_url", MagicMock(return_value=mock_pool))

    # Mock src.utils.cache functions
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)

    # JWT Mocking (should use the dummy keys from mock_settings)
    def mock_jose_jwt_encode(payload, key, algorithm, headers=None):
        # Generate a unique token string for each payload
        jti = payload.get("jti", str(uuid.uuid4()))
        token_type = payload.get("type", "access")
        fake_token_string = f"dummy_jwt_token_{token_type}_{jti}"
        _mock_jwt_payload_store[fake_token_string] = payload
        return fake_token_string

    def mock_jose_jwt_decode(token, key, algorithms, options=None):
        payload = _mock_jwt_payload_store.get(token)
        if payload:
            return payload
        raise JWTError("Invalid token or not found in mock store")

    monkeypatch.setattr("jose.jwt.encode", mock_jose_jwt_encode)
    monkeypatch.setattr("jose.jwt.decode", mock_jose_jwt_decode)

    # Mock TokenBlacklist methods
    from src.security.auth import token_blacklist
    monkeypatch.setattr(token_blacklist, "contains", AsyncMock(return_value=False))
    monkeypatch.setattr(token_blacklist, "add", AsyncMock(return_value=None)) # Mock add to do nothing


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
                user = _users_by_email.get(self.filter_val)
                if not user:
                    user = _users_by_id.get(self.filter_val)
                return user
            if self.model == APIKey:
                return _api_keys_by_hash.get(self.filter_val)
            return None

        def count(self):
            if self.model == User:
                return len(_users_by_email)
            return 0

        def offset(self, n):
            return self

        def limit(self, n):
            return self

        def all(self):
            if self.model == User:
                return list(_users_by_email.values())
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
            _users_by_email[obj.email] = obj
            _users_by_id[str(obj.id)] = obj
            _users_by_id[obj.id] = obj
        elif isinstance(obj, APIKey):
            if not obj.id:
                obj.id = uuid.uuid4()
            _api_keys_by_hash[obj.key_hash] = obj

    mock_session.add = MagicMock(side_effect=mock_add)
    mock_session.commit = MagicMock()
    
    def mock_refresh(obj):
        if isinstance(obj, User):
            if str(obj.id) in _users_by_id:
                refreshed_user = _users_by_id[str(obj.id)]
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
            elif obj.email in _users_by_email:
                refreshed_user = _users_by_email[obj.email]
                obj.id = refreshed_user.id
                obj.hashed_password = refreshed_user.hashed_password
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


@pytest.fixture
def unmocked_config_settings(monkeypatch):
    """
    Fixture to temporarily un-mock src.config.settings and src.config._settings
    for tests that need to load the real Settings instance.
    """
    import src.config
    import sys
    
    # Save current mocked state (which are MagicMocks from pytest_configure)
    mocked_settings_instance = src.config.settings
    mocked_Settings_class = src.config.Settings
    mocked_get_settings_func = src.config.get_settings
    
    # Force reload of the module to get real classes
    if "src.config" in sys.modules:
        del sys.modules["src.config"]
    
    import src.config
    importlib.reload(src.config)
    src.config._initialize_settings()

    yield

    # Restore mocks manually because pytest_configure won't run again
    # We use setattr on the reloaded module object
    src.config.settings = mocked_settings_instance
    src.config._settings = mocked_settings_instance
    src.config.Settings = mocked_Settings_class
    src.config.get_settings = mocked_get_settings_func