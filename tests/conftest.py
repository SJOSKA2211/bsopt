import pytest
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from _pytest.monkeypatch import MonkeyPatch # Import MonkeyPatch class
import uuid
from jose import JWTError
from src.database.models import User, APIKey
from datetime import datetime, timezone
import numpy as np
import importlib
from fastapi.testclient import TestClient

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

    # Create a mock settings object early and patch src.config.Settings
    mock_settings = MagicMock()
    # Populate mock_settings with all required values
    mock_settings.DATABASE_URL = "sqlite:///:memory:"
    mock_settings.DEBUG = False
    mock_settings.SLOW_QUERY_THRESHOLD_MS = 100
    mock_settings.ENVIRONMENT = "dev"
    mock_settings.LOG_LEVEL = "INFO"

    mock_settings.JWT_PRIVATE_KEY_PATH = ""
    mock_settings.JWT_PUBLIC_KEY_PATH = ""
    mock_settings.JWT_PRIVATE_KEY = "test-secret-key"
    mock_settings.JWT_PUBLIC_KEY = "test-secret-key"
    mock_settings.JWT_ALGORITHM = "HS256"
    mock_settings.REDIS_HOST = "localhost"
    mock_settings.REDIS_PORT = 6379
    mock_settings.REDIS_DB = 0
    mock_settings.REDIS_PASSWORD = None
    mock_settings.CORS_ORIGINS = ["http://localhost"]
    mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    mock_settings.REFRESH_TOKEN_EXPIRE_DAYS = 7
    mock_settings.BCRYPT_ROUNDS = 12 # Explicitly set to an integer
    mock_settings.PASSWORD_MIN_LENGTH = 8
    mock_settings.MFA_ENCRYPTION_KEY = "cUMkImRgwyuUNS_WDJPWOnJhlZlB_1cTOEMjtR2TMhU="
    mock_settings.ML_SERVICE_URL = "http://ml-service"
    mock_settings.rate_limit_tiers = {"free": 100, "pro": 1000, "enterprise": 0}
    mock_settings.ML_TRAINING_TEST_SIZE = 0.2
    mock_settings.ML_TRAINING_RANDOM_STATE = 42

    mpatch.setattr("src.config.Settings", MagicMock(return_value=mock_settings))
    mpatch.setattr("src.config.get_settings", MagicMock(return_value=mock_settings))

    import src.config
    src.config.settings = mock_settings
    src.config._settings = mock_settings

    # Mock src.database internals
    mock_db_engine = MagicMock()
    mpatch.setattr("src.database.get_engine", MagicMock(return_value=mock_db_engine))
    mpatch.setattr("src.database._SessionLocal", MagicMock())
    mpatch.setattr("src.database._initialize_db_components", MagicMock())


    # Mock Celery delay method globally
    from celery.app.task import Task
    
    Task.delay = MagicMock()
    Task.apply_async = MagicMock()

    # Mock psycopg2
    mock_psycopg2 = MagicMock()
    sys.modules["psycopg2"] = mock_psycopg2

    # Mock numba
    mock_numba = MagicMock()
    mock_numba.jit = lambda *args, **kwargs: lambda func: func
    mock_numba.njit = lambda *args, **kwargs: lambda func: func
    mock_numba.float64 = MagicMock()
    mock_numba.int32 = MagicMock()
    mock_numba.int64 = MagicMock()
    sys.modules["numba"] = mock_numba
    
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
    monkeypatch.setattr("redis.asyncio.ConnectionPool", mock_pool)
    monkeypatch.setattr("redis.asyncio.ConnectionPool.from_url", MagicMock(return_value=mock_pool))

    # Mock src.utils.cache functions
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)

    # JWT Mocking (should use the dummy keys from mock_settings)
    _mock_jwt_payload_store = {} # Reset for each fixture call

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