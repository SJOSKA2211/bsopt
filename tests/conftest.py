import sys
from unittest.mock import MagicMock, AsyncMock, patch

# ============================================================================ 
# EARLY MOCKS (Immediate Deception for Python 3.14 Compatibility)
# ============================================================================ 

# Version Info for Compatibility
version_tuple = (4, 6, 0)
version_str = "4.6.0"

# Create a base MagicMock that handles version comparisons and iteration
class VersionedMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.VERSION = version_tuple
        self.__version__ = version_str
        self.__path__ = []
        # 🚀 LEVEL 5: Spec Hallucination
        import importlib.machinery
        self.__spec__ = importlib.machinery.ModuleSpec(self._mock_name or "mock", None)
    
    def __gt__(self, other):
        if isinstance(other, int): return self.VERSION[0] > other
        return super().__gt__(other)
    def __ge__(self, other):
        if isinstance(other, int): return self.VERSION[0] >= other
        return super().__ge__(other)
    
    # Handle unpacking (e.g. host, port = urls.split(":"))
    def __iter__(self):
        return iter(["mock_host", "50051"])
    
    def split(self, sep=None, maxsplit=-1):
        return ["mock_host", "50051"]

# Mock heavy libraries
for mod in [
    "stable_baselines3", "stable_baselines3.common", "stable_baselines3.common.noise",
    "stable_baselines3.common.callbacks", "stable_baselines3.common.on_policy_algorithm",
    "stable_baselines3.common.base_class", "stable_baselines3.common.env_util",
    "stable_baselines3.common.monitor", "stable_baselines3.common.torch_layers",
    "stable_baselines3.common.policies", "stable_baselines3.common.distributions",
    "stable_baselines3.common.vec_env", "stable_baselines3.common.preprocessing",
    "stable_baselines3.td3", "stable_baselines3.td3.policies",
    "stable_baselines3.sac", "stable_baselines3.sac.policies",
    "stable_baselines3.ppo", "stable_baselines3.ppo.policies",
    "gymnasium", "gymnasium.core", "gymnasium.spaces", "gymnasium.envs", "optuna",
    "ray", "ray.tune", "ray.air", "ray.train", "ray.serve", 
    "ray.tune.search", "ray.tune.search.optuna", "ray.tune.schedulers",
    "authlib", "authlib.jose", "onnxruntime", "sklearn", "sklearn.ensemble",
    "sklearn.metrics", "sklearn.model_selection", "sklearn.preprocessing",
    "mlflow", "mlflow.pyfunc", "mlflow.models", "mlflow.pytorch", "mlflow.xgboost",
    "mlflow.data", "mlflow.tracking", "qiskit", "qiskit_aer", "qiskit.circuit",
    "qiskit.circuit.library", "cvxpy", "web3", "web3.providers", "eth_account",
    "prometheus_api_client", "prefect", "pytorch_forecasting", "pytorch_forecasting.data",
    "pytorch_forecasting.metrics", "selectolax", "selectolax.lexbor", "pandas_ta",
    "xgboost", "xgboost.dask", "lightning", "lightning.pytorch", "lightning.pytorch.callbacks",
    "flwr", "confluent_kafka.schema_registry", "confluent_kafka.schema_registry.avro"
]:
    sys.modules[mod] = VersionedMock(_mock_name=mod)

# Mock torch specifically
mock_torch = VersionedMock(_mock_name="torch")
mock_torch.cuda.is_available.return_value = False
mock_torch.device = MagicMock()
class MockTensor: pass
mock_torch.Tensor = MockTensor
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = VersionedMock(_mock_name="torch.nn")
sys.modules["torch.optim"] = VersionedMock(_mock_name="torch.optim")
sys.modules["torch.utils"] = VersionedMock(_mock_name="torch.utils")
sys.modules["torch.utils.data"] = VersionedMock(_mock_name="torch.utils.data")
sys.modules["torch.distributed"] = VersionedMock(_mock_name="torch.distributed")
sys.modules["torch.distributions"] = VersionedMock(_mock_name="torch.distributions")

# Mock redis (sync and async)
class MockRedisError(Exception): pass
mock_redis_mod = VersionedMock(_mock_name="redis")
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
        m = VersionedMock(_mock_name="pipeline")
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
mock_async_redis_mod = VersionedMock(_mock_name="redis.asyncio")
mock_async_redis_mod.Redis = MockRedis
sys.modules["redis.asyncio"] = mock_async_redis_mod

import pytest
import os
from _pytest.monkeypatch import MonkeyPatch
import uuid
import re
import jwt
from jwt.exceptions import PyJWTError as JWTError
from src.database.models import User, APIKey
from datetime import datetime, timezone
import numpy as np
import importlib
from fastapi.testclient import TestClient

_users_by_email = {}
_users_by_id = {}
_api_keys_by_hash = {}
_mock_jwt_payload_store = {}

def pytest_configure(config):
    mpatch = MonkeyPatch()
    os.environ["TESTING"] = "true"
    mock_settings = MagicMock()
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
    # Use real strings for properties that are split
    mock_settings.ML_SERVICE_GRPC_URL = "localhost:50051"
    mock_settings.ML_SERVICE_GRPC_URLS = "localhost:50051"
    mock_settings.ML_GRPC_POOL_SIZE = 1
    mock_settings.rate_limit_tiers = {"free": 100, "pro": 1000, "enterprise": 0}
    
    mpatch.setattr("src.config.Settings", MagicMock(return_value=mock_settings))
    import src.config
    src.config.settings = mock_settings
    src.config._settings = mock_settings

    mpatch.setattr("src.database.engine", MagicMock())
    mpatch.setattr("src.database.async_engine", AsyncMock())
    mpatch.setattr("src.database.SessionLocal", MagicMock())
    mpatch.setattr("src.database.AsyncSessionLocal", MagicMock())
    
    async def mock_get_async_db():
        mock_session = AsyncMock()
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
            user = list(_users_by_email.values())[-1] if _users_by_email else None
            return MockResult(user)
        mock_session.execute = AsyncMock(side_effect=mock_execute)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        def sync_add(obj):
            if isinstance(obj, User):
                if not obj.id: obj.id = uuid.uuid4()
                obj.is_verified = True
                _users_by_email[obj.email] = obj
                _users_by_id[str(obj.id)] = obj
        mock_session.add = MagicMock(side_effect=sync_add)
        yield mock_session
    mpatch.setattr("src.database.get_async_db", mock_get_async_db)

    from celery.app.task import Task
    Task.delay = MagicMock()
    Task.apply_async = MagicMock()

    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (True,)
    mock_cursor.fetchall.return_value = [("symbol",), ("market",), ("source_type",), ("close",)]
    mock_conn.cursor.return_value = mock_cursor
    mock_psycopg2.connect.return_value = mock_conn
    sys.modules["psycopg2"] = mock_psycopg2
    
    mock_confluent_kafka = VersionedMock(_mock_name="confluent_kafka")
    sys.modules["confluent_kafka"] = mock_confluent_kafka
    sys.modules["confluent_kafka.admin"] = VersionedMock(_mock_name="confluent_kafka.admin")
    sys.modules["confluent_kafka.schema_registry"] = VersionedMock(_mock_name="confluent_kafka.schema_registry")

@pytest.fixture(autouse=True)
def reset_shared_stores():
    _users_by_email.clear()
    _users_by_id.clear()
    _api_keys_by_hash.clear()
    _mock_jwt_payload_store.clear()

@pytest.fixture
def mock_redis_and_celery(monkeypatch):
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.publish = AsyncMock(return_value=1)
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr("redis.asyncio.Redis", MagicMock(return_value=mock_redis))
    monkeypatch.setattr("src.utils.cache.get_redis", lambda: mock_redis)

@pytest.fixture
def api_client(mock_db_session):
    from src.api.main import app
    from src.database import get_db
    app.dependency_overrides[get_db] = lambda: mock_db_session
    return TestClient(app)

@pytest.fixture
def mock_db_session():
    mock_session = MagicMock()
    class MockQuery:
        def __init__(self, model): self.model = model
        def filter(self, *args, **kwargs): return self
        def first(self): return list(_users_by_email.values())[-1] if _users_by_email else None
        def all(self): return list(_users_by_email.values())
    mock_session.query = MagicMock(side_effect=lambda m: MockQuery(m))
    def mock_add(obj):
        if isinstance(obj, User):
            if not obj.id: obj.id = uuid.uuid4()
            _users_by_email[obj.email] = obj
    mock_session.add = MagicMock(side_effect=mock_add)
    mock_session.commit = MagicMock()
    return mock_session

@pytest.fixture
def tight_tolerance(): return 1e-10
@pytest.fixture
def normal_tolerance(): return 1e-6
@pytest.fixture
def loose_tolerance(): return 1e-3
@pytest.fixture
def percentage_tolerance(): return 0.01

@pytest.fixture
def unmocked_config_settings(monkeypatch):
    import src.config
    mocked_settings_instance = src.config.settings
    if "src.config" in sys.modules: del sys.modules["src.config"]
    importlib.reload(src.config)
    yield
    src.config.settings = mocked_settings_instance