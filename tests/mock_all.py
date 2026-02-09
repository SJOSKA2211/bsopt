import sys
print("DEBUG: mock_all.py start")
from unittest.mock import AsyncMock, MagicMock


#  ADVANCED HIERARCHICAL MOCKING
def mock_module(name):
    if name not in sys.modules:
        # Check if parent is already mocked as MagicMock
        parts = name.split('.')
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            if parent in sys.modules and isinstance(sys.modules[parent], MagicMock):
                # If parent is a MagicMock, we should ensure the child is also accessible as attribute
                child = parts[i]
                m = MagicMock(_mock_name=name)
                setattr(sys.modules[parent], child, m)
                sys.modules[name] = m
                return m

        # Create a mock object that behaves like a package if it has submodules in MOCK_IF_MISSING
        m = MagicMock(_mock_name=name)
        m.__path__ = []
        # Create a spec to satisfy find_spec calls
        try:
            import importlib.machinery
            m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        except:
            pass
        sys.modules[name] = m
    return sys.modules[name]

# Only mock what's ABSOLUTELY necessary and likely to be missing
MOCK_IF_MISSING = [
    "cvxpy", "confluent_kafka", "confluent_kafka.admin", "confluent_kafka.schema_registry", 
    "confluent_kafka.schema_registry.avro",
    "selectolax", "selectolax.lexbor", "dask", "dask.distributed", "cvxopt", "xgboost.dask",
    "stable_baselines3", "stable_baselines3.common", "stable_baselines3.common.torch_layers",
    "stable_baselines3.td3", "stable_baselines3.td3.policies", "stable_baselines3.common.buffers",
    "stable_baselines3.common.callbacks", "stable_baselines3.common.noise",
    "gymnasium", "mlflow", "mlflow.tracking", "mlflow.pytorch", "mlflow.pyfunc", "mlflow.xgboost",
    "faker", "prometheus_api_client", "xgboost", 
    "prefect", "lightning", "lightning.pytorch", "lightning.pytorch.callbacks",
    "sendgrid", "sendgrid.helpers", "sendgrid.helpers.mail",
    "rich", "rich.box", "rich.console", "rich.panel", "rich.table", "rich.live", "rich.layout",
    "torch_geometric", "torch_geometric.nn", "torch_geometric.data",
    "pytorch_forecasting", "pytorch_forecasting.data", "pytorch_forecasting.metrics", "onnxruntime", "flwr",
    "ray.tune", "ray.tune.schedulers", "ray.tune.search", "ray.tune.search.optuna", "ray.air"
]

# Note: qiskit, qiskit_aer, flwr, numba, onnxruntime are now installed in God-Mode venv

heavy_prefixes = ["faker", "mlflow", "stable_baselines3", "xgboost", "confluent_kafka", "lightning", "prefect", "dask", "cvxopt", "cvxpy", "flwr", "pytorch_forecasting", "onnxruntime", "ray.tune", "ray.air"]

for mod in MOCK_IF_MISSING:
    try:
        # Don't try to import heavy ones, just mock
        if any(mod.startswith(p) for p in heavy_prefixes):
             raise ImportError
        if mod in sys.modules:
            continue
        print(f"DEBUG: Attempting to import {mod}")
        __import__(mod)
        print(f"DEBUG: Successfully imported {mod}")
    except (ImportError, Exception) as e:
        print(f"DEBUG: Failed to import {mod}: {e}")
        m = mock_module(mod)
        # Specific attribute fixes for mocks
        if mod == "xgboost":
            m.DMatrix = MagicMock()
        if mod == "faker":
            m.Faker = MagicMock(return_value=MagicMock())
        if mod == "mlflow":
            m.set_tracking_uri = MagicMock()
            m.start_run = MagicMock()
            m.end_run = MagicMock()

# Force Mock Ray (Heavy dependency)
if "ray" not in sys.modules:
    ray_mock = MagicMock(_mock_name="ray")
    ray_mock.__path__ = []
    ray_mock.init = MagicMock()
    ray_mock.remote = lambda x: x # Decorator pass-through
    ray_mock.get = MagicMock(return_value=None)
    ray_mock.put = MagicMock()
    ray_mock.shutdown = MagicMock()
    ray_mock.is_initialized = MagicMock(return_value=False)
    
    # Submodules
    ray_mock.train = MagicMock()
    ray_mock.train.torch = MagicMock()
    ray_mock.train.torch.prepare_model = MagicMock(side_effect=lambda x: x)
    ray_mock.train.torch.prepare_data_loader = MagicMock(side_effect=lambda x: x)
    ray_mock.train.report = MagicMock()
    ray_mock.train.get_context = MagicMock()
    ray_mock.train.get_context.return_value.get_local_rank.return_value = 0
    
    sys.modules["ray"] = ray_mock
    sys.modules["ray.train"] = ray_mock.train
    sys.modules["ray.train.torch"] = ray_mock.train.torch

# Ensure torch mock has version if it ends up being mocked
if "torch" in sys.modules and isinstance(sys.modules["torch"], MagicMock):
    sys.modules["torch"].__version__ = "2.0.0"
    sys.modules["torch"].__config__ = MagicMock()
    sys.modules["torch"].__config__.show.return_value = ""

# Special handling for Redis (always mock to avoid network)
r_client = MagicMock()
for m in ["get", "set", "setex", "publish", "incr", "expire", "delete", "pttl", "ping", "aclose", "exists"]:
    setattr(r_client, m, AsyncMock())

# Ensure pipeline also works
pipe_mock = MagicMock()
pipe_mock.execute = AsyncMock(return_value=[None, 0])
pipe_mock.get = MagicMock(return_value=pipe_mock)
pipe_mock.pttl = MagicMock(return_value=pipe_mock)
pipe_mock.incr = MagicMock(return_value=pipe_mock)
pipe_mock.expire = MagicMock(return_value=pipe_mock)
r_client.pipeline = MagicMock(return_value=pipe_mock)

redis_mock = MagicMock()
redis_mock.from_url.return_value = r_client
redis_mock.Redis.from_url.return_value = r_client
redis_mock.Redis.return_value = r_client
sys.modules["redis"] = sys.modules["redis.asyncio"] = sys.modules["redis.asyncio.client"] = redis_mock

import importlib.util  # noqa: E402

# Numba fallback
if importlib.util.find_spec("numba") is not None:
    pass
else:
    def jit_mock(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    numba_mock = MagicMock()
    numba_mock.jit = numba_mock.njit = jit_mock
    sys.modules["numba"] = numba_mock
