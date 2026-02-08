import sys
import types
from unittest.mock import AsyncMock, MagicMock


# Simple Mocking Strategy
def mock_module(name):
    if name not in sys.modules:
        # For complex hierarchical mocks, use a dummy module if it's mlflow or lightning
        if name in ["mlflow", "lightning", "lightning.pytorch"]:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
        else:
            m = MagicMock(_mock_name=name)
            sys.modules[name] = m
    return sys.modules[name]

# Only mock what's ABSOLUTELY necessary and likely to be missing
MOCK_IF_MISSING = [
    "cvxpy", "confluent_kafka", "confluent_kafka.admin", "confluent_kafka.schema_registry", 
    "selectolax", "selectolax.lexbor", "dask", "cvxopt",
    "stable_baselines3", "stable_baselines3.common", "stable_baselines3.common.torch_layers",
    "stable_baselines3.td3", "stable_baselines3.td3.policies", "stable_baselines3.common.buffers",
    "gymnasium", "mlflow", "faker", "prometheus_api_client", "xgboost", 
    "prefect", "lightning", "lightning.pytorch", "sendgrid", "sendgrid.helpers", "sendgrid.helpers.mail",
    "rich", "rich.box", "rich.console",
    "torch_geometric", "torch_geometric.nn", "torch_geometric.data",
    "flwr"
]

# Note: qiskit, qiskit_aer, flwr, numba, onnxruntime are now installed in God-Mode venv

for mod in MOCK_IF_MISSING:
    try:
        # Don't try to import heavy ones, just mock
        if mod in ["faker", "mlflow", "stable_baselines3", "xgboost", "confluent_kafka", "lightning", "prefect", "dask", "cvxopt", "cvxpy", "flwr"]:
            raise ImportError
        if mod in sys.modules:
            continue
        __import__(mod)
    except ImportError:
        m = mock_module(mod)
        # Specific attribute fixes for mocks
        if mod == "xgboost":
            m.DMatrix = MagicMock()
        if mod == "confluent_kafka":
            m.Consumer = MagicMock()
            m.Producer = MagicMock()
            m.KafkaError = MagicMock()
            m.schema_registry = mock_module("confluent_kafka.schema_registry")
        if mod == "lightning":
            m.pytorch = mock_module("lightning.pytorch")
            m.pytorch.callbacks = MagicMock()
            sys.modules["lightning.pytorch.callbacks"] = m.pytorch.callbacks
        if mod == "mlflow":
            m.set_tracking_uri = MagicMock()
            m.start_run = MagicMock()
            m.end_run = MagicMock()
            m.log_params = MagicMock()
            m.log_metrics = MagicMock()
            m.log_metric = MagicMock()
            m.log_artifact = MagicMock()
            # Handle submodules
            m.xgboost = MagicMock()
            m.pytorch = MagicMock()
            m.pyfunc = MagicMock()
            sys.modules["mlflow.xgboost"] = m.xgboost
            sys.modules["mlflow.pytorch"] = m.pytorch
            sys.modules["mlflow.pyfunc"] = m.pyfunc
        if mod == "faker":
            # Faker instance needs to be returned by Faker()
            m.Faker = MagicMock()
            m.Faker.return_value = MagicMock()
        if mod == "rich":
            m.box = MagicMock()
            m.console = MagicMock()
            sys.modules["rich.box"] = m.box
            sys.modules["rich.console"] = m.console
        if mod == "sendgrid":
            m.helpers = mock_module("sendgrid.helpers")
            m.helpers.mail = MagicMock()
            sys.modules["sendgrid.helpers.mail"] = m.helpers.mail

# Ensure torch mock has version if it ends up being mocked (though it should be installed)
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

# Force Mock Ray (Heavy dependency)
ray_mock = MagicMock()
ray_mock.init = MagicMock()
ray_mock.remote = lambda x: x # Decorator pass-through
ray_mock.get = MagicMock(return_value=None)
ray_mock.put = MagicMock()
ray_mock.shutdown = MagicMock()
ray_mock.is_initialized = MagicMock(return_value=False)

# Mock submodules
ray_mock.train = MagicMock()
ray_mock.train.torch = MagicMock()
ray_mock.train.torch.prepare_model = MagicMock(side_effect=lambda x: x)
ray_mock.train.torch.prepare_data_loader = MagicMock(side_effect=lambda x: x)
ray_mock.train.report = MagicMock()
ray_mock.train.get_context = MagicMock()
ray_mock.train.get_context.return_value.get_local_rank.return_value = 0

ray_mock.tune = MagicMock()
ray_mock.air = MagicMock()

sys.modules["ray"] = ray_mock
sys.modules["ray.train"] = ray_mock.train
sys.modules["ray.train.torch"] = ray_mock.train.torch
sys.modules["ray.tune"] = ray_mock.tune
sys.modules["ray.air"] = ray_mock.air
