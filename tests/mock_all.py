import sys
from unittest.mock import MagicMock, AsyncMock

# Version Info for Compatibility
version_tuple = (4, 6, 0)
version_str = "4.6.0"

class VersionedMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.VERSION = version_tuple
        self.__version__ = version_str
        self.__path__ = []
        import importlib.machinery
        self.__spec__ = importlib.machinery.ModuleSpec(self._mock_name or "mock", None)
    def __gt__(self, other):
        if isinstance(other, int): return self.VERSION[0] > other
        return super().__gt__(other)
    def __ge__(self, other):
        if isinstance(other, int): return self.VERSION[0] >= other
        return super().__ge__(other)
    def __iter__(self):
        return iter(["mock_host", "50051"])
    def split(self, sep=None, maxsplit=-1):
        return ["mock_host", "50051"]

class MockTensor: pass

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
    "ray.dag", "ray.experimental", "ray.rllib",
    "authlib", "authlib.jose", "onnxruntime", "sklearn", "sklearn.ensemble",
    "sklearn.metrics", "sklearn.model_selection", "sklearn.preprocessing",
    "sklearn.externals", "mlflow", "mlflow.pyfunc", "mlflow.models", 
    "mlflow.pytorch", "mlflow.xgboost", "mlflow.data", "mlflow.tracking", 
    "qiskit", "qiskit_aer", "qiskit.circuit", "qiskit.circuit.library", 
    "cvxpy", "web3", "web3.providers", "eth_account",
    "prometheus_api_client", "prefect", "pytorch_forecasting", "pytorch_forecasting.data",
    "pytorch_forecasting.metrics", "selectolax", "selectolax.lexbor", "pandas_ta",
    "xgboost", "xgboost.dask", "lightning", "lightning.pytorch", "lightning.pytorch.callbacks",
    "flwr", "confluent_kafka.schema_registry", "confluent_kafka.schema_registry.avro",
    "torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils", 
    "torch.utils.data", "torch.distributed", "torch.distributions"
]:
    if mod not in sys.modules:
        m = VersionedMock(_mock_name=mod)
        if mod == "torch":
            m.Tensor = MockTensor
            m.cuda.is_available.return_value = False
        sys.modules[mod] = m

# Fix submodules that are explicitly imported
sys.modules["torch.nn.functional"] = VersionedMock(_mock_name="torch.nn.functional")
sys.modules["torch.Tensor"] = MockTensor # CRITICAL for scipy
sys.modules["sklearn.ensemble"] = VersionedMock(_mock_name="sklearn.ensemble")
sys.modules["selectolax.lexbor"] = VersionedMock(_mock_name="selectolax.lexbor")
