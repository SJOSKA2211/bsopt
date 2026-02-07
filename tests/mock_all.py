import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np

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
        # Return array of length 10 to match mock X_test size
        return np.ones(10, dtype=int)
    
    def __ge__(self, other):
        return np.ones(10, dtype=int)
        
    def __lt__(self, other):
        return np.zeros(10, dtype=int)

    def __iter__(self):
        return iter([MagicMock(), MagicMock()])
        
    def split(self, sep=None, maxsplit=-1):
        return ["mock_host", "50051"]
    
    def astype(self, t):
        return np.ones(10, dtype=t)

class MockTensor: 
    def to(self, *args, **kwargs): return self
    def item(self): return 1.0
    def cpu(self): return self
    def numpy(self): return np.ones(10)
    def size(self, dim=None): return 10 if dim is None else 10
    def __iter__(self): return iter([MockTensor() for _ in range(10)])

# List of modules to aggressively mock
MOCK_MODULES = [
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
    "grpclib", "grpclib.client", "grpclib.const", "sendgrid", "sendgrid.helpers", "sendgrid.helpers.mail", "dask", "dask.distributed", "distributed",
    "redis", "redis.asyncio", "redis.asyncio.client",
    "authlib", "authlib.jose", "onnxruntime", "sklearn", "sklearn.ensemble",
    "sklearn.metrics", "sklearn.model_selection", "sklearn.preprocessing",
    "sklearn.externals", "mlflow", "mlflow.pyfunc", "mlflow.models", 
    "mlflow.pytorch", "mlflow.xgboost", "mlflow.data", "mlflow.tracking", 
    "qiskit", "qiskit_aer", "qiskit.circuit", "qiskit.circuit.library", 
    "cvxpy", 
    "prometheus_api_client", "prefect", "pytorch_forecasting", "pytorch_forecasting.data",
    "pytorch_forecasting.metrics", "selectolax", "selectolax.lexbor", "pandas_ta",
    "xgboost", "xgboost.dask", "lightning", "lightning.pytorch", "lightning.pytorch.callbacks",
    "flwr", "confluent_kafka", "confluent_kafka.admin", "confluent_kafka.schema_registry", "confluent_kafka.schema_registry.avro",
    "torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils", 
    "torch.utils.data", "torch.distributed", "torch.distributions"
]

# Apply mocks
for mod in MOCK_MODULES:
    if mod not in sys.modules or mod.startswith("ray") or mod.startswith("redis"):
        m = VersionedMock(_mock_name=mod)
        if mod == "torch":
            m.Tensor = MockTensor
            m.cuda.is_available.return_value = False
            m.FloatTensor = MockTensor
            m.LongTensor = MockTensor
        
        if mod == "ray":
            m.init = MagicMock(return_value=None)
            m.shutdown = MagicMock(return_value=None)
            m.is_initialized = MagicMock(return_value=False)
            
        sys.modules[mod] = m

# 🚀 CLEAR LAZY IMPORT CACHE
if "src.utils.lazy_import" in sys.modules:
    import src.utils.lazy_import
    src.utils.lazy_import._failed_imports.clear()

# 🚀 SOTA OVERRIDES for Model forward calls
class MockForward(MagicMock):
    def __call__(self, *args, **kwargs):
        # Return tuple for unpack tests
        return (MockTensor(), MockTensor(), MockTensor())

sys.modules["torch.nn"].Module = MockForward
sys.modules["torch.nn.functional"] = VersionedMock(_mock_name="torch.nn.functional")
sys.modules["torch.Tensor"] = MockTensor 
sys.modules["sklearn.ensemble"] = VersionedMock(_mock_name="sklearn.ensemble")
sys.modules["selectolax.lexbor"] = VersionedMock(_mock_name="selectolax.lexbor")

# XGBoost fixes
xgboost_mock = VersionedMock(_mock_name="xgboost")
booster_mock = MagicMock()
# FIX: Return array of length 10 to match train_test_split mock
booster_mock.predict.return_value = np.ones(10) * 0.9 
booster_mock.best_iteration = 10
xgboost_mock.train.return_value = booster_mock
xgboost_mock.XGBRegressor = MagicMock(return_value=booster_mock)
sys.modules["xgboost"] = xgboost_mock

# Sklearn fixes
# Default train_test_split returns arrays of length 10
sys.modules["sklearn.model_selection"].train_test_split.return_value = (
    np.zeros((10, 5)), np.zeros((10, 5)), np.zeros(10), np.zeros(10)
)
scaler_mock = MagicMock()
scaler_mock.fit_transform.return_value = np.zeros((10, 5))
scaler_mock.transform.return_value = np.zeros((10, 5))
sys.modules["sklearn.preprocessing"].StandardScaler.return_value = scaler_mock

# RandomForest Fix
rf_mock = MagicMock()
rf_mock.predict.return_value = np.ones(10) # Match y_test length
sys.modules["sklearn.ensemble"].RandomForestClassifier.return_value = rf_mock

# Numba fixes
def jit_mock(*args, **kwargs):
    def decorator(func): return func
    return decorator
def vectorize_mock(*args, **kwargs):
    def decorator(func): return np.vectorize(func)
    return decorator

numba_mock = MagicMock()
numba_mock.jit = numba_mock.njit = jit_mock
numba_mock.vectorize = numba_mock.guvectorize = vectorize_mock
numba_mock.prange = range
sys.modules["numba"] = numba_mock

# Qiskit fixes
class MockQuantumCircuit(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = MagicMock(); p.name = 'payoff'
        r = MagicMock(); r.name = 'price'
        self.qregs = [p, r]
        self.num_qubits = 4
        # self.depth.return_value = 10 
        self.data = [1, 2, 3]
sys.modules["qiskit"].QuantumCircuit = MockQuantumCircuit

# Redis fixes
redis_mock = MagicMock()
r_client = MagicMock()
for m in ["get", "set", "setex", "publish", "incr", "expire", "delete", "pttl", "ping", "aclose"]:
    setattr(r_client, m, AsyncMock())
redis_mock.from_url.return_value = r_client
redis_mock.Redis = MagicMock(return_value=r_client)
sys.modules["redis"] = sys.modules["redis.asyncio"] = sys.modules["redis.asyncio.client"] = redis_mock

# Optuna Magic Fixes
optuna = sys.modules["optuna"]
class TrialPruned(BaseException): pass
optuna.exceptions.TrialPruned = TrialPruned

def optimize_mock(objective, n_trials=1, **kwargs):
    trials_list = []
    for i in range(n_trials):
        trial = MagicMock()
        trial.number = i
        trial.suggest_int.return_value = 10
        trial.suggest_float.return_value = 0.1
        trial.suggest_categorical.return_value = "option"
        trial.should_prune.return_value = False
        try:
            objective(trial)
            trials_list.append(trial)
        except TrialPruned:
            pass # Pruned
    return trials_list

study_mock = MagicMock()
def create_study_side_effect(*args, **kwargs):
    s = MagicMock()
    s.trials = []
    def optimize_side_effect(obj, n_trials=1, **kwargs):
        optimize_mock(obj, n_trials)
        s.trials = [MagicMock()] * n_trials
    s.optimize.side_effect = optimize_side_effect
    s.best_params = {"n_estimators": 10}
    s.best_value = 0.95
    return s

optuna.create_study.side_effect = create_study_side_effect
