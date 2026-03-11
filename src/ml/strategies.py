from typing import Any

import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from src.ml.callbacks import EarlyStopping
from src.ml.utils.distributed import train_xgboost_distributed

logger = structlog.get_logger()


def init_collective_backend() -> None:
    """Initialize NCCL backend for multi-GPU training if available."""
    if not torch.cuda.is_available():
        return

    try:
        if not dist.is_initialized():
            # Use 'gloo' as fallback for CPU or if NCCL fails
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
            logger.info(
                "dist_backend_initialized", backend=backend, world_size=dist.get_world_size()
            )
    except Exception as e:
        logger.warning("dist_init_failed", error=str(e))


class TrainingStrategy:
    """Base interface for training strategies."""

    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        params: dict[str, Any],
        base_model: Any | None = None,
    ) -> Any:
        raise NotImplementedError

    def predict(self, model: Any, X: Any) -> Any:
        raise NotImplementedError

    def get_feature_importance(
        self, model: Any, feature_names: list[str]
    ) -> dict[str, float] | None:
        return None

    def export_onnx(self, model: Any, path: str, input_dim: int) -> None:
        """Standardized ONNX export interface."""
        pass


class ONNXOptimizationMixin:
    """
    High-Performance: Reusable ONNX optimization logic for strategies.
    """

    def export_onnx(self, model: Any, path: str, input_dim: int) -> None:
        import torch

        from src.ml.utils.optimization import export_to_onnx, quantize_onnx_model

        logger.info("optimizing_model_for_onnx", path=path)
        dummy_input = torch.randn(1, input_dim)

        # 1. Export standard
        export_to_onnx(model, dummy_input, path)

        # 2. Quantize
        quantized_path = path.replace(".onnx", ".int8.onnx")
        try:
            quantize_onnx_model(path, quantized_path)
        except Exception as e:
            logger.warning("quantization_skipped_in_mixin", error=str(e))


class XGBoostStrategy(TrainingStrategy):
    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        params: dict[str, Any],
        base_model: Any | None = None,
    ) -> Any:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_params = params.copy()
        n_estimators = xgb_params.pop("n_estimators", 100)
        xgb_params.pop("framework", None)

        evallist = [(dtest, "eval"), (dtrain, "train")]
        return xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evallist,
            early_stopping_rounds=10,
            verbose_eval=False,
            xgb_model=base_model,
        )

    def predict(self, model: Any, X: Any) -> Any:
        dtest = xgb.DMatrix(X)
        y_pred_prob = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        return (y_pred_prob > 0.5).astype(int)

    def get_feature_importance(
        self, model: Any, feature_names: list[str]
    ) -> dict[str, float] | None:
        importance = model.get_score(importance_type="weight")
        result = {}
        for i, name in enumerate(feature_names):
            key = f"f{i}"
            if key in importance:
                result[name] = float(importance[key])
        return result


class DaskXGBoostStrategy(TrainingStrategy):
    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        params: dict[str, Any],
        base_model: Any | None = None,
    ) -> Any:
        xgb_params = params.copy()
        xgb_params.pop("framework", None)
        dask_address = xgb_params.pop("dask_address", None)
        return train_xgboost_distributed(X_train, y_train, xgb_params, dask_address=dask_address)

    def predict(self, model: Any, X: Any) -> Any:
        dtest = xgb.DMatrix(X)
        y_pred_prob = model.predict(dtest)
        return (y_pred_prob > 0.5).astype(int)

    def get_feature_importance(
        self, model: Any, feature_names: list[str]
    ) -> dict[str, float] | None:
        importance = model.get_score(importance_type="weight")
        result = {}
        for i, name in enumerate(feature_names):
            key = f"f{i}"
            if key in importance:
                result[name] = float(importance[key])
        return result


class SklearnStrategy(TrainingStrategy):
    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        params: dict[str, Any],
        base_model: Any | None = None,
    ) -> Any:
        sk_params = params.copy()
        sk_params.pop("framework", None)
        model = RandomForestClassifier(**sk_params)
        model.fit(X_train, y_train)
        return model

    def predict(self, model: Any, X: Any) -> Any:
        return model.predict(X)

    def get_feature_importance(
        self, model: Any, feature_names: list[str]
    ) -> dict[str, float] | None:
        importances = model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances, strict=False)}


class PyTorchStrategy(TrainingStrategy, ONNXOptimizationMixin):
    class SimpleNet(nn.Module): # type: ignore
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),  # Regression output
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        params: dict[str, Any],
        base_model: Any | None = None,
    ) -> Any:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = params.get("epochs", 100)
        lr = params.get("lr", 0.001)
        batch_size = params.get("batch_size", 32)
        patience = params.get("early_stopping_patience", 10)

        # Prepare DataLoaders
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = self.SimpleNet(X_train.shape[1]).to(device)
        if base_model:
            model.load_state_dict(base_model.state_dict())

        # OPTIMIZED: Kernel Fusion via torch.compile
        if hasattr(torch, "compile") and device.type == "cuda":
            try:
                model = torch.compile(model)
                logger.info("pytorch_strategy_model_compiled")
            except Exception as e:
                logger.warning("pytorch_strategy_compile_failed", error=str(e))

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()  # Changed to MSE for regression
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    val_outputs = model(batch_X)
                    val_loss += criterion(val_outputs, batch_y).item()

            val_loss /= len(test_loader)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info("early_stopping_triggered", epoch=epoch)
                break

        return model

    def predict(self, model: Any, X: Any) -> Any:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            outputs = model(X_t)
            return outputs.cpu().numpy().flatten()


STRATEGY_MAP = {
    "xgboost": XGBoostStrategy,
    "sklearn": SklearnStrategy,
    "pytorch": PyTorchStrategy,
    "dask_xgboost": DaskXGBoostStrategy,
}


def get_strategy(framework: str) -> TrainingStrategy:
    return STRATEGY_MAP.get(framework, XGBoostStrategy)()
