from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from src.ml.utils.distributed import train_xgboost_distributed

logger = structlog.get_logger()


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
        y_pred_prob = model.predict(
            dtest, iteration_range=(0, model.best_iteration + 1)
        )
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
        return train_xgboost_distributed(
            X_train, y_train, xgb_params, dask_address=dask_address
        )

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
        return {name: float(imp) for name, imp in zip(feature_names, importances)}


class PyTorchStrategy(TrainingStrategy):
    class SimpleNet(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
            )

        def forward(self, x):
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
        epochs = params.get("epochs", 10)
        lr = params.get("lr", 0.01)

        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

        model = self.SimpleNet(X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_loss = float("inf")
        patience = 5
        trigger_times = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_t)
                val_loss = criterion(val_outputs, y_test_t)
                if val_loss < best_loss:
                    best_loss = val_loss
                    trigger_times = 0
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        logger.info("early_stopping_triggered", epoch=epoch)
                        break
        return model

    def predict(self, model: Any, X: Any) -> Any:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            outputs = model(X_t)
            return (outputs.cpu().numpy() > 0.5).astype(int).flatten()


STRATEGY_MAP = {
    "xgboost": XGBoostStrategy,
    "sklearn": SklearnStrategy,
    "pytorch": PyTorchStrategy,
    "dask_xgboost": DaskXGBoostStrategy,
}


def get_strategy(framework: str) -> TrainingStrategy:
    return STRATEGY_MAP.get(framework, XGBoostStrategy)()
