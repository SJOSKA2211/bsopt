import numpy as np
import structlog
import mlflow
import torch
import xgboost as xgb
from pathlib import Path
from typing import Any, dict, tuple
from sklearn.model_selection import train_test_split
from src.ml.training.base import BaseTrainer
from src.ml.models.neural_engine import NeuralPricingEngine
from src.math_kernel.models import BSParameters

logger = structlog.get_logger(__name__)

class ModelTrainer(BaseTrainer):
    """
    Unified Model Trainer for EquaFlow.
    Supports institutional-grade training for XGBoost and PyTorch models.
    """

    def __init__(self, study_name: str, tracking_uri: str | None = None) -> None:
        super().__init__(study_name=study_name, tracking_uri=tracking_uri)
        self.model: Any = None
        self.best_score: float = -float("inf")

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
        feature_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> float:
        """
        Execute training and evaluation using specified framework.
        """
        framework = params.get("framework", "xgboost").lower()
        logger.info("ml_training_started", framework=framework, samples=len(X))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with mlflow.start_run(nested=True) as run:
            self.log_params(params)
            if metadata:
                self.set_tags(metadata)

            if framework == "xgboost":
                score = self._train_xgboost(X_train, y_train, X_test, y_test, params)
            elif framework == "torch" or framework == "neural":
                score = self._train_torch(X_train, y_train, X_test, y_test, params)
            else:
                raise ValueError(f"Unsupported framework: {framework}")

            self.best_score = score
            self.log_metrics({"r2_score": score})
            logger.info("ml_training_complete", score=score)
            return score

    def _train_xgboost(self, X_train, y_train, X_test, y_test, params) -> float:
        """Train XGBoost model."""
        from sklearn.metrics import r2_score

        model_params = {
            "n_estimators": params.get("n_estimators", 100),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.1),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42
        }
        
        self.model = xgb.XGBRegressor(**model_params)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        score = r2_score(y_test, preds)
        
        mlflow.xgboost.log_model(self.model, "model")
        return float(score)

    def _train_torch(self, X_train, y_train, X_test, y_test, params) -> float:
        """Train PyTorch Neural Pricing Engine."""
        from sklearn.metrics import r2_score
        
        engine = NeuralPricingEngine()
        engine.train_model(
            inputs=X_train,
            targets=y_train.reshape(-1, 1),
            epochs=params.get("epochs", 10),
            batch_size=params.get("batch_size", 32),
            lr=params.get("lr", 0.001)
        )
        
        self.model = engine.model
        # Evaluation
        spots = X_test[:, 0] # Assuming first col is spot, etc. 
        # (This needs better feature mapping in a real scenario)
        
        # Simple R2 on raw outputs for now
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy().flatten()
        
        score = r2_score(y_test, preds)
        mlflow.pytorch.log_model(self.model, "model")
        return float(score)
