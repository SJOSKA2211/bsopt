import numpy as np
import structlog
import mlflow
import torch
import xgboost as xgb
from pathlib import Path
from typing import dict, tuple
from sklearn.model_selection import train_test_split
from src.ml.training.base import BaseTrainer, TrainingConfig, TrainingResult
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
        self.model: torch.nn.Module | xgb.XGBRegressor | None = None
        self.best_score: float = -float("inf")

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig,
        metadata: dict[str, str] | None = None
    ) -> TrainingResult:
        """
        Execute training and evaluation using specified framework.
        """
        framework = config.framework.lower()
        logger.info("ml_training_started", framework=framework, samples=len(X))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with mlflow.start_run(nested=True) as run:
            # Convert config to dict for mlflow logging
            self.log_params(msgspec.json.decode(msgspec.json.encode(config)))
            if metadata:
                self.set_tags(metadata)

            if framework == "xgboost":
                score = self._train_xgboost(X_train, y_train, X_test, y_test, config)
            elif framework == "torch" or framework == "neural":
                score = self._train_torch(X_train, y_train, X_test, y_test, config)
            else:
                raise ValueError(f"Unsupported framework: {framework}")

            self.best_score = score
            self.log_metrics({"r2_score": score})
            logger.info("ml_training_complete", score=score)
            
            return TrainingResult(
                score=score,
                metadata={"framework": framework, "samples": str(len(X))}
            )

    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, config: TrainingConfig) -> float:
        """Train XGBoost model."""
        from sklearn.metrics import r2_score

        model_params = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
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

    def _train_torch(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, config: TrainingConfig) -> float:
        """Train PyTorch Neural Pricing Engine."""
        from sklearn.metrics import r2_score
        
        engine = NeuralPricingEngine()
        engine.train_model(
            inputs=X_train,
            targets=y_train.reshape(-1, 1),
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr
        )
        
        self.model = engine.model
        # Evaluation
        # Simple R2 on raw outputs for now
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy().flatten()
        
        score = r2_score(y_test, preds)
        mlflow.pytorch.log_model(self.model, "model")
        return float(score)
