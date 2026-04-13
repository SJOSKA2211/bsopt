import os

import mlflow
import msgspec
import numpy as np
import structlog
import xgboost as xgb
from sklearn.model_selection import train_test_split

from typing import Any

from src.ml.training.base import BaseTrainer, TrainingConfig, TrainingResult
from src.ml.training.registry import training_registry

logger = structlog.get_logger(__name__)


class ModelTrainer(BaseTrainer):
    """
    Unified Model Trainer for Manifold.
    Supports Production-grade training for XGBoost and PyTorch models.
    """

    def __init__(self, study_name: str, tracking_uri: str | None = None) -> None:
        super().__init__(study_name=study_name, tracking_uri=tracking_uri)
        self.model: xgb.XGBRegressor | None = None
        self.best_score: float = -float("inf")

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig,
        metadata: dict[str, str] | None = None,
    ) -> TrainingResult:
        """
        Execute training and evaluation using specified framework with Optuna optimization.
        """
        import optuna
        framework = config.framework.lower()
        logger.info("ml_optimization_started", framework=framework, samples=len(X))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            trainer = training_registry.get_trainer(framework)
            return trainer(self, X_train, y_train, X_test, y_test, config, trial=trial)

        # Run Study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(os.getenv("N_TRIALS", "5")))
        
        self.best_score = study.best_value
        logger.info("optimization_complete", best_score=self.best_score, params=study.best_params)

        # Final train with best params
        with mlflow.start_run(nested=True):
            unified_params = msgspec.json.decode(msgspec.json.encode(config))
            unified_params.update(study.best_params)
            self.log_params(unified_params)
            
            trainer = training_registry.get_trainer(framework)
            cfg_obj = msgspec.json.decode(msgspec.json.encode(unified_params), type=TrainingConfig)
            score = trainer(self, X_train, y_train, X_test, y_test, cfg_obj)

            self.log_metrics({"best_r2_score": score})
            return TrainingResult(
                score=score, metadata={"framework": framework, "optimized": "true"}
            )

    @training_registry.register("xgboost")
    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: TrainingConfig,
        trial: Any | None = None,
    ) -> float:
        """Train XGBoost model with ONNX acceleration support."""
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        from sklearn.metrics import r2_score

        if trial:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        else:
            params = {
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }

        model_params = {**params, "n_jobs": -1, "random_state": 42}

        self.model = xgb.XGBRegressor(**model_params)
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        score = r2_score(y_test, preds)

        mlflow.xgboost.log_model(self.model, "model")
        
        # Convert to ONNX for ultra-low-latency serving
        try:
            initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
            onnx_model = onnxmltools.convert_xgboost(self.model, initial_types=initial_type, target_opset=15)
            artifact_path = "onnx_model"
            mlflow.onnx.log_model(onnx_model, artifact_path)
            logger.info("xgboost_onnx_export_success", artifact_path=artifact_path) # Log artifact path
        except Exception as e:
            logger.warning("xgboost_onnx_export_failed", error=str(e))
            
        return float(score)

    @training_registry.register("torch")
    def _train_torch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: TrainingConfig,
        trial: Any | None = None,
    ) -> float:
        """Train PyTorch Neural Pricing Engine with ONNX export."""
        import torch
        from sklearn.metrics import r2_score

        from src.ml.models.neural_engine import NeuralPricingEngine

        engine = NeuralPricingEngine()
        engine.train_model(
            inputs=X_train,
            targets=y_train.reshape(-1, 1),
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
        )

        self.model = engine.model
        # Evaluation
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy().flatten()

        score = r2_score(y_test, preds)
        mlflow.pytorch.log_model(self.model, "model")
        
        # Export to ONNX
        try:
            dummy_input = torch.randn(1, X_train.shape[1])
            torch.onnx.export(
                self.model, 
                dummy_input, 
                "model.onnx", 
                opset_version=15,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            mlflow.log_artifact("model.onnx", "onnx_model")
            logger.info("torch_onnx_export_success")
        except Exception as e:
            logger.warning("torch_onnx_export_failed", error=str(e))
            
        return float(score)