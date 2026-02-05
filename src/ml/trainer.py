import time
from collections.abc import Callable
from typing import Any

import optuna
import structlog
from sklearn.model_selection import train_test_split

from src.ml.evaluation.metrics import calculate_regression_metrics
from src.ml.serving.quantization import ModelQuantizer
from src.ml.strategies import get_strategy
from src.ml.tracker import ExperimentTracker

logger = structlog.get_logger()

class ModelTrainer:
    """
    Orchestrates the training process using Strategy Pattern and dedicated Tracker.
    """
    
    def __init__(self, study_name: str, storage: str = None, tracking_uri: str = None):
        self.tracker = ExperimentTracker(study_name, tracking_uri)
        self.storage = storage
        self.quantizer = ModelQuantizer()
        self.model = None
        self.best_params = {}

    def train_and_evaluate(self, X: Any, y: Any, params: dict[str, Any], feature_names: list[str] = None, dataset_metadata: dict[str, str] | None = None, base_model: Any | None = None, trial: optuna.Trial | None = None) -> float:
        """
        Trains a model using the specified strategy and evaluates regression performance.
        """
        framework = params.get("framework", "xgboost")
        strategy = get_strategy(framework)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            with self.tracker.start_run(nested=True):
                self.tracker.log_params(params)
                
                if dataset_metadata:
                    self.tracker.set_tags(dataset_metadata)
                
                if base_model:
                    self.tracker.set_tags({"warm_start": "true"})
                
                start_time = time.time()
                
                # Execute Strategy
                model = strategy.train(X_train, y_train, X_test, y_test, params, base_model=base_model)
                y_pred = strategy.predict(model, X_test)
                
                # 🚀 OPTIMIZATION: Use regression metrics
                metrics = calculate_regression_metrics(y_test, y_pred)
                duration = time.time() - start_time
                
                # Pruning based on R2 score
                if trial:
                    trial.report(metrics["r2"], step=1)
                    if trial.should_prune():
                        logger.info("trial_pruned", trial_id=trial.number)
                        raise optuna.exceptions.TrialPruned()
                
                # Logging
                self.tracker.log_metrics(metrics["mae"], metrics["rmse"], duration, framework)
                self.tracker.log_dict(metrics, "metrics.json")
                
                # Post-processing (Quantization)
                if framework == "pytorch":
                    quantized_model = self.quantizer.quantize_dynamic(model)
                    self.tracker.log_model(quantized_model, framework, "model_quantized")
                
                self.tracker.log_model(model, framework, "model")
                
                importance = strategy.get_feature_importance(model, feature_names)
                if importance:
                    self.tracker.log_feature_importance(importance, framework)
                
                self.model = model
                logger.info("model_trained", framework=framework, r2=metrics["r2"])
                
                return float(metrics["r2"])
                
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            self.tracker.log_error(framework, str(e))
            raise
        finally:
            self.tracker.push_to_gateway()

    def optimize(self, objective: Callable, n_trials: int = 20) -> optuna.study.Study:
        """
        Runs the hyperparameter optimization loop.
        """
        logger.info("optimization_started", study_name=self.tracker.study_name, n_trials=n_trials)
        
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            study_name=self.tracker.study_name, 
            direction="maximize", 
            storage=self.storage, 
            load_if_exists=True,
            pruner=pruner
        )
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info("optimization_completed", best_params=self.best_params, best_value=study.best_value)
        
        return study

# Alias for backward compatibility
InstrumentedTrainer = ModelTrainer
