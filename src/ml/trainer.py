import time
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import structlog

from src.ml.evaluation.metrics import ModelScorecard
from src.ml.strategies import get_strategy
from src.ml.tracker import ExperimentTracker
from src.ml.utils.validation import WalkForwardValidator

logger = structlog.get_logger()


class ModelTrainer:
    """
    Unified Model Trainer with Temporal Validation and Experiment Tracking.
    """

    def __init__(self, study_name: str, tracking_uri: str = None, n_splits: int = 5):
        self.tracker = ExperimentTracker(study_name, tracking_uri)
        self.n_splits = n_splits
        self.model = None

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
        feature_names: list[str] | None = None,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Executes a full Walk-Forward validation loop.
        """
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")

        framework = params.get("framework", "xgboost")
        from src.ml.strategies import STRATEGY_MAP
        if framework not in STRATEGY_MAP:
            raise ValueError(f"Unsupported framework: {framework}. Must be one of {list(STRATEGY_MAP.keys())}")


        strategy = get_strategy(framework)

        validator = WalkForwardValidator(n_splits=self.n_splits)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(validator.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            with self.tracker.start_run(nested=True):
                self.tracker.set_tags({"fold": str(fold), "framework": framework})
                if dataset_metadata:
                    self.tracker.set_tags(dataset_metadata)

                start = time.time()
                model = strategy.train(X_train, y_train, X_test, y_test, params)
                y_pred = strategy.predict(model, X_test)
                duration = time.time() - start

                scorecard = ModelScorecard(y_test, y_pred)
                metrics = scorecard.to_dict()
                scores.append(metrics["r2"])

                self.tracker.log_metrics(metrics["mae"], metrics["rmse"], duration, framework)

                if fold == self.n_splits - 1:
                    self.model = model
                    self.tracker.log_model(model, framework, "final_model")

        return float(np.mean(scores))

    def optimize(self, objective: Callable, n_trials: int = 20) -> optuna.study.Study:
        """Standard Optuna optimization wrapper."""
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study


# Aliases for compatibility
InstrumentedTrainer = ModelTrainer
PyTorchTrainer = ModelTrainer
