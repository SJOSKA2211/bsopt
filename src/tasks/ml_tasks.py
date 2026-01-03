"""
Machine Learning Tasks for Celery

Handles asynchronous ML model training and inference.
"""

import logging
import os
from typing import Optional

from src.ml.training.train import run_hyperparameter_optimization, train

from .celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, queue="ml")
def train_model_task(
    self,
    model_type: str = "xgboost",
    training_data: Optional[dict] = None,
    hyperparams: Optional[dict] = None,
):
    """
    Async task to train an ML model for option pricing.
    """
    logger.info(f"Training {model_type} model")

    try:
        # Set environment for MLflow tracking URI if not already set
        if not os.getenv("MLFLOW_TRACKING_URI"):
            os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

        import asyncio
        # Call the actual training function
        result_meta = asyncio.run(train(use_real_data=True, params=hyperparams, promote_threshold=0.95))

        result = {
            "task_id": self.request.id,
            "model_type": model_type,
            "status": "completed",
            "run_id": result_meta.get("run_id"),
            "metrics": result_meta.get("metrics"),
            "promoted": result_meta.get("promoted"),
        }

        logger.info(f"Model training completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"task_id": self.request.id, "status": "failed", "error": str(e)}


@celery_app.task(bind=True, queue="ml")
def hyperparameter_search_task(self, model_type: str, n_trials: int = 20):
    """
    Async task to perform hyperparameter optimization using Optuna.
    """
    logger.info(f"Starting hyperparameter search for {model_type} with {n_trials} trials")

    try:
        import asyncio
        optimization_result = asyncio.run(run_hyperparameter_optimization(use_real_data=True, n_trials=n_trials))

        return {
            "task_id": self.request.id,
            "model_type": model_type,
            "best_params": optimization_result["best_params"],
            "best_r2": optimization_result["best_r2"],
            "n_trials": n_trials,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Hyperparameter search error: {e}")
        return {"task_id": self.request.id, "status": "failed", "error": str(e)}
