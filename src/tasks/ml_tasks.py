"""
Machine Learning Tasks for Celery (Optimized)
"""

import asyncio
import os
import sys

import structlog

from src.utils.lazy_import import lazy_import

from .celery_app import celery_app

logger = structlog.get_logger(__name__)

# Lazy Import Map
_IMPORT_MAP = {
    "mlflow": "mlflow",
    "pd": "pandas",
    "train_fn": "src.ml.training.train.train",
    "run_hpo": "src.ml.training.train.run_hyperparameter_optimization",
    "collect_data": "src.ml.training.train.load_or_collect_data",
    "MLPipeline": "src.ml.pipeline.MLPipeline",
    "ModelQuantizer": "src.ml.serving.quantization.ModelQuantizer",
    "calc_metrics": "src.ml.evaluation.metrics.calculate_regression_metrics",
}


def _get_attr(name: str):
    return lazy_import(__name__, _IMPORT_MAP, name, sys.modules[__name__])


async def _run_async_safe(coro):
    """Safely run an async coroutine from a potentially synchronous worker."""
    import asyncio

    try:
        asyncio.get_running_loop()
        return await coro
    except RuntimeError:
        return asyncio.run(coro)


@celery_app.task(bind=True, queue="ml")
def train_model_task(
    self,
    model_type: str = "xgboost",
    hyperparams: dict | None = None,
):
    """Async task to train an ML model with lazy-loaded dependencies."""
    logger.info("training_model_start", model_type=model_type)
    train_fn = _get_attr("train_fn")

    try:
        # Call the actual training function
        result_meta = asyncio.run(
            train_fn(use_real_data=True, params=hyperparams, promote_threshold=0.95)
        )

        return {
            "task_id": self.request.id,
            "status": "completed",
            "run_id": result_meta.get("run_id"),
            "metrics": result_meta.get("metrics"),
            "promoted": result_meta.get("promoted"),
        }
    except Exception as e:
        logger.error("training_error", error=str(e), task_id=self.request.id)
        return {"status": "failed", "error": str(e)}


@celery_app.task(bind=True, queue="ml")
def monitor_drift_and_retrain_task(self):
    """Periodic task to monitor drift using lazy-loaded autonomous pipeline."""
    MLPipeline = _get_attr("MLPipeline")
    from src.config import settings

    try:
        config = {
            "api_key": os.getenv("POLYGON_API_KEY", "DEMO_KEY"),
            "db_url": settings.DATABASE_URL,
            "study_name": "autonomous_drift_retraining",
            "n_trials": 10,
            "framework": "xgboost",
        }
        pipeline = MLPipeline(config)
        study = asyncio.run(pipeline.run())

        if study:
            return {"status": "retrained", "best_value": study.best_value}
        return {"status": "no_drift_detected"}
    except Exception as e:
        logger.error("drift_monitoring_task_failed", error=str(e))
        return {"status": "failed", "error": str(e)}



@celery_app.task(bind=True, queue="ml")
def optimize_model_task(self, model_path: str, output_path: str):
    """
    Asynchronous task to quantize an ONNX model to INT8.
    """
    from src.ml.serving.quantization import ModelQuantizer

    logger.info("model_optimization_start", input=model_path)

    try:
        quantizer = ModelQuantizer()
        quantizer.quantize_onnx_model(model_path, output_path)

        logger.info("model_optimization_complete", output=output_path)
        return {"status": "success", "optimized_path": output_path}
    except Exception as e:
        logger.error("model_optimization_failed", error=str(e))
        return {"status": "failed", "error": str(e)}


@celery_app.task(bind=True, queue="ml")
def hyperparameter_search_task(self, model_type: str = "xgboost"):
    """Dummy hyperparameter search task."""
    logger.info("hyperparameter_search_start", model_type=model_type)
    return {"status": "success", "best_params": {}}


@celery_app.task(bind=True, queue="ml")
def evaluate_model_task(self, model_uri: str, dataset_path: str):
    """
    Asynchronous task to evaluate a specific model against a dataset.
    Useful for verification before promotion.
    """
    import mlflow
    import pandas as pd

    from src.ml.evaluation.metrics import calculate_regression_metrics

    logger.info("model_evaluation_start", model_uri=model_uri)

    try:
        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # Load data
        df = pd.read_parquet(dataset_path)
        X = df.drop(columns=["target"])
        y = df["target"]

        # Predict
        y_pred = model.predict(X)

        # Calculate metrics
        metrics = calculate_regression_metrics(y.values, y_pred)

        logger.info("model_evaluation_complete", metrics=metrics)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.error("model_evaluation_failed", error=str(e))
        return {"status": "failed", "error": str(e)}
