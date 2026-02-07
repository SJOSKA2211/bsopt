"""
Machine Learning Tasks for Celery

Handles asynchronous ML model training and inference.
"""

import os

import structlog

from src.ml.training.train import run_hyperparameter_optimization, train

from .celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(bind=True, queue="ml")
def train_model_task(
    self,
    model_type: str = "xgboost",
    training_data: dict | None = None,
    hyperparams: dict | None = None,
):
    """
    Async task to train an ML model for option pricing.
    """
    logger.info("training_model_start", model_type=model_type)

    try:
        # Set environment for MLflow tracking URI if not already set
        if not os.getenv("MLFLOW_TRACKING_URI"):
            os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

        import asyncio

        # Call the actual training function
        result_meta = asyncio.run(
            train(use_real_data=True, params=hyperparams, promote_threshold=0.95)
        )

        result = {
            "task_id": self.request.id,
            "model_type": model_type,
            "status": "completed",
            "run_id": result_meta.get("run_id"),
            "metrics": result_meta.get("metrics"),
            "promoted": result_meta.get("promoted"),
        }

        logger.info("model_training_completed", result=result)
        return result

    except Exception as e:
        logger.error("training_error", error=str(e), task_id=self.request.id)
        return {"task_id": self.request.id, "status": "failed", "error": str(e)}


@celery_app.task(bind=True, queue="ml")
def hyperparameter_search_task(self, model_type: str, n_trials: int = 20):
    """
    Async task to perform hyperparameter optimization using Optuna.
    """
    logger.info("hyperparameter_search_start", model_type=model_type, n_trials=n_trials)

    try:
        import asyncio

        optimization_result = asyncio.run(
            run_hyperparameter_optimization(use_real_data=True, n_trials=n_trials)
        )

        return {
            "task_id": self.request.id,
            "model_type": model_type,
            "best_params": optimization_result["best_params"],
            "best_r2": optimization_result["best_r2"],
            "n_trials": n_trials,
            "status": "completed",
        }

    except Exception as e:
        logger.error(
            "hyperparameter_search_error", error=str(e), task_id=self.request.id
        )
        return {"task_id": self.request.id, "status": "failed", "error": str(e)}


@celery_app.task(bind=True, queue="ml")
def monitor_drift_and_retrain_task(self):
    """
    Periodic task to monitor data/performance drift and trigger
    automated retraining if thresholds are breached.
    """
    import asyncio
    import os

    from src.config import settings
    from src.ml.autonomous_pipeline import AutonomousMLPipeline

    logger.info("drift_monitoring_task_started")

    try:
        # Load config from environment/settings
        config = {
            "api_key": os.getenv("POLYGON_API_KEY", "DEMO_KEY"),
            "provider": os.getenv("DATA_PROVIDER", "auto"),
            "db_url": settings.DATABASE_URL,
            "ticker": os.getenv("DEFAULT_TICKER", "AAPL"),
            "study_name": "autonomous_drift_retraining",
            "n_trials": 10,
            "framework": "xgboost",
        }

        pipeline = AutonomousMLPipeline(config)

        # Run the full autonomous pipeline
        study = asyncio.run(pipeline.run())

        if study:
            logger.info(
                "drift_monitoring_task_triggered_retraining",
                best_value=study.best_value,
                best_params=study.best_params,
            )
            return {"status": "retrained", "best_value": study.best_value}
        else:
            logger.info("drift_monitoring_task_no_retraining_needed")
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
