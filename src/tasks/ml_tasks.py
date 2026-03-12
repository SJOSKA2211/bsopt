"""
Machine Learning Tasks for Celery (Optimized)
"""

import sys
from typing import Any

import structlog

from src.ml.pipeline import MLPipeline
from src.utils.lazy_import import lazy_import

from .celery_app import MLTask, celery_app

logger = structlog.get_logger(__name__)

# Lazy Import Map
_IMPORT_MAP = {
    "mlflow": "mlflow",
    "pd": "pandas",
    "ModelQuantizer": "src.ml.serving.quantization.ModelQuantizer",
    "calc_metrics": "src.ml.evaluation.metrics.calculate_regression_metrics",
}


def _get_attr(name: str):
    return lazy_import(__name__, _IMPORT_MAP, name, sys.modules[__name__])


@celery_app.task(bind=True, base=MLTask, name="ml.run_autonomous_pipeline")
def run_pipeline_task(self, config: dict[str, Any]):
    """
    Celery task to run the autonomous ML pipeline.
    OPTIMIZED: Uses BaseAsyncTask for non-blocking execution.
    """
    logger.info("celery_task_started", task_id=self.request.id, ticker=config.get("ticker"))
    pipeline = MLPipeline(config)
    try:
        # OPTIMIZED: Use persistent loop from BaseAsyncTask
        model = self.run_async(pipeline.run())

        result = {
            "status": "success",
            "model_promoted": model is not None,
            "task_id": self.request.id,
        }

        logger.info("celery_task_completed", **result)
        return result

    except Exception as e:
        logger.error("celery_task_failed", error=str(e), task_id=self.request.id)
        # Re-raise so Celery marks it as failed
        raise e
    finally:
        try:
            self.run_async(pipeline.shutdown())
        except Exception:
            pass


@celery_app.task(bind=True, base=MLTask, queue="ml")
def train_model_task(
    self,
    ticker: str = "TSLA",
    model_type: str = "xgboost",
    hyperparams: dict | None = None,
):
    """Async task to train an ML model using the unified pipeline."""
    logger.info("training_model_start", ticker=ticker, model_type=model_type)

    try:
        config = hyperparams or {}
        config["ticker"] = ticker
        config["framework"] = model_type

        pipeline = MLPipeline(config)
        # OPTIMIZED: Use persistent loop from MLTask (BaseAsyncTask)
        model = self.run_async(pipeline.run(force=True))

        return {
            "task_id": self.request.id,
            "status": "completed",
            "ticker": ticker,
            "framework": model_type,
            "promoted": model is not None,
        }
    except Exception as e:
        logger.error("training_error", error=str(e), task_id=self.request.id)
        return {"status": "failed", "error": str(e)}


@celery_app.task(bind=True, base=MLTask, queue="ml")
def monitor_drift_and_retrain_task(self, ticker: str = "AAPL", mode: str = "regressor"):
    """
    Periodic task to monitor drift and trigger the optimized MLflow pipeline via Docker.
    Supports 'regressor' (single ticker) and 'cross_sectional' modes.
    """
    logger.info("drift_monitoring_task_started", ticker=ticker, mode=mode)

    try:
        import os
        import subprocess

        # Use the central startup script
        script_path = os.path.join(os.getcwd(), "scripts/start_mlflow_pipeline.sh")

        if mode == "cross_sectional":
            cmd = [
                "bash",
                script_path,
                "train_cross_sectional",
                "celery_cross_sectional_retrain",
                "-P", "epochs=5"
            ]
        else:
            cmd = [
                "bash",
                script_path,
                "train_regressor",
                f"celery_drift_{ticker}",
                "-P", f"ticker={ticker}",
                "-P", "n_trials=10",
            ]

        # Dispatch the job
        subprocess.check_call(cmd)

        return {"status": "retrained_job_dispatched", "ticker": ticker, "mode": mode}

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


@celery_app.task(bind=True, base=MLTask, queue="ml")
def check_threshold_and_retrain_task(self, ticker: str = "AAPL", force: bool = False, threshold: int = 50000, mode: str = "regressor"):
    """
    Checks if the database has accumulated enough new market data to justify retraining.
    """
    logger.info("check_threshold_and_retrain_start", ticker=ticker, threshold=threshold, mode=mode)

    try:
        if not force:
            from sqlalchemy import create_engine, text
            from src.config import settings
            
            engine = create_engine(settings.DATABASE_URL)
            with engine.connect() as conn:
                if mode == "cross_sectional":
                    # For cross-sectional, we check the total volume across all symbols
                    query = text("SELECT COUNT(*) FROM market_ticks")
                    count = conn.execute(query).scalar()
                else:
                    query = text("SELECT COUNT(*) FROM market_ticks WHERE symbol = :symbol")
                    count = conn.execute(query, {"symbol": ticker}).scalar()
                
                logger.info("data_volume_check", ticker=ticker, current_count=count, threshold=threshold, mode=mode)
                
                if count < threshold:
                    logger.info("retraining_skipped_insufficient_data", ticker=ticker, mode=mode)
                    return {"status": "skipped", "reason": "insufficient_data", "count": count}

        # Trigger the actual retraining
        return monitor_drift_and_retrain_task.delay(ticker=ticker, mode=mode).get()

    except Exception as e:
        logger.error("check_threshold_failed", error=str(e))
        return {"status": "failed", "error": str(e)}
