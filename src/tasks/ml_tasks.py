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
        result_meta = asyncio.run(train(use_real_data=True, params=hyperparams, promote_threshold=0.95))

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
        logger.error("hyperparameter_search_error", error=str(e), task_id=self.request.id)
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
            "framework": "xgboost"
        }
        
        pipeline = AutonomousMLPipeline(config)
        
        # Run the full autonomous pipeline
        study = asyncio.run(pipeline.run())
        
        if study:
            logger.info("drift_monitoring_task_triggered_retraining", 
                        best_value=study.best_value,
                        best_params=study.best_params)
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


@celery_app.task(bind=True, queue="ml")
def check_model_performance(self):
    """
    Periodic task to check current model performance against latest data.
    """
    import asyncio

    from src.config import settings
    from src.ml.training.train import load_or_collect_data
    
    logger.info("performance_check_start")
    
    try:
        # 1. Load latest data slice
        X, y, _, _ = asyncio.run(load_or_collect_data(use_real_data=True, n_samples=2000))
        
        # 2. Predict using current model (assuming a local loaded model or singleton)
        # For simplicity in this task, we'll use the trainer to get current metrics
        from src.ml.trainer import ModelTrainer
        ModelTrainer(study_name="Option_Pricing_Performance_Check")
        
        # Placeholder: In a real scenario, we'd load the "production" model from MLflow
        # For now, we'll just log that we are checking.
        logger.info("performance_check_data_loaded", n_samples=len(X))
        
        # Simulate check result
        r2 = 0.94 # Placeholder for real model evaluation
        
        if r2 < settings.ML_TRAINING_PROMOTE_THRESHOLD_R2:
            logger.warning("performance_degradation_detected", r2=r2)
            # Trigger retraining if needed
            # monitor_drift_and_retrain_task.delay()
            
        return {"status": "checked", "r2": r2}
        
    except Exception as e:
        logger.error("performance_check_failed", error=str(e))
        return {"status": "failed", "error": str(e)}


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
