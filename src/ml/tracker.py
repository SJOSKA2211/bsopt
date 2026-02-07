import os
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import structlog

from src.shared.observability import (
    MODEL_ACCURACY,
    MODEL_RMSE,
    TRAINING_DURATION,
    TRAINING_ERRORS,
    push_metrics,
)

logger = structlog.get_logger()

class ExperimentTracker:
    """
    Handles all observability, logging, and metrics for ML training.
    """
    def __init__(self, study_name: str, tracking_uri: str = None):
        self.study_name = study_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def start_run(self, nested: bool = True):
        return mlflow.start_run(nested=nested)

    def log_params(self, params: dict[str, Any]):
        mlflow.log_params(params)

    def set_tags(self, tags: dict[str, str]):
        mlflow.set_tags(tags)

    def log_metrics(self, accuracy: float, rmse: float, duration: float, framework: str):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("duration", duration)
        
        TRAINING_DURATION.labels(framework=framework).observe(duration)
        MODEL_ACCURACY.labels(framework=framework).set(accuracy)
        MODEL_RMSE.labels(model_type=framework, dataset="validation").set(rmse)

    def log_error(self, framework: str, error: str):
        TRAINING_ERRORS.labels(framework=framework).inc()
        logger.error("training_failed", framework=framework, error=error)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_model(self, model: Any, framework: str, artifact_path: str = "model"):
        if framework == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        elif framework == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif framework == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        else:
            mlflow.log_model(model, artifact_path) # Generic fallback

    def log_feature_importance(self, importance: dict[str, float], framework: str):
        plt.figure(figsize=(10, 6))
        names = list(importance.keys())
        values = list(importance.values())
        plt.barh(names, values)
        plt.title(f"Feature Importance ({framework})")
        plt.xlabel("Importance")
        
        temp_dir = tempfile.mkdtemp()
        plot_path = os.path.join(temp_dir, "feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
        
        self.log_artifact(plot_path)
        os.remove(plot_path)
        os.rmdir(os.path.dirname(plot_path))

    def push_to_gateway(self):
        push_metrics(job_name=self.study_name)
