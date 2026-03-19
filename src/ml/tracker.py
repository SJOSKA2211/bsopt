import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
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

    def __init__(self, study_name: str, tracking_uri: str | None = None) -> None:
        self.study_name = study_name
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def start_run(self, nested: bool = True) -> Any:
        """
        Starts an MLflow run, with support for nested runs and environment detection.
        """

        @contextmanager
        def run_context() -> Generator[Any, None, None]:
            active = mlflow.active_run()
            in_mlflow_run = "MLFLOW_RUN_ID" in os.environ

            # If already in a run, and nested is requested, try to start one.
            if active or in_mlflow_run:
                if nested:
                    try:
                        # OPTIMIZED: Use nested run if already in an active run
                        with mlflow.start_run(nested=True) as nested_run:
                            yield nested_run
                            return
                    except Exception as e:
                        logger.warning("nested_run_failed_using_existing", error=str(e))

                # If nesting fails or is not requested, yield the active run or a stub.
                yield active or mlflow.active_run()
            else:
                # Only set experiment if we are NOT already in a run to avoid conflicts
                if self.study_name:
                    try:
                        mlflow.set_experiment(self.study_name)
                    except Exception as e:
                        logger.warning("set_experiment_failed", error=str(e), study=self.study_name)

                with mlflow.start_run() as new_run:
                    yield new_run

        return run_context()

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Logs a dictionary as a JSON artifact."""
        mlflow.log_dict(dictionary, artifact_file)

    def log_metrics(self, accuracy: float, rmse: float, duration: float, framework: str) -> None:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("duration", duration)

        TRAINING_DURATION.labels(framework=framework).observe(duration)
        MODEL_ACCURACY.labels(framework=framework).set(accuracy)
        MODEL_RMSE.labels(model_type=framework, dataset="validation").set(rmse)

    def log_error(self, framework: str, error: str) -> None:
        TRAINING_ERRORS.labels(framework=framework).inc()
        logger.error("training_failed", framework=framework, error=error)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def log_model(self, model: Any, framework: str, artifact_path: str = "model") -> None:
        """Log the model to MLflow with optional ONNX conversion."""
        import mlflow
        import mlflow.sklearn
        import mlflow.xgboost
        import mlflow.pytorch

        logger.info("logging_model", framework=framework, path=artifact_path)

        if framework == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        elif framework == "pytorch" or framework == "torch":
            mlflow.pytorch.log_model(model, artifact_path)
        elif framework == "tensorflow" or framework == "keras":
            mlflow.tensorflow.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)

    def register_model(self, model_name: str, run_id: str, artifact_path: str = "model") -> Any:
        """Register the model in the MLflow Model Registry."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.register_model(model_uri, model_name)

    def transition_model_stage(self, model_name: str, version: int, stage: str) -> None:
        """Promote or rollback a model version in the registry."""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True if stage == "Production" else False
        )
        logger.info("model_stage_transitioned", name=model_name, version=version, stage=stage)

        # OPTIMIZED: Auto-export to ONNX for production inference
        try:
            from src.ml.strategies import get_strategy
            strategy = get_strategy(framework)
            onnx_path = os.path.join(tempfile.gettempdir(), f"{artifact_path}.onnx")
            # We assume a default input dim of 20 for now; in a real scenario
            # this would be passed or extracted from the model
            strategy.export_onnx(model, onnx_path, input_dim=20)
            if os.path.exists(onnx_path):
                mlflow.log_artifact(onnx_path, f"{artifact_path}_onnx")
        except Exception as e:
            logger.warning("onnx_auto_export_failed", error=str(e))

    def log_feature_importance(self, importance: dict[str, float], framework: str) -> None:
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

    def push_to_gateway(self) -> None:
        push_metrics(job_name=self.study_name)
