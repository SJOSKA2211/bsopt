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
    increment_counter,
    observe_latency,
    push_metrics,
    set_gauge,
)

logger = structlog.get_logger()


class ExperimentTracker:
    """
    Handles all observability, logging, and metrics for ML training.
    """

    def __init__(self, study_name: str, tracking_uri: str | None = None) -> None:
        from src.shared.config import settings

        self.study_name = study_name
        uri = tracking_uri or settings.tracking_uri
        mlflow.set_tracking_uri(uri)

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
                    # Injection of Galactic Governance Tags
                    import socket

                    mlflow.set_tags(
                        {
                            "bsopt.host": socket.gethostname(),
                            "bsopt.environment": os.getenv("ENVIRONMENT", "production"),
                            "bsopt.layer": "ML-Manifold",
                        }
                    )
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
        # OPTIMIZED: Batch MLflow metrics to reduce network overhead
        mlflow.log_metrics({"accuracy": accuracy, "rmse": rmse, "duration": duration})

        observe_latency(TRAINING_DURATION, duration, {"framework": framework})
        set_gauge(MODEL_ACCURACY, accuracy, {"framework": framework})
        set_gauge(MODEL_RMSE, rmse, {"model_type": framework, "dataset": "validation"})

    def log_error(self, framework: str, error: str) -> None:
        increment_counter(TRAINING_ERRORS, labels={"framework": framework})
        logger.error("training_failed", framework=framework, error=error)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def log_model(self, model: Any, framework: str, artifact_path: str = "model") -> None:
        """Log the model to MLflow with optional ONNX conversion."""
        import mlflow
        import mlflow.pytorch
        import mlflow.sklearn
        import mlflow.xgboost

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

    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        model: Any = None,
        framework: str = "sklearn",
    ) -> None:
        """Promote or rollback a model version in the registry."""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True if stage == "Production" else False,
        )
        logger.info("model_stage_transitioned", name=model_name, version=version, stage=stage)

        if stage == "Production" and model is not None:
            self.export_to_onnx(model, framework, f"{model_name}_v{version}.onnx")

    def export_to_onnx(self, model: Any, framework: str, filename: str) -> str | None:
        """Export a model to ONNX format for production inference (Galactic Optimized)."""

        with tempfile.TemporaryDirectory() as temp_dir:
            onx_path = os.path.join(temp_dir, filename)

            try:
                if framework == "xgboost":
                    import onnxmltools
                    from onnxmltools.convert.common.data_types import FloatTensorType

                    initial_type = [("float_input", FloatTensorType([None, 10]))]
                    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
                    onnxmltools.utils.save_model(onnx_model, onx_path)
                elif framework in ["pytorch", "torch"]:
                    import torch

                    dummy_input = torch.randn(1, 10)
                    torch.onnx.export(model, dummy_input, onx_path)
                elif framework == "sklearn":
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType

                    initial_type = [("float_input", FloatTensorType([None, 10]))]
                    onx = convert_sklearn(model, initial_types=initial_type)
                    with open(onx_path, "wb") as f:
                        f.write(onx.SerializeToString())
                else:
                    logger.warning("unsupported_framework_for_onnx", framework=framework)
                    return None

                self.log_artifact(onx_path)
                logger.info("onnx_model_exported", path=onx_path)
                # Note: The file will be deleted when the TemporaryDirectory context exits,
                # but MLflow has already uploaded it.
                return onx_path
            except Exception as e:
                logger.error("onnx_export_failed", error=str(e), framework=framework)
                return None

    def log_feature_importance(self, importance: dict[str, float], framework: str) -> None:
        """Saves and logs feature importance plots (Institutional Standard)."""
        plt.figure(figsize=(12, 8))
        names = list(importance.keys())
        values = list(importance.values())

        # Sort for better visual representation
        sorted_idx = [i for i, _ in sorted(enumerate(values), key=lambda x: x[1])]
        plt.barh([names[i] for i in sorted_idx], [values[i] for i in sorted_idx], color="royalblue")
        plt.title(f"Galactic Feature Importance ({framework})", fontsize=14)
        plt.xlabel("Importance Score")
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = os.path.join(temp_dir, "feature_importance.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.log_artifact(plot_path)

    def push_to_gateway(self) -> None:
        push_metrics(job_name=self.study_name)
