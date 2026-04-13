import os

import structlog

logger = structlog.get_logger(__name__)


def check_model_loaded() -> bool:
    """Verifies that the ONNX model is loaded and session is active."""
    from src.ml.serving.onnx_serving import model_server

    return model_server is not None


def check_mlflow_connection() -> bool:
    """Verifies connectivity to the MLflow tracking server."""
    try:
        from mlflow.tracking import MlflowClient

        from src.shared.config import settings

        client = MlflowClient(tracking_uri=settings.tracking_uri)
        client.search_experiments(max_results=1)
        return True
    except Exception as e:
        logger.error("health_mlflow_check_failed", error=str(e))
        return False


def get_serving_health() -> dict:
    """Aggregates neural pricing serving health components."""
    model_ok = check_model_loaded()
    mlflow_ok = check_mlflow_connection()

    # Inference readiness only requires the model to be loaded.
    # MLflow connectivity is advisory (used for training/logging, not serving).
    status = "healthy" if model_ok else "degraded"

    return {
        "status": status,
        "model_loaded": model_ok,
        "mlflow_connected": mlflow_ok,
        "service": "ml-inference-service",
        "model_path": os.getenv("ONNX_MODEL_PATH", "unknown"),
    }