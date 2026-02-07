import logging
import os
import subprocess

import click
import mlflow
import uvicorn

logger = logging.getLogger(__name__)

# Basic MLflow setup for client operations
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


@click.command()
@click.option(
    "--model-name",
    required=True,
    help="Registered MLflow model name",
)
@click.option("--model-version", type=str, default="latest", help="Model version")
@click.option("--port", type=int, default=5001, help="Port to serve on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to bind to")
@click.option(
    "--backend",
    type=click.Choice(["onnx", "mlflow"]),
    default="onnx",
    help="Serving backend",
)
@click.option(
    "--onnx-path", type=str, help="Path to ONNX model file (required for onnx backend)"
)
def serve_model(
    model_name: str,
    model_version: str,
    port: int,
    host: str,
    backend: str,
    onnx_path: str,
):
    """
    Serves a model using either high-performance ONNX Runtime or standard MLflow.
    """
    logger.info(
        f"Starting model server: model={model_name}, version={model_version}, backend={backend}"
    )

    if backend == "onnx":
        if not onnx_path:
            # Try to infer path or download from MLflow
            logger.info("Attempting to locate ONNX artifact from MLflow...")
            onnx_path = f"models/{model_name}.onnx"
            if not os.path.exists(onnx_path):
                logger.error("ONNX model file not found. Please provide --onnx-path.")
                return

        os.environ["ONNX_MODEL_PATH"] = onnx_path
        logger.info(f"Launching high-performance ONNX server on {host}:{port}")
        uvicorn.run(
            "src.ml.serving.onnx_serving:app", host=host, port=port, log_level="info"
        )

    else:
        # Legacy MLflow subprocess serving
        model_uri = f"models:/{model_name}/{model_version}"
        command = [
            "mlflow",
            "models",
            "serve",
            "-m",
            model_uri,
            "-p",
            str(port),
            "-h",
            host,
        ]
        logger.info(f"Executing legacy MLflow command: {' '.join(command)}")
        subprocess.run(command)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    serve_model()
