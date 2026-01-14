import logging
import os

import click
import mlflow

logger = logging.getLogger(__name__)

# Basic MLflow setup for client operations
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


@click.command()
@click.option(
    "--model-name",
    required=True,
    help='Registered MLflow model name (e.g., "XGBoost_Pricing_Enterprise")',
)
@click.option(
    "--model-version", type=str, default="latest", help='Model version (e.g., "1" or "latest")'
)
@click.option("--port", type=int, default=5001, help="Port to serve the model on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to bind the serving process to")
@click.option("--workers", type=int, default=1, help="Number of worker processes")
def serve_model(model_name: str, model_version: str, port: int, host: str, workers: int):
    """
    Serves a registered MLflow model using `mlflow models serve`.
    """
    logger.info(
        f"Attempting to serve model '{model_name}' version '{model_version}' on {host}:{port}"
    )

    # Construct the model URI
    model_uri = f"models:/{model_name}/{model_version}"

    try:
        # Use subprocess to run the mlflow models serve command
        # This allows us to control the serving process outside of Python's main process
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
            "--workers",
            str(workers),
        ]

        # Log the command being executed for debugging
        logger.info(f"Executing serving command: {' '.join(command)}")

        # Start the serving process. It will block until terminated.
        # For a production scenario, you might want to run this in a detached process
        # or use a process manager (e.g., Gunicorn, systemd)
        import subprocess

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # Print output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end="")

        process.wait()  # Wait for the process to terminate

        if process.returncode != 0:
            logger.error(f"Model serving process exited with non-zero code: {process.returncode}")
            raise Exception("Model serving failed.")

    except FileNotFoundError:
        logger.error("MLflow CLI not found. Make sure MLflow is installed and in your PATH.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during model serving: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    serve_model()
