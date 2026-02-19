"""
MLflow Model Promotion and App Dispatch script.
Handles transitioning model versions in the registry and notifying the application layer.
"""

import argparse

import mlflow
import structlog

logger = structlog.get_logger()


def promote_model(model_name, run_id, stage="Production"):
    client = mlflow.tracking.MlflowClient()

    # 1. Register model if not already
    model_uri = f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri, model_name)

    # 2. Transition stage
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info("model_promoted", name=model_name, version=version.version, stage=stage)

    # 3. App Dispatch (Notify system of update)
    # This triggers a repository dispatch or a direct API call to reload models
    notify_app_of_update(model_name, version.version)


def notify_app_of_update(model_name, version):
    # Simulated dispatch logic
    logger.info("app_notification_sent", model=model_name, version=version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--stage", default="Production")
    args = parser.parse_args()

    promote_model(args.model, args.run_id, args.stage)
