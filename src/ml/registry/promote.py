"""
MLflow Model Promotion and App Dispatch script.
Handles transitioning model versions in the registry and notifying the application layer.
"""

import argparse

import mlflow
import structlog

logger = structlog.get_logger()


def promote_model(model_name: str, run_id: str, stage: str = "Production") -> None:
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


def notify_app_of_update(model_name: str, version: str) -> None:
    """
    Triggers a reload in the serving layer.
    """
    import asyncio

    import httpx

    serving_url = "http://api:8000/ml/reload"
    logger.info("notifying_serving_layer", model=model_name, version=version, url=serving_url)

    async def trigger() -> None:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    serving_url, json={"model_name": model_name, "version": version}
                )
                if resp.status_code == 200:
                    logger.info("serving_layer_reloaded", status=resp.status_code)
                else:
                    logger.error(
                        "serving_layer_reload_failed_triggering_rollback",
                        status=resp.status_code,
                        text=resp.text,
                    )
                    from src.ml.utils.rollback import rollback_model

                    rollback_model(model_name)
        except Exception as e:
            logger.error("serving_layer_notification_error_triggering_rollback", error=str(e))
            from src.ml.utils.rollback import rollback_model

            rollback_model(model_name)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(trigger())
        else:
            loop.run_until_complete(trigger())
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--stage", default="Production")
    args = parser.parse_args()

    promote_model(args.model, args.run_id, args.stage)