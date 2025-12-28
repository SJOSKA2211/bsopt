import logging

from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def rollback_model(model_name: str, target_stage: str = "Production"):
    """
    Rollback model to the previous version in the specified stage.
    """
    client = MlflowClient()
    try:
        # Get all versions of the model
        versions = client.get_latest_versions(model_name, stages=[target_stage])
        if not versions:
            logger.warning(f"No models found in stage {target_stage}")
            return False

        current_version = versions[0].version

        # Get all versions to find the one before current
        all_versions = client.search_model_versions(f"name='{model_name}'")
        sorted_versions = sorted(all_versions, key=lambda x: int(x.version), reverse=True)

        previous_version = None
        for v in sorted_versions:
            if int(v.version) < int(current_version):
                previous_version = v.version
                break

        if previous_version:
            logger.info(
                f"Rolling back {model_name} from version {current_version} to {previous_version}"
            )
            client.transition_model_version_stage(
                name=model_name,
                version=previous_version,
                stage=target_stage,
                archive_existing_versions=True,
            )
            return True
        else:
            logger.warning("No previous version found to rollback to.")
            return False

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False


def monitor_and_rollback(model_name: str, metric_name: str, threshold: float):
    """
    Monitor model performance and trigger rollback if below threshold.
    """
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return

    latest = versions[0]
    run = client.get_run(latest.run_id)
    metric_value = run.data.metrics.get(metric_name)

    if metric_value and metric_value < threshold:
        logger.warning(
            f"Performance degradation detected: {metric_name}={metric_value} < {threshold}"
        )
        rollback_model(model_name)
