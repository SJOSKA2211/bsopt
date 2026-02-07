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
        sorted_versions = sorted(
            all_versions, key=lambda x: int(x.version), reverse=True
        )

        previous_version = None
        for v in sorted_versions:
            if int(v.version) < int(current_version):
                previous_version = v.version
                break

        if previous_version:
            logger.info(
                f"Rolling back {model_name} from version {current_version} to {previous_version}"
            )

            # 1. Transition current production model to 'Archived'
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Archived",
                archive_existing_versions=False,  # Don't archive other models if any
            )
            logger.info(
                f"Model {model_name} version {current_version} moved to Archived."
            )

            # 2. Transition previous version to the target_stage (e.g., 'Production')
            client.transition_model_version_stage(
                name=model_name,
                version=previous_version,
                stage=target_stage,
                archive_existing_versions=False,  # We explicitly archived the current version above
            )
            logger.info(
                f"Model {model_name} version {previous_version} moved to {target_stage}."
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
        logger.info(
            f"No model found in Production stage for {model_name}. Skipping monitoring."
        )
        return

    if len(versions) > 1:
        logger.warning(
            f"Multiple models found in Production stage for {model_name}. "
            "Monitoring will use the first one. Please ensure only one model "
            "is in Production for consistent behavior."
        )

    latest = versions[0]
    run = client.get_run(latest.run_id)
    metric_value = run.data.metrics.get(metric_name)

    if (
        metric_value is not None and metric_value < threshold
    ):  # Check for None explicitly
        logger.warning(
            f"Performance degradation detected: {metric_name}={metric_value} < {threshold}. "
            f"Triggering rollback for model {model_name} (version {latest.version})."
        )
        rollback_model(model_name)
    elif metric_value is None:
        logger.warning(
            f"Metric '{metric_name}' not found for model {model_name} version {latest.version}. Skipping monitoring."
        )
