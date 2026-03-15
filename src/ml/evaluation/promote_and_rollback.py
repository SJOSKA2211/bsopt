import argparse
import mlflow
import structlog
from src.ml.evaluation.compare_models import compare_models
from src.config import settings

logger = structlog.get_logger(__name__)

def automate_deployment(model_name: str, challenger_run_id: str):
    """
    Automates the Champion-Challenger promotion cycle.
    If challenger wins, it is promoted to 'Production'.
    Otherwise, the existing champion is retained.
    """
    client = mlflow.tracking.MlflowClient()
    
    logger.info("deployment_cycle_started", model=model_name, challenger=challenger_run_id)
    
    # 1. Compare models
    should_promote = compare_models(model_name, challenger_run_id)
    
    if should_promote:
        logger.info("promoting_new_model", model=model_name, run_id=challenger_run_id)
        
        # Identify the latest version for this run
        versions = client.search_model_versions(f"run_id='{challenger_run_id}'")
        if not versions:
            # Register if not exists
            model_uri = f"runs:/{challenger_run_id}/model"
            mv = client.create_model_version(model_name, model_uri, challenger_run_id)
            version = mv.version
        else:
            version = versions[0].version
            
        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info("promotion_successful", version=version)
    else:
        logger.warning("rollback_triggered_keeping_champion", model=model_name)
        # Register but leave in Staging or mark as rejected
        versions = client.search_model_versions(f"run_id='{challenger_run_id}'")
        if versions:
            client.transition_model_version_stage(
                name=model_name,
                version=versions[0].version,
                stage="None" # Rejected/Archived
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(settings.tracking_uri)
    automate_deployment(args.model, args.run_id)
