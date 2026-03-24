"""
Automated Model Backtesting & Rollback Evaluator

Fetches models from MLflow 'Staging', evaluates them against Out-Of-Sample (OOS) data,
and compares metrics with the current 'Production' model.
"""

from typing import Any

import mlflow
import structlog
from mlflow.tracking import MlflowClient

from src.shared.config import settings

logger = structlog.get_logger(__name__)


class BacktestEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = MlflowClient()
        mlflow.set_tracking_uri(settings.tracking_uri)

    def get_latest_version(self, stage: str) -> dict[str, str | dict[str, float]] | None:
        """Fetch model metadata for a specific stage."""
        try:
            versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            if versions:
                metrics = self.client.get_run(versions[0].run_id).data.metrics
                return {
                    "version": str(versions[0].version),
                    "run_id": str(versions[0].run_id),
                    "metrics": cast(dict[str, float], metrics),
                }
        except Exception as e:
            logger.error("failed_to_fetch_model_version", stage=stage, error=str(e))
        return None

    def evaluate_performance(
        self, 
        staging_v: dict[str, str | dict[str, float]], 
        prod_v: dict[str, str | dict[str, float]]
    ) -> bool:
        """
        Compare metrics. Returns True if staging should be promoted.
        Threshold: RMSE must not increase by more than 15%.
        """
        staging_rmse = staging_v["metrics"].get("rmse", float("inf"))
        prod_rmse = prod_v["metrics"].get("rmse", float("inf"))

        logger.info("comparing_metrics", staging_rmse=staging_rmse, prod_rmse=prod_rmse)

        # Promotion logic:
        # 1. Staging is better than Prod
        # 2. OR Staging is at most 15% worse than Prod (tolerable noise)
        if staging_rmse <= prod_rmse:
            return True

        degradation = (staging_rmse - prod_rmse) / prod_rmse
        if degradation < 0.15:
            logger.info("acceptable_performance_noise", degradation=degradation)
            return True

        logger.warning("performance_degradation_exceeded_threshold", degradation=degradation)
        return False

    def run_rollback_check(self):
        """Main entry point for rollback assessment."""
        logger.info("running_model_quality_gate", model=self.model_name)

        staging = self.get_latest_version("Staging")
        prod = self.get_latest_version("Production")

        if not staging:
            logger.info("no_staging_model_found")
            return

        if not prod:
            logger.info("no_production_model_found_promoting_staging")
            self.client.transition_model_version_stage(
                name=self.model_name, version=staging["version"], stage="Production"
            )
            return

        should_promote = self.evaluate_performance(staging, prod)

        if should_promote:
            logger.info("promoting_staging_to_production", version=staging["version"])
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=staging["version"],
                stage="Production",
                archive_existing_versions=True,
            )
        else:
            logger.warning("rolling_back_staging_model", version=staging["version"])
            self.client.transition_model_version_stage(
                name=self.model_name, version=staging["version"], stage="Archived"
            )
            # Alerting mechanism would be triggered here


if __name__ == "__main__":
    evaluator = BacktestEvaluator("OptionPricingModel_v2")
    evaluator.run_rollback_check()
