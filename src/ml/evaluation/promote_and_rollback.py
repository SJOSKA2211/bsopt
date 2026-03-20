"""
Model Promotion and Rollback System

Handles champion-challenger deployment with automated rollback
if performance degrades beyond acceptable thresholds.

Features:
- Automated comparison against production model
- Sharpe ratio, max drawdown, and accuracy comparisons
- Graceful rollback on degradation
- MLflow integration for model registry
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import structlog
from mlflow.tracking import MlflowClient

from src.shared.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    sharpe_ratio: float
    max_drawdown: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mape: float
    rmse: float
    mean_return: float
    volatility: float


@dataclass
class RollbackConfig:
    """Configuration for rollback decisions."""
    sharpe_degradation_threshold: float = 0.05
    max_drawdown_threshold: float = 0.10
    accuracy_degradation_threshold: float = 0.05
    rollback_cooldown_seconds: int = 3600
    min_sample_size: int = 100


class ModelComparator:
    """
    Compares candidate model against production model.
    """

    def __init__(self, model_name: str, client: MlflowClient | None = None):
        self.model_name = model_name
        self.client = client or MlflowClient()

    def get_production_model(self) -> dict[str, Any] | None:
        """Get current production model version."""
        try:
            prod_versions = self.client.search_model_versions(
                f"name='{self.model_name}' AND stage='Production'"
            )
            if prod_versions:
                version = prod_versions[0]
                run = self.client.get_run(version.run_id)
                return {
                    "version": version.version,
                    "run_id": version.run_id,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "creation_timestamp": version.creation_timestamp,
                }
        except Exception as e:
            logger.warning("no_production_model_found", model=self.model_name, error=str(e))

        return None

    def get_candidate_model(self, run_id: str) -> dict[str, Any]:
        """Get candidate model by run ID."""
        run = self.client.get_run(run_id)
        return {
            "run_id": run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "status": run.info.status,
        }

    def extract_sharpe_ratio(self, metrics: dict[str, float]) -> float:
        """Extract or calculate Sharpe ratio from metrics."""
        if "sharpe_ratio" in metrics:
            return metrics["sharpe_ratio"]

        if "mean_return" in metrics and "volatility" in metrics and metrics["volatility"] > 0:
            return metrics["mean_return"] / metrics["volatility"]

        return 0.0

    def extract_max_drawdown(self, metrics: dict[str, float]) -> float:
        """Extract max drawdown from metrics."""
        return metrics.get("max_drawdown", 0.0)

    def extract_accuracy(self, metrics: dict[str, float]) -> float:
        """Extract accuracy from metrics."""
        return metrics.get("accuracy", 0.0)

    def compare_models(
        self,
        candidate_run_id: str,
        config: RollbackConfig,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Compare candidate model against production model.

        Returns:
            Tuple of (should_deploy, comparison_details)
        """
        production = self.get_production_model()
        candidate = self.get_candidate_model(candidate_run_id)

        if not production:
            logger.info("no_production_model_using_candidate", candidate_run_id=candidate_run_id)
            return True, {"reason": "no_production_model"}

        prod_metrics = production["metrics"]
        cand_metrics = candidate["metrics"]

        prod_sharpe = self.extract_sharpe_ratio(prod_metrics)
        cand_sharpe = self.extract_sharpe_ratio(cand_metrics)

        prod_drawdown = self.extract_max_drawdown(prod_metrics)
        cand_drawdown = self.extract_max_drawdown(cand_metrics)

        prod_accuracy = self.extract_accuracy(prod_metrics)
        cand_accuracy = self.extract_accuracy(cand_metrics)

        sharpe_degradation = (prod_sharpe - cand_sharpe) / prod_sharpe if prod_sharpe != 0 else 0
        drawdown_change = cand_drawdown - prod_drawdown
        accuracy_change = cand_accuracy - prod_accuracy

        comparison = {
            "production": {
                "sharpe_ratio": prod_sharpe,
                "max_drawdown": prod_drawdown,
                "accuracy": prod_accuracy,
            },
            "candidate": {
                "sharpe_ratio": cand_sharpe,
                "max_drawdown": cand_drawdown,
                "accuracy": cand_accuracy,
            },
            "degradation": {
                "sharpe_degradation_pct": sharpe_degradation * 100,
                "drawdown_change": drawdown_change,
                "accuracy_change": accuracy_change,
            },
        }

        should_deploy = True
        rollback_reasons = []

        if prod_sharpe > 0 and sharpe_degradation > config.sharpe_degradation_threshold:
            should_deploy = False
            rollback_reasons.append(f"Sharpe ratio degraded by {sharpe_degradation * 100:.2f}%")

        if cand_drawdown > prod_drawdown + config.max_drawdown_threshold:
            should_deploy = False
            rollback_reasons.append(
                f"Max drawdown increased by {drawdown_change * 100:.2f}%"
            )

        if prod_accuracy > 0 and accuracy_change < -config.accuracy_degradation_threshold:
            should_deploy = False
            rollback_reasons.append(f"Accuracy degraded by {abs(accuracy_change) * 100:.2f}%")

        comparison["rollback_reasons"] = rollback_reasons
        comparison["should_deploy"] = should_deploy

        if should_deploy:
            logger.info(
                "candidate_approved",
                model=self.model_name,
                candidate_run_id=candidate_run_id,
                sharpe=cand_sharpe,
            )
        else:
            logger.warning(
                "candidate_rejected",
                model=self.model_name,
                candidate_run_id=candidate_run_id,
                reasons=rollback_reasons,
            )

        return should_deploy, comparison


class ModelPromoter:
    """
    Handles model promotion and rollback operations.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = MlflowClient()
        self.comparator = ModelComparator(model_name, self.client)

    def promote_candidate(
        self,
        run_id: str,
        config: RollbackConfig | None = None,
    ) -> dict[str, Any]:
        """
        Promote candidate model to production.

        Args:
            run_id: MLflow run ID for candidate model
            config: Rollback configuration

        Returns:
            Dictionary with promotion details
        """
        config = config or RollbackConfig()

        should_deploy, comparison = self.comparator.compare_models(run_id, config)

        if not should_deploy:
            return {
                "status": "rejected",
                "comparison": comparison,
            }

        try:
            # Check existing versions to ensure model is registered
            self.client.search_model_versions(
                f"name='{self.model_name}'"
            )

            model_uri = f"runs:/{run_id}/model"
            mv = self.client.create_model_version(
                self.model_name,
                model_uri,
                run_id,
            )
            version = mv.version

            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )

            self._backup_production_model()

            logger.info(
                "model_promoted",
                model=self.model_name,
                version=version,
                run_id=run_id,
            )

            return {
                "status": "promoted",
                "version": version,
                "run_id": run_id,
                "comparison": comparison,
            }

        except Exception as e:
            logger.error(
                "promotion_failed",
                model=self.model_name,
                run_id=run_id,
                error=str(e),
            )
            raise

    def rollback_to_production(self) -> dict[str, Any]:
        """
        Rollback to previous production model.

        Returns:
            Dictionary with rollback details
        """
        try:
            archived_versions = self.client.search_model_versions(
                f"name='{self.model_name}' AND stage='Archived'"
            )

            if archived_versions:
                latest_archived = archived_versions[0]

                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=latest_archived.version,
                    stage="Production",
                    archive_existing_versions=True,
                )

                logger.warning(
                    "rollback_completed",
                    model=self.model_name,
                    rolled_back_to_version=latest_archived.version,
                )

                return {
                    "status": "rolled_back",
                    "version": latest_archived.version,
                }

            logger.error("no_archived_versions_for_rollback")
            return {"status": "failed", "reason": "no_archived_versions"}

        except Exception as e:
            logger.error("rollback_failed", error=str(e))
            raise

    def _backup_production_model(self) -> None:
        """Backup current production model artifacts."""
        backup_dir = Path(settings.MODEL_ARTIFACT_DIR) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        prod_versions = self.client.search_model_versions(
            f"name='{self.model_name}' AND stage='Production'"
        )
        if prod_versions:
            version = prod_versions[0]
            try:
                local_path = self.client.download_artifacts(
                    version.run_id,
                    "model",
                    dst_path=str(backup_dir / f"v{version.version}"),
                )
                logger.info("production_model_backed_up", path=local_path)
            except Exception as e:
                logger.warning("backup_failed", error=str(e))


def automate_deployment(
    model_name: str,
    challenger_run_id: str,
    config: RollbackConfig | None = None,
) -> dict[str, Any]:
    """
    CLI-friendly deployment automation.

    Args:
        model_name: Name of registered model
        challenger_run_id: Run ID of candidate model
        config: Rollback configuration

    Returns:
        Deployment result dictionary
    """
    mlflow.set_tracking_uri(settings.tracking_uri)

    promoter = ModelPromoter(model_name)
    return promoter.promote_candidate(challenger_run_id, config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Registered model name")
    parser.add_argument("--run-id", required=True, help="Candidate run ID")
    parser.add_argument(
        "--sharpe-threshold",
        type=float,
        default=0.05,
        help="Max Sharpe ratio degradation (fraction)",
    )
    parser.add_argument(
        "--dd-threshold",
        type=float,
        default=0.10,
        help="Max drawdown increase (fraction)",
    )
    args = parser.parse_args()

    config = RollbackConfig(
        sharpe_degradation_threshold=args.sharpe_threshold,
        max_drawdown_threshold=args.dd_threshold,
    )

    result = automate_deployment(args.model, args.run_id, config)
    print(f"\nDeployment Result: {result}")
