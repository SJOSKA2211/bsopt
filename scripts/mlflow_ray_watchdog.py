"""
MLflow-Ray Watchdog Service

Monitors MLflow runs for failures and automatically triggers Ray job respawns
with optimized resource allocation or backoff strategies.
"""

import asyncio
from typing import Any

import mlflow
import structlog
from mlflow.entities import RunStatus

from src.ml.distributed_training import BSOptDistributedTrainer

logger = structlog.get_logger(__name__)


class MLflowRayWatchdog:
    """
    Persistent watchdog for Ray training jobs.
    """

    def __init__(self, experiment_name: str, check_interval: int = 60):
        self.experiment_name = experiment_name
        self.check_interval = check_interval
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        self._experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self._experiment:
            mlflow.create_experiment(experiment_name)
            self._experiment = mlflow.get_experiment_by_name(experiment_name)

    async def monitor(self):
        """
        Main monitoring loop.
        """
        logger.info("watchdog_started", experiment=self.experiment_name)

        while True:
            try:
                # 1. Fetch recent failed or killed runs
                runs = mlflow.search_runs(
                    experiment_ids=[self._experiment.experiment_id],
                    filter_string=f"status = '{RunStatus.to_string(RunStatus.FAILED)}' OR status = '{RunStatus.to_string(RunStatus.KILLED)}'",
                    order_by=["start_time DESC"],
                    max_results=5,
                )

                for _, run in runs.iterrows():
                    run_id = run["run_id"]
                    # Check if we already tried to recover this run (tag-based)
                    tags = mlflow.get_run(run_id).data.tags
                    if "recovered_by_watchdog" in tags:
                        continue

                    logger.warning("failed_run_detected", run_id=run_id, status=run["status"])

                    # 2. Trigger auto-recovery
                    await self.recover_run(run)

            except Exception as e:
                logger.error("watchdog_loop_error", error=str(e))

            await asyncio.sleep(self.check_interval)

    async def recover_run(self, run: Any):
        """
        Respawn the Ray job with adjusted parameters.
        """
        run_id = run["run_id"]
        logger.info("initiating_recovery", run_id=run_id)

        # Mark as recovered to avoid duplicate respawns
        mlflow.set_tag("recovered_by_watchdog", "true", run_id=run_id)

        # Extract params from failed run to preserve configuration
        params = mlflow.get_run(run_id).data.params

        # ⚡ OOM Recovery Strategy: Reduce batch size AND Increase allocated memory per worker
        if "batch_size" in params:
            new_batch_size = max(8, int(params["batch_size"]) // 2)
            params["batch_size"] = str(new_batch_size)
            logger.info(
                "recovery_strategy_applied",
                original_batch=params["batch_size"],
                new_batch=new_batch_size,
            )

        # 🚀 Resource Negotiation: Request MORE memory per worker if previous failed
        # This is a hint to the BSOptDistributedTrainer
        config = {k: self._infer_type(v) for k, v in params.items()}
        config["_recovery_attempt"] = True
        config["resources_per_worker"] = {"CPU": 1, "memory": 4 * 1024 * 1024 * 1024}  # 4GB floor

        # 3. Respawn Job via Ray
        try:
            trainer = BSOptDistributedTrainer()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, trainer.run, config)

            logger.info("recovery_job_submitted", run_id=run_id)
        except Exception as e:
            logger.error("recovery_submission_failed", run_id=run_id, error=str(e))

    def _infer_type(self, val: str) -> Any:
        if val.isdigit():
            return int(val)
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        return val


if __name__ == "__main__":
    watchdog = MLflowRayWatchdog("distributed_dt_v1")
    asyncio.run(watchdog.monitor())
