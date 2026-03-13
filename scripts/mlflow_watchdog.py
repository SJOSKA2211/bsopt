"""
MLflow & Ray Watchdog
====================
Autonomously monitors training runs and recovers from hangs or crashes.
Implements the 'Self-Healing MLOps' pattern.
"""

import os
import time

import structlog

logger = structlog.get_logger(__name__)

# Config
CHECK_INTERVAL = 60  # seconds
MAX_RUN_TIME = 3600  # 1 hour
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def get_runs_by_status(status):
    """Retrieves runs by status from MLflow API."""
    import requests

    try:
        resp = requests.get(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search",
            params={"filter": f"status = '{status}'"},
        )
        if resp.status_code == 200:
            return resp.json().get("runs", [])
    except Exception as e:
        logger.error("mlflow_connection_failed", error=str(e))
    return []


def kill_and_restart(run_id, ticker, adapt_params=False):
    """Terminates a runaway Ray job and restarts it via Celery, optionally adapting parameters."""
    logger.warning("restarting_job", run_id=run_id, ticker=ticker, adapt_params=adapt_params)

    # 1. Terminate Ray job if possible and mark FAILED in MLflow
    import requests

    requests.post(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/update",
        json={"run_id": run_id, "status": "FAILED", "end_time": int(time.time() * 1000)},
    )

    # 2. Adapt parameters if failed (e.g. reduce batch size, lower learning rate)
    kwargs = {"ticker": ticker}
    if adapt_params:
        kwargs["batch_size"] = 32  # Example adaptation
        kwargs["learning_rate"] = 0.0001
        logger.info("parameters_adapted", **kwargs)

    # 3. Trigger restart via Celery
    from src.tasks.ml_tasks import run_cross_sectional_training

    run_cross_sectional_training.delay(**kwargs)
    logger.info("job_restart_triggered", ticker=ticker)


def main():
    logger.info("watchdog_started", interval=CHECK_INTERVAL)
    while True:
        # Check hanging runs
        running_runs = get_runs_by_status("RUNNING")
        for run in running_runs:
            start_time = int(run["info"]["start_time"]) / 1000
            run_id = run["info"]["run_id"]
            ticker = next(
                (t["value"] for t in run["data"].get("tags", []) if t["key"] == "ticker"), "unknown"
            )

            elapsed = time.time() - start_time
            if elapsed > MAX_RUN_TIME:
                logger.warning("runaway_job_detected", run_id=run_id, ticker=ticker)
                kill_and_restart(run_id, ticker, adapt_params=False)

        # Check failed runs for auto-recovery
        failed_runs = get_runs_by_status("FAILED")
        for run in failed_runs:
            # Only respawn recently failed runs to avoid infinite loops
            end_time = int(run["info"]["end_time"]) / 1000
            run_id = run["info"]["run_id"]
            ticker = next(
                (t["value"] for t in run["data"].get("tags", []) if t["key"] == "ticker"), "unknown"
            )

            # If failed within the last CHECK_INTERVAL, we adapt and respawn
            if time.time() - end_time < CHECK_INTERVAL:
                logger.warning("failed_job_detected", run_id=run_id, ticker=ticker)
                kill_and_restart(run_id, ticker, adapt_params=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
