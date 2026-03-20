import os
import time

import requests
import structlog

logger = structlog.get_logger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
RAY_DASHBOARD_URL = os.getenv("RAY_DASHBOARD_URL", "http://ray-head:8265")


def check_ray_health():
    """Poll Ray dashboard for status."""
    try:
        resp = requests.get(f"{RAY_DASHBOARD_URL}/api/jobs/", timeout=5)
        if resp.status_code == 200:
            return True
    except Exception as e:
        logger.error("ray_dashboard_unreachable", error=str(e))
    return False


def check_mlflow_status():
    """Poll MLflow for experiment status."""
    try:
        resp = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def monitor_and_heal():
    """Self-healing loop."""
    logger.info("mlflow_watchdog_started", tracking_uri=MLFLOW_TRACKING_URI)

    while True:
        if not check_ray_health():
            logger.warning("ray_head_down_attempting_respawn")
            # In a production k8s/docker environment, this might trigger a restart
            # Here we log and could potentially call a self-healing script

        if not check_mlflow_status():
            logger.error("mlflow_down_system_degraded")

        # Check for OOM jobs or failed Ray instances
        # Logic to auto-adjust parameters and respawn would go here

        time.sleep(30)


if __name__ == "__main__":
    monitor_and_heal()
