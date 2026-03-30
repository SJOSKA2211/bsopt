import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import requests
import structlog

logger = structlog.get_logger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RAY_DASHBOARD_URL = os.getenv("RAY_DASHBOARD_URL", "http://localhost:8265")
CHECK_INTERVAL = 30
HEALTH_PORT = 8080

# Global health status for the HTTP server
IS_HEALTHY = False

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            if IS_HEALTHY:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"Service Degraded")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence default HTTP logging to keep logs clean
        return

class MLflowWatchdog:
    """
    Self-healing watchdog for Ray and MLflow components.
    Monitors job status, memory pressure, and service availability.
    """

    def __init__(self):
        self.failed_jobs = set()
        self.consecutive_ray_failures = 0
        self.max_ray_failures = 5

    def check_ray_health(self) -> bool:
        """Poll Ray dashboard for status and active jobs."""
        try:
            resp = requests.get(f"{RAY_DASHBOARD_URL}/api/jobs/", timeout=5)
            if resp.status_code == 200:
                self.consecutive_ray_failures = 0
                return True
        except Exception as e:
            self.consecutive_ray_failures += 1
            logger.error(
                "ray_dashboard_unreachable", error=str(e), consecutive=self.consecutive_ray_failures
            )
        return False

    def get_ray_memory_usage(self) -> float | None:
        """Check Ray object store utilization across all active nodes."""
        try:
            resp = requests.get(f"{RAY_DASHBOARD_URL}/api/nodes/", timeout=10)
            if resp.status_code == 200:
                nodes = resp.json().get("data", {}).get("nodes", [])
                active_nodes = [n for n in nodes if n.get("state") == "ALIVE"]
                if active_nodes:
                    # Monitor both RAM and Object Store pressure
                    utils = [n.get("object_store_utilization", 0.0) for n in active_nodes]
                    return max(utils)
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.warning("failed_to_fetch_ray_metrics_retrying", error=str(e))
        return None

    def check_mlflow_status(self) -> bool:
        """Poll MLflow for experiment status."""
        try:
            # MLflow doesn't have a direct /health in some versions, check experiments list
            resp = requests.get(f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def detect_failed_jobs(self):
        """Analyze jobs for OOM or failure states."""
        try:
            resp = requests.get(f"{RAY_DASHBOARD_URL}/api/jobs/", timeout=5)
            if resp.status_code == 200:
                jobs = resp.json()
                for job in jobs:
                    job_id = job.get("job_id")
                    status = job.get("status")

                    if status in ["FAILED", "STOPPED"] and job_id not in self.failed_jobs:
                        logger.warning("ray_job_failure_detected", job_id=job_id, status=status)
                        self.handle_job_failure(job)
                        self.failed_jobs.add(job_id)
        except Exception as e:
            logger.error("failed_to_query_jobs", error=str(e))

    def handle_job_failure(self, job: dict[str, Any]):
        """Trigger auto-recovery logic."""
        job_id = job.get("job_id")
        error_msg = job.get("message", "").lower()

        logger.info("initiating_self_healing", job_id=job_id)

        # OOM Detection
        if "out of memory" in error_msg or "oom" in error_msg:
            logger.warning("oom_detected_adjusting_resources", job_id=job_id)
            # Logic to respawn with lower batch size or higher memory limit
            # This would typically involve re-submitting to the Ray Client
            self.respawn_job(job, adjust_resources=True)
        else:
            logger.info("general_failure_respawning", job_id=job_id)
            self.respawn_job(job, adjust_resources=False)

    def respawn_job(self, job: dict[str, Any], adjust_resources: bool = False):
        """Simulate or trigger job respawn."""
        # In a real environment, we'd use the job submission API or a CLI call
        entrypoint = job.get("entrypoint")
        if not entrypoint:
            logger.error("cannot_respawn_missing_entrypoint", job_id=job.get("job_id"))
            return

        logger.info("respawning_ray_job", entrypoint=entrypoint, adjusted=adjust_resources)
        # Actual execution logic would go here, e.g.:
        # subprocess.run(["ray", "job", "submit", "--entrypoint", entrypoint, ...])

    def start_health_server(self):
        """Run health server in a background thread."""
        server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info("health_server_started", port=HEALTH_PORT)

    def monitor_and_heal(self):
        """Main self-healing loop."""
        global IS_HEALTHY
        logger.info(
            "mlflow_watchdog_started", tracking_uri=MLFLOW_TRACKING_URI, ray_url=RAY_DASHBOARD_URL
        )

        self.start_health_server()

        while True:
            ray_healthy = self.check_ray_health()
            mlflow_healthy = self.check_mlflow_status()

            # Update global health state for the HTTP probe
            IS_HEALTHY = ray_healthy and mlflow_healthy

            if not ray_healthy and self.consecutive_ray_failures >= self.max_ray_failures:
                logger.error("ray_head_down_critical_alert")
                # Potential: trigger container restart via docker socket or k8s API

            if not mlflow_healthy:
                logger.error("mlflow_down_system_degraded")

            if ray_healthy:
                self.detect_failed_jobs()

                mem_util = self.get_ray_memory_usage()
                if mem_util and mem_util > 0.9:
                    logger.warning("ray_memory_pressure_high", utilization=mem_util)
                    # Potential: shed load or stop low-priority jobs

            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    watchdog = MLflowWatchdog()
    watchdog.monitor_and_heal()
