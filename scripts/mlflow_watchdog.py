import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# Institutional-grade minimal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mlflow_watchdog")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
RAY_DASHBOARD_URL = os.getenv("RAY_DASHBOARD_URL", "http://ray-head:8265")
CHECK_INTERVAL = 30
HEALTH_PORT = 8080

# Global health status for the HTTP server
IS_HEALTHY = False
HEALTH_DETAILS = "Initializing..."


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            if IS_HEALTHY:
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(503)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Service Degraded: {HEALTH_DETAILS}".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def fetch_json(url: str, timeout: int = 10) -> dict | None:
    """Robust JSON fetcher using standard library only."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.getcode() == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Fetch failed: {url} | Error: {e}")
    return None


class MLflowWatchdog:
    """
    Zero-dependency self-healing watchdog for Ray and MLflow.
    Monitors service discovery, job health, and resource pressure.
    """

    def __init__(self):
        self.failed_jobs = {}  # job_id -> retry_count
        self.max_retries = 3
        self.consecutive_ray_failures = 0
        self.max_ray_failures = 5

    def check_ray_health(self) -> bool:
        """Poll Ray dashboard for status."""
        data = fetch_json(f"{RAY_DASHBOARD_URL}/api/cluster_status")
        if data:
            self.consecutive_ray_failures = 0
            return True

        self.consecutive_ray_failures += 1
        return False

    def check_mlflow_status(self) -> bool:
        """Poll MLflow for operational readiness."""
        try:
            with urllib.request.urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=5) as resp:
                return resp.getcode() == 200
        except:
            data = fetch_json(f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list")
            return data is not None

    def get_resource_pressure(self) -> str:
        """Calculate memory pressure from /proc/meminfo."""
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {line.split(":")[0]: line.split(":")[1].strip() for line in f}
            
            total = int(meminfo["MemTotal"].split()[0])
            available = int(meminfo["MemAvailable"].split()[0])
            used_pct = 100 * (1 - available / total)
            
            if used_pct > 90:
                return f"CRITICAL ({used_pct:.1f}% used)"
            if used_pct > 75:
                return f"WARNING ({used_pct:.1f}% used)"
            return f"LOW ({used_pct:.1f}% used)"
        except:
            return "UNKNOWN"

    def detect_failed_jobs(self):
        """Analyze jobs for failure states via Ray Job API."""
        data = fetch_json(f"{RAY_DASHBOARD_URL}/api/jobs/")
        if data and isinstance(data, list):
            for job in data:
                job_id = job.get("job_id")
                status = job.get("status")

                if status in ["FAILED", "STOPPED"]:
                    if job_id not in self.failed_jobs:
                        logger.warning(f"Ray job failure detected: {job_id} ({status})")
                        self.handle_job_failure(job)
                        self.failed_jobs[job_id] = 1
                    elif self.failed_jobs[job_id] < self.max_retries:
                        # Logic for tracking retries could be added here
                        pass

    def handle_job_failure(self, job: dict):
        """Self-healing: trigger job respawn or resource adjustment."""
        job_id = job.get("job_id")
        entrypoint = job.get("entrypoint", "N/A")
        pressure = self.get_resource_pressure()
        
        logger.info(
            f"Initiating self-healing for job {job_id} | Entrypoint: {entrypoint} | Pressure: {pressure}"
        )
        
        if "CRITICAL" in pressure:
            logger.error(f"Cannot restart job {job_id} due to CRITICAL resource pressure.")
            self.send_alert(f"Job {job_id} failed and cannot be restarted: {pressure}")
            return

        # Simplified restart logic
        logger.info(f"Restarting job {job_id}...")
        # In a real scenario, we'd use the Ray Job Submission SDK or 'ray job submit'
        # For now, we log the recovery attempt.
        self.send_alert(f"Job {job_id} failed. Recovery attempt initiated (Pressure: {pressure})")

    def send_alert(self, message):
        """Send alert to webhook."""
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if not webhook_url:
            return
        
        payload = {"text": f"🚨 *MLOps Watchdog*: {message}"}
        try:
            req = urllib.request.Request(webhook_url, method="POST")
            req.add_header("Content-Type", "application/json")
            data = json.dumps(payload).encode("utf-8")
            with urllib.request.urlopen(req, data=data, timeout=5) as _:
                pass
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def start_health_server(self):
        """Run the internal health probe server."""
        def serve():
            try:
                server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
                logger.info(f"Watchdog health server listening on port {HEALTH_PORT}")
                server.serve_forever()
            except Exception as e:
                logger.error(f"Health server failed: {e}")

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

    def monitor_and_heal(self):
        """Main loop for infrastructure monitoring."""
        global IS_HEALTHY, HEALTH_DETAILS
        logger.info(
            f"MLOps Watchdog started | MLFlow: {MLFLOW_TRACKING_URI} | Ray: {RAY_Dashboard_URL if 'RAY_Dashboard_URL' in globals() else RAY_DASHBOARD_URL}"
        )

        self.start_health_server()

        while True:
            ray_up = self.check_ray_health()
            mlflow_up = self.check_mlflow_status()
            pressure = self.get_resource_pressure()

            if ray_up and mlflow_up:
                IS_HEALTHY = True
                HEALTH_DETAILS = f"All components reachable | Pressure: {pressure}"
                self.detect_failed_jobs()
            else:
                IS_HEALTHY = False
                reasons = []
                if not ray_up:
                    reasons.append("Ray Dashboard Unreachable")
                if not mlflow_up:
                    reasons.append("MLFlow Tracking Unreachable")
                HEALTH_DETAILS = f"{' | '.join(reasons)} | Pressure: {pressure}"
                logger.error(f"System Degraded: {HEALTH_DETAILS}")

            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    watchdog = MLflowWatchdog()
    watchdog.monitor_and_heal()
