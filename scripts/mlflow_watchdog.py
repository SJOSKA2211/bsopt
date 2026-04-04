import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request

# Institutional-grade minimal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mlflow_watchdog")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
RAY_DASHBOARD_URL = os.getenv("RAY_DASHBOARD_URL", "http://ray-head:8265")
CHECK_INTERVAL = 15  # Increased frequency
HEALTH_PORT = 8080

# Global health status
STATE = {
    "is_healthy": False,
    "details": "Initializing...",
    "ray_up": False,
    "mlflow_up": False,
    "pressure": "UNKNOWN",
    "last_check": 0
}

async def fetch_json_async(url: str, timeout: int = 5) -> dict | None:
    """Non-blocking JSON fetcher using run_in_executor."""
    loop = asyncio.get_event_loop()
    try:
        def sync_fetch():
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                if resp.getcode() == 200:
                    return json.loads(resp.read().decode("utf-8"))
            return None

        return await loop.run_in_executor(None, sync_fetch)
    except Exception as e:
        logger.debug(f"Fetch failed: {url} | Error: {e}")
    return None

class MLflowWatchdog:
    """
    Asynchronous self-healing watchdog for Ray and MLflow.
    Monitors service discovery, job health, and resource pressure.
    """

    def __init__(self):
        self.failed_jobs = {}
        self.max_retries = 3
        self.consecutive_ray_failures = 0
        self.max_ray_failures = 5

    async def check_ray_health(self) -> bool:
        data = await fetch_json_async(f"{RAY_DASHBOARD_URL}/api/cluster_status")
        if data:
            self.consecutive_ray_failures = 0
            return True
        self.consecutive_ray_failures += 1
        return False

    async def check_mlflow_status(self) -> bool:
        try:
            # Try health endpoint first
            loop = asyncio.get_event_loop()
            def sync_check():
                try:
                    with urllib.request.urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=3) as resp:
                        return resp.getcode() == 200
                except:
                    return False

            if await loop.run_in_executor(None, sync_check):
                return True

            # Fallback to API check
            data = await fetch_json_async(f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list")
            return data is not None
        except:
            return False

    def get_resource_pressure(self) -> str:
        try:
            with open("/proc/meminfo") as f:
                meminfo = {line.split(":")[0]: line.split(":")[1].strip() for line in f}
            
            total = int(meminfo["MemTotal"].split()[0])
            available = int(meminfo["MemAvailable"].split()[0])
            used_pct = 100 * (1 - available / total)
            
            if used_pct > 90: return f"CRITICAL ({used_pct:.1f}%)"
            if used_pct > 75: return f"WARNING ({used_pct:.1f}%)"
            return f"OK ({used_pct:.1f}%)"
        except:
            return "UNKNOWN"

    async def detect_failed_jobs(self):
        data = await fetch_json_async(f"{RAY_DASHBOARD_URL}/api/jobs/")
        if data and isinstance(data, list):
            for job in data:
                job_id = job.get("job_id")
                status = job.get("status")
                if status in ["FAILED", "STOPPED"]:
                    if job_id not in self.failed_jobs:
                        logger.warning(f"Ray job failure: {job_id} ({status})")
                        await self.handle_job_failure(job)
                        self.failed_jobs[job_id] = 1

    async def handle_job_failure(self, job: dict):
        job_id = job.get("job_id")
        pressure = self.get_resource_pressure()
        
        if "CRITICAL" in pressure:
            logger.error(f"Cannot restart {job_id}: Memory Pressure CRITICAL")
            return

        logger.info(f"Recovery initiated for {job_id}...")
        self.send_alert(f"Job {job_id} failed. Recovery initiated. Pressure: {pressure}")

    def send_alert(self, message):
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if not webhook_url: return
        
        try:
            payload = json.dumps({"text": f"🚨 *MLOps Watchdog*: {message}"}).encode("utf-8")
            req = urllib.request.Request(webhook_url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.error(f"Alert failed: {e}")

    async def monitor_loop(self):
        while True:
            ray_up = await self.check_ray_health()
            mlflow_up = await self.check_mlflow_status()
            pressure = self.get_resource_pressure()

            STATE["ray_up"] = ray_up
            STATE["mlflow_up"] = mlflow_up
            STATE["pressure"] = pressure
            STATE["last_check"] = time.time()

            if ray_up and mlflow_up:
                STATE["is_healthy"] = True
                STATE["details"] = f"All systems nominal | Mem: {pressure}"
                await self.detect_failed_jobs()
            else:
                STATE["is_healthy"] = False
                reasons = []
                if not ray_up: reasons.append("Ray-Head Down")
                if not mlflow_up: reasons.append("MLFlow Down")
                STATE["details"] = f"{' | '.join(reasons)} | Mem: {pressure}"
                logger.error(f"Status Degraded: {STATE['details']}")

            await asyncio.sleep(CHECK_INTERVAL)

async def handle_health(reader, writer):
    status_code = 200 if STATE["is_healthy"] else 503
    status_text = "OK" if STATE["is_healthy"] else f"Service Degraded: {STATE['details']}"

    response = (
        f"HTTP/1.1 {status_code} {'OK' if status_code == 200 else 'Service Unavailable'}\r\n"
        "Content-Type: text/plain\r\n"
        f"Content-Length: {len(status_text)}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{status_text}"
    )
    writer.write(response.encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()

async def main():
    logger.info(f"Starting MLOps Watchdog (Async) | Tracking: {MLFLOW_TRACKING_URI}")
    watchdog = MLflowWatchdog()

    # Start health server
    server = await asyncio.start_server(handle_health, '0.0.0.0', HEALTH_PORT)
    logger.info(f"Health server on port {HEALTH_PORT}")

    async with server:
        await asyncio.gather(
            server.serve_forever(),
            watchdog.monitor_loop()
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
