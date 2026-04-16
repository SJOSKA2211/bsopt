import asyncio
import json
import os
import subprocess
import time

import structlog

setup_logging = False
try:
    from src.shared.observability import setup_logging as sl
    sl()
    setup_logging = True
except ImportError:
    pass

logger = structlog.get_logger(__name__)

def check_heartbeat(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            content = f.read().strip()
        try:
            # Try to parse as JSON first
            data = json.loads(content)
            ts = data.get("time", 0.0)
            delta = time.time() - ts
            return delta < 60
        except json.JSONDecodeError:
            # Fallback to plain timestamp
            ts = float(content)
            delta = time.time() - ts
            return delta < 60
    except Exception:
        return False

async def run_until_healthy(max_retries: int = 30, retry_interval: int = 5):
    """
    Ensures Ingestion Service is up and running.
    Attempts to start it via docker-compose if heartbeat is missing or stale.
    """
    heartbeat_path = "/tmp/ingestion_heartbeat"
    print(f" Checking Ingestion Health ({heartbeat_path})...")

    for i in range(max_retries):
        if check_heartbeat(heartbeat_path):
            print(" Ingestion Service is HEALTHY and READY.")
            return True

        if i == 0:
            print("️ Ingestion not healthy. Attempting to start via docker-compose...")
            try:
                compose_path = "infrastructure/orchestration/docker-compose.yml"
                if not os.path.exists(compose_path):
                    compose_path = "../../infrastructure/orchestration/docker-compose.yml"

                subprocess.run(
                    ["docker-compose", "-f", compose_path, "up", "-d", "ingestion-service"],
                    check=True,
                    capture_output=True,
                )
                print(" started_via_docker_compose")
            except Exception as e:
                print(f" Failed to run docker-compose: {str(e)}")

        print(f"⏳ Waiting for Ingestion... (Attempt {i + 1}/{max_retries})")
        await asyncio.sleep(retry_interval)

    print(" Ingestion Service failed to become healthy within the timeout.")
    return False

if __name__ == "__main__":
    if not asyncio.run(run_until_healthy()):
        exit(1)
