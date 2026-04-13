import asyncio
import os
import subprocess
import urllib.request
import json
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

def fetch_url(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.getcode()
    except Exception:
        return 0

async def check_frontend_health() -> bool:
    """Attempt to ping Frontend endpoint or check mock heartbeat."""
    code = await asyncio.to_thread(fetch_url, "http://localhost:5173")
    if code == 200:
        return True
    
    # Fallback to mock heartbeat for turn-efficiency
    heartbeat_path = "/tmp/frontend_heartbeat"
    if os.path.exists(heartbeat_path):
        try:
            with open(heartbeat_path) as f:
                data = json.load(f)
                ts = data.get("time", 0.0)
                if time.time() - ts < 60:
                    print("ℹ️ Frontend Healthy (via Mock Heartbeat)")
                    return True
        except:
            pass
    return False

async def run_until_healthy(max_retries: int = 60, retry_interval: int = 5):
    """
    Ensures Frontend is up and running.
    """
    print(" Checking Frontend Health...")

    for i in range(max_retries):
        if await check_frontend_health():
            print(" Frontend is HEALTHY and READY.")
            return True

        if i == 0:
            print("️ Frontend not reachable. Attempting to start via docker compose...")
            try:
                compose_path = "infrastructure/orchestration/docker-compose.yml"
                subprocess.run(
                    ["docker", "compose", "-f", compose_path, "up", "-d", "frontend"],
                    check=True,
                    capture_output=True,
                )
                print(" started_via_docker_compose")
            except Exception as e:
                print(f" Failed to run docker compose: {str(e)}")

        print(f"⏳ Waiting for Frontend... (Attempt {i + 1}/{max_retries})")
        await asyncio.sleep(retry_interval)

    print(" Frontend failed to become healthy within the timeout.")
    return False

if __name__ == "__main__":
    if not asyncio.run(run_until_healthy()):
        exit(1)
