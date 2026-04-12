import asyncio
import os
import subprocess
import urllib.request
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

async def check_minio_health() -> bool:
    """Attempt to ping MinIO health endpoints."""
    # Check both live and cluster health
    live_code = await asyncio.to_thread(fetch_url, "http://localhost:9000/minio/health/live")
    cluster_code = await asyncio.to_thread(fetch_url, "http://localhost:9000/minio/health/cluster")
    
    if live_code == 200 and cluster_code == 200:
        return True
    return False

async def run_until_healthy(max_retries: int = 30, retry_interval: int = 5):
    """
    Ensures MinIO is up and running.
    Attempts to start it via docker-compose if connection fails initially.
    """
    print("🔍 Checking MinIO Health...")

    for i in range(max_retries):
        if await check_minio_health():
            print("✅ MinIO is HEALTHY and READY.")
            return True

        if i == 0:
            print("⚠️ MinIO not reachable. Attempting to start via docker-compose...")
            try:
                compose_path = "infrastructure/orchestration/docker-compose.yml"
                if not os.path.exists(compose_path):
                    compose_path = "../../infrastructure/orchestration/docker-compose.yml"

                subprocess.run(
                    ["docker-compose", "-f", compose_path, "up", "-d", "minio"],
                    check=True,
                    capture_output=True,
                )
                print("🚀 started_via_docker_compose")
            except Exception as e:
                print(f"🚨 Failed to run docker-compose: {str(e)}")

        print(f"⏳ Waiting for MinIO... (Attempt {i + 1}/{max_retries})")
        await asyncio.sleep(retry_interval)

    print("❌ MinIO failed to become healthy within the timeout.")
    return False

if __name__ == "__main__":
    if not asyncio.run(run_until_healthy()):
        exit(1)
