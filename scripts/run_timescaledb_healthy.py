import asyncio
import os
import subprocess

import structlog
from sqlalchemy import text

from src.database import get_async_engine

setup_logging = False
try:
    from src.shared.observability import setup_logging as sl

    sl()
    setup_logging = True
except ImportError:
    pass

logger = structlog.get_logger(__name__)


async def check_db_health() -> bool:
    """Attempt to connect and run a simple query."""
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.debug("timescaledb_health_check_failed", error=str(e))
        return False


async def run_until_healthy(max_retries: int = 30, retry_interval: int = 5):
    """
    Ensures TimescaleDB is up and running.
    Attempts to start it via docker-compose if connection fails initially.
    """
    print("🔍 Checking TimescaleDB Health...")

    for i in range(max_retries):
        if await check_db_health():
            print("✅ TimescaleDB is HEALTHY and READY.")
            return True

        if i == 0:
            print("⚠️ TimescaleDB not reachable. Attempting to start via docker-compose...")
            try:
                # Determine the correct compose file
                compose_path = "infrastructure/orchestration/docker-compose.yml"
                if not os.path.exists(compose_path):
                    # Fallback if run from different dir
                    compose_path = "../../infrastructure/orchestration/docker-compose.yml"

                subprocess.run(
                    ["docker-compose", "-f", compose_path, "up", "-d", "postgres"],
                    check=True,
                    capture_output=True,
                )
                print("🚀 started_via_docker_compose")
            except Exception as e:
                print(f"🚨 Failed to run docker-compose: {str(e)}")

        print(f"⏳ Waiting for TimescaleDB... (Attempt {i + 1}/{max_retries})")
        await asyncio.sleep(retry_interval)

    print("❌ TimescaleDB failed to become healthy within the timeout.")
    return False


if __name__ == "__main__":
    if not asyncio.run(run_until_healthy()):
        exit(1)
