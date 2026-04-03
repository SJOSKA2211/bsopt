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
            # Check basic connectivity
            await conn.execute(text("SELECT 1"))
            # Check for TimescaleDB extension as a proxy for "Full Health" in this platform
            await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            )
            return True
    except Exception as e:
        logger.debug("postgres_health_check_failed", error=str(e))
        return False


async def run_until_healthy(max_retries: int = 30, retry_interval: int = 5):
    """
    Ensures Postgres is up and running.
    Attempts to start it via docker-compose if connection fails initially.
    """
    print("🔍 Checking Postgres Health...")

    for i in range(max_retries):
        if await check_db_health():
            print("✅ Postgres is HEALTHY and READY.")
            return True

        if i == 0:
            print("⚠️ Postgres not reachable. Attempting to start via docker-compose...")
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

        print(f"⏳ Waiting for Postgres... (Attempt {i + 1}/{max_retries})")
        await asyncio.sleep(retry_interval)

    print("❌ Postgres failed to become healthy within the timeout.")
    return False


if __name__ == "__main__":
    if not asyncio.run(run_until_healthy()):
        exit(1)
