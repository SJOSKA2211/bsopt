import asyncio
import os
import subprocess
import time
import structlog
from src.shared.utils.cache import get_redis_client

setup_logging = False
try:
    from src.shared.observability import setup_logging as sl
    sl()
    setup_logging = True
except ImportError:
    pass

logger = structlog.get_logger(__name__)

async def check_redis_health() -> bool:
    """Attempt to ping Redis."""
    try:
        redis = await get_redis_client()
        return await redis.ping()
    except Exception as e:
        logger.debug("redis_health_check_failed", error=str(e))
        return False

async def run_until_healthy(max_retries: int = 30, retry_interval: int = 5):
    """
    Ensures Redis is up and running. 
    Attempts to start it via docker-compose if connection fails initially.
    """
    print("🔍 Checking Redis Health...")
    
    for i in range(max_retries):
        if await check_redis_health():
            print("✅ Redis is HEALTHY and READY.")
            return True
        
        if i == 0:
            print("⚠️ Redis not reachable. Attempting to start via docker-compose...")
            try:
                # Determine the correct compose file
                compose_path = "infrastructure/orchestration/docker-compose.yml"
                if not os.path.exists(compose_path):
                    # Fallback if run from different dir
                    compose_path = "../../infrastructure/orchestration/docker-compose.yml"
                
                subprocess.run(
                    ["docker-compose", "-f", compose_path, "up", "-d", "redis"],
                    check=True,
                    capture_output=True
                )
                print("🚀 started_via_docker_compose")
            except Exception as e:
                print(f"🚨 Failed to run docker-compose: {str(e)}")
        
        print(f"⏳ Waiting for Redis... (Attempt {i+1}/{max_retries})")
        await asyncio.sleep(retry_interval)
    
    print("❌ Redis failed to become healthy within the timeout.")
    return False

if __name__ == "__main__":
    if not asyncio.run(run_until_healthy()):
        exit(1)
