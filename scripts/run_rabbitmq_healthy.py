import asyncio
import os
import subprocess

import structlog

from src.shared.rabbitmq import RabbitMQManager

setup_logging = False
try:
    from src.shared.observability import setup_logging as sl

    sl()
    setup_logging = True
except ImportError:
    pass

logger = structlog.get_logger(__name__)


async def check_rabbitmq_health(manager: RabbitMQManager) -> bool:
    """Attempt to connect and perform a basic operation."""
    try:
        await manager.connect()
        # If connect() succeeds, it has already declared queues, which is a good health signal.
        return not manager.connection.is_closed
    except Exception as e:
        logger.debug("rabbitmq_health_check_failed", error=str(e))
        return False


async def run_until_healthy(max_retries: int = 30, retry_interval: int = 5):
    """
    Ensures RabbitMQ is up and running.
    Attempts to start it via docker-compose if connection fails initially.
    """
    manager = RabbitMQManager()

    print(" Checking RabbitMQ Health...")

    for i in range(max_retries):
        if await check_rabbitmq_health(manager):
            print(" RabbitMQ is HEALTHY and READY.")
            await manager.close()
            return True

        if i == 0:
            print("️ RabbitMQ not reachable. Attempting to start via docker-compose...")
            try:
                # Determine the correct compose file
                compose_path = "infrastructure/orchestration/docker-compose.yml"
                if not os.path.exists(compose_path):
                    # Fallback if run from different dir
                    compose_path = "../../infrastructure/orchestration/docker-compose.yml"

                subprocess.run(
                    ["docker-compose", "-f", compose_path, "up", "-d", "rabbitmq"],
                    check=True,
                    capture_output=True,
                )
                print(" started_via_docker_compose")
            except Exception as e:
                print(f" Failed to run docker-compose: {str(e)}")

        print(f"⏳ Waiting for RabbitMQ... (Attempt {i + 1}/{max_retries})")
        await asyncio.sleep(retry_interval)

    print(" RabbitMQ failed to become healthy within the timeout.")
    return False


if __name__ == "__main__":
    if not asyncio.run(run_until_healthy()):
        exit(1)
