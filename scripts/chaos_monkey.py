import time
import subprocess
import random
import os
import structlog

logger = structlog.get_logger(__name__)

# List of critical containers to target for chaos engineering
# We exclude the 'postgres' and 'redis' to maintain state, or include them for full resilience tests.
TARGET_CONTAINERS = [
    "bsopt-api-1",
    "bsopt-auth-service-1",
    "bsopt-ml-inference-1",
    "bsopt-worker-1",
    "bsopt-nse-scraper-1",
    "bsopt-envoy-1",
]


def kill_container(container_name):
    """Randomly kill a container to test system resilience."""
    try:
        logger.warning("chaos_monkey_attack_initiated", target=container_name)
        subprocess.run(["docker", "kill", container_name], check=True)
        logger.info("chaos_monkey_attack_success", target=container_name)
    except subprocess.CalledProcessError:
        # Try podman if docker fails
        try:
            subprocess.run(["podman", "kill", container_name], check=True)
            logger.info("chaos_monkey_attack_success", target=container_name, engine="podman")
        except Exception as e:
            logger.error("chaos_monkey_attack_failed", target=container_name, error=str(e))


def monitor_recovery(container_name, timeout=60):
    """Monitor if the system (docker restart policy) recovers the container."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Check if container is running
            output = (
                subprocess.check_output(
                    ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
                )
                .decode()
                .strip()
            )
            if output == "true":
                logger.info(
                    "chaos_monkey_recovery_verified",
                    target=container_name,
                    duration=time.time() - start_time,
                )
                return True
        except Exception:
            try:
                output = (
                    subprocess.check_output(
                        ["podman", "inspect", "-f", "{{.State.Running}}", container_name]
                    )
                    .decode()
                    .strip()
                )
                if output == "true":
                    logger.info(
                        "chaos_monkey_recovery_verified",
                        target=container_name,
                        duration=time.time() - start_time,
                    )
                    return True
            except Exception:
                pass
        time.sleep(5)

    logger.error("chaos_monkey_recovery_failed", target=container_name, timeout=timeout)
    return False


def chaos_loop(interval=300):
    """Continuous Chaos Engineering Loop."""
    logger.info("chaos_monkey_started", interval=interval, targets=TARGET_CONTAINERS)

    while True:
        target = random.choice(TARGET_CONTAINERS)
        kill_container(target)

        # Give it a moment to realize it's dead and for restart policy to kick in
        time.sleep(10)
        monitor_recovery(target)

        # Wait for the next chaos event
        wait_time = random.randint(interval // 2, interval * 2)
        logger.info("chaos_monkey_sleeping", next_attack_in=wait_time)
        time.sleep(wait_time)


if __name__ == "__main__":
    # Ensure we don't run chaos in production by accident without a flag
    if os.getenv("CHAOS_ENABLED") == "1":
        chaos_loop()
    else:
        logger.error(
            "chaos_monkey_aborted", reason="CHAOS_ENABLED environment variable not set to 1"
        )
