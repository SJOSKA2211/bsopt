import os
import random
import subprocess
import time

import structlog

logger = structlog.get_logger(__name__)


# List of critical containers to target for chaos engineering
# We exclude the 'postgres' and 'redis' to maintain state, or include them for full resilience tests.
def get_dynamic_targets():
    """Discover all bsopt-related running containers."""
    try:
        output = (
            subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"])
            .decode()
            .strip()
            .split("\n")
        )
        return [n for n in output if "bsopt" in n]
    except Exception:
        try:
            output = (
                subprocess.check_output(["podman", "ps", "--format", "{{.Names}}"])
                .decode()
                .strip()
                .split("\n")
            )
            return [n for n in output if "bsopt" in n]
        except Exception:
            return []


def log_chaos_event(target, event_type):
    """Push chaos event to Prometheus Pushgateway."""
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        g = Gauge("chaos_event_total", "Total chaos events", ["target", "type"], registry=registry)
        g.labels(target=target, type=event_type).inc()
        push_to_gateway("pushgateway:9091", job="chaos_monkey", registry=registry)
    except Exception as e:
        logger.warning("pushgateway_not_reachable", error=str(e))


def kill_container(container_name):
    """Randomly kill a container and log the event."""
    try:
        logger.warning("chaos_monkey_attack_initiated", target=container_name)
        log_chaos_event(container_name, "kill")
        subprocess.run(["docker", "kill", container_name], check=True)
        logger.info("chaos_monkey_attack_success", target=container_name)
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["podman", "kill", container_name], check=True)
            logger.info("chaos_monkey_attack_success", target=container_name, engine="podman")
        except Exception as e:
            logger.error("chaos_monkey_attack_failed", target=container_name, error=str(e))


def monitor_recovery(container_name, timeout=60):
    """Wait for a container to return to 'running' state."""
    logger.info("monitoring_recovery", target=container_name)
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            output = (
                subprocess.check_output(
                    ["docker", "inspect", "-f", "{{.State.Running}}", container_name]
                )
                .decode()
                .strip()
            )
            if output == "true":
                logger.info("recovery_detected", target=container_name)
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
                    logger.info("recovery_detected", target=container_name, engine="podman")
                    return True
            except Exception:
                pass
        time.sleep(2)
    logger.error("recovery_timeout", target=container_name)
    return False


def chaos_loop(interval=300):
    """Continuous Chaos Engineering Loop with dynamic discovery."""
    logger.info("chaos_monkey_started", interval=interval)

    while True:
        targets = get_dynamic_targets()
        if not targets:
            logger.warning("no_targets_found_waiting")
            time.sleep(60)
            continue

        target = random.choice(targets)
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
