import random
import subprocess
import time

import structlog

logger = structlog.get_logger(__name__)

class ChaosMonkey:
    """
    EquaFlow Chaos Monkey.
    Randomly disrupts the system to ensure institutional resilience.
    """

    def __init__(self, target_containers=None):
        self.target_containers = target_containers or [
            "api",
            "worker",
            "auth-service",
            "redis",
            "postgres",
        ]

    def wreak_havoc(self):
        logger.info("chaos_monkey_started")
        while True:
            action = random.choice(["kill_container", "inject_latency", "pause_service"])
            target = random.choice(self.target_containers)

            if action == "kill_container":
                self._kill(target)
            elif action == "inject_latency":
                self._latency(target)
            elif action == "pause_service":
                self._pause(target)

            sleep_time = random.randint(30, 120)
            logger.info("chaos_monkey_resting", next_action_in=sleep_time)
            time.sleep(sleep_time)

    def _kill(self, container):
        logger.warning("chaos_action_kill_container", container=container)
        subprocess.run(f"docker compose kill {container}", shell=True)
        time.sleep(5)
        subprocess.run(f"docker compose start {container}", shell=True)

    def _latency(self, container):
        logger.warning("chaos_action_inject_latency", container=container)
        # Uses tc (traffic control) inside container if possible, or network disruption
        pass

    def _pause(self, container):
        logger.warning("chaos_action_pause_container", container=container)
        subprocess.run(f"docker compose pause {container}", shell=True)
        time.sleep(random.randint(5, 15))
        subprocess.run(f"docker compose unpause {container}", shell=True)

if __name__ == "__main__":
    monkey = ChaosMonkey()
    # monkey.wreak_havoc()
