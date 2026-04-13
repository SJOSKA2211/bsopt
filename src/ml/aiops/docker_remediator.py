import asyncio
import re
import subprocess
from typing import Any

import docker
import structlog

logger = structlog.get_logger()

ALLOWED_SERVICES = {
    "api",
    "worker",
    "redis",
    "postgres",
    "rabbitmq",
    "ml-inference",  # Updated from neural-pricing
    "portfolio",
    "scraper",
    "envoy",
    "frontend",
}

SERVICE_NAME_REGEX = re.compile(r"^[a-z0-9-]+$")


class DockerRemediator:
    """
    Advanced Docker remediator supporting restarts and autonomous scaling.
    Deterministic: Requires docker-py library.
    """

    def __init__(self) -> None:
        self.client: Any = None
        try:
            self.client = docker.from_env()
            logger.info("docker_remediator_init", status="success")
        except Exception as e:
            logger.error("docker_remediator_init", status="failure", error=str(e))
            # If we don't have a client, we will fallback to CLI, which is fine,
            # but we no longer hide the fact that we expected the SDK.
            self.client = None

    async def _run_cmd(self, cmd: list[str]) -> bool:
        """Helper to run shell commands in the background."""
        # Note: cmd is a list, reducing shell injection risks.
        try:
            # Wrap in to_thread since subprocess.run is blocking
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, check=True
            )
            logger.info(
                "docker_remediator_cmd_success", cmd=" ".join(cmd), output=result.stdout[:200]
            )
            return True
        except Exception as e:
            logger.error("docker_remediator_cmd_failed", cmd=" ".join(cmd), error=str(e))
            return False

    def _validate_service(self, service_name: str) -> bool:
        """Validates service name against allowlist and format."""
        if service_name not in ALLOWED_SERVICES:
            logger.error("docker_remediator_invalid_service", service=service_name)
            return False
        if not SERVICE_NAME_REGEX.match(service_name):
            logger.error("docker_remediator_invalid_format", service=service_name)
            return False
        return True

    async def restart_service(self, service_name: str) -> bool:
        """Restarts a service asynchronously."""
        if not self._validate_service(service_name):
            return False

        try:
            if self.client:
                # Explicitly use bsopt prefix to prevent attacking arbitrary containers
                container = await asyncio.to_thread(
                    self.client.containers.get, f"bsopt-{service_name}-1"
                )
                await asyncio.to_thread(container.restart)
                logger.info("docker_remediator_restart_sdk_success", service=service_name)
                return True
        except Exception as e:
            logger.warning(
                "docker_remediator_restart_sdk_failed", service=service_name, error=str(e)
            )

        # Fallback to compose CLI
        return await self._run_cmd(["docker", "compose", "restart", service_name])

    async def scale_service(self, service_name: str, replicas: int) -> bool:
        """
        Autonomous scaling (Asynchronous).
        Prioritizes Docker SDK for direct scaling if possible.
        """
        logger.warning(
            "docker_remediator_scale_initiated",
            service=service_name,
            target_replicas=replicas,
        )

        if not self._validate_service(service_name):
            return False

        # Ensure replicas is a sane integer
        try:
            replicas_int = int(replicas)
            if replicas_int < 1 or replicas_int > 5:
                logger.error("docker_remediator_invalid_scale", count=replicas_int)
                return False
        except (ValueError, TypeError):
            return False

        cmd = [
            "docker",
            "compose",
            "up",
            "-d",
            "--scale",
            f"{service_name}={replicas_int}",
            service_name,
        ]

        return await self._run_cmd(cmd)

    def close(self) -> None:
        """Release Docker SDK resources."""
        if self.client:
            try:
                self.client.close()
                logger.info("docker_remediator_closed")
            except Exception as e:
                logger.error("docker_remediator_close_error", error=str(e))