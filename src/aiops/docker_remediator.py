import structlog
import docker
import subprocess
import os

logger = structlog.get_logger()

class DockerRemediator:
    """
    Advanced Docker remediator supporting restarts and autonomous scaling.
    """
    def __init__(self):
        try:
            self.client = docker.from_env()
            logger.info("docker_remediator_init", status="success")
        except Exception as e:
            logger.error("docker_remediator_init", status="failure", error=str(e))
            self.client = None

    def restart_service(self, service_name: str) -> bool:
        if not self.client: return False
        try:
            # Try to get by name (usually service_name_1 in compose)
            container = self.client.containers.get(f"{service_name}-1")
            container.restart()
            logger.info("docker_remediator_restart", service=service_name, status="success")
            return True
        except Exception:
            # Fallback to shell if container name varies
            try:
                subprocess.run(["docker", "compose", "restart", service_name], check=True)
                return True
            except Exception as e:
                logger.error("docker_remediator_restart_failed", service=service_name, error=str(e))
                return False

    def scale_service(self, service_name: str, replicas: int) -> bool:
        """
        Autonomous scaling via Docker Compose.
        """
        logger.warning("docker_remediator_scale_initiated", service=service_name, target_replicas=replicas)
        try:
            # Use docker-compose command directly for scaling
            cmd = ["docker", "compose", "up", "-d", "--scale", f"{service_name}={replicas}", service_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("docker_remediator_scale_success", service=service_name, output=result.stdout[:200])
            return True
        except subprocess.CalledProcessError as e:
            logger.error("docker_remediator_scale_failed", service=service_name, error=e.stderr)
            return False
        except Exception as e:
            logger.error("docker_remediator_scale_error", service=service_name, error=str(e))
            return False
