import docker
import structlog

logger = structlog.get_logger()

class DockerRemediator:
    def __init__(self):
        try:
            self.client = docker.from_env()
            logger.info("docker_remediator_init", status="success", message="Docker client initialized.")
        except Exception as e:
            logger.error("docker_remediator_init", status="failure", error=str(e), message="Failed to initialize Docker client.")
            raise

    def restart_service(self, service_name: str) -> bool:
        if not service_name:
            raise ValueError("Service name cannot be empty.")

        try:
            container = self.client.containers.get(service_name)
            logger.info("docker_remediator_restart", service=service_name, status="found", container_id=container.id)
            container.restart()
            logger.info("docker_remediator_restart", service=service_name, status="success", container_id=container.id, message="Service restarted successfully.")
            return True
        except docker.errors.NotFound:
            logger.warning("docker_remediator_restart", service=service_name, status="not_found", message="Container not found.")
            return False
        except Exception as e:
            logger.error("docker_remediator_restart", service=service_name, status="failure", error=str(e), message="Failed to restart service.")
            return False