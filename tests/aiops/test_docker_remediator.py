from unittest.mock import MagicMock, call, patch

import pytest

import docker
from src.aiops.docker_remediator import DockerRemediator


@patch("src.aiops.docker_remediator.logger")
@patch("src.aiops.docker_remediator.docker.from_env")
class TestDockerRemediator:
    def test_docker_remediator_init_success(self, mock_from_env, mock_logger):
        """Test successful initialization of DockerRemediator."""
        mock_from_env.return_value = MagicMock()
        remediator = DockerRemediator()
        assert remediator.client is not None
        mock_logger.info.assert_called_once_with(
            "docker_remediator_init", status="success", message="Docker client initialized."
        )
        mock_logger.error.assert_not_called()

    def test_docker_remediator_init_failure(self, mock_from_env, mock_logger):
        """Test that DockerRemediator initialization raises an exception on Docker client failure."""
        mock_from_env.side_effect = Exception("Docker client connection failed")
        with pytest.raises(Exception, match="Docker client connection failed"):
            DockerRemediator()
        mock_logger.error.assert_called_once_with(
            "docker_remediator_init", status="failure", error="Docker client connection failed", message="Failed to initialize Docker client."
        )
        mock_logger.info.assert_not_called()

    def test_docker_remediator_restart_service_success(self, mock_from_env, mock_logger):
        """Test successful service restart."""
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        
        mock_container = MagicMock()
        mock_container.id = "abc123def456"
        mock_client.containers.get.return_value = mock_container
        
        remediator = DockerRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        service_name = "test_service"
        
        result = remediator.restart_service(service_name)
        
        mock_client.containers.get.assert_called_once_with(service_name)
        mock_container.restart.assert_called_once()
        assert result
        mock_logger.info.assert_has_calls([
            call("docker_remediator_restart", service=service_name, status="found", container_id=mock_container.id),
            call("docker_remediator_restart", service=service_name, status="success", container_id=mock_container.id, message="Service restarted successfully.")
        ])
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_docker_remediator_restart_service_not_found(self, mock_from_env, mock_logger):
        """Test service restart when container is not found."""
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        
        mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")
        
        remediator = DockerRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        service_name = "non_existent_service"
        
        result = remediator.restart_service(service_name)
        
        mock_client.containers.get.assert_called_once_with(service_name)
        assert not result
        mock_logger.warning.assert_called_once_with(
            "docker_remediator_restart", service=service_name, status="not_found", message="Container not found."
        )
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()


    def test_docker_remediator_restart_service_get_failure(self, mock_from_env, mock_logger):
        """Test service restart when container.get fails with a general exception."""
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        
        mock_client.containers.get.side_effect = Exception("Generic Docker error during get")
        
        remediator = DockerRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        service_name = "failing_service"
        
        result = remediator.restart_service(service_name)
        
        mock_client.containers.get.assert_called_once_with(service_name)
        assert not result
        mock_logger.error.assert_called_once_with(
            "docker_remediator_restart", service=service_name, status="failure", error="Generic Docker error during get", message="Failed to restart service."
        )
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()

    def test_docker_remediator_restart_action_failure(self, mock_from_env, mock_logger):
        """Test service restart when container.restart fails with a general exception."""
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        
        mock_container = MagicMock()
        mock_container.id = "abc123def456"
        mock_client.containers.get.return_value = mock_container
        mock_container.restart.side_effect = Exception("Generic Docker error during restart")
        
        remediator = DockerRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        service_name = "failing_service_restart"
        
        result = remediator.restart_service(service_name)
        
        mock_client.containers.get.assert_called_once_with(service_name)
        mock_container.restart.assert_called_once()
        assert not result
        mock_logger.info.assert_called_once_with(
            "docker_remediator_restart", service=service_name, status="found", container_id=mock_container.id
        )
        mock_logger.error.assert_called_once_with(
            "docker_remediator_restart", service=service_name, status="failure", error="Generic Docker error during restart", message="Failed to restart service."
        )
        mock_logger.warning.assert_not_called()


    def test_docker_remediator_restart_service_empty_name_raises_error(self, mock_from_env, mock_logger):
        """Test service restart with empty service name."""
        mock_from_env.return_value = MagicMock()
        remediator = DockerRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        with pytest.raises(ValueError, match="Service name cannot be empty."):
            remediator.restart_service("")
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_logger.warning.assert_not_called()