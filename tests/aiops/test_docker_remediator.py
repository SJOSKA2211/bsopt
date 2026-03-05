from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import docker
from src.aiops.docker_remediator import DockerRemediator


@patch("src.aiops.docker_remediator.logger")
@patch("src.aiops.docker_remediator.docker.from_env")
class TestDockerRemediator:
    @pytest.mark.asyncio
    async def test_docker_remediator_init_success(self, mock_from_env, mock_logger):
        """Test successful initialization of DockerRemediator."""
        mock_from_env.return_value = MagicMock()
        remediator = DockerRemediator()
        assert remediator.client is not None
        mock_logger.info.assert_any_call(
            "docker_remediator_init",
            status="success",
        )

    @pytest.mark.asyncio
    async def test_docker_remediator_init_failure(self, mock_from_env, mock_logger):
        """Test that DockerRemediator initialization logs error on failure."""
        mock_from_env.side_effect = Exception("Docker client connection failed")
        remediator = DockerRemediator()
        assert remediator.client is None
        mock_logger.error.assert_called_once_with(
            "docker_remediator_init",
            status="failure",
            error="Docker client connection failed",
        )

    @pytest.mark.asyncio
    async def test_docker_remediator_restart_service_success(self, mock_from_env, mock_logger):
        """Test successful service restart."""
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.id = "abc123def456"
        mock_container.restart = MagicMock()
        mock_client.containers.get.return_value = mock_container

        remediator = DockerRemediator()
        mock_logger.reset_mock()
        service_name = "api" # Use allowed service

        result = await remediator.restart_service(service_name)

        mock_client.containers.get.assert_called_once_with(f"bsopt-{service_name}-1")
        assert result is True
        mock_logger.info.assert_any_call(
            "docker_remediator_restart_sdk_success",
            service=service_name,
        )

    @pytest.mark.asyncio
    async def test_docker_remediator_restart_service_not_found_fallback(self, mock_from_env, mock_logger):
        """Test service restart fallback to shell when SDK fails."""
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")

        remediator = DockerRemediator()
        mock_logger.reset_mock()
        service_name = "worker"

        with patch("src.aiops.docker_remediator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="restarted")
            result = await remediator.restart_service(service_name)
            
            assert result is True
            mock_run.assert_called_once()
            assert "restart" in mock_run.call_args[0][0]

    @pytest.mark.asyncio
    async def test_docker_remediator_restart_service_invalid(self, mock_from_env, mock_logger):
        """Test service restart with invalid service name."""
        remediator = DockerRemediator()
        result = await remediator.restart_service("malicious; rm -rf")
        assert result is False
        mock_logger.error.assert_any_call("docker_remediator_invalid_service", service="malicious; rm -rf")

    @pytest.mark.asyncio
    async def test_scale_service_success(self, mock_from_env, mock_logger):
        remediator = DockerRemediator()
        with patch("src.aiops.docker_remediator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="scaled")
            result = await remediator.scale_service("api", 3)
            assert result is True
            mock_run.assert_called_once()
            assert "--scale" in mock_run.call_args[0][0]
            assert "api=3" in mock_run.call_args[0][0]
