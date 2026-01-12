import pytest
from unittest.mock import MagicMock, patch
from src.aiops.docker_remediator import DockerRemediator # Assuming this path

@patch("src.aiops.docker_remediator.docker.from_env")
def test_docker_remediator_init(mock_from_env):
    """Test initialization of DockerRemediator."""
    mock_from_env.return_value = MagicMock()
    remediator = DockerRemediator()
    assert remediator.client is not None

@patch("src.aiops.docker_remediator.docker.from_env")
def test_docker_remediator_restart_service_success(mock_from_env):
    """Test successful service restart."""
    mock_client = MagicMock()
    mock_from_env.return_value = mock_client
    
    mock_container = MagicMock()
    mock_client.containers.get.return_value = mock_container
    
    remediator = DockerRemediator()
    service_name = "test_service"
    
    result = remediator.restart_service(service_name)
    
    mock_client.containers.get.assert_called_once_with(service_name)
    mock_container.restart.assert_called_once()
    assert result

@patch("src.aiops.docker_remediator.docker.from_env")
def test_docker_remediator_restart_service_not_found(mock_from_env):
    """Test service restart when container is not found."""
    mock_client = MagicMock()
    mock_from_env.return_value = mock_client
    
    mock_client.containers.get.side_effect = Exception("Container not found") # Simulate NotFound
    
    remediator = DockerRemediator()
    service_name = "non_existent_service"
    
    result = remediator.restart_service(service_name)
    
    mock_client.containers.get.assert_called_once_with(service_name)
    assert not result

@patch("src.aiops.docker_remediator.docker.from_env")
def test_docker_remediator_restart_service_failure(mock_from_env):
    """Test service restart when container.get fails with a general exception."""
    mock_client = MagicMock()
    mock_from_env.return_value = mock_client
    
    mock_client.containers.get.side_effect = Exception("Generic Docker error during get")
    
    remediator = DockerRemediator()
    service_name = "failing_service"
    
    result = remediator.restart_service(service_name)
    
    mock_client.containers.get.assert_called_once_with(service_name)
    assert not result

@patch("src.aiops.docker_remediator.docker.from_env")
@patch("src.aiops.docker_remediator.logger")
def test_docker_remediator_restart_action_failure(mock_logger, mock_from_env):
    """Test service restart when container.restart fails with a general exception."""
    mock_client = MagicMock()
    mock_from_env.return_value = mock_client
    
    mock_container = MagicMock()
    mock_client.containers.get.return_value = mock_container
    mock_container.restart.side_effect = Exception("Generic Docker error during restart")
    
    remediator = DockerRemediator()
    service_name = "failing_service_restart"
    
    result = remediator.restart_service(service_name)
    
    mock_client.containers.get.assert_called_once_with(service_name)
    mock_container.restart.assert_called_once()
    assert not result
    mock_logger.error.assert_called_once()

@patch("src.aiops.docker_remediator.docker.from_env")
def test_docker_remediator_restart_service_empty_name_raises_error(mock_from_env):
    """Test service restart with empty service name."""
    mock_from_env.return_value = MagicMock()
    remediator = DockerRemediator()
    with pytest.raises(ValueError, match="Service name cannot be empty."):
        remediator.restart_service("")

@patch("src.aiops.docker_remediator.docker.from_env")
def test_docker_remediator_init_failure(mock_from_env):
    """Test that DockerRemediator initialization raises an exception on Docker client failure."""
    mock_from_env.side_effect = Exception("Docker client connection failed")
    with pytest.raises(Exception, match="Docker client connection failed"):
        DockerRemediator()
