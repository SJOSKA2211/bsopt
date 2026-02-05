from unittest.mock import patch

import pytest

from src.ml.federated_learning.coordinator import FederatedLearningCoordinator


@pytest.fixture
def mock_flwr_server():
    with patch("src.ml.federated_learning.coordinator.fl.server.start_server") as mock:
        yield mock

def test_coordinator_initialization():
    """Verify coordinator initializes with correct strategy."""
    coordinator = FederatedLearningCoordinator(strategy_name="FedAvg")
    assert coordinator.strategy_name == "FedAvg"
    assert coordinator.server_address == "0.0.0.0:8080"

def test_coordinator_start_server(mock_flwr_server):
    """Verify coordinator starts the Flower server with correct parameters."""
    coordinator = FederatedLearningCoordinator()
    coordinator.start(num_rounds=3)
    
    mock_flwr_server.assert_called_once()
    _, kwargs = mock_flwr_server.call_args
    assert kwargs["config"].num_rounds == 3
    assert "strategy" in kwargs
