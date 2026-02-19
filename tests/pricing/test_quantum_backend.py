import os
from unittest.mock import MagicMock, patch

import pytest

from src.pricing.quantum_backend import QuantumBackendManager


class TestQuantumBackendManager:
    @patch.dict(os.environ, {"IBM_QUANTUM_TOKEN": "test_token"})
    @patch("src.pricing.quantum_backend.IBM_PROVIDER_AVAILABLE", True)
    @patch("src.pricing.quantum_backend.IBMProvider")
    def test_get_backend_remote(self, mock_ibm_provider):
        """Test getting a remote backend using IBMProvider."""
        # Setup mock
        mock_provider_instance = MagicMock()
        mock_ibm_provider.return_value = mock_provider_instance

        mock_backend = MagicMock()
        mock_provider_instance.get_backend.return_value = mock_backend

        # Initialize manager
        manager = QuantumBackendManager()

        # Get backend
        backend = manager.get_backend(backend_name="ibmq_qasm_simulator")

        # Assertions
        mock_ibm_provider.assert_called_once_with(token="test_token")
        mock_provider_instance.get_backend.assert_called_once_with(
            "ibmq_qasm_simulator"
        )
        assert backend == mock_backend

    def test_get_backend_local(self):
        """Test getting a local simulator backend."""
        manager = QuantumBackendManager()
        backend = manager.get_backend(backend_name="aer_simulator")

        assert backend.name == "aer_simulator"

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.pricing.quantum_backend.IBM_PROVIDER_AVAILABLE", True)
    def test_missing_token_error(self):
        """Test error when token is missing for remote backend."""
        manager = QuantumBackendManager()

        with pytest.raises(
            ValueError, match="IBM_QUANTUM_TOKEN environment variable not set"
        ):
            manager.get_backend(backend_name="ibmq_qasm_simulator")
