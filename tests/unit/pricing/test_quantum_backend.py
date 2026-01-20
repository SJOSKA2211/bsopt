import pytest
import os
from src.pricing.quantum_backend import QuantumBackendManager
from qiskit_aer import AerSimulator

def test_get_local_backend():
    manager = QuantumBackendManager()
    backend = manager.get_backend("aer_simulator")
    assert isinstance(backend, AerSimulator)

def test_get_remote_backend_no_provider(mocker):
    # Mock IBM_PROVIDER_AVAILABLE to False
    mocker.patch("src.pricing.quantum_backend.IBM_PROVIDER_AVAILABLE", False)
    manager = QuantumBackendManager()
    with pytest.raises(ImportError, match="qiskit-ibm-provider is not installed"):
        manager.get_backend("ibm_brisbane")

def test_get_remote_backend_no_token(mocker):
    # Ensure IBM_PROVIDER_AVAILABLE is True
    mocker.patch("src.pricing.quantum_backend.IBM_PROVIDER_AVAILABLE", True)
    # Clear environment variable
    if "IBM_QUANTUM_TOKEN" in os.environ:
        del os.environ["IBM_QUANTUM_TOKEN"]
    
    manager = QuantumBackendManager()
    with pytest.raises(ValueError, match="IBM_QUANTUM_TOKEN environment variable not set"):
        manager.get_backend("ibm_brisbane")

def test_get_remote_backend_success(mocker):
    mocker.patch("src.pricing.quantum_backend.IBM_PROVIDER_AVAILABLE", True)
    os.environ["IBM_QUANTUM_TOKEN"] = "mock_token"
    
    mock_provider_class = mocker.patch("src.pricing.quantum_backend.IBMProvider")
    mock_provider_instance = mock_provider_class.return_value
    mock_backend = mocker.Mock()
    mock_provider_instance.get_backend.return_value = mock_backend
    
    manager = QuantumBackendManager()
    backend = manager.get_backend("ibm_brisbane")
    
    assert backend == mock_backend
    mock_provider_class.assert_called_once_with(token="mock_token")
    mock_provider_instance.get_backend.assert_called_once_with("ibm_brisbane")

def test_get_remote_backend_provider_init_failure(mocker):
    mocker.patch("src.pricing.quantum_backend.IBM_PROVIDER_AVAILABLE", True)
    os.environ["IBM_QUANTUM_TOKEN"] = "mock_token"
    
    mocker.patch("src.pricing.quantum_backend.IBMProvider", side_effect=Exception("Init failed"))
    
    manager = QuantumBackendManager()
    with pytest.raises(Exception, match="Init failed"):
        manager.get_backend("ibm_brisbane")

def test_get_remote_backend_retrieval_failure(mocker):
    mocker.patch("src.pricing.quantum_backend.IBM_PROVIDER_AVAILABLE", True)
    os.environ["IBM_QUANTUM_TOKEN"] = "mock_token"
    
    mock_provider_class = mocker.patch("src.pricing.quantum_backend.IBMProvider")
    mock_provider_instance = mock_provider_class.return_value
    mock_provider_instance.get_backend.side_effect = Exception("Retrieval failed")
    
    manager = QuantumBackendManager()
    with pytest.raises(Exception, match="Retrieval failed"):
        manager.get_backend("ibm_brisbane")
