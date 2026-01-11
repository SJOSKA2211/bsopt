import os
import structlog
from qiskit_aer import AerSimulator
from typing import Any

try:
    from qiskit_ibm_provider import IBMProvider
    IBM_PROVIDER_AVAILABLE = True
except ImportError:
    IBMProvider = None
    IBM_PROVIDER_AVAILABLE = False

logger = structlog.get_logger()

class QuantumBackendManager:
    """
    Manages connections to Quantum Backends (Local Simulators or IBM Quantum Hardware).
    """
    def __init__(self):
        self.provider = None

    def get_backend(self, backend_name: str = "aer_simulator") -> Any:
        """
        Retrieves a quantum backend.
        
        Args:
            backend_name: Name of the backend (e.g., 'aer_simulator', 'ibmq_qasm_simulator', 'ibm_brisbane').
            
        Returns:
            A Qiskit Backend instance.
        """
        if backend_name == "aer_simulator":
            logger.info("using_local_backend", backend="aer_simulator")
            return AerSimulator()
        
        # Assume anything else is an IBM Quantum backend
        if not IBM_PROVIDER_AVAILABLE:
            raise ImportError("qiskit-ibm-provider is not installed. Cannot access remote backends.")
            
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            raise ValueError("IBM_QUANTUM_TOKEN environment variable not set. Cannot access remote backends.")
            
        if self.provider is None:
            try:
                self.provider = IBMProvider(token=token)
                logger.info("ibm_provider_initialized")
            except Exception as e:
                logger.error("ibm_provider_init_failed", error=str(e))
                raise

        try:
            backend = self.provider.get_backend(backend_name)
            logger.info("using_remote_backend", backend=backend_name)
            return backend
        except Exception as e:
            logger.error("backend_retrieval_failed", backend=backend_name, error=str(e))
            raise
