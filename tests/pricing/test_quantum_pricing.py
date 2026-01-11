import pytest
import numpy as np
from qiskit import QuantumCircuit
# We anticipate the module structure based on PRD
from src.pricing.quantum_pricing import QuantumOptionPricer

class TestQuantumPricing:
    
    def test_create_stock_price_distribution_structure(self):
        """Verify the output is a QuantumCircuit with correct dimensions."""
        pricer = QuantumOptionPricer()
        S0 = 100.0
        mu = 0.05
        sigma = 0.2
        T = 1.0
        num_qubits = 3
        
        qc, prices = pricer.create_stock_price_distribution(S0, mu, sigma, T, num_qubits)
        
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == num_qubits
        assert len(prices) == 2**num_qubits
        
    def test_distribution_normalization(self):
        """Verify that the encoded amplitudes create a valid probability distribution."""
        pricer = QuantumOptionPricer()
        S0 = 100.0
        mu = 0.05
        sigma = 0.2
        T = 1.0
        num_qubits = 4
        
        qc, prices = pricer.create_stock_price_distribution(S0, mu, sigma, T, num_qubits)
        
        # We can't easily extract statevector without running simulation, 
        # but the method should theoretically setup the circuit.
        # For this unit test, we might check if the circuit has initialization instructions.
        
        # Assuming the implementation uses qc.initialize(amplitudes, qr)
        # We can inspect the circuit instructions
        
        assert len(qc.data) > 0
        instruction = qc.data[0].operation
        assert instruction.name == "initialize"
        
        # Check if params (amplitudes) sum to 1 (squared)
        amplitudes = instruction.params
        probs = np.abs(amplitudes)**2
        assert np.isclose(np.sum(probs), 1.0)
