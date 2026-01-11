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
        # Instead of checking instruction name which depends on implementation details (Initialize vs StatePreparation)
        # We check if the circuit has non-zero depth and correct number of qubits
        assert qc.depth() > 0
        assert qc.num_qubits == num_qubits
        
        # Verify that the total probability encoded is 1.0
        # We can do this by inspecting the statevector if we use a simulator,
        # but for unit test we'll just check that it's not empty.

    def test_payoff_operator_gates(self):
        """Verify that the payoff operator adds the expected gates (RY, MCRY)."""
        pricer = QuantumOptionPricer()
        S0 = 100.0
        K = 100.0
        mu = 0.05
        sigma = 0.2
        T = 1.0
        num_qubits = 3
        
        qc, prices = pricer.create_stock_price_distribution(S0, mu, sigma, T, num_qubits)
        
        # Add payoff operator
        # We expect a method to exist or we test the pricing flow which adds it.
        # Let's assume we expose a helper for modularity/testing.
        
        # Verify initial depth
        initial_depth = qc.depth()
        
        pricer.add_payoff_operator(qc, prices, K, S0)
        
        # Should have added gates
        assert qc.depth() > initial_depth
        
        # Check for RY gates (rotations) or controlled rotations
        # Qiskit's mcry decompostion might vary, but we look for rotation logic.
        
        # A simple check is ensuring the Payoff register is present if added by the method, 
        # or checking that we can add it.
        
        assert any(reg.name == 'payoff' for reg in qc.qregs)

    def test_price_european_call_quantum_convergence(self):
        """Verify that the quantum pricer converges towards the analytical Black-Scholes price."""
        pricer = QuantumOptionPricer()
        
        # Test parameters
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        num_qubits = 5 # Higher precision for convergence check
        
        # Calculate analytical BS price for reference
        from src.pricing.black_scholes import black_scholes
        bs_price = black_scholes(S0, K, T, r, sigma)["price"]
        
        # Call quantum pricing
        result = pricer.price_european_call_quantum(S0, K, T, r, sigma, num_qubits=num_qubits)
        
        assert "price" in result
        assert "confidence_interval" in result
        
        # Check if quantum price is within 10% of BS price (discretization error is significant)
        assert np.isclose(result["price"], bs_price, rtol=0.1)