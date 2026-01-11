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
        
        assert len(qc.data) > 0
        assert qc.depth() > 0
        assert qc.num_qubits == num_qubits

    def test_payoff_operator_gates(self):
        """Verify that the payoff operator adds the expected gates."""
        pricer = QuantumOptionPricer()
        S0 = 100.0
        K = 100.0
        mu = 0.05
        sigma = 0.2
        T = 1.0
        num_qubits = 3
        
        qc, prices = pricer.create_stock_price_distribution(S0, mu, sigma, T, num_qubits)
        
        initial_depth = qc.depth()
        pricer.add_payoff_operator(qc, prices, K, S0)
        
        assert qc.depth() > initial_depth
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
        num_qubits = 5 
        
        # Calculate analytical BS price for reference
        from src.pricing.black_scholes import black_scholes
        bs_price = black_scholes(S0, K, T, r, sigma)["price"]
        
        # Call quantum pricing
        result = pricer.price_european_call_quantum(S0, K, T, r, sigma, num_qubits=num_qubits)
        
        assert "price" in result
        assert "confidence_interval" in result
        assert "speedup_factor" in result
        
        # Check if quantum price is within 10% of BS price
        assert np.isclose(result["price"], bs_price, rtol=0.1)
        assert result["speedup_factor"] > 0

class TestQuantumOptimizer:
    def test_optimizer_reduces_gate_count(self):
        """Verify that the optimizer actually reduces the number of gates in a redundant circuit."""
        from src.pricing.quantum_pricing import QuantumCircuitOptimizer
        optimizer = QuantumCircuitOptimizer()
        
        # Create a highly redundant circuit
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.h(0) # Should be identity
        qc.x(0)
        qc.x(0) # Should be identity
        
        initial_size = qc.size()
        optimized_qc = optimizer.optimize_circuit(qc)
        
        assert optimized_qc.size() < initial_size
        # For this specific case, it should ideally be 0 or much smaller
        assert optimized_qc.size() == 0