import pytest
import numpy as np
import os
from src.pricing.quantum_pricing import (
    QuantumOptionPricer,
    QuantumCircuitOptimizer,
    HybridQuantumClassicalPricer
)
from qiskit import QuantumCircuit, QuantumRegister

def test_quantum_option_pricer_init():
    # Local simulator
    pricer = QuantumOptionPricer(use_real_quantum=False)
    assert not pricer.use_real_quantum
    
    # Fallback case - should not crash
    pricer_fallback = QuantumOptionPricer(use_real_quantum=True, backend_name="invalid")
    assert pricer_fallback.backend is not None

def test_add_payoff_operator_coverage():
    pricer = QuantumOptionPricer()
    prices = np.array([100.0])
    
    # Cover line 90: reuse payoff register
    qc = QuantumCircuit(QuantumRegister(1, 'price'), QuantumRegister(1, 'payoff'))
    pricer.add_payoff_operator(qc, prices, 100, 100)
    assert len(qc.qregs) == 2
    
    # Cover line 106: price_qubits fallback to qregs[0]
    qc2 = QuantumCircuit(QuantumRegister(1, 'unknown'))
    pricer.add_payoff_operator(qc2, prices, 100, 100)
    assert any(reg.name == 'payoff' for reg in qc2.qregs)

def test_price_european_call_quantum_errors(mocker):
    pricer = QuantumOptionPricer()
    # Mock QuantumCircuitOptimizer to return a circuit without payoff register
    mocker.patch("src.pricing.quantum_pricing.QuantumCircuitOptimizer.optimize_circuit", return_value=QuantumCircuit(1))
    
    with pytest.raises(ValueError, match="Payoff qubit not found"):
        pricer.price_european_call_quantum(100, 100, 1.0, 0.05, 0.2, num_qubits=2)

def test_create_stock_price_distribution():
    pricer = QuantumOptionPricer()
    qc, prices = pricer.create_stock_price_distribution(100, 0.05, 0.2, 1.0, num_qubits=3)
    assert len(prices) == 8
    assert isinstance(qc, QuantumCircuit)

def test_add_payoff_operator():
    pricer = QuantumOptionPricer()
    num_qubits = 3
    qc, prices = pricer.create_stock_price_distribution(100, 0.05, 0.2, 1.0, num_qubits=num_qubits)
    
    # Test adding payoff register
    pricer.add_payoff_operator(qc, prices, 100, 100)
    assert any(reg.name == 'payoff' for reg in qc.qregs)
    
    # Test error if price register missing
    empty_qc = QuantumCircuit()
    with pytest.raises(ValueError, match="Price register not found"):
        pricer.add_payoff_operator(empty_qc, prices, 100, 100)

def test_price_european_call_quantum():
    pricer = QuantumOptionPricer()
    # Use very small num_qubits for speed
    res = pricer.price_european_call_quantum(100, 100, 1.0, 0.05, 0.2, num_qubits=2)
    assert "price" in res
    assert res["price"] >= 0
    assert "confidence_interval" in res

def test_quantum_circuit_optimizer():
    optimizer = QuantumCircuitOptimizer()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    optimized_qc = optimizer.optimize_circuit(qc)
    assert isinstance(optimized_qc, QuantumCircuit)

def test_hybrid_pricer_adaptive():
    hybrid = HybridQuantumClassicalPricer()
    
    # Classical route
    res_classical = hybrid.price_option_adaptive(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, num_underlyings=1, accuracy=0.1)
    assert isinstance(res_classical, tuple) # MonteCarloEngine.price_european returns tuple
    
    # Quantum route (Dimension > 3)
    res_quantum = hybrid.price_option_adaptive(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, num_underlyings=4, accuracy=0.1, num_qubits=2)
    assert "price" in res_quantum
    
    # Quantum route (Accuracy < 1%)
    res_quantum_acc = hybrid.price_option_adaptive(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, num_underlyings=1, accuracy=0.001, num_qubits=2)
    assert "price" in res_quantum_acc

def test_quantum_pricer_backend_env(mocker):
    os.environ["QUANTUM_BACKEND"] = "mock_backend"
    mock_get_backend = mocker.patch("src.pricing.quantum_backend.QuantumBackendManager.get_backend")
    
    QuantumOptionPricer(use_real_quantum=True)
    mock_get_backend.assert_called_with("mock_backend")
