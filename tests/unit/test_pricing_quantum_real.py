import pytest

from src.pricing.quantum_pricing import QISKIT_AVAILABLE, QuantumOptionPricer


def test_qiskit_availability():
    """Verify Qiskit is installed and detected."""
    assert QISKIT_AVAILABLE is True, "Qiskit should be available in Phase 4"


def test_quantum_pricing_aer_execution():
    """Verify that QuantumOptionPricer uses AerSimulator and not fallback."""
    if not QISKIT_AVAILABLE:
        pytest.skip("Qiskit not available")

    pricer = QuantumOptionPricer(use_real_quantum=False)

    # Simple call option parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    # Run pricing (low qubits for speed in test)
    result = pricer.price_european_call_quantum(S0, K, T, r, sigma, num_qubits=3)

    # Verify backend
    # The current implementation returns a dict. If fallback is used, it adds "backend": "analytical_fallback"
    # If successful, it returns "price", "confidence_interval", "num_queries", "speedup_factor"

    print(f"Quantum Result: {result}")

    assert (
        "backend" not in result or result["backend"] != "analytical_fallback"
    ), f"Pricer used fallback backend! Error: {result.get('error')}"

    assert "price" in result
    assert result["price"] > 0
    assert "num_queries" in result
