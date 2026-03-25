from unittest.mock import MagicMock, patch

import pytest

from src.math_kernel.quantum_pricing import (
    HybridQuantumClassicalPricer,
    QuantumOptionPricer,
)

@pytest.fixture
def quantum_pricer():
    return QuantumOptionPricer(use_real_quantum=False)

def test_quantum_option_pricer_initialization(quantum_pricer):
    assert quantum_pricer.backend is not None
    assert quantum_pricer.optimizer is not None

def test_create_stock_price_distribution(quantum_pricer):
    S0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    qc, prices = quantum_pricer.create_stock_price_distribution(S0, mu, sigma, T, num_qubits=3)

    assert len(prices) == 2**3
    assert qc.num_qubits >= 3
    # Check if price distribution is centered roughly around S0
    assert prices[0] < S0 < prices[-1]

def test_math_fallback_logic(quantum_pricer):
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    result = quantum_pricer._math_fallback(S0, K, T, r, sigma)

    assert "price" in result
    assert result["price"] > 0
    assert result["backend"] == "analytical_fallback"

@patch("src.math_kernel.quantum_pricing.QISKIT_AVAILABLE", True)
@patch("src.math_kernel.quantum_pricing.IterativeAmplitudeEstimation")
@patch("src.math_kernel.quantum_pricing.StatevectorSampler")
def test_price_european_call_quantum_success(mock_sampler, mock_iae_class, quantum_pricer):
    # Setup mock IAE result
    mock_iae = mock_iae_class.return_value
    mock_result = MagicMock()
    mock_result.estimation = 0.5
    mock_result.confidence_interval = [0.45, 0.55]
    mock_result.num_oracle_queries = 100
    mock_iae.estimate.return_value = mock_result

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    # Mock add_payoff_operator to avoid heavy circuit ops
    quantum_pricer.add_payoff_operator = MagicMock()

    result = quantum_pricer.price_european_call_quantum(S0, K, T, r, sigma, num_qubits=3)

    assert result["price"] > 0
    assert "confidence_interval" in result
    assert result["num_queries"] == 100

def test_hybrid_pricer_selection():
    hybrid = HybridQuantumClassicalPricer()
    hybrid.quantum_pricer.price_european_call_quantum = MagicMock(return_value={"price": 20.0})
    hybrid.classical_pricer.price_european = MagicMock(return_value={"price": 10.0})

    # 1. Simple case -> Classical
    res1 = hybrid.price_option_adaptive(S0=100, K=100, T=1, r=0.05, sigma=0.2, num_underlyings=1)
    assert res1["price"] == 10.0

    # 2. Complex case -> Quantum
    res2 = hybrid.price_option_adaptive(S0=100, K=100, T=1, r=0.05, sigma=0.2, num_underlyings=5)
    assert res2["price"] == 20.0
