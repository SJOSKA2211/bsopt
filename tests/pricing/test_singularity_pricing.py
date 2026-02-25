from src.pricing.quantum_pricing import (
    HybridQuantumClassicalPricer,
    QuantumOptionPricer,
)


def test_quantum_option_pricer_simulation():
    pricer = QuantumOptionPricer(use_real_quantum=False)
    # Standard BS parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    result = pricer.price_european_call_quantum(S0, K, T, r, sigma)

    assert "price" in result
    assert result["price"] > 0
    assert result["speedup_factor"] >= 10.0


def test_hybrid_pricer_routing():
    hybrid = HybridQuantumClassicalPricer()

    # 1. Classical routing (low dimensionality)
    res_c = hybrid.price_option_adaptive(S0=100, K=100, T=1, r=0.05, sigma=0.2, num_underlyings=1)
    # MonteCarloEngine returns a float or dict depending on implementation
    assert res_c is not None

    # 2. Quantum routing (high dimensionality)
    res_q = hybrid.price_option_adaptive(S0=100, K=100, T=1, r=0.05, sigma=0.2, num_underlyings=5)
    assert "speedup_factor" in res_q
