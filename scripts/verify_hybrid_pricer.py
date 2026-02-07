from src.pricing.quantum_pricing import HybridQuantumClassicalPricer


def verify_hybrid_pricer():
    print("Initializing HybridQuantumClassicalPricer...")
    pricer = HybridQuantumClassicalPricer()
    
    # Common parameters
    base_params = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "num_qubits": 5
    }
    
    print("\nCase 1: Low Dimensionality (1 underlying), Low Accuracy (5%)")
    params1 = {**base_params, "num_underlyings": 1, "accuracy": 0.05}
    result1 = pricer.price_option_adaptive(**params1)
    print(f"Result: {result1}")
    method1 = result1.get("method", "quantum") # Quantum doesn't explicitly return 'method' yet in my implementation but classical does
    if "classical" in str(method1).lower():
        print("✅ Correctly routed to Classical Engine.")
    else:
        print("❌ Unexpectedly routed to Quantum Engine.")

    print("\nCase 2: High Dimensionality (4 underlyings), Low Accuracy (5%)")
    params2 = {**base_params, "num_underlyings": 4, "accuracy": 0.05}
    result2 = pricer.price_option_adaptive(**params2)
    # Quantum result has 'num_queries' and 'estimation'
    if "num_queries" in result2:
        print("✅ Correctly routed to Quantum Engine.")
    else:
        print("❌ Unexpectedly routed to Classical Engine.")

    print("\nCase 3: Low Dimensionality (1 underlying), High Accuracy (0.5%)")
    params3 = {**base_params, "num_underlyings": 1, "accuracy": 0.005}
    result3 = pricer.price_option_adaptive(**params3)
    if "num_queries" in result3:
        print("✅ Correctly routed to Quantum Engine.")
    else:
        print("❌ Unexpectedly routed to Classical Engine.")

if __name__ == "__main__":
    verify_hybrid_pricer()
