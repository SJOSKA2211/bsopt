from src.pricing.black_scholes import black_scholes
from src.pricing.quantum_pricing import QuantumOptionPricer


def verify_pricing():
    print("Initializing QuantumOptionPricer...")
    pricer = QuantumOptionPricer()
    
    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    num_qubits = 5
    
    print(f"Calculating analytical BS price for S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}...")
    bs_result = black_scholes(S0, K, T, r, sigma)
    bs_price = bs_result["price"]
    print(f"Analytical BS Price: ${bs_price:.4f}")
    
    print("\nRunning Quantum Amplitude Estimation...")
    result = pricer.price_european_call_quantum(S0, K, T, r, sigma, num_qubits=num_qubits)
    
    q_price = result["price"]
    ci = result["confidence_interval"]
    speedup = result["speedup_factor"]
    
    print(f"Quantum Estimated Price: ${q_price:.4f}")
    print(f"95% Confidence Interval: [${ci[0]:.4f}, ${ci[1]:.4f}]")
    print(f"Theoretical Speedup Factor: {speedup:.2f}x")
    
    diff_pct = abs(q_price - bs_price) / bs_price * 100
    print(f"\nDifference: {diff_pct:.2f}%")
    
    if diff_pct < 10:
        print("\n✅ SUCCESS: Quantum price is within 10% tolerance.")
    else:
        print("\n⚠️ WARNING: Difference is high. This may be due to coarse discretization (5 qubits).")

if __name__ == "__main__":
    verify_pricing()
