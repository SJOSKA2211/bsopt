import time

import pandas as pd

from src.pricing.black_scholes import black_scholes
from src.pricing.quantum_pricing import HybridQuantumClassicalPricer


def run_benchmarks():
    print("🚀 Starting E2E Quantum-Classical Pricing Benchmarks")
    pricer = HybridQuantumClassicalPricer()

    # Parameters
    params = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "num_qubits": 5,
    }

    results = []

    # 1. Analytical (Ground Truth)
    start = time.time()
    bs_result = black_scholes(
        params["S0"], params["K"], params["T"], params["r"], params["sigma"]
    )
    end = time.time()
    results.append(
        {
            "Engine": "Analytical (BS)",
            "Price": bs_result["price"],
            "Latency (ms)": (end - start) * 1000,
            "Accuracy": "100% (Truth)",
        }
    )

    # 2. Classical Monte Carlo (Adaptive Routing - forced)
    start = time.time()
    # num_underlyings=1 and accuracy=0.05 routes to Classical
    classical_result = pricer.price_option_adaptive(
        **params, num_underlyings=1, accuracy=0.05
    )
    end = time.time()
    results.append(
        {
            "Engine": "Classical MC",
            "Price": classical_result["price"],
            "Latency (ms)": (end - start) * 1000,
            "Accuracy": f"{100 - abs(classical_result['price'] - bs_result['price'])/bs_result['price']*100:.2f}%",
        }
    )

    # 3. Quantum Amplitude Estimation (Adaptive Routing - forced)
    start = time.time()
    # num_underlyings=4 routes to Quantum
    quantum_result = pricer.price_option_adaptive(
        **params, num_underlyings=4, accuracy=0.05
    )
    end = time.time()
    results.append(
        {
            "Engine": "Quantum QAE",
            "Price": quantum_result["price"],
            "Latency (ms)": (end - start) * 1000,
            "Accuracy": f"{100 - abs(quantum_result['price'] - bs_result['price'])/bs_result['price']*100:.2f}%",
        }
    )

    df = pd.DataFrame(results)
    print("\nBenchmark Results Summary:")
    print(df.to_string(index=False))

    print("\n✅ Benchmarking Complete.")


if __name__ == "__main__":
    run_benchmarks()
