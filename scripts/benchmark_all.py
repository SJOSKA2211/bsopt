import time
import numpy as np
from src.pricing.vol_surface import SVIModel
from src.pricing.quant_utils import generate_paths_v2, batch_bs_price_jit_v2
from src.trading.risk_kernels import _full_risk_check_v2_kernel

def benchmark_quant():
    print("--- Quant Benchmarks ---")
    s0, k, t, v, r, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.02
    n_paths = 100_000
    n_steps = 252

    start = time.perf_counter()
    generate_paths_v2(s0, t, v, r, q, n_paths, n_steps)
    end = time.perf_counter()
    print(f"Path Generation (MC, {n_paths} paths): {(end-start)*1000:.2f}ms")

    n_options = 10_000
    spots = np.full(n_options, 100.0)
    strikes = np.full(n_options, 100.0)
    times = np.full(n_options, 1.0)
    vols = np.full(n_options, 0.2)
    rates = np.full(n_options, 0.05)
    divs = np.full(n_options, 0.02)
    is_calls = np.ones(n_options, dtype=bool)

    start = time.perf_counter()
    batch_bs_price_jit_v2(spots, strikes, times, vols, rates, divs, is_calls)
    end = time.perf_counter()
    print(f"Batch BS Price ({n_options} options): {(end-start)*1000:.2f}ms")

def benchmark_risk():
    print("\n--- Risk Benchmarks ---")
    state = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    limits = np.array([10000.0, 5000.0, 5000.0], dtype=np.float64)
    
    n_iterations = 1_000_000
    start = time.perf_counter()
    for _ in range(n_iterations):
        _full_risk_check_v2_kernel(100.0, 10, 1, 1.5, 0.1, 0.05, state, limits)
    end = time.perf_counter()
    avg_ns = ((end-start) / n_iterations) * 1e9
    print(f"Multi-Greeks Risk Kernel: {avg_ns:.2f}ns per check")

if __name__ == "__main__":
    benchmark_quant()
    benchmark_risk()
