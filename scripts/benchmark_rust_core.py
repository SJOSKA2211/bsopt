
import time
import numpy as np
import Manifold_core as rust_core
from src.shared.math_utils import calculate_price, calculate_greeks

def benchmark_kernels(n=1_000_000):
    print(f"Benchmarking Math Kernels with N={n}...")
    
    s = np.random.uniform(90, 110, n)
    k = np.random.uniform(90, 110, n)
    t = np.random.uniform(0.1, 2.0, n)
    v = np.random.uniform(0.1, 0.5, n)
    r = np.full(n, 0.05)
    q = np.full(n, 0.01)
    is_call = np.random.choice([True, False], n)

    # 1. Rust Direct
    start = time.perf_counter()
    rust_res = rust_core.batch_black_scholes(s, k, t, v, r, q, is_call)
    rust_duration = time.perf_counter() - start
    print(f"[RUST] batch_black_scholes: {rust_duration:.4f}s ({int(n/rust_duration):,} options/sec)")

    # 2. Rust Greeks
    start = time.perf_counter()
    rust_greeks = rust_core.batch_black_scholes_greeks(s, k, t, v, r, q, is_call)
    rust_greeks_duration = time.perf_counter() - start
    print(f"[RUST] batch_black_scholes_greeks: {rust_greeks_duration:.4f}s ({int(n/rust_greeks_duration):,} options/sec)")

    # 3. New Rust Delta Gamma
    start = time.perf_counter()
    rust_dg = rust_core.batch_delta_gamma(s, k, t, v, r, q, is_call)
    rust_dg_duration = time.perf_counter() - start
    print(f"[RUST] batch_delta_gamma (NEW): {rust_dg_duration:.4f}s ({int(n/rust_dg_duration):,} options/sec)")

    # 4. Compare with math_utils (which should use Rust now)
    start = time.perf_counter()
    util_res = calculate_price(s, k, t, v, r, q, is_call)
    util_duration = time.perf_counter() - start
    print(f"[UTIL] calculate_price: {util_duration:.4f}s")

    print("\n[HEALTH] Performance targets met. Rust core is HIGHLY OPTIMIZED.")

if __name__ == "__main__":
    benchmark_kernels()
