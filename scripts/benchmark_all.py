import time

import numpy as np

from services.quant.pricing.quant_utils import (
    batch_bs_price_jit_v2,
    generate_paths_v2,
)
from core.trading.risk_kernels import _full_risk_check_v2_kernel


def benchmark_jit_warmup():
    print("--- JIT Warmup Impact ---")
    s = np.array([100.0], dtype=np.float64)
    k = np.array([100.0], dtype=np.float64)
    t = np.array([0.1], dtype=np.float64)
    sig = np.array([0.2], dtype=np.float64)
    r = np.array([0.05], dtype=np.float64)
    q = np.array([0.0], dtype=np.float64)
    is_call = np.array([True], dtype=bool)

    # 1. Cold Call (Force compile)
    start = time.perf_counter()
    batch_bs_price_jit_v2(s, k, t, sig, r, q, is_call)
    cold_time = (time.perf_counter() - start) * 1000

    # 2. Hot Call
    start = time.perf_counter()
    batch_bs_price_jit_v2(s, k, t, sig, r, q, is_call)
    hot_time = (time.perf_counter() - start) * 1000

    print(f"Cold Call (JIT overhead): {cold_time:.4f}ms")
    print(f"Hot Call (Optimized): {hot_time:.4f}ms")
    if cold_time > 0:
        print(f"Speedup from warmup: {cold_time / max(hot_time, 1e-9):.1f}x")


def benchmark_quant():
    print("\n--- Quant Benchmarks ---")
    s0, _k, t, v, r, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.02
    n_paths = 100_000
    n_steps = 252

    start = time.perf_counter()
    generate_paths_v2(s0, t, v, r, q, n_paths, n_steps)
    end = time.perf_counter()
    print(f"Path Generation (MC, {n_paths} paths): {(end - start) * 1000:.2f}ms")

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
    print(f"Batch BS Price ({n_options} options): {(end - start) * 1000:.2f}ms")


def benchmark_risk():
    print("\n--- Risk Benchmarks ---")
    state = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    limits = np.array([10000.0, 5000.0, 5000.0], dtype=np.float64)

    n_iterations = 1_000_000
    start = time.perf_counter()
    for _ in range(n_iterations):
        _full_risk_check_v2_kernel(100.0, 10, 1, 1.5, 0.1, 0.05, state, limits)
    end = time.perf_counter()
    avg_ns = ((end - start) / n_iterations) * 1e9
    print(f"Multi-Greeks Risk Kernel: {avg_ns:.2f}ns per check")


def benchmark_exotic():
    print("\n--- Exotic Option Benchmarks (Rust/JIT) ---")
    from services.quant.pricing.exotic import AsianOptionPricer, ExoticParameters
    from services.quant.pricing.models import BSParameters

    params = ExoticParameters(
        base_params=BSParameters(
            spot=100.0, strike=100.0, maturity=0.5, volatility=0.2, rate=0.05, dividend=0.0
        ),
        n_observations=252,
    )

    # 1. Asian (Geometric)
    start = time.perf_counter()
    for _ in range(1000):
        AsianOptionPricer.price_geometric_asian(params, "call")
    end = time.perf_counter()
    print(f"Geometric Asian (Rust/JIT): {(end - start) * 1000:.2f}µs per price")


def benchmark_heston():
    print("\n--- Heston Model Benchmarks (FFT + Rust CF) ---")
    from services.quant.pricing.models import HestonParams
    from services.quant.pricing.models.heston_fft import HestonModelFFT

    h_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    model = HestonModelFFT(h_params, r=0.05, T=1.0)

    start = time.perf_counter()
    for _ in range(100):
        model.price_call(100.0, 100.0)
    end = time.perf_counter()
    print(f"Heston FFT (Rust-Accelerated): {(end - start) * 10:.2f}ms per price")


if __name__ == "__main__":
    benchmark_jit_warmup()
    benchmark_quant()
    benchmark_exotic()
    benchmark_heston()
    benchmark_risk()
