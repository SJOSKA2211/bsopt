import asyncio
import os
import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


async def benchmark_rust_kernel():
    """Benchmark the Rust-accelerated math kernel."""
    print("🔹 Benchmarking Rust Math Kernel (CPU Parallel)...")
    try:
        import equaflow_core

        n = 100_000
        S = np.random.uniform(90, 110, n)
        K = np.random.uniform(90, 110, n)
        T = np.random.uniform(0.1, 2.0, n)
        sigma = np.random.uniform(0.1, 0.5, n)
        r = np.full(n, 0.05)
        q = np.zeros(n)
        is_call = np.random.choice([True, False], n)

        start = time.perf_counter()
        _ = equaflow_core.batch_black_scholes(S, K, T, sigma, r, q, is_call)
        duration = time.perf_counter() - start

        print(f"   - Batch Size: {n:,}")
        print(f"   - Total Time: {duration:.4f} s")
        print(f"   - Per Opt:    {duration / n * 1_000_000:.4f} μs")
        return duration
    except ImportError:
        print("   ⚠️  equaflow_core not installed. Skipping.")
        return None


async def benchmark_cupy_kernel():
    """Benchmark the GPU-accelerated math kernel."""
    print("🔹 Benchmarking CuPy Math Kernel (GPU)...")
    try:
        import cupy as cp

        from src.math_kernel.kernels import black_scholes_cupy

        n = 1_000_000
        S = cp.random.uniform(90, 110, n)
        K = cp.random.uniform(90, 110, n)
        T = cp.random.uniform(0.1, 2.0, n)
        sigma = cp.random.uniform(0.1, 0.5, n)
        r = cp.full(n, 0.05)

        # Warmup
        _ = black_scholes_cupy(S, K, T, sigma, r)
        cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()
        _ = black_scholes_cupy(S, K, T, sigma, r)
        cp.cuda.Stream.null.synchronize()
        duration = time.perf_counter() - start

        print(f"   - Batch Size: {n:,}")
        print(f"   - Total Time: {duration:.4f} s")
        print(f"   - Per Opt:    {duration / n * 1_000_000:.4f} μs")
        return duration
    except (ImportError, Exception) as e:
        print(f"   ⚠️  GPU/CuPy not available: {str(e)[:50]}. Skipping.")
        return None


async def benchmark_mmap_parser():
    """Benchmark binary tick parsing throughput."""
    print("🔹 Benchmarking Zero-Copy MMap Tick Parsing...")
    try:
        import equaflow_core

        # Create a dummy 32MB file (1M ticks of 32 bytes)
        file_path = "/tmp/benchmark_ticks.bin"
        with open(file_path, "wb") as f:  # noqa: ASYNC230
            f.write(os.urandom(1024 * 1024 * 32))

        buffer = equaflow_core.TickDataBuffer(file_path)
        start = time.perf_counter()
        _ = buffer.parse_ticks_32b(0, 1_000_000)
        duration = time.perf_counter() - start

        throughput = (32) / duration  # MB/s
        print("   - Count:      1,000,000 ticks")
        print(f"   - Throughput: {throughput:.2f} MB/s")
        print(f"   - Latency:    {duration * 1000:.4f} ms")
        os.remove(file_path)
        return duration
    except (ImportError, Exception) as e:
        print(f"   ⚠️  MMap benchmark failed: {str(e)}. Skipping.")
        return None


async def run_suite():
    print("=" * 60)
    print("EquaFlow Institutional Performance Report")
    print("=" * 60)
    await benchmark_rust_kernel()
    print("-" * 40)
    await benchmark_cupy_kernel()
    print("-" * 40)
    await benchmark_mmap_parser()
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_suite())
