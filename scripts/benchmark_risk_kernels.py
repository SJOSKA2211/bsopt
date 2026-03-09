import time

import numpy as np
import structlog

from src.trading.risk_kernels import IncrementalDeltaTracker, _validate_order_kernel

logger = structlog.get_logger()


def benchmark():
    # 1. Benchmark Base Silicon Risk Check
    N = 1000000
    start = time.perf_counter()
    for _ in range(N):
        _validate_order_kernel(100.5, 10, 1)
    duration = time.perf_counter() - start
    print(
        f"Base Silicon Risk Check: {N} iterations in {duration:.4f}s ({duration / N * 1e9:.2f} ns/op)"
    )

    # 2. Benchmark Incremental Delta Tracker (Python wrapper around Numba)
    tracker = IncrementalDeltaTracker(max_net_delta=10000.0)
    start = time.perf_counter()
    for _ in range(N):
        tracker.validate_and_update(1.5)
    duration = time.perf_counter() - start
    print(
        f"Incremental Delta Tracker (Python Wrapper): {N} iterations in {duration:.4f}s ({duration / N * 1e9:.2f} ns/op)"
    )

    # 4. Benchmark Combined God-Tier Risk Kernel (Simulation of absolute hot loop)
    from src.trading.risk_kernels import _full_risk_check_kernel

    state = np.array([0.0], dtype=np.float64)
    start = time.perf_counter()
    for _ in range(N):
        _full_risk_check_kernel(100.5, 10, 1, 1.5, state)
    duration = time.perf_counter() - start
    print(
        f"Combined God-Tier Risk Kernel: {N} iterations in {duration:.4f}s ({duration / N * 1e9:.2f} ns/op)"
    )


if __name__ == "__main__":
    benchmark()
