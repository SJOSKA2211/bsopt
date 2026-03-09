import time

import structlog

from src.trading.risk_kernels import _validate_incremental_delta_kernel, _validate_order_kernel

logger = structlog.get_logger()


def benchmark_risk_path():
    # 1. Warmup
    for _ in range(1000):
        _validate_order_kernel(100.0, 10, 1)
        _validate_incremental_delta_kernel(5000.0, 10.5, 10000.0)

    # 2. Benchmark Order Kernel
    iters = 1_000_000
    start = time.perf_counter()
    for _ in range(iters):
        _validate_order_kernel(100.0, 10, 1)
    end = time.perf_counter()
    order_latency_ns = ((end - start) / iters) * 1e9

    # 3. Benchmark Delta Kernel
    start = time.perf_counter()
    for _ in range(iters):
        _validate_incremental_delta_kernel(5000.0, 10.5, 10000.0)
    end = time.perf_counter()
    delta_latency_ns = ((end - start) / iters) * 1e9

    logger.info(
        "risk_path_benchmark_results",
        order_check_ns=round(order_latency_ns, 2),
        delta_check_ns=round(delta_latency_ns, 2),
        total_check_ns=round(order_latency_ns + delta_latency_ns, 2),
    )


if __name__ == "__main__":
    benchmark_risk_path()
