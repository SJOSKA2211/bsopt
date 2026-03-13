"""
EquaFlow GPU Mathematical Kernels
Optimized for NVIDIA CUDA via Numba.
"""

import math
import time

import numpy as np
import structlog
from numba import cuda, float64, void

logger = structlog.get_logger(__name__)

# Constants
INV_SQRT2PI = 0.3989422804014327
INV_SQRT2 = 0.7071067811865476


@cuda.jit(device=True)
def _cuda_normal_cdf(x: float) -> float:
    """Device-only function for Normal CDF calculation."""
    abs_x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * abs_x)
    d = INV_SQRT2PI * math.exp(-x * x / 2.0)
    prob = (
        d
        * t
        * (
            0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
    )

    if x > 0:
        return 1.0 - prob
    return prob


@cuda.jit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]))
def bs_price_kernel(S, K, T, sigma, r, q, out):
    """Vectorized Black-Scholes pricing kernel."""
    idx = cuda.grid(1)
    if idx < S.size:
        s_val = S[idx]
        k_val = K[idx]
        t_val = max(T[idx], 1e-9)
        sig_val = max(sigma[idx], 1e-9)
        r_val = r[idx]
        q_val = q[idx]

        vol_sqrt_t = sig_val * math.sqrt(t_val)
        d1 = (
            math.log(s_val / k_val) + (r_val - q_val + 0.5 * sig_val * sig_val) * t_val
        ) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t

        exp_qT = math.exp(-q_val * t_val)
        exp_rT = math.exp(-r_val * t_val)

        cdf_d1 = _cuda_normal_cdf(d1)
        cdf_d2 = _cuda_normal_cdf(d2)

        # We assume call for this kernel, or could pass is_call array
        out[idx] = s_val * exp_qT * cdf_d1 - k_val * exp_rT * cdf_d2


def gpu_price_batch(S, K, T, sigma, r, q):
    """Wrapper to launch CUDA kernel for batch pricing."""
    n = S.size
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # Device allocations
    d_S = cuda.to_device(S)
    d_K = cuda.to_device(K)
    d_T = cuda.to_device(T)
    d_sigma = cuda.to_device(sigma)
    d_r = cuda.to_device(r)
    d_q = cuda.to_device(q)
    d_out = cuda.device_array(n, dtype=np.float64)

    start = time.perf_counter()
    bs_price_kernel[blocks_per_grid, threads_per_block](d_S, d_K, d_T, d_sigma, d_r, d_q, d_out)
    cuda.synchronize()
    duration = time.perf_counter() - start

    logger.info("cuda_kernel_executed", duration_ms=duration * 1000, n=n)
    return d_out.copy_to_host()


def profile_gpu_memory():
    """Profiles GPU memory usage via Numba/CUDA."""
    try:
        mem = cuda.current_context().get_memory_info()
        # free, total = mem
        return {
            "gpu_free_mb": mem[0] / 1024**2,
            "gpu_total_mb": mem[1] / 1024**2,
            "gpu_used_pct": (1 - mem[0] / mem[1]) * 100,
        }
    except Exception:
        return {"error": "GPU not available or driver missing"}
