import math

import numpy as np
import structlog
from numba import cuda, float64, njit, prange, vectorize

from src.utils.memory import profile_gpu_memory

logger = structlog.get_logger(__name__)

# --- GPU Acceleration (Numba CUDA) ---

@vectorize([float64(float64)], target="cuda")
def erf_cuda(x: float) -> float:
    """Error function approximation for CUDA."""
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    res = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return math.copysign(res, x)

@cuda.jit(device=True)
def _ncdf_cuda(x: float) -> float:
    """Device-side normal CDF."""
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865476))

@cuda.jit
def black_scholes_kernel(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
    out: np.ndarray
):
    """Full Black-Scholes GPU Kernel."""
    i = cuda.grid(1)
    if i < S.size:
        if T[i] <= 0:
            out[i] = max(0.0, S[i] - K[i]) if is_call[i] else max(0.0, K[i] - S[i])
            return

        sqrtT = math.sqrt(T[i])
        d1 = (math.log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * v[i]**2) * T[i]) / (v[i] * sqrtT)
        d2 = d1 - v[i] * sqrtT

        if is_call[i]:
            out[i] = S[i] * math.exp(-q[i] * T[i]) * _ncdf_cuda(d1) - K[i] * math.exp(-r[i] * T[i]) * _ncdf_cuda(d2)
        else:
            out[i] = K[i] * math.exp(-r[i] * T[i]) * _ncdf_cuda(-d2) - S[i] * math.exp(-q[i] * T[i]) * _ncdf_cuda(-d1)

@profile_gpu_memory
def price_options_gpu(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray
) -> np.ndarray:
    """Institutional-grade GPU pricing with memory safety."""
    n = S.size
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # Device allocation
    d_S = cuda.to_device(S)
    d_K = cuda.to_device(K)
    d_T = cuda.to_device(T)
    d_v = cuda.to_device(v)
    d_r = cuda.to_device(r)
    d_q = cuda.to_device(q)
    d_is_call = cuda.to_device(is_call)
    d_out = cuda.device_array(n, dtype=np.float64)

    try:
        black_scholes_kernel[blocks_per_grid, threads_per_block](
            d_S, d_K, d_T, d_v, d_r, d_q, d_is_call, d_out
        )
        cuda.synchronize()
        return d_out.copy_to_host()
    finally:
        # Prevent memory leaks
        del d_S, d_K, d_T, d_v, d_r, d_q, d_is_call, d_out

# --- Stochastic Modeling (Numerical Methods) ---

@njit
def gbm_step_rk4(S: float, mu: float, sigma: float, dt: float, dW: float) -> float:
    """4th-order Runge-Kutta step for GBM ODE."""
    k1 = mu * S
    k2 = mu * (S + 0.5 * k1 * dt)
    k3 = mu * (S + 0.5 * k2 * dt)
    k4 = mu * (S + k3 * dt)
    
    S_new = S + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    S_new += sigma * S * dW
    return max(0.0, S_new)

@njit(parallel=True)
def simulate_gbm_rk4(S0: float, mu: float, sigma: float, T: float, steps: int, paths: int) -> np.ndarray:
    """Simulate multiple GBM paths using RK4-like integration."""
    dt = T / steps
    results = np.zeros((paths, steps + 1))
    results[:, 0] = S0
    
    for p in prange(paths):
        curr_S = S0
        for i in range(1, steps + 1):
            dW = np.random.normal(0, math.sqrt(dt))
            curr_S = gbm_step_rk4(curr_S, mu, sigma, dt, dW)
            results[p, i] = curr_S
    return results
