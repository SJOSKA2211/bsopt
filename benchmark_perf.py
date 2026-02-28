import time
import numpy as np
from numba import njit, prange
import sys
import os

# Set required environment variables before imports
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["JWT_SECRET"] = "dummy"
os.environ["MFA_ENCRYPTION_KEY"] = "dummykey" * 8
os.environ["DATABASE_URL"] = "postgresql://dummy"
os.environ["PROJECT_NAME"] = "dummy"

sys.path.append("/app")

from src.shared.math_utils import calculate_price_core, calculate_greeks_core

@njit(cache=True, fastmath=True)
def _vec_price_impl_serial(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    n = len(flat_s)
    flat_res = np.empty(n, dtype=np.float64)
    for i in range(n):
        flat_res[i] = calculate_price_core(
            flat_s[i], flat_k[i], flat_t[i], flat_sigma[i], flat_r[i], flat_q[i], flat_is_call[i]
        )
    return flat_res

@njit(cache=True, fastmath=True, parallel=True)
def _vec_price_impl_parallel(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    n = len(flat_s)
    flat_res = np.empty(n, dtype=np.float64)
    for i in prange(n):
        flat_res[i] = calculate_price_core(
            flat_s[i], flat_k[i], flat_t[i], flat_sigma[i], flat_r[i], flat_q[i], flat_is_call[i]
        )
    return flat_res

@njit(cache=True, fastmath=True)
def _vec_greeks_impl_serial(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    n = len(flat_s)
    f_delta = np.empty(n, dtype=np.float64)
    f_gamma = np.empty(n, dtype=np.float64)
    f_theta = np.empty(n, dtype=np.float64)
    f_vega = np.empty(n, dtype=np.float64)
    f_rho = np.empty(n, dtype=np.float64)
    for i in range(n):
        d, g, th, v, rh = calculate_greeks_core(
            flat_s[i], flat_k[i], flat_t[i], flat_sigma[i], flat_r[i], flat_q[i], flat_is_call[i]
        )
        f_delta[i] = d
        f_gamma[i] = g
        f_theta[i] = th
        f_vega[i] = v
        f_rho[i] = rh
    return f_delta, f_gamma, f_theta, f_vega, f_rho

@njit(cache=True, fastmath=True, parallel=True)
def _vec_greeks_impl_parallel(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    n = len(flat_s)
    f_delta = np.empty(n, dtype=np.float64)
    f_gamma = np.empty(n, dtype=np.float64)
    f_theta = np.empty(n, dtype=np.float64)
    f_vega = np.empty(n, dtype=np.float64)
    f_rho = np.empty(n, dtype=np.float64)
    for i in prange(n):
        d, g, th, v, rh = calculate_greeks_core(
            flat_s[i], flat_k[i], flat_t[i], flat_sigma[i], flat_r[i], flat_q[i], flat_is_call[i]
        )
        f_delta[i] = d
        f_gamma[i] = g
        f_theta[i] = th
        f_vega[i] = v
        f_rho[i] = rh
    return f_delta, f_gamma, f_theta, f_vega, f_rho


def run_benchmark():
    N = 1000000
    s = np.random.rand(N)*100
    k = np.random.rand(N)*100
    t = np.random.rand(N)*2
    sigma = np.random.rand(N)*0.5
    r = np.random.rand(N)*0.1
    q = np.random.rand(N)*0.05
    is_call = np.random.rand(N) > 0.5

    # Warmup
    _vec_price_impl_serial(s[:10], k[:10], t[:10], sigma[:10], r[:10], q[:10], is_call[:10])
    _vec_price_impl_parallel(s[:10], k[:10], t[:10], sigma[:10], r[:10], q[:10], is_call[:10])
    _vec_greeks_impl_serial(s[:10], k[:10], t[:10], sigma[:10], r[:10], q[:10], is_call[:10])
    _vec_greeks_impl_parallel(s[:10], k[:10], t[:10], sigma[:10], r[:10], q[:10], is_call[:10])

    print("Testing Price")
    start = time.time()
    _vec_price_impl_serial(s, k, t, sigma, r, q, is_call)
    print(f"Serial: {time.time()-start:.4f}s")

    start = time.time()
    _vec_price_impl_parallel(s, k, t, sigma, r, q, is_call)
    print(f"Parallel: {time.time()-start:.4f}s")

    print("\nTesting Greeks")
    start = time.time()
    _vec_greeks_impl_serial(s, k, t, sigma, r, q, is_call)
    print(f"Serial: {time.time()-start:.4f}s")

    start = time.time()
    _vec_greeks_impl_parallel(s, k, t, sigma, r, q, is_call)
    print(f"Parallel: {time.time()-start:.4f}s")

if __name__ == "__main__":
    run_benchmark()
