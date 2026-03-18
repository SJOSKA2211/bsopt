import numpy as np
import cupy as cp
from typing import Union, cast
try:
    from pyo3_runtime import import_module
except ImportError:
    import_module = None

# Attempt to load the Rust src.shared manifold binding
try:
    if import_module:
        equaflow_core = import_module("equaflow_core")
    else:
        equaflow_core = None
except ImportError:
    equaflow_core = None

def gpu_black_scholes(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    V: Union[float, np.ndarray],
    is_call: bool = True
) -> np.ndarray:
    """
    Institutional-grade GPU-accelerated Black-Scholes using CuPy.
    C = S0 * N(d1) - K * e^(-rT) * N(d2)
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        R: Risk-free rate
        V: Volatility
        is_call: True for Call, False for Put
        
    Returns:
        np.ndarray: Vectorized option prices
    """
    S_gpu = cp.asarray(S, dtype=cp.float64)
    K_gpu = cp.asarray(K, dtype=cp.float64)
    T_gpu = cp.asarray(T, dtype=cp.float64)
    R_gpu = cp.asarray(R, dtype=cp.float64)
    V_gpu = cp.asarray(V, dtype=cp.float64)

    # Standard d1, d2 formulation (Institutional Grade)
    d1 = (cp.log(S_gpu / K_gpu) + (R_gpu + 0.5 * V_gpu**2) * T_gpu) / (V_gpu * cp.sqrt(T_gpu))
    d2 = d1 - V_gpu * cp.sqrt(T_gpu)

    def norm_cdf(x: cp.ndarray) -> cp.ndarray:
        return 0.5 * (1 + cp.erf(x / cp.sqrt(2.0)))

    if is_call:
        price = S_gpu * norm_cdf(d1) - K_gpu * cp.exp(-R_gpu * T_gpu) * norm_cdf(d2)
    else:
        price = K_gpu * cp.exp(-R_gpu * T_gpu) * norm_cdf(-d2) - S_gpu * norm_cdf(-d1)

    return cast(np.ndarray, cp.asnumpy(price))

def runge_kutta_4(
    S0: Union[float, np.ndarray],
    mu: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    T: float,
    dt: float,
    steps: int
) -> np.ndarray:
    """
    4th-order Runge-Kutta (RK4) solver for Geometric Brownian Motion (GBM):
    dSt = mu*St*dt + sigma*St*dWt
    
    This solver handles the deterministic drift part via RK4 and integrates
    the stochastic diffusion component.
    
    Args:
        S0: Initial spot price(s)
        mu: Drift coefficient
        sigma: Diffusion coefficient
        T: Total time
        dt: Time step
        steps: Total steps
        
    Returns:
        np.ndarray: Final spot price vector
    """
    S = cp.asarray(S0, dtype=cp.float64)
    mu_gpu = cp.asarray(mu, dtype=cp.float64)
    sigma_gpu = cp.asarray(sigma, dtype=cp.float64)
    
    for _ in range(steps):
        # Deterministic Drift (RK4): f(t, S) = mu * S
        k1 = mu_gpu * S
        k2 = mu_gpu * (S + 0.5 * dt * k1)
        k3 = mu_gpu * (S + 0.5 * dt * k2)
        k4 = mu_gpu * (S + dt * k3)
        
        drift = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Stochastic Diffusion (Euler-Maruyama integration)
        dW = cp.random.normal(0, cp.sqrt(dt), S.shape).astype(cp.float64)
        diffusion = sigma_gpu * S * dW
        
        S = S + drift + diffusion
        
    return cast(np.ndarray, cp.asnumpy(S))

def hybrid_compute_bs(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    V: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Hybrid Compute Manifold: Prefers Rust (equaflow_core) for low-latency CPU 
    vectorization, and falls back to CuPy for massively parallel GPU execution.
    """
    if equaflow_core and hasattr(equaflow_core, "black_scholes_vectorized"):
        return cast(np.ndarray, equaflow_core.black_scholes_vectorized(S, K, T, R, V))
    else:
        return gpu_black_scholes(S, K, T, R, V)


def mmap_accelerated_runge_kutta_4(
    mmap_file_path: str,
    mu: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    T: float,
    dt: float,
    steps: int
) -> np.ndarray:
    """
    Zero-copy data ingestion pipeline into GPU solvers.
    Uses PyO3 mmap reader from equaflow_core to pull ticks and computes RK4 on GPU.
    """
    if not equaflow_core or not hasattr(equaflow_core, "mmap_parse_ticks"):
        raise RuntimeError("equaflow_core is missing mmap_parse_ticks native hook.")

    # 1. Zero-copy to Numpy
    S0_numpy = equaflow_core.mmap_parse_ticks(mmap_file_path)

    # 2. Host-to-Device transfer
    # 3. GPU execution of ODE
    return runge_kutta_4(S0_numpy, mu, sigma, T, dt, steps)

