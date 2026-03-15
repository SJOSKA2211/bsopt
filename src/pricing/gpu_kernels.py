import cupy as cp
from pyo3_runtime import import_module

# Attempt to load the Rust core manifold binding
try:
    equaflow_core = import_module("equaflow_core")
except ImportError:
    equaflow_core = None

def gpu_black_scholes(S, K, T, R, V, is_call=True):
    """
    Institutional-grade GPU-accelerated Black-Scholes using CuPy.
    C = S0 * N(d1) - K * e^(-rT) * N(d2)
    """
    S = cp.asarray(S)
    K = cp.asarray(K)
    T = cp.asarray(T)
    R = cp.asarray(R)
    V = cp.asarray(V)

    d1 = (cp.log(S / K) + (R + 0.5 * V**2) * T) / (V * cp.sqrt(T))
    d2 = d1 - V * cp.sqrt(T)

    def norm_cdf(x):
        return 0.5 * (1 + cp.erf(x / cp.sqrt(2)))

    if is_call:
        price = S * norm_cdf(d1) - K * cp.exp(-R * T) * norm_cdf(d2)
    else:
        price = K * cp.exp(-R * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    return cp.asnumpy(price)

def runge_kutta_4(S0, mu, sigma, T, dt, steps):
    """
    4th-order Runge-Kutta solver for Geometric Brownian Motion:
    dSt = mu*St*dt + sigma*St*dWt
    Approximation for the ODE part: f(t, S) = mu * S
    (Note: dWt is the stochastic part, RK4 usually applies to the deterministic drift)
    """
    S = cp.asarray(S0)
    mu = cp.asarray(mu)
    sigma = cp.asarray(sigma)
    
    # We use RK4 for the drift component mu*S*dt
    # For GBM, we often use the exact solution for the stochastic part, 
    # but the directive implies solving it as an ODE/SDE.
    # Here we implement the RK4 step for the drift:
    
    for _ in range(steps):
        k1 = mu * S
        k2 = mu * (S + 0.5 * dt * k1)
        k3 = mu * (S + 0.5 * dt * k2)
        k4 = mu * (S + dt * k3)
        
        drift = (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Stochastic component (Euler-Maruyama step integrated)
        dW = cp.random.normal(0, cp.sqrt(dt), S.shape)
        diffusion = sigma * S * dW
        
        S = S + drift + diffusion
        
    return cp.asnumpy(S)

def hybrid_compute_bs(S, K, T, R, V):
    """
    Hybrid Execution: Prefers Rust for CPU parsing/logic, CuPy for heavy GPU math.
    """
    if equaflow_core:
        # If we have many small tasks, Rust vectors are faster
        return equaflow_core.black_scholes_vectorized(S, K, T, R, V)
    else:
        # Fallback/Primary for massive parallelization
        return gpu_black_scholes(S, K, T, R, V)
