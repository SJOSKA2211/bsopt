import cupy as cp
import structlog

logger = structlog.get_logger(__name__)

def black_scholes_cupy(
    S: cp.ndarray,
    K: cp.ndarray,
    T: cp.ndarray,
    sigma: cp.ndarray,
    r: cp.ndarray,
    is_call: bool = True,
) -> cp.ndarray:
    """
    Production-grade vectorized Black-Scholes using CuPy.

    C = S0 * N(d1) - K * e^(-rT) * N(d2)
    d1 = (ln(S0/K) + (r + sigma^2/2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    # Use CuPy's built-in normal CDF approximation
    from cupyx.scipy.special import ndtr

    d1 = (cp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * cp.sqrt(T))
    d2 = d1 - sigma * cp.sqrt(T)

    if is_call:
        return S * ndtr(d1) - K * cp.exp(-r * T) * ndtr(d2)
    else:
        return K * cp.exp(-r * T) * ndtr(-d2) - S * ndtr(-d1)

def runge_kutta_4_gbm(
    S: cp.ndarray, mu: cp.ndarray, sigma: cp.ndarray, dt: float, dW: cp.ndarray
) -> cp.ndarray:
    """
    4th-order Runge-Kutta for Geometric Brownian Motion.
    dSt = mu * St * dt + sigma * St * dWt
    """

    # Deterministic part (mu * S)
    def f(s):
        return mu * s

    k1 = f(S)
    k2 = f(S + 0.5 * k1 * dt)
    k3 = f(S + 0.5 * k2 * dt)
    k4 = f(S + k3 * dt)

    # Update deterministic part + stochastic diffusion
    S_new = S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) + sigma * S * dW
    return cp.maximum(S_new, 0.0)
