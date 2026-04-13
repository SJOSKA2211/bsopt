import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def black_scholes_numpy(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    is_call: bool = True,
) -> np.ndarray:
    """
    Vectorized Black-Scholes using NumPy.
    """
    from scipy.special import ndtr

    safe_t = np.maximum(T, 1e-9)
    safe_sigma = np.maximum(sigma, 1e-9)
    sqrt_t = np.sqrt(safe_t)

    d1 = (np.log(S / K) + (r + 0.5 * safe_sigma**2) * safe_t) / (safe_sigma * sqrt_t)
    d2 = d1 - safe_sigma * sqrt_t

    if is_call:
        return S * ndtr(d1) - K * np.exp(-r * safe_t) * ndtr(d2)
    else:
        return K * np.exp(-r * safe_t) * ndtr(-d2) - S * ndtr(-d1)


def runge_kutta_4_gbm(
    S: np.ndarray, mu: np.ndarray, sigma: np.ndarray, dt: float, dW: np.ndarray
) -> np.ndarray:
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
    return np.maximum(S_new, 0.0)