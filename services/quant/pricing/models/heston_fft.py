from typing import Any, cast

import numpy as np
import structlog
from numba import complex128, float64, prange

from services.pricing.models import HestonParams
from services.shared.math_utils import njit_engine

logger = structlog.get_logger()


@njit_engine(
    complex128[:, :](
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
        float64[:],
    ),
    cache=True,
    fastmath=True,
    parallel=True,
)
def _heston_cf_kernel(
    v: np.ndarray[Any, np.dtype[np.float64]],
    k: np.ndarray[Any, np.dtype[np.float64]],
    alpha: np.ndarray[Any, np.dtype[np.float64]],
    T: np.ndarray[Any, np.dtype[np.float64]],
    r: np.ndarray[Any, np.dtype[np.float64]],
    v0: np.ndarray[Any, np.dtype[np.float64]],
    kappa: np.ndarray[Any, np.dtype[np.float64]],
    theta: np.ndarray[Any, np.dtype[np.float64]],
    sigma: np.ndarray[Any, np.dtype[np.float64]],
    rho: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """
    Fused Numba kernel for the Heston Characteristic Function.
    Avoids NumPy broadcasting overhead and uses machine-code complex math.
    """
    n_v = v.shape[0]
    n_batch = k.shape[0]
    res = np.zeros((n_v, n_batch), dtype=np.complex128)

    for i in prange(n_v):
        for j in range(n_batch):
            u_v = v[i] - (alpha[j] + 1) * 1j
            xi = kappa[j] - sigma[j] * rho[j] * u_v * 1j
            d = np.sqrt(xi**2 + sigma[j] ** 2 * (u_v**2 + 1j * u_v))
            g = (xi + d) / (xi - d)

            exp_dT = np.exp(d * T[j])
            G = (1.0 - g * exp_dT) / (1.0 - g)

            A = (kappa[j] * theta[j] / sigma[j] ** 2) * ((xi + d) * T[j] - 2.0 * np.log(G))
            B = (v0[j] / sigma[j] ** 2) * (xi + d) * (1.0 - exp_dT) / (1.0 - g * exp_dT)

            phi = np.exp(A + B)

            num = np.exp(-1j * v[i] * k[j]) * phi
            den = alpha[j] ** 2 + alpha[j] - v[i] ** 2 + 1j * (2 * alpha[j] + 1) * v[i]
            res[i, j] = num / den

    return res


try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = structlog.get_logger()
...


def _heston_integrand_vectorized(
    v: np.ndarray[Any, np.dtype[np.float64]],
    k: np.ndarray[Any, np.dtype[np.float64]],
    alpha: np.ndarray[Any, np.dtype[np.float64]],
    T: np.ndarray[Any, np.dtype[np.float64]],
    r: np.ndarray[Any, np.dtype[np.float64]],
    v0: np.ndarray[Any, np.dtype[np.float64]],
    kappa: np.ndarray[Any, np.dtype[np.float64]],
    theta: np.ndarray[Any, np.dtype[np.float64]],
    sigma: np.ndarray[Any, np.dtype[np.float64]],
    rho: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Delegates to the JIT-compiled or Rust kernel.
    """
    if CORE_AVAILABLE:
        try:
            cf = bsopt_core.heston_characteristic_function(
                v, k, alpha, T, r, v0, kappa, theta, sigma, rho
            )
            return np.real(cf)
        except Exception as e:
            logger.warning("rust_heston_cf_failed_falling_back", error=str(e))

    return cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.real(_heston_cf_kernel(v, k, alpha, T, r, v0, kappa, theta, sigma, rho)),
    )


def batch_heston_price_jit(
    spots: np.ndarray[Any, np.dtype[np.float64]],
    strikes: np.ndarray[Any, np.dtype[np.float64]],
    maturities: np.ndarray[Any, np.dtype[np.float64]],
    rates: np.ndarray[Any, np.dtype[np.float64]],
    v0s: np.ndarray[Any, np.dtype[np.float64]],
    kappas: np.ndarray[Any, np.dtype[np.float64]],
    thetas: np.ndarray[Any, np.dtype[np.float64]],
    sigmas: np.ndarray[Any, np.dtype[np.float64]],
    rhos: np.ndarray[Any, np.dtype[np.float64]],
    is_calls: np.ndarray[Any, np.dtype[np.bool_]],
    out: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    """
    Advanced vectorized batch pricing. ZERO Python loops.
    """
    k = np.log(strikes / spots)
    # alpha = 1.5 for call, -2.5 for put usually, but let's stick to 1.5 and use parity
    alpha = np.full_like(k, 1.5)

    upper_bound = 250.0
    n_steps = 2000
    h = upper_bound / n_steps
    v = np.linspace(0, upper_bound, n_steps + 1).astype(np.float64)

    f_v = _heston_integrand_vectorized(
        v, k, alpha, maturities, rates, v0s, kappas, thetas, sigmas, rhos
    )

    weights = np.ones(n_steps + 1, dtype=np.float64)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    weights_col = cast(np.ndarray[Any, np.dtype[np.float64]], weights.reshape(-1, 1))

    integrals = (h / 3.0) * np.sum(f_v * weights_col, axis=0)

    price_vals = (np.exp(-alpha * k) / np.pi) * integrals
    discounted_prices = np.exp(-rates * maturities) * spots * price_vals

    # For puts, use put-call parity: P = C - S + K*exp(-rT)
    put_prices = discounted_prices - spots + strikes * np.exp(-rates * maturities)

    final_prices = np.where(is_calls, discounted_prices, put_prices)

    intrinsics = np.where(
        is_calls, np.maximum(spots - strikes, 0.0), np.maximum(strikes - spots, 0.0)
    )
    out[:] = np.maximum(final_prices, intrinsics)


class HestonModelFFT:
    """
    Heston Model using vectorized FFT and Simpson integration.
    """

    MAX_INTEGRATION_BOUND = 250.0
    MIN_PRICE = 1e-10

    def __init__(
        self,
        params: HestonParams | None = None,
        r: float | None = None,
        T: float | None = None,
    ) -> None:
        self.params = params
        self.r = r
        self.T = T

    def price_call(self, S0: float, K: float) -> float:
        """Single option pricing for backward compatibility."""
        if self.params is None or self.r is None or self.T is None:
            raise ValueError("Model not fully initialized")

        out = np.zeros(1, dtype=np.float64)
        batch_heston_price_jit(
            np.array([S0], dtype=np.float64),
            np.array([K], dtype=np.float64),
            np.array([self.T], dtype=np.float64),
            np.array([self.r], dtype=np.float64),
            np.array([self.params.v0], dtype=np.float64),
            np.array([self.params.kappa], dtype=np.float64),
            np.array([self.params.theta], dtype=np.float64),
            np.array([self.params.sigma], dtype=np.float64),
            np.array([self.params.rho], dtype=np.float64),
            np.array([True], dtype=bool),
            out,
        )
        return float(out[0])

    def price_put(self, S0: float, K: float) -> float:
        """Single option pricing for backward compatibility."""
        if self.params is None or self.r is None or self.T is None:
            raise ValueError("Model not fully initialized")

        out = np.zeros(1, dtype=np.float64)
        batch_heston_price_jit(
            np.array([S0], dtype=np.float64),
            np.array([K], dtype=np.float64),
            np.array([self.T], dtype=np.float64),
            np.array([self.r], dtype=np.float64),
            np.array([self.params.v0], dtype=np.float64),
            np.array([self.params.kappa], dtype=np.float64),
            np.array([self.params.theta], dtype=np.float64),
            np.array([self.params.sigma], dtype=np.float64),
            np.array([self.params.rho], dtype=np.float64),
            np.array([False], dtype=bool),
            out,
        )
        return float(out[0])

    def price_surface_fft(
        self, S0: float, K_min: float, K_max: float, N: int = 1024
    ) -> dict[float, float]:
        """
        O(N log N) multi-strike pricing using vectorized FFT.
        """
        if self.params is None or self.r is None or self.T is None:
            raise ValueError("Model must be initialized with params, r, and T for surface pricing.")

        p = self.params
        alpha = 1.5
        eta = 0.25
        lambda_grid = (2 * np.pi) / (N * eta)
        b = (N * lambda_grid) / 2

        v = np.arange(N) * eta
        k_grid = -b + np.arange(N) * lambda_grid

        u = v - (alpha + 1) * 1j
        xi = p.kappa - p.sigma * p.rho * u * 1j
        d = np.sqrt(xi**2 + p.sigma**2 * (u**2 + 1j * u))
        g = (xi + d) / (xi - d)

        dT = d * self.T
        exp_dT = np.exp(np.clip(dT.real, -100, 100) + 1j * dT.imag)

        g_exp_dT = g * exp_dT
        G = (1.0 - g_exp_dT) / np.maximum(1e-18, (1.0 - g))

        A = (p.kappa * p.theta / p.sigma**2) * (
            (xi + d) * self.T - 2.0 * np.log(np.maximum(1e-18, G))
        )
        B = (p.v0 / p.sigma**2) * (xi + d) * (1.0 - exp_dT) / np.maximum(1e-18, 1.0 - g_exp_dT)

        phi = np.exp(A + B)
        psi = (np.exp(-self.r * self.T) * phi) / (
            alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
        )

        w = (eta / 3.0) * (3 + (-1) ** (np.arange(N) + 1))
        w[0] = eta / 3.0

        phi_values = np.exp(1j * v * b) * psi * w
        x_fft = np.fft.fft(phi_values)

        prices = np.real(np.exp(-alpha * k_grid) / np.pi * x_fft) * S0
        strikes = S0 * np.exp(k_grid)

        mask = (strikes >= K_min) & (strikes <= K_max)
        filtered_strikes = strikes[mask]
        filtered_prices = np.maximum(prices[mask], self.MIN_PRICE)

        return dict(zip(filtered_strikes.tolist(), filtered_prices.tolist(), strict=False))
