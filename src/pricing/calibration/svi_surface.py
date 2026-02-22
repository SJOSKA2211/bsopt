import numpy as np
import structlog
from numba import njit
from scipy.optimize import minimize

logger = structlog.get_logger()

@njit(cache=True, fastmath=True)
def _raw_svi_kernel(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Vectorized JIT SVI formula."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))

class SVISurface:
    """
    SVI (Stochastic Volatility Inspired) surface parameterization.
    OPTIMIZED: Vectorized JIT kernels for calibration.
    """
    @staticmethod
    def fit_svi_slice(
        log_strikes: np.ndarray, total_variances: np.ndarray, T: float
    ) -> tuple[float, ...]:
        """Fit SVI using vectorized JIT objective."""

        def objective(params):
            a, b, rho, m, sigma = params
            # Fast validation
            if b < 0 or abs(rho) >= 1 or (a + b * sigma * np.sqrt(1 - rho**2)) < 0:
                return 1e10
            
            # Vectorized kernel call (O(1) from Python)
            model_var = _raw_svi_kernel(log_strikes, a, b, rho, m, sigma)
            return np.sum((total_variances - model_var) ** 2)

        initial = [np.median(total_variances) * 0.5, 0.1, -0.3, 0.0, 0.1]
        bounds = [(0, 2.0), (0, 1.0), (-0.99, 0.99), (-1.0, 1.0), (0.01, 1.0)]

        result = minimize(objective, initial, bounds=bounds, method="L-BFGS-B")
        return tuple(result.x)

    @staticmethod
    def get_implied_vol(
        S0: float, K: float, T: float, params: tuple[float, ...]
    ) -> float:
        k = np.log(K / S0)
        # Using _raw_svi_kernel as raw_svi logic seems to be what was intended by raw_svi call in original code,
        # but original code had SVISurface.raw_svi which wasn't defined in the file provided.
        # Assuming _raw_svi_kernel is the intended implementation.
        total_variance = _raw_svi_kernel(np.array([k]), *params)[0]
        return np.sqrt(max(total_variance, 1e-6) / T)
