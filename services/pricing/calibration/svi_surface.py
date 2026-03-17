from typing import Any, cast

import numpy as np
import structlog
from scipy.optimize import minimize

try:
    import bsopt_core

    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

from services.shared.math_utils import njit_engine

logger = structlog.get_logger()


@njit_engine(cache=True, fastmath=True)
def _raw_svi_kernel(
    k: np.ndarray[Any, np.dtype[np.float64]],
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Vectorized JIT SVI formula."""
    return cast(
        np.ndarray[Any, np.dtype[np.float64]],
        a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2)),
    )


class SVISurface:
    """
    SVI (Stochastic Volatility Inspired) surface parameterization.
    OPTIMIZED: Vectorized JIT kernels for calibration.
    """

    @staticmethod
    def raw_svi(
        k: float | np.ndarray[Any, np.dtype[np.float64]],
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Public interface for SVI total variance calculation."""
        if np.isscalar(k):
            k_val = float(cast(float, k))
            return float(a + b * (rho * (k_val - m) + np.sqrt((k_val - m) ** 2 + sigma**2)))

        k_arr = np.asanyarray(k, dtype=np.float64)
        return _raw_svi_kernel(k_arr, a, b, rho, m, sigma)

    @staticmethod
    def fit_svi_slice(
        log_strikes: np.ndarray[Any, np.dtype[np.float64]],
        total_variances: np.ndarray[Any, np.dtype[np.float64]],
        T: float,
    ) -> tuple[float, ...]:
        """Fit SVI using vectorized JIT objective or Rust Core."""
        initial = np.array([np.median(total_variances) * 0.5, 0.1, -0.3, 0.0, 0.1])

        if _CORE_AVAILABLE:
            try:
                # Convert variances back to vols for Rust calibrator which fits in vol space (more robust)
                vols = np.sqrt(np.maximum(total_variances / T, 1e-9))
                weights = np.ones_like(vols)

                res = bsopt_core.calibrate_svi_rust(
                    log_strikes.astype(np.float64),
                    vols.astype(np.float64),
                    weights.astype(np.float64),
                    float(T),
                    list(initial),
                )
                return tuple(res)
            except Exception as e:
                logger.warning("rust_svi_calibration_failed_falling_back", error=str(e))

        def objective(params: np.ndarray[Any, np.dtype[np.float64]]) -> float:
            a, b, rho, m, sigma = params
            # Fast validation
            if b < 0 or abs(rho) >= 1 or (a + b * sigma * np.sqrt(1 - rho**2)) < 0:
                return 1e10

            # Vectorized kernel call (O(1) from Python)
            model_var = _raw_svi_kernel(log_strikes, a, b, rho, m, sigma)
            return float(np.sum((total_variances - model_var) ** 2))

        bounds = [(0, 2.0), (0, 1.0), (-0.99, 0.99), (-1.0, 1.0), (0.01, 1.0)]

        result = minimize(objective, initial, bounds=bounds, method="L-BFGS-B")
        return tuple(result.x)

    @staticmethod
    def get_implied_vol(S0: float, K: float, T: float, params: tuple[float, ...]) -> float:
        k = float(np.log(K / S0))
        # params: (a, b, rho, m, sigma)
        total_variance = SVISurface.raw_svi(
            k, params[0], params[1], params[2], params[3], params[4]
        )
        return float(np.sqrt(max(float(cast(float, total_variance)), 1e-6) / T))
