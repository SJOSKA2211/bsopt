from typing import Any, cast

import numpy as np
import structlog
from scipy.spatial.distance import cdist

try:
    import Manifold_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = structlog.get_logger(__name__)


def _gaussian_kernel_matrix(
    x: np.ndarray[Any, np.dtype[np.float64]], y: np.ndarray[Any, np.dtype[np.float64]], sigma: float
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Optimized Gaussian RBF kernel calculation using NumPy/SciPy."""
    gamma = 1.0 / (2.0 * sigma**2)
    # cdist computes squared Euclidean distance efficiently
    dist_sq = cdist(x, y, "sqeuclidean")
    return cast(np.ndarray[Any, np.dtype[np.float64]], np.exp(-gamma * dist_sq))


def calculate_mmd(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]],
    sigma: float = 1.0,
) -> float:
    """
    Maximum Mean Discrepancy (MMD) multivariate distance.
    Uses Rust src.shared for sub-microsecond calculation if available.
    """
    from src.shared.observability import MMD_DRIFT_SCORE

    if CORE_AVAILABLE:
        try:
            # We use Any to avoid mypy complaining about dynamic module
            val = float(
                cast(Any, Manifold_core).calculate_mmd(
                    x.astype(np.float64), y.astype(np.float64), float(sigma)
                )
            )
            MMD_DRIFT_SCORE.set(val)
            return val
        except Exception as e:
            logger.warning("rust_mmd_calculation_failed_falling_back", error=str(e))

    n = x.shape[0]
    m = y.shape[0]

    k_xx = _gaussian_kernel_matrix(x, x, sigma)
    k_yy = _gaussian_kernel_matrix(y, y, sigma)
    k_xy = _gaussian_kernel_matrix(x, y, sigma)

    # MMD^2 = 1/n^2 * sum(K_xx) + 1/m^2 * sum(K_yy) - 2/(nm) * sum(K_xy)
    # Subtracting diagonal from K_xx and K_yy for unbiased estimator
    sum_xx = (np.sum(k_xx) - n) / (n * (n - 1)) if n > 1 else 0.0
    sum_yy = (np.sum(k_yy) - m) / (m * (m - 1)) if m > 1 else 0.0
    sum_xy = float(np.mean(k_xy))

    mmd_sq = sum_xx + sum_yy - 2 * sum_xy
    val = float(np.sqrt(max(mmd_sq, 0.0)))
    MMD_DRIFT_SCORE.set(val)
    return val


class MultivariateDriftDetector:
    """
    High-dimensional drift detector using MMD.
    Sensitive to correlations and manifold collapse.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def detect_drift(
        self,
        baseline_x: np.ndarray[Any, np.dtype[np.float64]],
        current_x: np.ndarray[Any, np.dtype[np.float64]],
    ) -> tuple[bool, float]:
        """Detect drift between two multivariate samples."""
        mmd_val = calculate_mmd(baseline_x, current_x, sigma=1.0)
        is_drifted = bool(mmd_val > self.threshold)

        if is_drifted:
            logger.warning("multivariate_drift_detected", mmd=mmd_val, threshold=self.threshold)

        return is_drifted, mmd_val
