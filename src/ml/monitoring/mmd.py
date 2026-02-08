import numpy as np
import structlog
from scipy.spatial.distance import cdist

logger = structlog.get_logger(__name__)

def _gaussian_kernel_matrix(x, y, sigma):
    """Optimized Gaussian RBF kernel calculation using NumPy/SciPy."""
    gamma = 1.0 / (2.0 * sigma**2)
    # cdist computes squared Euclidean distance efficiently
    dist_sq = cdist(x, y, 'sqeuclidean')
    return np.exp(-gamma * dist_sq)

def calculate_mmd(x, y, sigma=1.0):
    """
    Maximum Mean Discrepancy (MMD) multivariate distance.
    Measures the distance between two distributions in RKHS.
    """
    n = x.shape[0]
    m = y.shape[0]
    
    k_xx = _gaussian_kernel_matrix(x, x, sigma)
    k_yy = _gaussian_kernel_matrix(y, y, sigma)
    k_xy = _gaussian_kernel_matrix(x, y, sigma)
    
    # MMD^2 = 1/n^2 * sum(K_xx) + 1/m^2 * sum(K_yy) - 2/(nm) * sum(K_xy)
    # Subtracting diagonal from K_xx and K_yy for unbiased estimator
    sum_xx = (np.sum(k_xx) - n) / (n * (n - 1)) if n > 1 else 0.0
    sum_yy = (np.sum(k_yy) - m) / (m * (m - 1)) if m > 1 else 0.0
    sum_xy = np.mean(k_xy)
    
    mmd_sq = sum_xx + sum_yy - 2 * sum_xy
    return np.sqrt(max(mmd_sq, 0.0))

class MultivariateDriftDetector:
    """
    High-dimensional drift detector using MMD.
    Sensitive to correlations and manifold collapse.
    """
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def detect_drift(self, baseline_x: np.ndarray, current_x: np.ndarray) -> tuple[bool, float]:
        """Detect drift between two multivariate samples."""
        mmd_val = calculate_mmd(baseline_x, current_x, sigma=1.0)
        is_drifted = mmd_val > self.threshold
        
        if is_drifted:
            logger.warning("multivariate_drift_detected", mmd=mmd_val, threshold=self.threshold)
            
        return is_drifted, mmd_val
