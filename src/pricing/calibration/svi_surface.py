import numpy as np
import structlog
from scipy.optimize import minimize

logger = structlog.get_logger()


class SVISurface:
    """
    SVI (Stochastic Volatility Inspired) surface parameterization.
    """

    @staticmethod
    def raw_svi(
        k: float, a: float, b: float, rho: float, m: float, sigma: float
    ) -> float:
        """
        Raw SVI formula: w(k) = a + b(ρ(k-m) + √[(k-m)² + σ²])
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))

    @staticmethod
    def validate_no_arbitrage(params: tuple[float, ...]) -> bool:
        """Check Gatheral-Jacquier no-arbitrage conditions."""
        a, b, rho, m, sigma = params
        if b < 0:
            return False
        if abs(rho) >= 1:
            return False
        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return False
        if b * (1 + abs(rho)) >= 4:
            return False
        return True

    @staticmethod
    def fit_svi_slice(
        log_strikes: np.ndarray, total_variances: np.ndarray, T: float
    ) -> tuple[float, ...]:
        """Fit SVI to a single maturity slice."""

        def objective(params):
            a, b, rho, m, sigma = params
            if not SVISurface.validate_no_arbitrage(params):
                return 1e10
            model_var = np.array(
                [SVISurface.raw_svi(k, a, b, rho, m, sigma) for k in log_strikes]
            )
            return np.sum((total_variances - model_var) ** 2)

        initial = [np.median(total_variances) * 0.5, 0.1, -0.3, 0.0, 0.1]
        bounds = [(0, 2.0), (0, 1.0), (-0.99, 0.99), (-1.0, 1.0), (0.01, 1.0)]

        result = minimize(objective, initial, bounds=bounds, method="SLSQP")
        return tuple(result.x)

    @staticmethod
    def get_implied_vol(
        S0: float, K: float, T: float, params: tuple[float, ...]
    ) -> float:
        k = np.log(K / S0)
        total_variance = SVISurface.raw_svi(k, *params)
        return np.sqrt(max(total_variance, 1e-6) / T)
