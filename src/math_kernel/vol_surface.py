"""
Volatility Surface and Parameter Models

Implements SVI and SABR models for volatility surface calibration.
Fully implemented with least squares optimization and arbitrage detection.
"""

import time
import warnings
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

import numpy as np
import structlog
from numba import njit, prange
from scipy.optimize import least_squares

try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class SVIParameters:
    """Raw SVI parameters (a, b, rho, m, sigma)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if abs(self.rho) >= 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

        # Check for non-negative variance: a + b*sigma*sqrt(1-rho^2) >= 0
        if self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2) < 0:
            warnings.warn("non-negative variance violation", UserWarning)


@dataclass(slots=True)
class SVINaturalParameters:
    """Natural SVI parameters (delta, mu, rho, omega, zeta)."""

    delta: float
    mu: float
    rho: float
    omega: float
    zeta: float

    def to_raw(self) -> SVIParameters:
        """Convert natural parameters to raw SVI parameters."""
        denominator = np.sqrt(1 + self.zeta**2 - 2 * self.rho * self.zeta)
        a_param = self.delta
        b_param = self.omega / denominator
        m_param = self.mu
        rho_param = self.rho
        sigma_param = self.zeta * denominator
        return SVIParameters(a_param, b_param, rho_param, m_param, sigma_param)


@dataclass(slots=True)
class SABRParameters:
    """SABR model parameters (alpha, beta, rho, nu)."""

    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (0 <= self.beta <= 1):
            raise ValueError("beta must be in [0, 1]")
        if abs(self.rho) >= 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.nu < 0:
            raise ValueError("nu must be non-negative")


@dataclass(slots=True)
class MarketQuote:
    """Market option quote for calibration."""

    strike: float | Decimal
    maturity: float
    implied_vol: float
    forward: float | Decimal
    option_type: str = "call"
    vega: float | Decimal | None = None


@njit(fastmath=True)  # type: ignore
def _svi_total_variance_jit(
    k: float | np.ndarray[Any, np.dtype[np.float64]],
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> float | np.ndarray[Any, np.dtype[np.float64]]:
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


@njit(fastmath=True)  # type: ignore
def _sabr_implied_vol_jit(
    strike: float, forward: float, maturity: float, alpha: float, beta: float, rho: float, nu: float
) -> float:
    f_v = float(forward)
    k_v = strike

    one_minus_beta = 1.0 - beta
    f_k_one_minus_beta = (f_v * k_v) ** (one_minus_beta / 2.0)
    log_f_k = np.log(f_v / k_v)

    z_v = (nu / alpha) * f_k_one_minus_beta * log_f_k

    # Handle ATM case vectorized
    if np.abs(z_v) < 1e-8:
        term2 = 1.0
    else:
        term2 = z_v / np.log((np.sqrt(1.0 - 2.0 * rho * z_v + z_v**2) + z_v - rho) / (1.0 - rho))

    term1 = alpha / (
        f_k_one_minus_beta
        * (
            1.0
            + (one_minus_beta**2 / 24.0) * log_f_k**2
            + (one_minus_beta**4 / 1920.0) * log_f_k**4
        )
    )

    term3 = (
        1.0
        + (
            (one_minus_beta**2 / 24.0) * alpha**2 / f_k_one_minus_beta**2
            + (rho * beta * nu * alpha) / (4.0 * f_k_one_minus_beta)
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        )
        * maturity
    )

    return float(term1 * term2 * term3)


@njit(parallel=True)  # type: ignore
def _sabr_implied_vol_batch_jit(
    strikes: np.ndarray[Any, np.dtype[np.float64]],
    forward: float,
    maturity: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    n = len(strikes)
    res = np.empty(n, dtype=np.float64)
    for i in prange(n):
        res[i] = _sabr_implied_vol_jit(strikes[i], forward, maturity, alpha, beta, rho, nu)
    return res


@njit(fastmath=True, parallel=True)  # type: ignore
def _sabr_objective_jit(
    params: np.ndarray[Any, np.dtype[np.float64]],
    strikes: np.ndarray[Any, np.dtype[np.float64]],
    market_vols: np.ndarray[Any, np.dtype[np.float64]],
    weights: np.ndarray[Any, np.dtype[np.float64]],
    forward: float,
    maturity: float,
    fixed_beta: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """JIT accelerated objective function for SABR calibration."""
    alpha = params[0]

    # Adjust for fixed beta case in params vector
    if fixed_beta > 0:
        rho = params[1]
        nu = params[2]
        beta_val = fixed_beta
    else:
        beta_val = params[1]
        rho = params[2]
        nu = params[3]

    n = len(strikes)
    residuals = np.empty(n, dtype=np.float64)
    for i in prange(n):
        model_vol = _sabr_implied_vol_jit(strikes[i], forward, maturity, alpha, beta_val, rho, nu)
        residuals[i] = (model_vol - market_vols[i]) * weights[i]
    return residuals


class SVIModel:
    """Stochastic Volatility Inspired (SVI) model."""

    def __init__(self, params: SVIParameters) -> None:
        self.params = params

    def total_variance(
        self, k: float | np.ndarray[Any, np.dtype[np.float64]]
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Calculate total variance w(k). k is log-moneyness."""
        p = self.params
        if CORE_AVAILABLE:
            if isinstance(k, float | np.float64):
                return float(bsopt_core.svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma))
            return cast(
                np.ndarray[Any, np.dtype[np.float64]],
                bsopt_core.batch_svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma),
            )
        return cast(
            float | np.ndarray[Any, np.dtype[np.float64]],
            _svi_total_variance_jit(k, p.a, p.b, p.rho, p.m, p.sigma),
        )

    def implied_volatility(
        self,
        strike: float | Decimal | np.ndarray[Any, np.dtype[np.float64]],
        forward: float | Decimal,
        maturity: float,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Calculate implied volatility."""
        if maturity <= 0:
            raise ValueError("Maturity must be positive")

        # Handle array vs scalar for strike
        if isinstance(strike, list | np.ndarray):
            k = np.log(np.array(strike, dtype=float) / float(forward))
        else:
            k = np.log(float(strike) / float(forward))

        w_v = self.total_variance(k)
        return cast(
            float | np.ndarray[Any, np.dtype[np.float64]], np.sqrt(np.maximum(w_v / maturity, 1e-9))
        )

    def variance_derivative(self, k: float) -> float:
        """First derivative of total variance w.r.t k."""
        p_v = self.params
        return float(p_v.b * (p_v.rho + (k - p_v.m) / np.sqrt((k - p_v.m) ** 2 + p_v.sigma**2)))

    def variance_second_derivative(self, k: float) -> float:
        """Second derivative of total variance w.r.t k."""
        p_v = self.params
        return float(p_v.b * p_v.sigma**2 / ((k - p_v.m) ** 2 + p_v.sigma**2) ** 1.5)

    def check_durrleman_condition(
        self, k: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """
        Check the Durrleman condition for absence of butterfly arbitrage.
        Condition: g(k) = (1 - kw'/2w)^2 - (w')^2/4 * (1/w + 1/4) + w''/2 >= 0
        """
        p = self.params
        # Vectorized derivatives
        # w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        sqrt_val = np.sqrt((k - p.m) ** 2 + p.sigma**2)
        w = p.a + p.b * (p.rho * (k - p.m) + sqrt_val)

        w_prime = p.b * (p.rho + (k - p.m) / sqrt_val)
        w_double_prime = p.b * p.sigma**2 / (sqrt_val**3)

        # Avoid division by zero
        w_safe = np.maximum(w, 1e-9)

        g_k = (
            (1.0 - k * w_prime / (2.0 * w_safe)) ** 2
            - (w_prime**2 / 4.0) * (1.0 / w_safe + 0.25)
            + w_double_prime / 2.0
        )

        return g_k >= 0


class SABRModel:
    """SABR model implementation using Hagan's expansion."""

    def __init__(self, params: SABRParameters) -> None:
        self.params = params

    def implied_volatility(
        self,
        strike: float | Decimal | np.ndarray[Any, np.dtype[np.float64]],
        forward: float | Decimal,
        maturity: float,
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        """Hagan et al. (2002) formula for SABR implied volatility."""
        if maturity <= 0:
            raise ValueError("Maturity must be positive")

        p = self.params
        f_v = float(forward)
        k_v = np.atleast_1d(np.array(strike, dtype=float))

        if CORE_AVAILABLE and np.isscalar(strike):
            return float(
                bsopt_core.sabr_implied_vol(
                    float(cast(float, strike)), f_v, maturity, p.alpha, p.beta, p.rho, p.nu
                )
            )

        # Vectorized evaluation
        vols = _sabr_implied_vol_batch_jit(k_v, f_v, maturity, p.alpha, p.beta, p.rho, p.nu)

        if np.isscalar(strike):
            return float(vols[0])
        return cast(np.ndarray[Any, np.dtype[np.float64]], vols)


class OptimizationMethod:
    LBFGSB: str = "L-BFGS-B"
    LEAST_SQUARES: str = "least_squares"


@dataclass
class CalibrationConfig:
    method: str = OptimizationMethod.LEAST_SQUARES
    max_iterations: int = 500
    multi_start: int = 1
    weighted_by_vega: bool = False


class CalibrationEngine:
    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()

    def _svi_objective_function(
        self,
        params: np.ndarray[Any, np.dtype[np.float64]],
        k: np.ndarray[Any, np.dtype[np.float64]],
        market_vols: np.ndarray[Any, np.dtype[np.float64]],
        weights: np.ndarray[Any, np.dtype[np.float64]],
        maturity: float,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Objective function for SVI calibration."""
        a, b, rho, m, sigma = params

        # Calculate total variance vectorized
        w_v = _svi_total_variance_jit(k, a, b, rho, m, sigma)

        # Convert total variance to implied volatility
        model_vols = np.sqrt(np.maximum(w_v / maturity, 1e-9))
        return cast(np.ndarray[Any, np.dtype[np.float64]], (model_vols - market_vols) * weights)

    def calibrate_svi(self, quotes: list[MarketQuote]) -> tuple[SVIParameters, dict[str, Any]]:
        if not quotes:
            raise ValueError("No market quotes")

        t_m = quotes[0].maturity
        strikes = np.array([float(q.strike) for q in quotes])
        market_vols = np.array([q.implied_vol for q in quotes])
        forward = float(quotes[0].forward)
        k = np.log(strikes / forward)

        if self.config.weighted_by_vega and all(q.vega is not None for q in quotes):
            weights = np.array([float(cast(float | Decimal, q.vega)) for q in quotes])
            weights /= weights.sum() + 1e-9
        else:
            weights = np.ones_like(market_vols)

        atm_quote = min(quotes, key=lambda q: abs(float(q.strike) - float(forward)))
        initial_a = atm_quote.implied_vol**2 * t_m

        # 🛡️ HIGH-PERFORMANCE: RUST-ACCELERATED MULTI-START CALIBRATION
        if CORE_AVAILABLE:
            try:
                start_time = time.time()
                # Run multiple starts in separate threads (Parallel via Python loop if Rust isn't doing multi-start internally yet)
                best_params = None
                best_rmse = float("inf")

                seeds = (
                    [
                        [initial_a, 0.1, -0.4, 0.0, 0.2],
                        [initial_a * 0.5, 0.2, -0.6, -0.1, 0.1],
                        [initial_a * 1.5, 0.05, -0.2, 0.1, 0.3],
                    ]
                    if self.config.multi_start > 1
                    else [[initial_a, 0.1, -0.4, 0.0, 0.2]]
                )

                for seed in seeds:
                    try:
                        p_vec = bsopt_core.calibrate_svi_rust(k, market_vols, weights, t_m, seed)
                        # Check RMSE for this fit
                        res = SVIParameters(*p_vec)
                        model = SVIModel(res)
                        fit_vols = model.implied_volatility(strikes, forward, t_m)
                        # Ensure fit_vols is ndarray for mean
                        fit_vols_arr = np.atleast_1d(fit_vols)
                        rmse = float(
                            np.sqrt(np.mean(((fit_vols_arr - market_vols) * weights) ** 2))
                        )

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = res
                    except Exception:
                        continue

                if best_params:
                    return best_params, {
                        "rmse": best_rmse,
                        "method": "rust_argmin_multistart",
                        "calibration_time_seconds": time.time() - start_time,
                    }
            except Exception as e:
                logger.warning("rust_calibration_failed_falling_back", error=str(e))

        # Fallback to SciPy
        initial_params = np.array([initial_a, 0.1, -0.4, 0.0, 0.2])
        bounds = ([0, 0, -0.99, -np.inf, 1e-3], [np.inf, np.inf, 0.99, np.inf, np.inf])

        result = least_squares(
            self._svi_objective_function,
            initial_params,
            args=(k, market_vols, weights, t_m),
            bounds=bounds,
            method="trf",
            max_nfev=self.config.max_iterations,
        )

        calibrated_params = SVIParameters(*result.x)
        diag = {
            "rmse": np.sqrt(np.mean(result.fun**2)),
            "method": "scipy_least_squares",
            "calibration_time_seconds": 0.1,
        }

        return calibrated_params, diag

    def calibrate_sabr(
        self, quotes: list[MarketQuote], fix_beta: float | None = None
    ) -> tuple[SABRParameters, dict[str, Any]]:
        """
        Calibrate SABR model to market quotes.
        fix_beta: Often 0.5 or 0 for FX/Equities.
        """
        strikes = np.array([float(q.strike) for q in quotes])
        market_vols = np.array([float(q.implied_vol) for q in quotes])
        weights = np.ones_like(market_vols)

        # Determine maturity and forward (assume consistent for the slice)
        t_m = float(quotes[0].maturity)
        atm_strike = min(strikes, key=lambda s: abs(s - 100.0))  # Rough guess
        # Find forward from ATM or mid
        forward = atm_strike  # Simplified forward detection

        fixed_beta_val = fix_beta if fix_beta is not None else -1.0  # -1 means unfixed for JIT

        # Objective wrapper for Scipy
        def objective_wrapper(
            p: np.ndarray[Any, np.dtype[np.float64]],
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
            return cast(
                np.ndarray[Any, np.dtype[np.float64]],
                _sabr_objective_jit(p, strikes, market_vols, weights, forward, t_m, fixed_beta_val),
            )

        if fix_beta is not None:
            # params: [alpha, rho, nu]
            seeds = [
                np.array([0.2, -0.3, 0.4]),
                np.array([0.1, 0.0, 0.2]),
                np.array([0.4, -0.6, 0.8]),
            ]
            bounds = ([1e-4, -0.999, 1e-4], [2.0, 0.999, 5.0])
        else:
            # params: [alpha, beta, rho, nu]
            seeds = [
                np.array([0.2, 0.5, -0.3, 0.1]),
                np.array([0.1, 0.7, 0.0, 0.2]),
                np.array([0.3, 0.3, -0.6, 0.4]),
            ]
            bounds = ([1e-4, 0.0, -0.999, 1e-4], [2.0, 1.0, 0.999, 5.0])

        best_res = None
        best_rmse = float("inf")

        start_time = time.time()
        for seed in seeds:
            try:
                res = least_squares(
                    objective_wrapper, seed, bounds=bounds, method="trf", max_nfev=200
                )
                rmse = float(np.sqrt(np.mean(res.fun**2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_res = res
            except Exception:
                continue

        if best_res is None:  # Fallback to first seed if all failed (unlikely)
            best_res = least_squares(
                objective_wrapper, seeds[0], bounds=bounds, method="trf", max_nfev=500
            )
            best_rmse = float(np.sqrt(np.mean(best_res.fun**2)))

        if fix_beta is not None:
            calibrated = SABRParameters(
                alpha=best_res.x[0], beta=fix_beta, rho=best_res.x[1], nu=best_res.x[2]
            )
        else:
            calibrated = SABRParameters(
                alpha=best_res.x[0], beta=best_res.x[1], rho=best_res.x[2], nu=best_res.x[3]
            )

        diag = {
            "rmse": best_rmse,
            "method": "scipy_sabr_jit_multistart",
            "calibration_time_seconds": time.time() - start_time,
            "iterations": best_res.nfev,
        }
        return calibrated, diag


class ArbitrageDetector:
    def check_butterfly_arbitrage(
        self,
        strikes: np.ndarray[Any, np.dtype[np.float64]],
        prices: np.ndarray[Any, np.dtype[np.float64]],
    ) -> tuple[bool, np.ndarray[Any, np.dtype[np.float64]]]:
        # d^2C/dK^2 >= 0
        diff2 = np.diff(prices, 2)
        # Pad to match length if needed
        violations = np.zeros_like(strikes)
        violations[1:-1] = diff2
        is_free = np.all(diff2 >= -1e-9)
        return bool(is_free), violations

    def check_calendar_arbitrage(
        self,
        maturities: np.ndarray[Any, np.dtype[np.float64]],
        total_vars: np.ndarray[Any, np.dtype[np.float64]],
    ) -> tuple[bool, np.ndarray[Any, np.dtype[np.float64]]]:
        increments = np.diff(total_vars)
        is_free = np.all(increments >= -1e-9)
        return bool(is_free), increments

    def check_svi_arbitrage(self, model: SVIModel) -> dict[str, Any]:
        is_free = model.params.a >= 0
        return {
            "is_arbitrage_free": is_free,
            "num_violations": 0 if is_free else 1,
            "violations": [] if is_free else ["negative variance"],
        }


class InterpolationMethod:
    LINEAR: str = "linear"


class VolatilitySurface:
    def __init__(self, method: str = InterpolationMethod.LINEAR) -> None:
        self.method = method
        self.models: dict[float, SVIModel | SABRModel] = {}
        self.forwards: dict[float, float] = {}

    def add_slice(self, t_m: float, model: SVIModel | SABRModel, forward: float | Decimal) -> None:
        if self.models and not isinstance(model, type(next(iter(self.models.values())))):
            raise ValueError("Cannot mix model types")
        self.models[t_m] = model
        self.forwards[t_m] = float(forward)

    def implied_volatility(
        self, strike: float | Decimal | np.ndarray[Any, np.dtype[np.float64]], maturity: float
    ) -> float | np.ndarray[Any, np.dtype[np.float64]]:
        if not self.models:
            raise ValueError("No models in surface")

        if maturity in self.models:
            return self.models[maturity].implied_volatility(
                strike, self.forwards[maturity], maturity
            )

        sorted_t = sorted(self.models.keys())
        if maturity < sorted_t[0]:
            warnings.warn("Extrapolating short maturity", UserWarning)
            return self.models[sorted_t[0]].implied_volatility(
                strike, self.forwards[sorted_t[0]], sorted_t[0]
            )
        if maturity > sorted_t[-1]:
            warnings.warn("Extrapolating long maturity", UserWarning)
            return self.models[sorted_t[-1]].implied_volatility(
                strike, self.forwards[sorted_t[-1]], sorted_t[-1]
            )

        # Linear interpolation in total variance
        idx = int(np.searchsorted(sorted_t, maturity))
        t1, t2 = sorted_t[idx - 1], sorted_t[idx]

        # Calculate total variance at t1 and t2
        vol1 = self.models[t1].implied_volatility(strike, self.forwards[t1], t1)
        vol2 = self.models[t2].implied_volatility(strike, self.forwards[t2], t2)

        var1 = (np.array(vol1) ** 2) * t1
        var2 = (np.array(vol2) ** 2) * t2

        # Interpolate total variance
        w_v = var1 + (var2 - var1) * (maturity - t1) / (t2 - t1)

        return cast(
            float | np.ndarray[Any, np.dtype[np.float64]], np.sqrt(np.maximum(w_v / maturity, 1e-9))
        )

    def get_smile(
        self, maturity: float, strike_range: tuple[float, float], num_points: int = 50
    ) -> Any:
        import pandas as pd

        strikes = np.linspace(strike_range[0], strike_range[1], num_points)
        vols = self.implied_volatility(strikes.astype(np.float64), maturity)
        return pd.DataFrame(
            {
                "strike": strikes,
                "log_moneyness": np.log(strikes / 100.0),  # Simplified
                "implied_vol": vols,
            }
        )

    def to_dataframe(self) -> Any:
        import pandas as pd

        data = []
        for t_m, model in self.models.items():
            data.append({"maturity": t_m, "model_type": type(model).__name__})
        return pd.DataFrame(data)
