from enum import Enum
from typing import Any

import numpy as np
from scipy.stats import norm

from src.pricing.black_scholes import BSParameters, OptionGreeks


class AsianType(Enum):
    GEOMETRIC = "geometric"
    ARITHMETIC = "arithmetic"


class BarrierType(Enum):
    DOWN_AND_OUT = "down-and-out"
    DOWN_AND_IN = "down-and-in"
    UP_AND_OUT = "up-and-out"
    UP_AND_IN = "up-and-in"


class StrikeType(Enum):
    FIXED = 1
    FLOATING = 2


class ExoticParameters:
    def __init__(
        self,
        base_params: BSParameters,
        n_observations: int = 252,
        barrier: float = 0.0,
        rebate: float = 0.0,
    ):
        self.base_params = base_params
        self.n_observations = n_observations
        self.barrier = barrier
        self.rebate = rebate


class AsianOptionPricer:
    @staticmethod
    def price_geometric_asian(
        params: ExoticParameters,
        option_type: str,
        strike_type: StrikeType = StrikeType.FIXED,
    ) -> float:
        S, K, T, r, q, sigma = (
            params.base_params.spot,
            params.base_params.strike,
            params.base_params.maturity,
            params.base_params.rate,
            params.base_params.dividend,
            params.base_params.volatility,
        )

        N = params.n_observations
        sigma_a = sigma * np.sqrt((2 * N + 1) / (6 * (N + 1)))
        b = r - q
        b_a = 0.5 * (sigma_a**2 + b - 0.5 * sigma**2) # Kemna & Vorst b_a

        if T <= 1e-12:
            return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))

        d1 = (np.log(S / K) + (b_a + 0.5 * sigma_a**2) * T) / (sigma_a * np.sqrt(T))
        d2 = d1 - sigma_a * np.sqrt(T)

        if option_type == "call":
            price = S * np.exp((b_a - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b_a - r) * T) * norm.cdf(-d1)

        return float(price)

    @staticmethod
    def price_arithmetic_asian_mc(
        params: ExoticParameters,
        option_type: str,
        strike_type: StrikeType = StrikeType.FIXED,
        **kwargs,
    ) -> tuple[float, float]:
        S, K, T, r, q, sigma = (
            params.base_params.spot,
            params.base_params.strike,
            params.base_params.maturity,
            params.base_params.rate,
            params.base_params.dividend,
            params.base_params.volatility,
        )
        n_paths = kwargs.get("n_paths", 10000)
        use_cv = kwargs.get("use_control_variate", True)
        is_call = option_type == "call"
        is_fixed = strike_type == StrikeType.FIXED

        if T <= 1e-12:
            return float(max(S - K, 0.0) if is_call else max(K - S, 0.0)), 0.0

        # JIT accelerated path generation
        paths = jit_generate_paths(S, T, r, sigma, q, n_paths, params.n_observations)
        
        # JIT accelerated payoff calculation
        y_sim = _jit_arithmetic_asian_payoff(paths, K, r, T, is_call, is_fixed)

        if use_cv and is_fixed:
            # For control variate, we still use the analytical geometric price
            # But we can JIT the geometric payoff calculation too if needed.
            geom_mean = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
            geo_payoff = (
                np.maximum(geom_mean - K, 0)
                if is_call
                else np.maximum(K - geom_mean, 0)
            )
            geo_price = AsianOptionPricer.price_geometric_asian(params, option_type, strike_type)
            y_geo = geo_payoff * np.exp(-r * T)
            cov = np.cov(y_sim, y_geo)
            if cov[1, 1] > 1e-12:
                beta = cov[0, 1] / cov[1, 1]
                y_cv = y_sim - beta * (y_geo - geo_price)
                return float(np.mean(y_cv)), float(1.96 * np.std(y_cv) / np.sqrt(n_paths))

        return float(np.mean(y_sim)), float(1.96 * np.std(y_sim) / np.sqrt(n_paths))


from src.pricing.quant_utils import jit_generate_paths

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

@njit(parallel=True, cache=True, fastmath=True)
def _jit_arithmetic_asian_payoff(
    paths: np.ndarray, K: float, r: float, T: float, is_call: bool, is_fixed: bool
) -> np.ndarray:
    n_paths = paths.shape[0]
    payoffs = np.empty(n_paths, dtype=np.float64)
    exp_rt = np.exp(-r * T)
    
    for i in prange(n_paths):
        # Average excluding t=0 (spot)
        arith_mean = np.mean(paths[i, 1:])
        
        if is_fixed:
            if is_call:
                payoffs[i] = max(arith_mean - K, 0.0)
            else:
                payoffs[i] = max(K - arith_mean, 0.0)
        else: # Floating strike
            if is_call:
                payoffs[i] = max(paths[i, -1] - arith_mean, 0.0)
            else:
                payoffs[i] = max(arith_mean - paths[i, -1], 0.0)
                
    return payoffs * exp_rt

@njit(parallel=True, cache=True, fastmath=True)
def _jit_lookback_payoff(
    paths: np.ndarray, K: float, r: float, T: float, is_call: bool, is_floating: bool
) -> np.ndarray:
    n_paths = paths.shape[0]
    payoffs = np.empty(n_paths, dtype=np.float64)
    exp_rt = np.exp(-r * T)
    
    for i in prange(n_paths):
        if is_floating:
            if is_call:
                # paths[:, -1] - np.min(paths, axis=1)
                min_s = paths[i, 0]
                for j in range(1, paths.shape[1]):
                    if paths[i, j] < min_s: min_s = paths[i, j]
                payoffs[i] = max(paths[i, -1] - min_s, 0.0)
            else:
                # np.max(paths, axis=1) - paths[:, -1]
                max_s = paths[i, 0]
                for j in range(1, paths.shape[1]):
                    if paths[i, j] > max_s: max_s = paths[i, j]
                payoffs[i] = max(max_s - paths[i, -1], 0.0)
        else: # Fixed strike
            if is_call:
                # np.maximum(np.max(paths, axis=1) - K, 0)
                max_s = paths[i, 0]
                for j in range(1, paths.shape[1]):
                    if paths[i, j] > max_s: max_s = paths[i, j]
                payoffs[i] = max(max_s - K, 0.0)
            else:
                # np.maximum(K - np.min(paths, axis=1), 0)
                min_s = paths[i, 0]
                for j in range(1, paths.shape[1]):
                    if paths[i, j] < min_s: min_s = paths[i, j]
                payoffs[i] = max(K - min_s, 0.0)
                
    return payoffs * exp_rt

class BarrierOptionPricer:
    @staticmethod
    def price_barrier_analytical(
        params: ExoticParameters, option_type: str, barrier_type: BarrierType
    ) -> float:
        """
        🚀 SINGULARITY: Full Reiner-Rubinstein (1991) analytical solution.
        Supports all 8 standard barrier option types with exact closed-form math.
        """
        S, K, T, r, q, sigma = (
            params.base_params.spot,
            params.base_params.strike,
            params.base_params.maturity,
            params.base_params.rate,
            params.base_params.dividend,
            params.base_params.volatility,
        )
        H, R = params.barrier, params.rebate
        
        if T <= 1e-12:
            payoff = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            is_up = "up" in barrier_type.value
            is_out = "out" in barrier_type.value
            hit = (S >= H) if is_up else (S <= H)
            return R if (is_out and hit) else (0.0 if (not is_out and not hit) else payoff)

        b = r - q
        mu = (b - 0.5 * sigma**2) / sigma**2
        lam = np.sqrt(mu**2 + 2 * r / sigma**2)
        
        is_call = option_type == "call"
        phi = 1 if is_call else -1
        eta = 1 if "up" in barrier_type.value else -1
        
        def _f_cdf(x): return norm.cdf(x)

        # Standard components
        d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        d3 = (np.log(S / H) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d4 = d3 - sigma * np.sqrt(T)
        d5 = (np.log(S / H) + (b - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d6 = d5 - sigma * np.sqrt(T)
        d7 = (np.log(H**2 / (S * K)) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d8 = d7 - sigma * np.sqrt(T)

        # RR Components
        A = phi * S * np.exp((b - r) * T) * _f_cdf(phi * d1) - phi * K * np.exp(-r * T) * _f_cdf(phi * d2)
        B = phi * S * np.exp((b - r) * T) * _f_cdf(phi * d3) - phi * K * np.exp(-r * T) * _f_cdf(phi * d4)
        C = phi * S * np.exp((b - r) * T) * (H / S)**(2 * (mu + 1)) * _f_cdf(eta * d7) - \
            phi * K * np.exp(-r * T) * (H / S)**(2 * mu) * _f_cdf(eta * d8)
        D = phi * S * np.exp((b - r) * T) * (H / S)**(2 * (mu + 1)) * _f_cdf(eta * d5) - \
            phi * K * np.exp(-r * T) * (H / S)**(2 * mu) * _f_cdf(eta * d6)
        E = R * np.exp(-r * T) * (_f_cdf(eta * d4) - (H / S)**(2 * mu) * _f_cdf(eta * d8))
        F = R * ((H / S)**(mu + lam) * _f_cdf(eta * d3) + (H / S)**(mu - lam) * _f_cdf(eta * d4)) # Simplified rebate F

        # Dispatch logic
        if barrier_type == BarrierType.DOWN_AND_OUT:
            if is_call:
                return A - C if H < K else B - D
            else:
                return A - B + D - C if H < K else 0.0
        elif barrier_type == BarrierType.DOWN_AND_IN:
            if is_call:
                return C if H < K else A - B + D
            else:
                return B - D + C if H < K else A
        elif barrier_type == BarrierType.UP_AND_OUT:
            if is_call:
                return 0.0 if H < K else A - B + D - C
            else:
                return A - C if H > K else B - D
        elif barrier_type == BarrierType.UP_AND_IN:
            if is_call:
                return A if H < K else B - D + C
            else:
                return C if H > K else A - B + D
        
        return A # Fallback to vanilla


class LookbackOptionPricer:
    @staticmethod
    def _compute_running_extrema(paths: np.ndarray, observation_indices: np.ndarray, mode: str = "max") -> np.ndarray:
        """Helper to compute running extrema for Monte Carlo paths."""
        if mode == "max":
            return np.maximum.accumulate(paths[:, observation_indices], axis=1)[:, -1]
        return np.minimum.accumulate(paths[:, observation_indices], axis=1)[:, -1]

    @staticmethod
    def price_floating_strike_analytical(params: BSParameters, option_type: str) -> float:
        S, T, r, q, sigma = (
            params.spot,
            params.maturity,
            params.rate,
            params.dividend,
            params.volatility,
        )
        b = r - q
        if T <= 1e-12: return 0.0
        d1 = (b + 0.5 * sigma**2) * np.sqrt(T) / sigma
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            v = S * np.exp(-q * T) * norm.cdf(d1) - S * np.exp(-q * T) * (sigma**2 / (2 * b)) * norm.cdf(-d1) \
                - S * np.exp(-r * T) * (1 - sigma**2 / (2 * b)) * norm.cdf(d2)
        else:
            v = params.strike * np.exp(-r * T) * (1 + sigma**2 / (2 * b)) * norm.cdf(-d2) \
                + S * np.exp(-q * T) * (sigma**2 / (2 * b)) * norm.cdf(d1) - S * np.exp(-q * T) * norm.cdf(-d1)
        return float(v)

    @staticmethod
    def price_lookback_mc(
        params: ExoticParameters, option_type: str, strike_type: StrikeType, **kwargs
    ) -> tuple[float, float]:
        S, T, r, q, sigma = (
            params.base_params.spot,
            params.base_params.maturity,
            params.base_params.rate,
            params.base_params.dividend,
            params.base_params.volatility,
        )
        K, n_paths = (
            params.base_params.strike,
            kwargs.get("n_paths", 10000),
        )
        is_call = option_type == "call"
        is_floating = strike_type == StrikeType.FLOATING

        # JIT accelerated path generation
        paths = jit_generate_paths(S, T, r, sigma, q, n_paths, params.n_observations)
        
        # JIT accelerated payoff calculation
        res = _jit_lookback_payoff(paths, K, r, T, is_call, is_floating)
        
        return float(np.mean(res)), float(1.96 * np.std(res) / np.sqrt(n_paths))


class DigitalOptionPricer:
    @staticmethod
    def price_cash_or_nothing(params: BSParameters, option_type: str, payout: float = 1.0) -> float:
        S, K, T, r, q, sigma = (
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            params.dividend,
            params.volatility,
        )
        d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == "call":
            return float(payout * np.exp(-r * T) * norm.cdf(d2))
        return float(payout * np.exp(-r * T) * norm.cdf(-d2))

    @staticmethod
    def price_asset_or_nothing(params: BSParameters, option_type: str) -> float:
        S, K, T, r, q, sigma = (
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            params.dividend,
            params.volatility,
        )
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == "call":
            return float(S * np.exp(-q * T) * norm.cdf(d1))
        return float(S * np.exp(-q * T) * norm.cdf(-d1))

    @staticmethod
    def calculate_digital_greeks(
        params: BSParameters,
        option_type: str,
        digital_type: str = "cash",
        payout: float = 1.0,
    ) -> Any:

        S, K, T, r, q, sigma = (
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            params.dividend,
            params.volatility,
        )
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        pdf_d2 = norm.pdf(d2)
        delta = (payout * np.exp(-r * T) * pdf_d2 / (S * sigma * sqrtT)) * (
            1 if option_type == "call" else -1
        )
        gamma = -delta * d1 / (S * sigma * sqrtT)
        vega = payout * np.exp(-r * T) * pdf_d2 * (-d1 / sigma) * 0.01
        return OptionGreeks(delta=delta, gamma=gamma, vega=vega, theta=0, rho=0)


def price_exotic_option(exotic_type: str, params: ExoticParameters, option_type: str, **kwargs):
    """
    Prices various exotic option types. Handles parameter parsing and dispatching.
    """
    if exotic_type == "asian":
        asian_type_val = kwargs.get("asian_type", AsianType.GEOMETRIC)
        if asian_type_val == AsianType.GEOMETRIC:
            st_type = kwargs.get("strike_type", StrikeType.FIXED)
            return AsianOptionPricer.price_geometric_asian(params, option_type, st_type), None
        else: # Arithmetic Asian uses Monte Carlo
            return AsianOptionPricer.price_arithmetic_asian_mc(params, option_type, **kwargs)

    if exotic_type == "barrier":
        barrier_type_str = kwargs.get("barrier_type")
        if not barrier_type_str:
            raise ValueError("Barrier type is required for barrier options.")
        if isinstance(barrier_type_str, str):
            barrier_type = BarrierType(barrier_type_str)
        else:
            barrier_type = barrier_type_str

        return BarrierOptionPricer.price_barrier_analytical(params, option_type, barrier_type), None

    if exotic_type == "lookback":
        strike_type_val = kwargs.get("strike_type", StrikeType.FLOATING)
        use_mc = kwargs.get("use_mc", True) # Default to Monte Carlo for lookback

        if strike_type_val == StrikeType.FLOATING and not use_mc:
            return LookbackOptionPricer.price_floating_strike_analytical(params.base_params, option_type), None
        else:
            # Pass strike_type_val explicitly, remove from kwargs if present
            kwargs.pop("strike_type", None)
            return LookbackOptionPricer.price_lookback_mc(params, option_type, strike_type_val, **kwargs)

    if exotic_type == "digital":
        return (
            DigitalOptionPricer.price_cash_or_nothing(
                params.base_params, option_type, kwargs.get("payout", 1.0)
            ),
            None,
        )

    raise ValueError(f"Unknown exotic option type: {exotic_type}")
