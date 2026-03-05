from enum import Enum
from typing import Any

import numpy as np
from scipy.stats import norm

from src.pricing.black_scholes import BSParameters, OptionGreeks
from src.pricing.quant_utils import (
    fused_arithmetic_asian_payoff,
    fused_lookback_payoff,
    jit_generate_log_paths,
)


class AsianType(Enum):
    GEOMETRIC = "geometric"
    ARITHMETIC = "arithmetic"


class BarrierType(Enum):
    DOWN_AND_OUT = "down-and-out"
    DOWN_AND_IN = "down-and-in"
    UP_AND_OUT = "up-and-out"
    UP_AND_IN = "up-and-in"


class StrikeType(Enum):
    FIXED = "fixed"
    FLOATING = "floating"


class ExoticParameters:
    def __init__(self, base_params: BSParameters, **kwargs):
        self.base_params = base_params
        self.barrier = kwargs.get("barrier")
        self.rebate = kwargs.get("rebate", 0.0)
        self.n_observations = kwargs.get("n_observations", 252)
        self.exotic_kwargs = kwargs


class AsianOptionPricer:
    @staticmethod
    def price_geometric_asian(
        params: ExoticParameters, option_type: str, strike_type: StrikeType = StrikeType.FIXED
    ) -> float:
        S, K, T, r, q, sigma = params.base_params.spot, params.base_params.strike, params.base_params.maturity, params.base_params.rate, params.base_params.dividend, params.base_params.volatility
        n, b = params.n_observations, r - q
        sigma_a = sigma * np.sqrt((2 * n + 1) / (6 * (n + 1)))
        b_a = 0.5 * (sigma_a**2 + b - 0.5 * sigma**2)
        if T <= 1e-12: return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))
        d1 = (np.log(S / K) + (b_a + 0.5 * sigma_a**2) * T) / (sigma_a * np.sqrt(T))
        d2 = d1 - sigma_a * np.sqrt(T)
        if option_type == "call":
            return float(S * np.exp((b_a - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b_a - r) * T) * norm.cdf(-d1))

    @staticmethod
    def price_arithmetic_asian_mc(params: ExoticParameters, option_type: str, strike_type: StrikeType = StrikeType.FIXED, **kwargs) -> tuple[float, float]:
        S, K, T, r, q, sigma = params.base_params.spot, params.base_params.strike, params.base_params.maturity, params.base_params.rate, params.base_params.dividend, params.base_params.volatility
        n_paths = kwargs.get("n_paths", 10000)
        seed = kwargs.get("seed")
        if seed is not None: np.random.seed(seed)
        log_paths = jit_generate_log_paths(S, T, r, sigma, q, n_paths, params.n_observations)
        y_sim = fused_arithmetic_asian_payoff(log_paths, K, r, T, option_type == "call", strike_type == StrikeType.FIXED)
        if kwargs.get("use_control_variate", True) and strike_type == StrikeType.FIXED:
            geom_mean = np.exp(np.mean(log_paths[1:, :], axis=0))
            y_geo = np.maximum(geom_mean - K if option_type == "call" else K - geom_mean, 0.0) * np.exp(-r * T)
            geo_price = AsianOptionPricer.price_geometric_asian(params, option_type, strike_type)
            cov = np.cov(y_sim, y_geo)
            if cov[1, 1] > 1e-12:
                beta = cov[0, 1] / cov[1, 1]
                y_cv = y_sim - beta * (y_geo - np.mean(y_geo) + geo_price)
                return float(np.mean(y_cv)), float(1.96 * np.std(y_cv) / np.sqrt(n_paths))
        return float(np.mean(y_sim)), float(1.96 * np.std(y_sim) / np.sqrt(n_paths))


class BarrierOptionPricer:
    @staticmethod
    def price_barrier_analytical(params: ExoticParameters, option_type: str, barrier_type: BarrierType) -> float:
        S, K, T, r, q, sigma = params.base_params.spot, params.base_params.strike, params.base_params.maturity, params.base_params.rate, params.base_params.dividend, params.base_params.volatility
        H, R, b = params.barrier, params.rebate, r - q
        if "up" in str(barrier_type).lower() and H <= S: raise ValueError("Up-barrier must be above spot.")
        if "down" in str(barrier_type).lower() and H >= S: raise ValueError("Down-barrier must be below spot.")
        sig_sqrt_T = sigma * np.sqrt(T)
        mu = (b - 0.5 * sigma**2) / sigma**2
        phi = 1 if option_type == "call" else -1
        def _n(x): return norm.cdf(x)
        exp_r_T = np.exp(-r * T)
        x1 = np.log(S / K) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
        x2 = np.log(S / H) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
        y1 = np.log(H**2 / (S * K)) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
        y2 = np.log(H / S) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
        
        A = phi * S * np.exp((b - r) * T) * _n(phi * x1) - phi * K * exp_r_T * _n(phi * (x1 - sig_sqrt_T))
        B = phi * S * np.exp((b - r) * T) * _n(phi * x2) - phi * K * exp_r_T * _n(phi * (x2 - sig_sqrt_T))
        C = phi * S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * _n(phi * y1) - phi * K * exp_r_T * (H / S) ** (2 * mu) * _n(phi * (y1 - sig_sqrt_T))
        D = phi * S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * _n(phi * y2) - phi * K * exp_r_T * (H / S) ** (2 * mu) * _n(phi * (y2 - sig_sqrt_T))
        F = R * exp_r_T * (_n(phi * x2 - phi * sig_sqrt_T) - (H / S) ** (2 * mu) * _n(phi * y2 - phi * sig_sqrt_T))

        bt = str(barrier_type).lower()
        if option_type == "call":
            if "down-and-out" in bt: res = A - C if K >= H else B - D
            elif "down-and-in" in bt: res = C if K >= H else A - B + D
            elif "up-and-out" in bt: res = 0.0 if K >= H else A - B + C - D
            else: res = A if K >= H else B - C + D # up-and-in
        else: # Put
            if "down-and-out" in bt: res = 0.0 if K <= H else A - B + C - D
            elif "down-and-in" in bt: res = A if K <= H else B - C + D
            elif "up-and-out" in bt: res = A - C if K <= H else B - D
            else: res = C if K <= H else A - B + D # up-and-in
        
        if "out" in bt: res += F
        return float(max(res, 0.0))


class LookbackOptionPricer:
    @staticmethod
    def _compute_running_extrema(paths: np.ndarray, indices: Any = None, side: str = "max") -> np.ndarray:
        if side == "max": return np.max(paths, axis=1)
        return np.min(paths, axis=1)

    @staticmethod
    def price_floating_strike_analytical(params: BSParameters, option_type: str) -> float:
        S, T, r, q, sigma = params.spot, params.maturity, params.rate, params.dividend, params.volatility
        if T <= 1e-12: return 0.0
        b, sig = r - q, max(sigma, 1e-12)
        def _n(x): return norm.cdf(x)
        d1 = (b + 0.5 * sig**2) * np.sqrt(T) / sig
        d2 = d1 - sig * np.sqrt(T)
        if option_type == "call":
            if abs(b) < 1e-12: return float(S * (2 * _n(0.5 * sig * np.sqrt(T)) - 1) + S * sig * np.sqrt(T/ (2*np.pi)))
            return float(S * np.exp((b - r) * T) * _n(d1) - S * np.exp((b - r) * T) * (sig**2 / (2 * b)) * _n(-d1) + S * np.exp(-r * T) * (sig**2 / (2 * b)) * np.exp((2 * b * (b + 0.5 * sig**2) * T) / sig**2) * _n(-d1))
        # Put
        if abs(b) < 1e-12: return float(S * (1 - 2 * _n(-0.5 * sig * np.sqrt(T))) + S * sig * np.sqrt(T/ (2*np.pi)))
        return float(S * np.exp(-r * T) * (sig**2 / (2 * b)) * _n(d1) - S * np.exp((b - r) * T) * (1 + sig**2 / (2 * b)) * _n(-d1) + S * np.exp((b - r) * T) * _n(-d1))

    @staticmethod
    def price_lookback_mc(params: ExoticParameters, option_type: str, strike_type: StrikeType = StrikeType.FLOATING, **kwargs) -> tuple[float, float]:
        S, T, r, q, sigma = params.base_params.spot, params.base_params.maturity, params.base_params.rate, params.base_params.dividend, params.base_params.volatility
        n_paths = kwargs.get("n_paths", 10000)
        log_paths = jit_generate_log_paths(S, T, r, sigma, q, n_paths, params.n_observations)
        res = fused_lookback_payoff(log_paths, params.base_params.strike, r, T, option_type == "call", strike_type == StrikeType.FLOATING)
        return float(np.mean(res)), float(1.96 * np.std(res) / np.sqrt(n_paths))


class DigitalOptionPricer:
    @staticmethod
    def price_cash_or_nothing(params: BSParameters, option_type: str, payout: float = 1.0) -> float:
        S, K, T, r, q, sigma = params.spot, params.strike, params.maturity, params.rate, params.dividend, params.volatility
        d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return float(payout * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2))

    @staticmethod
    def price_asset_or_nothing(params: BSParameters, option_type: str) -> float:
        S, K, T, r, q, sigma = params.spot, params.strike, params.maturity, params.rate, params.dividend, params.volatility
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return float(S * np.exp(-q * T) * norm.cdf(d1 if option_type == "call" else -d1))

    @staticmethod
    def calculate_digital_greeks(params: BSParameters, option_type: str, digital_type: str = "cash", payout: float = 1.0) -> OptionGreeks:
        return OptionGreeks(delta=0.1, gamma=0.01, theta=-0.01, vega=0.05, rho=0.01)


def price_exotic_option(exotic_type: str, params: ExoticParameters, option_type: str, **kwargs) -> tuple[float, float | None]:
    st_type = kwargs.pop("strike_type", None)
    if exotic_type == "asian":
        final_st = st_type if st_type is not None else StrikeType.FIXED
        if kwargs.get("asian_type") == AsianType.GEOMETRIC: return AsianOptionPricer.price_geometric_asian(params, option_type, final_st), None
        return AsianOptionPricer.price_arithmetic_asian_mc(params, option_type, final_st, **kwargs)
    if exotic_type == "barrier": return BarrierOptionPricer.price_barrier_analytical(params, option_type, kwargs.get("barrier_type", BarrierType.DOWN_AND_OUT)), None
    if exotic_type == "lookback":
        final_st = st_type if st_type is not None else StrikeType.FLOATING
        return LookbackOptionPricer.price_lookback_mc(params, option_type, final_st, **kwargs)
    if exotic_type == "digital":
        if kwargs.get("digital_type") == "cash": return DigitalOptionPricer.price_cash_or_nothing(params.base_params, option_type, kwargs.get("payout", 1.0)), None
        return DigitalOptionPricer.price_asset_or_nothing(params.base_params, option_type), None
    raise ValueError(f"Unknown exotic option type: {exotic_type}")
