from enum import Enum
from typing import Any

import numpy as np
from numba import njit

try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

from src.math_kernel.black_scholes import BSParameters, OptionGreeks
from src.math_kernel.quant_utils import (
    fast_normal_cdf_v2,
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

@njit(fastmath=True)
def _price_geometric_asian_jit(S, K, T, r, q, sigma, n, is_call):
    """JIT Accelerated Geometric Asian Pricing."""
    if T <= 1e-12:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    b = r - q
    sigma_a = sigma * np.sqrt((2.0 * n + 1.0) / (6.0 * (n + 1.0)))
    b_a = 0.5 * (sigma_a**2 + b - 0.5 * sigma**2)

    sqrt_T = np.sqrt(T)
    vol_sqrt_T = sigma_a * sqrt_T
    d1 = (np.log(S / K) + (b_a + 0.5 * sigma_a**2) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T

    exp_rT = np.exp(-r * T)
    exp_ba_r_T = np.exp((b_a - r) * T)

    if is_call:
        return S * exp_ba_r_T * fast_normal_cdf_v2(d1) - K * exp_rT * fast_normal_cdf_v2(d2)
    return K * exp_rT * fast_normal_cdf_v2(-d2) - S * exp_ba_r_T * fast_normal_cdf_v2(-d1)

@njit(fastmath=True)
def _price_barrier_analytical_jit(S, K, T, r, q, sigma, H, R, barrier_type_idx, is_call):
    """
    JIT Accelerated Barrier Option Pricing.
    barrier_type_idx: 0: down-and-out, 1: down-and-in, 2: up-and-out, 3: up-and-in
    """
    b = r - q
    sig_sqrt_T = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    phi = 1 if is_call else -1

    exp_r_T = np.exp(-r * T)
    exp_b_r_T = np.exp((b - r) * T)

    x1 = np.log(S / K) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
    x2 = np.log(S / H) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
    y1 = np.log(H**2 / (S * K)) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
    y2 = np.log(H / S) / sig_sqrt_T + (mu + 1) * sig_sqrt_T

    def _n(x):
        return fast_normal_cdf_v2(x)

    A = phi * S * exp_b_r_T * _n(phi * x1) - phi * K * exp_r_T * _n(phi * (x1 - sig_sqrt_T))
    B = phi * S * exp_b_r_T * _n(phi * x2) - phi * K * exp_r_T * _n(phi * (x2 - sig_sqrt_T))
    C = phi * S * exp_b_r_T * (H / S) ** (2 * (mu + 1)) * _n(phi * y1) - phi * K * exp_r_T * (
        H / S
    ) ** (2 * mu) * _n(phi * (y1 - sig_sqrt_T))
    D = phi * S * exp_b_r_T * (H / S) ** (2 * (mu + 1)) * _n(phi * y2) - phi * K * exp_r_T * (
        H / S
    ) ** (2 * mu) * _n(phi * (y2 - sig_sqrt_T))
    F = (
        R
        * exp_r_T
        * (_n(phi * x2 - phi * sig_sqrt_T) - (H / S) ** (2 * mu) * _n(phi * y2 - phi * sig_sqrt_T))
    )

    res = 0.0
    if is_call:
        if barrier_type_idx == 0:  # down-and-out
            res = A - C if K >= H else B - D
        elif barrier_type_idx == 1:  # down-and-in
            res = C if K >= H else A - B + D
        elif barrier_type_idx == 2:  # up-and-out
            res = 0.0 if K >= H else A - B + C - D
        else:  # up-and-in
            res = A if K >= H else B - C + D
    else:  # Put
        if barrier_type_idx == 0:  # down-and-out
            res = 0.0 if K <= H else A - B + C - D
        elif barrier_type_idx == 1:  # down-and-in
            res = A if K <= H else B - C + D
        elif barrier_type_idx == 2:  # up-and-out
            res = A - C if K <= H else B - D
        else:  # up-and-in
            res = C if K <= H else A - B + D

    if barrier_type_idx % 2 == 0:  # "out" types are 0 and 2
        res += F
    return max(res, 0.0)

class AsianOptionPricer:
    @staticmethod
    def price_geometric_asian(
        params: ExoticParameters, option_type: str, strike_type: StrikeType = StrikeType.FIXED
    ) -> float:
        if CORE_AVAILABLE:
            try:
                return float(
                    bsopt_core.geometric_asian_price(
                        params.base_params.spot,
                        params.base_params.strike,
                        params.base_params.maturity,
                        params.base_params.rate,
                        params.base_params.dividend,
                        params.base_params.volatility,
                        float(params.n_observations),
                        option_type.lower() == "call",
                    )
                )
            except Exception:
                pass

        return float(
            _price_geometric_asian_jit(
                params.base_params.spot,
                params.base_params.strike,
                params.base_params.maturity,
                params.base_params.rate,
                params.base_params.dividend,
                params.base_params.volatility,
                params.n_observations,
                option_type.lower() == "call",
            )
        )

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
        seed = kwargs.get("seed")
        if seed is not None:
            np.random.seed(seed)
        log_paths = jit_generate_log_paths(S, T, r, sigma, q, n_paths, params.n_observations)
        y_sim = fused_arithmetic_asian_payoff(
            log_paths, K, r, T, option_type == "call", strike_type == StrikeType.FIXED
        )
        if kwargs.get("use_control_variate", True) and strike_type == StrikeType.FIXED:
            geom_mean = np.exp(np.mean(log_paths[1:, :], axis=0))
            y_geo = np.maximum(
                geom_mean - K if option_type == "call" else K - geom_mean, 0.0
            ) * np.exp(-r * T)
            geo_price = AsianOptionPricer.price_geometric_asian(params, option_type, strike_type)
            cov = np.cov(y_sim, y_geo)
            if cov[1, 1] > 1e-12:
                beta = cov[0, 1] / cov[1, 1]
                y_cv = y_sim - beta * (y_geo - np.mean(y_geo) + geo_price)
                return float(np.mean(y_cv)), float(1.96 * np.std(y_cv) / np.sqrt(n_paths))
        return float(np.mean(y_sim)), float(1.96 * np.std(y_sim) / np.sqrt(n_paths))

class BarrierOptionPricer:
    @staticmethod
    def price_barrier_analytical(
        params: ExoticParameters, option_type: str, barrier_type: BarrierType
    ) -> float:
        bt_str = str(barrier_type).lower()
        if "down-and-out" in bt_str:
            bt_idx = 0
        elif "down-and-in" in bt_str:
            bt_idx = 1
        elif "up-and-out" in bt_str:
            bt_idx = 2
        else:
            bt_idx = 3

        if CORE_AVAILABLE:
            try:
                return float(
                    bsopt_core.barrier_option_price(
                        params.base_params.spot,
                        params.base_params.strike,
                        params.base_params.maturity,
                        params.base_params.rate,
                        params.base_params.dividend,
                        params.base_params.volatility,
                        float(params.barrier),
                        float(params.rebate),
                        bt_idx,
                        option_type.lower() == "call",
                    )
                )
            except Exception:
                pass

        return float(
            _price_barrier_analytical_jit(
                params.base_params.spot,
                params.base_params.strike,
                params.base_params.maturity,
                params.base_params.rate,
                params.base_params.dividend,
                params.base_params.volatility,
                params.barrier,
                params.rebate,
                bt_idx,
                option_type.lower() == "call",
            )
        )

@njit(fastmath=True)
def _price_lookback_floating_strike_jit(S, T, r, q, sigma, is_call):
    """JIT Accelerated Lookback Floating Strike Pricing."""
    if T <= 1e-12:
        return 0.0
    b = r - q
    sig = max(sigma, 1e-12)
    sig_sqrt_T = sig * np.sqrt(T)
    d1 = (b + 0.5 * sig**2) * np.sqrt(T) / sig

    def _n(x):
        return fast_normal_cdf_v2(x)

    exp_ba_r_T = np.exp((b - r) * T)
    exp_rT = np.exp(-r * T)

    if is_call:
        if abs(b) < 1e-12:
            return S * (2.0 * _n(0.5 * sig_sqrt_T) - 1.0) + S * sig * np.sqrt(T / (2.0 * np.pi))
        term1 = S * exp_ba_r_T * _n(d1)
        term2 = S * exp_ba_r_T * (sig**2 / (2.0 * b)) * _n(-d1)
        term3 = (
            S
            * exp_rT
            * (sig**2 / (2.0 * b))
            * np.exp((2.0 * b * (b + 0.5 * sig**2) * T) / sig**2)
            * _n(-d1)
        )
        return term1 - term2 + term3
    # Put
    if abs(b) < 1e-12:
        return S * (1.0 - 2.0 * _n(-0.5 * sig_sqrt_T)) + S * sig * np.sqrt(T / (2.0 * np.pi))
    term1 = S * exp_rT * (sig**2 / (2.0 * b)) * _n(d1)
    term2 = S * exp_ba_r_T * (1.0 + sig**2 / (2.0 * b)) * _n(-d1)
    term3 = S * exp_ba_r_T * _n(-d1)
    return term1 - term2 + term3

@njit(fastmath=True)
def _price_digital_cash_or_nothing_jit(S, K, T, r, q, sigma, payout, is_call):
    """JIT Accelerated Digital Cash-or-Nothing Pricing."""
    sqrt_T = np.sqrt(T)
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return payout * np.exp(-r * T) * fast_normal_cdf_v2(d2 if is_call else -d2)

@njit(fastmath=True)
def _price_digital_asset_or_nothing_jit(S, K, T, r, q, sigma, is_call):
    """JIT Accelerated Digital Asset-or-Nothing Pricing."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return S * np.exp(-q * T) * fast_normal_cdf_v2(d1 if is_call else -d1)

class LookbackOptionPricer:
    @staticmethod
    def _compute_running_extrema(
        paths: np.ndarray, indices: Any = None, side: str = "max"
    ) -> np.ndarray:
        if side == "max":
            return np.max(paths, axis=1)
        return np.min(paths, axis=1)

    @staticmethod
    def price_floating_strike_analytical(params: BSParameters, option_type: str) -> float:
        return float(
            _price_lookback_floating_strike_jit(
                params.spot,
                params.maturity,
                params.rate,
                params.dividend,
                params.volatility,
                option_type.lower() == "call",
            )
        )

    @staticmethod
    def price_lookback_mc(
        params: ExoticParameters,
        option_type: str,
        strike_type: StrikeType = StrikeType.FLOATING,
        **kwargs,
    ) -> tuple[float, float]:
        S, T, r, q, sigma = (
            params.base_params.spot,
            params.base_params.maturity,
            params.base_params.rate,
            params.base_params.dividend,
            params.base_params.volatility,
        )
        n_paths = kwargs.get("n_paths", 10000)
        log_paths = jit_generate_log_paths(S, T, r, sigma, q, n_paths, params.n_observations)
        res = fused_lookback_payoff(
            log_paths,
            params.base_params.strike,
            r,
            T,
            option_type == "call",
            strike_type == StrikeType.FLOATING,
        )
        return float(np.mean(res)), float(1.96 * np.std(res) / np.sqrt(n_paths))

class DigitalOptionPricer:
    @staticmethod
    def price_cash_or_nothing(params: BSParameters, option_type: str, payout: float = 1.0) -> float:
        if CORE_AVAILABLE:
            try:
                return float(
                    bsopt_core.digital_option_price(
                        params.spot,
                        params.strike,
                        params.maturity,
                        params.rate,
                        params.dividend,
                        params.volatility,
                        payout,
                        option_type.lower() == "call",
                        True,  # is_cash_or_nothing
                    )
                )
            except Exception:
                pass

        return float(
            _price_digital_cash_or_nothing_jit(
                params.spot,
                params.strike,
                params.maturity,
                params.rate,
                params.dividend,
                params.volatility,
                payout,
                option_type.lower() == "call",
            )
        )

    @staticmethod
    def price_asset_or_nothing(params: BSParameters, option_type: str) -> float:
        if CORE_AVAILABLE:
            try:
                return float(
                    bsopt_core.digital_option_price(
                        params.spot,
                        params.strike,
                        params.maturity,
                        params.rate,
                        params.dividend,
                        params.volatility,
                        0.0,  # payout ignored
                        option_type.lower() == "call",
                        False,  # is_cash_or_nothing
                    )
                )
            except Exception:
                pass

        return float(
            _price_digital_asset_or_nothing_jit(
                params.spot,
                params.strike,
                params.maturity,
                params.rate,
                params.dividend,
                params.volatility,
                option_type.lower() == "call",
            )
        )

    @staticmethod
    def calculate_digital_greeks(
        params: BSParameters, option_type: str, digital_type: str = "cash", payout: float = 1.0
    ) -> OptionGreeks:
        """Analytical Greeks for Digital Options (Cash-or-Nothing / Asset-or-Nothing)."""
        from src.math_kernel.quant_utils import fast_normal_cdf_v2, fast_normal_pdf_v2

        S, K, T, r, q, sigma = (
            params.spot,
            params.strike,
            params.maturity,
            params.rate,
            params.dividend,
            params.volatility,
        )
        if T <= 1e-12:
            return OptionGreeks(0, 0, 0, 0, 0)

        sqrt_T = np.sqrt(T)
        sig_sqrt_T = sigma * sqrt_T
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sig_sqrt_T
        d2 = d1 - sig_sqrt_T

        n_d1 = fast_normal_pdf_v2(d1)
        n_d2 = fast_normal_pdf_v2(d2)
        N_d1 = fast_normal_cdf_v2(d1)
        N_d2 = fast_normal_cdf_v2(d2)

        is_call = option_type.lower() == "call"
        phi = 1 if is_call else -1

        if digital_type == "cash":
            # Cash-or-Nothing
            price_factor = payout * np.exp(-r * T)
            delta = price_factor * n_d2 * (phi / (S * sig_sqrt_T))
            gamma = -price_factor * d1 * n_d2 * (1.0 / (S**2 * sig_sqrt_T * sig_sqrt_T))
            vega = price_factor * n_d2 * (-d1 / sigma)
            theta = r * price_factor * (N_d2 if is_call else (1 - N_d2)) - price_factor * n_d2 * (
                d1 / (2 * T)
            )
            rho = -T * price_factor * (N_d2 if is_call else (1 - N_d2)) + price_factor * n_d2 * (
                sqrt_T / sigma
            )
        else:
            # Asset-or-Nothing
            price_factor = np.exp(-q * T)
            delta = (
                price_factor * (N_d1 if is_call else (1 - N_d1))
                + price_factor * phi * n_d1 / sig_sqrt_T
            )
            gamma = -price_factor * n_d1 * d2 / (S * sigma**2 * T)
            vega = S * price_factor * n_d1 * (d2 / sigma)
            theta = q * S * price_factor * (
                N_d1 if is_call else (1 - N_d1)
            ) - S * price_factor * n_d1 * (d2 / (2 * T))
            rho = S * price_factor * n_d1 * (sqrt_T / sigma)

        return OptionGreeks(
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            rho=float(rho),
        )

def price_exotic_option(
    exotic_type: str, params: ExoticParameters, option_type: str, **kwargs
) -> tuple[float, float | None]:
    st_type = kwargs.pop("strike_type", None)
    if exotic_type == "asian":
        final_st = st_type if st_type is not None else StrikeType.FIXED
        if kwargs.get("asian_type") == AsianType.GEOMETRIC:
            return AsianOptionPricer.price_geometric_asian(params, option_type, final_st), None
        return AsianOptionPricer.price_arithmetic_asian_mc(params, option_type, final_st, **kwargs)
    if exotic_type == "barrier":
        return BarrierOptionPricer.price_barrier_analytical(
            params, option_type, kwargs.get("barrier_type", BarrierType.DOWN_AND_OUT)
        ), None
    if exotic_type == "lookback":
        final_st = st_type if st_type is not None else StrikeType.FLOATING
        return LookbackOptionPricer.price_lookback_mc(params, option_type, final_st, **kwargs)
    if exotic_type == "digital":
        if kwargs.get("digital_type") == "cash":
            return DigitalOptionPricer.price_cash_or_nothing(
                params.base_params, option_type, kwargs.get("payout", 1.0)
            ), None
        return DigitalOptionPricer.price_asset_or_nothing(params.base_params, option_type), None
    raise ValueError(f"Unknown exotic option type: {exotic_type}")
