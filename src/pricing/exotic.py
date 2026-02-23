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
        b_a = 0.5 * (sigma_a**2 + b - 0.5 * sigma**2)  # Kemna & Vorst b_a

        if T <= 1e-12:
            return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))

        d1 = (np.log(S / K) + (b_a + 0.5 * sigma_a**2) * T) / (sigma_a * np.sqrt(T))
        d2 = d1 - sigma_a * np.sqrt(T)

        if option_type == "call":
            price = S * np.exp((b_a - r) * T) * norm.cdf(d1) - K * np.exp(
                -r * T
            ) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(
                (b_a - r) * T
            ) * norm.cdf(-d1)

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

        from src.pricing.quant_utils import fused_arithmetic_asian_payoff, jit_generate_log_paths

        log_paths = jit_generate_log_paths(
            S, T, r, sigma, q, n_paths, params.n_observations
        )

        # OPTIMIZED: Fused kernel call (no large paths matrix allocation)
        y_sim = fused_arithmetic_asian_payoff(log_paths, K, r, T, is_call, is_fixed)

        if use_cv and is_fixed:
            # log_paths is (n_steps+1, n_paths). paths[:, 1:] corresponds to log_paths[1:, :].T
            geom_mean = np.exp(np.mean(log_paths[1:, :], axis=0))
            geo_payoff = (
                np.maximum(geom_mean - K, 0)
                if is_call
                else np.maximum(K - geom_mean, 0)
            )
            geo_price = AsianOptionPricer.price_geometric_asian(
                params, option_type, strike_type
            )
            y_geo = geo_payoff * np.exp(-r * T)
            cov = np.cov(y_sim, y_geo)
            if cov[1, 1] > 1e-12:
                beta = cov[0, 1] / cov[1, 1]
                y_cv = y_sim - beta * (y_geo - geo_price)
                return float(np.mean(y_cv)), float(
                    1.96 * np.std(y_cv) / np.sqrt(n_paths)
                )

        return float(np.mean(y_sim)), float(1.96 * np.std(y_sim) / np.sqrt(n_paths))


def _jit_arithmetic_asian_payoff(
    paths: np.ndarray, K: float, r: float, T: float, is_call: bool, is_fixed: bool
) -> np.ndarray:
    exp_rt = np.exp(-r * T)
    arith_mean = np.mean(paths[:, 1:], axis=1)

    if is_fixed:
        if is_call:
            payoffs = np.maximum(arith_mean - K, 0.0)
        else:
            payoffs = np.maximum(K - arith_mean, 0.0)
    else:  # Floating strike
        if is_call:
            payoffs = np.maximum(paths[:, -1] - arith_mean, 0.0)
        else:
            payoffs = np.maximum(arith_mean - paths[:, -1], 0.0)

    return payoffs * exp_rt


def _jit_lookback_payoff(
    paths: np.ndarray, K: float, r: float, T: float, is_call: bool, is_floating: bool
) -> np.ndarray:
    exp_rt = np.exp(-r * T)

    if is_floating:
        if is_call:
            min_s = np.min(paths, axis=1)
            payoffs = np.maximum(paths[:, -1] - min_s, 0.0)
        else:
            max_s = np.max(paths, axis=1)
            payoffs = np.maximum(max_s - paths[:, -1], 0.0)
    else:  # Fixed strike
        if is_call:
            max_s = np.max(paths, axis=1)
            payoffs = np.maximum(max_s - K, 0.0)
        else:
            min_s = np.min(paths, axis=1)
            payoffs = np.maximum(K - min_s, 0.0)

    return payoffs * exp_rt


class BarrierOptionPricer:
    @staticmethod
    def price_barrier_analytical(
        params: ExoticParameters, option_type: str, barrier_type: BarrierType
    ) -> float:
        """
        Implementation of Reiner-Rubinstein (1991) formulas.
        Follows Haug (2007) Table 4-4 carefully.
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

        # Validation
        if "up" in barrier_type.value and H < S:
            raise ValueError("Up-barrier must be above spot")
        if "down" in barrier_type.value and H > S:
            raise ValueError("Down-barrier must be below spot")

        if T <= 1e-12:
            payoff = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            is_up = "up" in barrier_type.value
            is_out = "out" in barrier_type.value
            hit = (S >= H) if is_up else (S <= H)
            return float(
                R if (is_out and hit) else (0.0 if (not is_out and not hit) else payoff)
            )

        b = r - q
        sig = max(sigma, 1e-12)
        mu = (b - 0.5 * sig**2) / sig**2
        lam = np.sqrt(mu**2 + 2 * r / sig**2)

        phi = 1 if option_type == "call" else -1
        eta = 1 if "down" in barrier_type.value else -1

        sqrt_T = np.sqrt(T)
        sig_sqrt_T = sig * sqrt_T

        # Consistent RR/Haug components
        x1 = np.log(S / K) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
        x2 = np.log(S / H) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
        y1 = np.log(H**2 / (S * K)) / sig_sqrt_T + (mu + 1) * sig_sqrt_T
        y2 = np.log(H / S) / sig_sqrt_T + (mu + 1) * sig_sqrt_T

        def _n(x):
            return norm.cdf(x)

        exp_br_T = np.exp((b - r) * T)
        exp_r_T = np.exp(-r * T)

        A = phi * S * exp_br_T * _n(phi * x1) - phi * K * exp_r_T * _n(
            phi * (x1 - sig_sqrt_T)
        )
        B = phi * S * exp_br_T * _n(phi * x2) - phi * K * exp_r_T * _n(
            phi * (x2 - sig_sqrt_T)
        )
        C = phi * S * exp_br_T * (H / S) ** (2 * (mu + 1)) * _n(
            eta * y1
        ) - phi * K * exp_r_T * (H / S) ** (2 * mu) * _n(eta * (y1 - sig_sqrt_T))
        D = phi * S * exp_br_T * (H / S) ** (2 * (mu + 1)) * _n(
            eta * y2
        ) - phi * K * exp_r_T * (H / S) ** (2 * mu) * _n(eta * (y2 - sig_sqrt_T))

        # Rebate components E and F
        E = (
            R
            * exp_r_T
            * (
                _n(eta * (x2 - sig_sqrt_T))
                - (H / S) ** (2 * mu) * _n(eta * (y2 - sig_sqrt_T))
            )
        )
        F = 0.0
        if R != 0:
            F = R * (
                (H / S) ** (mu + lam)
                * _n(eta * (np.log(H / S) / sig_sqrt_T + lam * sig_sqrt_T))
                + (H / S) ** (mu - lam)
                * _n(eta * (np.log(H / S) / sig_sqrt_T - lam * sig_sqrt_T))
            )

        price = 0.0
        is_call = option_type == "call"

        # RR Dispatch (Haug Table 4-4)
        if is_call:
            if barrier_type == BarrierType.DOWN_AND_IN:
                price = C if H <= K else A - B + D
            elif barrier_type == BarrierType.DOWN_AND_OUT:
                price = A - C if H <= K else B - D
            elif barrier_type == BarrierType.UP_AND_IN:
                price = A if H <= K else B - C + D
            elif barrier_type == BarrierType.UP_AND_OUT:
                price = 0.0 if H <= K else A - B + C - D
        else:  # Put
            if barrier_type == BarrierType.DOWN_AND_IN:
                price = A if H >= K else B - C + D
            elif barrier_type == BarrierType.DOWN_AND_OUT:
                price = 0.0 if H >= K else A - B + C - D
            elif barrier_type == BarrierType.UP_AND_IN:
                price = C if H >= K else A - B + D
            elif barrier_type == BarrierType.UP_AND_OUT:
                price = A - C if H >= K else B - D

        return max(float(price + E + F), 0.0)


class LookbackOptionPricer:
    @staticmethod
    def _compute_running_extrema(
        paths: np.ndarray, observation_indices: np.ndarray, mode: str = "max"
    ) -> np.ndarray:
        """Helper to compute running extrema for Monte Carlo paths."""
        if mode == "max":
            return np.max(paths[:, observation_indices], axis=1)
        return np.min(paths[:, observation_indices], axis=1)

    @staticmethod
    def price_floating_strike_analytical(
        params: BSParameters, option_type: str
    ) -> float:
        S, T, r, q, sigma = (
            params.spot,
            params.maturity,
            params.rate,
            params.dividend,
            params.volatility,
        )
        b = r - q
        if T <= 1e-12:
            return 0.0

        b_eff = b if abs(b) > 1e-12 else 1e-12
        sig = max(sigma, 1e-12)

        def _n(x):
            return norm.cdf(x)

        d1 = (b_eff + 0.5 * sig**2) * np.sqrt(T) / sig
        d2 = d1 - sig * np.sqrt(T)
        if option_type == "call":
            v = (
                S * np.exp(-q * T) * _n(d1)
                - S * np.exp(-q * T) * (sig**2 / (2 * b_eff)) * _n(-d1)
                - S * np.exp(-r * T) * (1 - sig**2 / (2 * b_eff)) * _n(d2)
            )
        else:
            v = (
                S * np.exp(-r * T) * (1 + sig**2 / (2 * b_eff)) * _n(-d2)
                + S * np.exp(-q * T) * (sig**2 / (2 * b_eff)) * _n(d1)
                - S * np.exp(-q * T) * _n(-d1)
            )
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

        from src.pricing.quant_utils import fused_lookback_payoff, jit_generate_log_paths

        log_paths = jit_generate_log_paths(
            S, T, r, sigma, q, n_paths, params.n_observations
        )

        # OPTIMIZED: Fused kernel call
        res = fused_lookback_payoff(log_paths, K, r, T, is_call, is_floating)

        return float(np.mean(res)), float(1.96 * np.std(res) / np.sqrt(n_paths))


def _n(x):
    return norm.cdf(x)


class DigitalOptionPricer:
    @staticmethod
    def price_cash_or_nothing(
        params: BSParameters, option_type: str, payout: float = 1.0
    ) -> float:
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


def price_exotic_option(
    exotic_type: str, params: ExoticParameters, option_type: str, **kwargs
):
    """
    Prices various exotic option types.
    """
    if exotic_type == "asian":
        asian_type_val = kwargs.get("asian_type", AsianType.GEOMETRIC)
        if asian_type_val == AsianType.GEOMETRIC:
            st_type = kwargs.get("strike_type", StrikeType.FIXED)
            return (
                AsianOptionPricer.price_geometric_asian(params, option_type, st_type),
                None,
            )
        else:
            return AsianOptionPricer.price_arithmetic_asian_mc(
                params, option_type, **kwargs
            )

    if exotic_type == "barrier":
        barrier_type_str = kwargs.get("barrier_type")
        if not barrier_type_str:
            raise ValueError("Barrier type is required for barrier options.")
        if isinstance(barrier_type_str, str):
            barrier_type = BarrierType(barrier_type_str)
        else:
            barrier_type = barrier_type_str

        return (
            BarrierOptionPricer.price_barrier_analytical(
                params, option_type, barrier_type
            ),
            None,
        )

    if exotic_type == "lookback":
        strike_type_val = kwargs.get("strike_type", StrikeType.FLOATING)
        use_mc = kwargs.get("use_mc", True)

        if strike_type_val == StrikeType.FLOATING and not use_mc:
            return (
                LookbackOptionPricer.price_floating_strike_analytical(
                    params.base_params, option_type
                ),
                None,
            )
        else:
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop("strike_type", None)
            return LookbackOptionPricer.price_lookback_mc(
                params, option_type, strike_type_val, **kwargs_copy
            )

    if exotic_type == "digital":
        return (
            DigitalOptionPricer.price_cash_or_nothing(
                params.base_params, option_type, kwargs.get("payout", 1.0)
            ),
            None,
        )

    raise ValueError(f"Unknown exotic option type: {exotic_type}")
