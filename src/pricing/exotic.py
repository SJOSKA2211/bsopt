"Exotic Options Pricing Module"

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple, cast

import numpy as np
from scipy.stats import norm

from src.pricing.black_scholes import BlackScholesEngine, BSParameters


class AsianType(Enum):
    GEOMETRIC = auto()
    ARITHMETIC = auto()


class StrikeType(Enum):
    FIXED = auto()
    FLOATING = auto()


class BarrierType(Enum):
    DOWN_AND_OUT = auto()
    DOWN_AND_IN = auto()
    UP_AND_OUT = auto()
    UP_AND_IN = auto()


@dataclass
class ExoticParameters:
    base_params: BSParameters
    n_observations: int = 252
    barrier: float = 0.0
    rebate: float = 0.0

    def __post_init__(self):
        self.barrier = np.float64(self.barrier)
        self.rebate = np.float64(self.rebate)
        if self.barrier < 0:
            raise ValueError("Barrier must be positive")
        if self.rebate < 0:
            raise ValueError("Rebate must be non-negative")
        if self.n_observations < 2:
            raise ValueError("n_observations must be >= 2")


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
        b_a = 0.5 * (b - 0.5 * sigma**2 + sigma_a**2)

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
        **kwargs
    ) -> Tuple[float, float]:
        S, K, T, r, q, sigma = (
            params.base_params.spot,
            params.base_params.strike,
            params.base_params.maturity,
            params.base_params.rate,
            params.base_params.dividend,
            params.base_params.volatility,
        )
        n_paths, seed = kwargs.get("n_paths", 10000), kwargs.get("seed")
        use_cv = kwargs.get("use_control_variate", True)
        st = kwargs.get("strike_type", StrikeType.FIXED)

        if T <= 1e-12:
            return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)), 0.0

        dt = T / params.n_observations
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_paths, params.n_observations))
        paths = np.zeros((n_paths, params.n_observations + 1), dtype=np.float64)
        paths[:, 0] = S
        drift, diff = (r - q - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)

        for t in range(1, params.n_observations + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diff * z[:, t - 1])

        arith_mean = np.mean(paths[:, 1:], axis=1)
        payoff = np.maximum(arith_mean - K, 0) if option_type == "call" else \
            np.maximum(K - arith_mean, 0)
        y_sim = payoff * np.exp(-r * T)

        if use_cv:
            geom_mean = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
            geo_payoff = np.maximum(geom_mean - K, 0) if option_type == "call" else \
                np.maximum(K - geom_mean, 0)
            geo_price = AsianOptionPricer.price_geometric_asian(params, option_type, st)
            x_sim = geo_payoff * np.exp(-r * T)
            cov_mat = np.cov(y_sim, x_sim)
            beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 1e-12 else 1.0
            adj_payoff = y_sim - beta * (x_sim - geo_price)
            return float(np.mean(adj_payoff)), float(1.96 * np.std(adj_payoff) / np.sqrt(n_paths))

        return float(np.mean(y_sim)), float(1.96 * np.std(y_sim) / np.sqrt(n_paths))


class BarrierOptionPricer:
    @staticmethod
    def price_barrier_analytical(
        params: ExoticParameters,
        option_type: str,
        barrier_type: BarrierType
    ) -> float:
        S, K, T, r, q, sigma = (
            params.base_params.spot,
            params.base_params.strike,
            params.base_params.maturity,
            params.base_params.rate,
            params.base_params.dividend,
            params.base_params.volatility,
        )
        H, R = params.barrier, params.rebate
        if (barrier_type in [BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN] and H <= S):
            raise ValueError("Up-barrier must be strictly above spot")
        if (barrier_type in [BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN] and H >= S):
            raise ValueError("Down-barrier must be strictly below spot")

        b, sqrtT = r - q, np.sqrt(T)
        mu = (b - 0.5 * sigma**2) / sigma**2
        lam = np.sqrt(mu**2 + 2 * r / sigma**2)

        def x1(S, K):
            return np.log(S/K)/(sigma*sqrtT) + (mu+1)*sigma*sqrtT

        def x2(S, H):
            return np.log(S/H)/(sigma*sqrtT) + (mu+1)*sigma*sqrtT

        def y1(S, H, K):
            return np.log(H**2/(S*K))/(sigma*sqrtT) + (mu+1)*sigma*sqrtT

        def y2(S, H):
            return np.log(H/S)/(sigma*sqrtT) + (mu+1)*sigma*sqrtT

        def f1(phi, S, K):
            return phi*S*np.exp((b-r)*T)*norm.cdf(phi*x1(S,K)) - \
                phi*K*np.exp(-r*T)*norm.cdf(phi*(x1(S,K)-sigma*sqrtT))

        def f2(phi, S, H):
            return phi*S*np.exp((b-r)*T)*norm.cdf(phi*x2(S,H)) - \
                phi*K*np.exp(-r*T)*norm.cdf(phi*(x2(S,H)-sigma*sqrtT))

        def f3(phi, eta, S, H, K):
            return phi*S*np.exp((b-r)*T)*(H/S)**(2*(mu+1))*norm.cdf(eta*y1(S,H,K)) - \
                phi*K*np.exp(-r*T)*(H/S)**(2*mu)*norm.cdf(eta*(y1(S,H,K)-sigma*sqrtT))

        def f4(phi, eta, S, H):
            return phi*S*np.exp((b-r)*T)*(H/S)**(2*(mu+1))*norm.cdf(eta*y2(S,H)) - \
                phi*K*np.exp(-r*T)*(H/S)**(2*mu)*norm.cdf(eta*(y2(S,H)-sigma*sqrtT))

        def f5(eta):
            return R*np.exp(-r*T)*(norm.cdf(eta*(x2(S,H)-sigma*sqrtT)) - \
                                   (H/S)**(2*mu)*norm.cdf(eta*(y2(S,H)-sigma*sqrtT)))

        def f6(eta):
            z_val = np.log(H/S)/(sigma*sqrtT) + lam*sigma*sqrtT
            return R*((H/S)**(mu+lam)*norm.cdf(eta*z_val) + \
                      (H/S)**(mu-lam)*norm.cdf(eta*(z_val-2*lam*sigma*sqrtT)))

        price = 0.0
        if option_type == "call":
            if barrier_type == BarrierType.DOWN_AND_IN:
                price = f3(1, 1, S, H, K) + f5(1) if K >= H else \
                    f1(1, S, K) - f2(1, S, H) + f4(1, 1, S, H) + f5(1)
            elif barrier_type == BarrierType.DOWN_AND_OUT:
                price = f1(1, S, K) - f3(1, 1, S, H, K) + f6(1) if K >= H else \
                    f2(1, S, H) - f4(1, 1, S, H) + f6(1)
            elif barrier_type == BarrierType.UP_AND_IN:
                price = f1(1, S, K) + f5(-1) if K >= H else \
                    f2(1, S, H) - f3(1, -1, S, H, K) + f4(1, -1, S, H) + f5(-1)
            elif barrier_type == BarrierType.UP_AND_OUT:
                price = f6(-1) if K >= H else \
                    f1(1, S, K) - f2(1, S, H) + f3(1, -1, S, H, K) - f4(1, -1, S, H) + f6(-1)
        else:  # put
            if barrier_type == BarrierType.DOWN_AND_IN:
                price = f2(-1, S, H) - f3(-1, 1, S, H, K) + f4(-1, 1, S, H) + f5(1) if K >= H \
                    else f1(-1, S, K) + f5(1)
            elif barrier_type == BarrierType.DOWN_AND_OUT:
                price = f1(-1, S, K) - f2(-1, S, H) + f3(-1, 1, S, H, K) - f4(-1, 1, S, H) + f6(1) \
                    if K >= H else f6(1)
            elif barrier_type == BarrierType.UP_AND_IN:
                price = f1(-1, S, K) - f2(-1, S, H) + f4(-1, -1, S, H) + f5(-1) if K >= H else \
                    f3(-1, -1, S, H, K) + f5(-1)
            elif barrier_type == BarrierType.UP_AND_OUT:
                price = f2(-1, S, H) - f4(-1, -1, S, H) + f6(-1) if K >= H else \
                    f1(-1, S, K) - f3(-1, -1, S, H, K) + f6(-1)
        return float(max(0.0, price))


class LookbackOptionPricer:
    @staticmethod
    def price_floating_strike_analytical(params: BSParameters, option_type: str) -> float:
        S, T, r, q, sigma = (
            params.spot, params.maturity, params.rate, params.dividend, params.volatility
        )
        b = r - q
        if abs(b) < 1e-10:
            b = 1e-10
        d1 = (b + 0.5 * sigma**2) * np.sqrt(T) / sigma
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            v_c = S*np.exp(-q*T)*norm.cdf(d1) - S*np.exp(-r*T)*norm.cdf(d2)
            price = v_c + S*np.exp(-r*T)*(sigma**2/(2*b)) * (np.exp(b*T)*norm.cdf(d1) - norm.cdf(d2))
        else:
            v_p = S*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
            price = v_p + S*np.exp(-q*T)*(sigma**2/(2*b)) * (np.exp(-b*T)*norm.cdf(-d2) - norm.cdf(-d1))
        return float(price)

    @staticmethod
    def price_lookback_mc(
        params: ExoticParameters,
        option_type: str,
        strike_type: StrikeType,
        **kwargs
    ) -> Tuple[float, float]:
        S, T, r, q, sigma = (
            params.base_params.spot, params.base_params.maturity,
            params.base_params.rate, params.base_params.dividend, params.base_params.volatility
        )
        K, n_paths, seed = params.base_params.strike, kwargs.get("n_paths", 10000), kwargs.get("seed")
        dt = T / params.n_observations
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_paths, params.n_observations))
        paths = np.zeros((n_paths, params.n_observations + 1), dtype=np.float64)
        paths[:, 0] = S
        drift, diff = (r-q-0.5*sigma**2)*dt, sigma*np.sqrt(dt)
        for t in range(1, params.n_observations + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diff * z[:, t-1])
        if strike_type == StrikeType.FLOATING:
            payoff = paths[:, -1] - np.min(paths, axis=1) if option_type == "call" \
                else np.max(paths, axis=1) - paths[:, -1]
        else:
            payoff = np.max(paths, axis=1) - K if option_type == "call" else K - np.min(paths, axis=1)
        v_val = payoff * np.exp(-r * T)
        return float(np.mean(v_val)), float(1.96 * np.std(v_val) / np.sqrt(n_paths))

    @staticmethod
    def _compute_running_extrema(paths: np.ndarray, indices: np.ndarray, type: str = "max") -> np.ndarray:
        if type == "max":
            return cast(np.ndarray, np.max(paths[:, indices], axis=1))
        return cast(np.ndarray, np.min(paths[:, indices], axis=1))


class DigitalOptionPricer:
    @staticmethod
    def price_cash_or_nothing(params: BSParameters, option_type: str, payout: float = 1.0) -> float:
        S, K, T, r, q, sigma = (
            params.spot, params.strike, params.maturity, params.rate, params.dividend, params.volatility
        )
        if T <= 1e-12:
            return float(payout if (option_type == "call" and S > K) or \
                         (option_type == "put" and S < K) else 0.0)
        d2 = (np.log(S/K) + (r-q-0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return float(payout*np.exp(-r*T)*norm.cdf(d2 if option_type == "call" else -d2))

    @staticmethod
    def price_asset_or_nothing(params: BSParameters, option_type: str) -> float:
        S, K, T, r, q, sigma = (
            params.spot, params.strike, params.maturity, params.rate, params.dividend, params.volatility
        )
        if T <= 1e-12:
            return float(S if (option_type == "call" and S > K) or \
                         (option_type == "put" and S < K) else 0.0)
        d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return float(S*np.exp(-q*T)*norm.cdf(d1 if option_type == "call" else -d1))

    @staticmethod
    def calculate_digital_greeks(
        params: BSParameters,
        option_type: str,
        digital_type: str,
        payout: float = 1.0
    ) -> Any:
        from src.pricing.black_scholes import OptionGreeks
        S, K, T, r, q, sigma = (
            params.spot, params.strike, params.maturity, params.rate, params.dividend, params.volatility
        )
        sqrtT = np.sqrt(T)
        d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*sqrtT)
        d2 = d1 - sigma*sqrtT
        pdf_d2 = norm.pdf(d2)
        delta = (payout*np.exp(-r*T)*pdf_d2 / (S*sigma*sqrtT)) * (1 if option_type == "call" else -1)
        gamma = -delta * d1 / (S*sigma*sqrtT)
        vega = payout*np.exp(-r*T)*pdf_d2*(-d1/sigma) * 0.01
        return OptionGreeks(delta=delta, gamma=gamma, vega=vega, theta=0, rho=0)


def price_exotic_option(exotic_type: str, params: ExoticParameters, option_type: str, **kwargs):
    if exotic_type == "asian":
        at = kwargs.pop("asian_type", AsianType.GEOMETRIC)
        if at == AsianType.GEOMETRIC:
            return AsianOptionPricer.price_geometric_asian(params, option_type, **kwargs), None
        return AsianOptionPricer.price_arithmetic_asian_mc(params, option_type, **kwargs)
    if exotic_type == "barrier":
        return BarrierOptionPricer.price_barrier_analytical(
            params, option_type, kwargs["barrier_type"]),
    if exotic_type == "lookback":
        st_type = kwargs.pop("strike_type", StrikeType.FLOATING)
        if st_type == StrikeType.FLOATING and not kwargs.pop("use_mc", True):
            return LookbackOptionPricer.price_floating_strike_analytical(
                params.base_params, option_type), None
        return LookbackOptionPricer.price_lookback_mc(params, option_type, st_type, **kwargs)
    if exotic_type == "digital":
        return DigitalOptionPricer.price_cash_or_nothing(
            params.base_params, option_type, kwargs.get("payout", 1.0)), None
    raise ValueError(f"Unknown option_class: {exotic_type}")