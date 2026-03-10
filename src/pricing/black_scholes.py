from typing import Any

import numpy as np
import structlog

from src.pricing.models import BSParameters, OptionGreeks
from src.shared.math_utils import calculate_greeks, calculate_price

try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = structlog.get_logger(__name__)


class BlackScholesEngine:
    """
    Black-Scholes option pricing engine.
    Optimized: Uses Rust 'bsopt_core' if available, falls back to Numba JIT.
    """

    @staticmethod
    def _extract_params(params: Any | None = None, **kwargs) -> tuple:
        """Helper to extract parameters."""
        if params:
            s = getattr(params, "spot", kwargs.get("spot"))
            k = getattr(params, "strike", kwargs.get("strike"))
            t = getattr(params, "maturity", kwargs.get("maturity"))
            v = getattr(params, "volatility", kwargs.get("volatility"))
            r = getattr(params, "rate", kwargs.get("rate"))
            d = getattr(params, "dividend", kwargs.get("dividend", 0.0))
        else:
            s = kwargs.get("spot")
            k = kwargs.get("strike")
            t = kwargs.get("maturity")
            v = kwargs.get("volatility")
            r = kwargs.get("rate")
            d = kwargs.get("dividend", 0.0)

        # Basic validation
        if any(x is None for x in [s, k, t, v, r]):
            raise ValueError(
                "Missing required parameters (spot, strike, maturity, volatility, rate)"
            )

        # Convert inputs to at least 1D numpy arrays first
        s_arr = np.atleast_1d(s).astype(np.float64)
        k_arr = np.atleast_1d(k).astype(np.float64)
        t_arr = np.atleast_1d(t).astype(np.float64)
        v_arr = np.atleast_1d(v).astype(np.float64)
        r_arr = np.atleast_1d(r).astype(np.float64)
        d_arr = np.atleast_1d(d).astype(np.float64)

        # Broadcast all to a common shape
        # np.broadcast() will raise an error if they are not compatible
        try:
            target_shape = np.broadcast(s_arr, k_arr, t_arr, v_arr, r_arr, d_arr).shape
        except ValueError as e:
            raise ValueError(f"Parameters cannot be broadcast to a common shape: {e}")

        def _to_arr(arr):
            if arr.shape != target_shape:
                return np.broadcast_to(arr, target_shape).copy()
            return arr

        return (
            _to_arr(s_arr),
            _to_arr(k_arr),
            _to_arr(t_arr),
            _to_arr(v_arr),
            _to_arr(r_arr),
            _to_arr(d_arr),
        )

    @staticmethod
    def price_options(
        spot: float | np.ndarray | Any | None = None,
        strike: float | np.ndarray | None = None,
        maturity: float | np.ndarray | None = None,
        volatility: float | np.ndarray | None = None,
        rate: float | np.ndarray | None = None,
        dividend: float | np.ndarray = 0.0,
        option_type: str | np.ndarray = "call",
        params: Any | None = None,
        **kwargs,
    ) -> float | np.ndarray:
        """
        Calculate European option prices using Black-Scholes formula (JIT Accelerated).
        """
        # If first arg is BSParameters, shift it to params
        if hasattr(spot, "spot") and params is None:
            params = spot
            spot = None

        S, K, T, sigma, r, q = BlackScholesEngine._extract_params(
            params,
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            rate=rate,
            dividend=dividend,
        )

        # Vectorized option type handling
        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"

        if CORE_AVAILABLE:
            try:
                # Optimized Rust path
                if S.size > 1:
                    # Ensure is_call is broadcast to S.shape for Rust core
                    is_call_arr = np.atleast_1d(is_call).astype(bool)
                    if is_call_arr.shape != S.shape:
                        is_call_arr = np.broadcast_to(is_call_arr, S.shape).copy()

                    # Ensure 1D arrays for the Rust batch function
                    return bsopt_core.batch_black_scholes(
                        S.ravel(),
                        K.ravel(),
                        T.ravel(),
                        sigma.ravel(),
                        r.ravel(),
                        q.ravel(),
                        is_call_arr.ravel(),
                    ).reshape(S.shape)

                # Scalar path
                return bsopt_core.black_scholes_price(
                    float(S[0]),
                    float(K[0]),
                    float(T[0]),
                    float(sigma[0]),
                    float(r[0]),
                    float(q[0]),
                    bool(is_call),
                )
            except Exception as e:
                logger.warning("rust_core_pricing_failed_falling_back", error=str(e))

        # The shared math utility handles broadcasting and returns either scalar or array
        if kwargs.get("out") is not None:
            from src.pricing.quant_utils import batch_bs_price_jit_v2_out

            # Ensure is_call is an array for Numba batch path
            if np.isscalar(is_call):
                is_call_arr = np.full(S.shape, is_call, dtype=bool)
            else:
                is_call_arr = np.asanyarray(is_call).astype(bool)

            batch_bs_price_jit_v2_out(S, K, T, sigma, r, q, is_call_arr, kwargs["out"])
            return kwargs["out"]

        return calculate_price(S, K, T, sigma, r, q, is_call)

    @staticmethod
    def calculate_greeks(
        spot: float | np.ndarray | Any | None = None,
        strike: float | np.ndarray | None = None,
        maturity: float | np.ndarray | None = None,
        volatility: float | np.ndarray | None = None,
        rate: float | np.ndarray | None = None,
        dividend: float | np.ndarray = 0.0,
        option_type: str | np.ndarray = "call",
        params: Any | None = None,
        **kwargs,
    ) -> OptionGreeks:
        """
        Calculate Greeks for European options (Accelerated).
        """
        if hasattr(spot, "spot") and params is None:
            params = spot
            spot = None

        S, K, T, sigma, r, q = BlackScholesEngine._extract_params(
            params,
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            rate=rate,
            dividend=dividend,
        )

        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"

        if CORE_AVAILABLE:
            try:
                if S.size == 1:
                    # Optimized Rust scalar path
                    res = bsopt_core.black_scholes_greeks(
                        float(S[0]),
                        float(K[0]),
                        float(T[0]),
                        float(sigma[0]),
                        float(r[0]),
                        float(q[0]),
                        bool(is_call),
                    )
                    return OptionGreeks(
                        delta=res.delta,
                        gamma=res.gamma,
                        theta=res.theta,
                        vega=res.vega,
                        rho=res.rho,
                    )
                else:
                    # Optimized Rust batch path
                    is_call_arr = np.atleast_1d(is_call).astype(bool)
                    if is_call_arr.shape != S.shape:
                        is_call_arr = np.broadcast_to(is_call_arr, S.shape).copy()

                    d, g, th, v, rh = bsopt_core.batch_black_scholes_greeks(
                        S.ravel(),
                        K.ravel(),
                        T.ravel(),
                        sigma.ravel(),
                        r.ravel(),
                        q.ravel(),
                        is_call_arr.ravel(),
                    )
                    return OptionGreeks(
                        delta=d.reshape(S.shape),
                        gamma=g.reshape(S.shape),
                        theta=th.reshape(S.shape),
                        vega=v.reshape(S.shape),
                        rho=rh.reshape(S.shape),
                    )
            except Exception as e:
                logger.warning("rust_core_greeks_failed_falling_back", error=str(e))

        if "out_delta" in kwargs:
            from src.pricing.quant_utils import batch_greeks_jit_v2_out

            # Ensure is_call is an array for Numba batch path
            if np.isscalar(is_call):
                is_call_arr = np.full(S.shape, is_call, dtype=bool)
            else:
                is_call_arr = np.asanyarray(is_call).astype(bool)

            batch_greeks_jit_v2_out(
                S,
                K,
                T,
                sigma,
                r,
                q,
                is_call_arr,
                kwargs["out_delta"],
                kwargs["out_gamma"],
                kwargs["out_theta"],
                kwargs["out_vega"],
                kwargs["out_rho"],
            )
            return OptionGreeks(
                delta=kwargs["out_delta"],
                gamma=kwargs["out_gamma"],
                theta=kwargs["out_theta"],
                vega=kwargs["out_vega"],
                rho=kwargs["out_rho"],
            )

        delta, gamma, theta, vega, rho = calculate_greeks(S, K, T, sigma, r, q, is_call)

        return OptionGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


    @staticmethod
    def price_call(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type="call"))

    @staticmethod
    def price_put(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type="put"))

    @staticmethod
    def price_batch(S, K, T, sigma, r, dividend, option_types) -> np.ndarray:
        """Vectorized pricing returning an array."""
        return BlackScholesEngine.price_options(
            spot=S,
            strike=K,
            maturity=T,
            volatility=sigma,
            rate=r,
            dividend=dividend,
            option_type=option_types,
        )

    @staticmethod
    def price_batch_greeks(S, K, T, sigma, r, dividend, is_call=True) -> tuple[np.ndarray, ...]:
        """
        Specialized batch Greek calculation for GreekEngine.
        Returns a tuple of arrays: (delta, gamma, theta, vega, rho)
        """
        # Ensure arrays for broadcasting
        n = len(S)
        K_arr = np.full(n, K) if np.isscalar(K) else np.asanyarray(K)
        T_arr = np.full(n, T) if np.isscalar(T) else np.asanyarray(T)
        sig_arr = np.full(n, sigma) if np.isscalar(sigma) else np.asanyarray(sigma)
        r_arr = np.full(n, r) if np.isscalar(r) else np.asanyarray(r)
        q_arr = np.full(n, dividend) if np.isscalar(dividend) else np.asanyarray(dividend)

        if np.isscalar(is_call):
            is_call_arr = np.full(n, is_call, dtype=bool)
        else:
            is_call_arr = np.asanyarray(is_call).astype(bool)

        from src.pricing.quant_utils import batch_greeks_jit_v2

        # Note: batch_greeks_jit_v2 returns (delta, gamma, vega, theta, rho)
        # But GreekEngine expects (delta, gamma, theta, vega, rho)
        d, g, v, th, rh = batch_greeks_jit_v2(S, K_arr, T_arr, sig_arr, r_arr, q_arr, is_call_arr)
        return d, g, th, v, rh

    @staticmethod
    def calculate_greeks_batch(**kwargs) -> dict[str, np.ndarray]:
        """Vectorized Greeks calculation returning a dictionary."""
        greeks = BlackScholesEngine.calculate_greeks(**kwargs)
        return {
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "theta": greeks.theta,
            "vega": greeks.vega,
            "rho": greeks.rho,
        }

    @staticmethod
    def verify_put_call_parity(S, K, T, r, call_price, put_price, q=0.0):
        lhs = np.asanyarray(call_price) - np.asanyarray(put_price)
        rhs = np.asanyarray(S) * np.exp(-np.asanyarray(q) * np.asanyarray(T)) - np.asanyarray(
            K
        ) * np.exp(-np.asanyarray(r) * np.asanyarray(T))
        return np.allclose(lhs, rhs, atol=1e-5)

    @classmethod
    def price(
        cls, params: BSParameters | None = None, option_type: str = "call", **kwargs
    ) -> float:
        """Class method for backward compatibility and PricingStrategy interface."""
        return float(cls.price_options(params=params, option_type=option_type, **kwargs))


def black_scholes(*args, **kwargs):
    result = BlackScholesEngine.price_options(*args, **kwargs)
    if len(args) == 5 or "params" in kwargs:
        return {"price": result}
    return result


def verify_put_call_parity(
    params_or_S, K=None, T=None, r=None, call_price=None, put_price=None, q=0.0
):
    """Module-level parity verifier for test compatibility."""
    if hasattr(params_or_S, "spot") and K is None:
        p = params_or_S
        cp = BlackScholesEngine.price_options(params=p, option_type="call")
        pp = BlackScholesEngine.price_options(params=p, option_type="put")
        return BlackScholesEngine.verify_put_call_parity(
            p.spot, p.strike, p.maturity, p.rate, cp, pp, p.dividend
        )
    return BlackScholesEngine.verify_put_call_parity(params_or_S, K, T, r, call_price, put_price, q)
