from typing import Any, cast

import numpy as np
import structlog

from services.pricing.models import BSParameters, OptionGreeks
from core.shared.math_utils import calculate_greeks, calculate_price

from .base import PricingStrategy

try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = structlog.get_logger(__name__)


class BlackScholesEngine(PricingStrategy):
    """
    Black-Scholes option pricing engine.
    Optimized: Uses Rust 'bsopt_core' if available, falls back to Numba JIT.
    """

    @staticmethod
    def _extract_params(
        params: Any | None = None, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        try:
            target_shape = np.broadcast(s_arr, k_arr, t_arr, v_arr, r_arr, d_arr).shape
        except ValueError as e:
            raise ValueError(f"Parameters cannot be broadcast to a common shape: {e}")

        def _to_arr(arr: np.ndarray) -> np.ndarray:
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
        **kwargs: Any,
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
                    is_call_arr = np.atleast_1d(is_call).astype(bool)
                    if is_call_arr.shape != S.shape:
                        is_call_arr = np.broadcast_to(is_call_arr, S.shape).copy()

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

        # GPU Acceleration Path (Numba CUDA)
        if S.size > 100:  # Threshold for GPU overhead
            try:
                from services.pricing.cuda_kernels import price_options_gpu

                is_call_arr = np.atleast_1d(is_call).astype(bool)
                if is_call_arr.shape != S.shape:
                    is_call_arr = np.broadcast_to(is_call_arr, S.shape).copy()

                return price_options_gpu(S, K, T, sigma, r, q, is_call_arr)
            except Exception as e:
                logger.warning("gpu_kernel_failed_falling_back", error=str(e))

        # Vectorized numpy fallback
        res = calculate_price(S, K, T, sigma, r, q, is_call)

        if kwargs.get("out") is not None:
            out_arr = kwargs["out"]
            if isinstance(res, np.ndarray):
                np.copyto(out_arr, res)
            else:
                out_arr.fill(res)
            return out_arr

        return res

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
        **kwargs: Any,
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

        # Vectorized numpy fallback
        delta, gamma, theta, vega, rho = calculate_greeks(S, K, T, sigma, r, q, is_call)

        if "out_delta" in kwargs:

            def _copy(dst, src):
                if isinstance(src, np.ndarray):
                    np.copyto(dst, src)
                else:
                    dst.fill(src)

            _copy(kwargs["out_delta"], delta)
            _copy(kwargs["out_gamma"], gamma)
            _copy(kwargs["out_theta"], theta)
            _copy(kwargs["out_vega"], vega)
            _copy(kwargs["out_rho"], rho)

            return OptionGreeks(
                delta=kwargs["out_delta"],
                gamma=kwargs["out_gamma"],
                theta=kwargs["out_theta"],
                vega=kwargs["out_vega"],
                rho=kwargs["out_rho"],
            )

        return OptionGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

    @staticmethod
    def price_call(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type="call"))

    @staticmethod
    def price_put(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type="put"))

    @staticmethod
    def price_batch(
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        r: np.ndarray,
        dividend: np.ndarray,
        option_types: np.ndarray,
    ) -> np.ndarray:
        """Vectorized pricing returning an array."""
        return cast(
            np.ndarray,
            BlackScholesEngine.price_options(
                spot=S,
                strike=K,
                maturity=T,
                volatility=sigma,
                rate=r,
                dividend=dividend,
                option_type=option_types,
            ),
        )

    def price_european(
        self, params: BSParameters, option_type: str = "call", **kwargs: Any
    ) -> float:
        """Implementation of PricingStrategy interface."""
        return float(self.price_options(params=params, option_type=option_type, **kwargs))


def black_scholes(*args: Any, **kwargs: Any) -> Any:
    result = BlackScholesEngine.price_options(*args, **kwargs)
    if len(args) == 5 or "params" in kwargs:
        return {"price": result}
    return result
