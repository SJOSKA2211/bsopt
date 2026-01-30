import numpy as np
from scipy.stats import norm
from typing import Union, Optional, Dict, Any
from src.pricing.models import BSParameters, OptionGreeks
from .base import PricingStrategy
from .quant_utils import batch_bs_price_jit, batch_greeks_jit

class BlackScholesEngine(PricingStrategy):
    def price(self, params: Optional[BSParameters] = None, option_type: str = "call", **kwargs) -> Union[float, np.ndarray]:
        """Instance method for PricingStrategy interface. Delegates to optimized static method."""
        return self.price_options(params=params, option_type=option_type, **kwargs)

    @staticmethod
    def calculate_greeks(params: Optional[BSParameters] = None, option_type: str = "call", **kwargs) -> OptionGreeks:
        """Implementation of PricingStrategy.calculate_greeks. Delegates to optimized static method."""
        return BlackScholesEngine.calculate_greeks_batch(params=params, option_type=option_type, **kwargs)

    @staticmethod
    def price_call(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type='call'))

    @staticmethod
    def price_put(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type='put'))

    @staticmethod
    def calculate_greeks_static(params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Compatibility wrapper."""
        return BlackScholesEngine.calculate_greeks_batch(params=params, option_type=option_type)

    @staticmethod
    def price_options(
        spot: Union[float, np.ndarray, None] = None,
        strike: Union[float, np.ndarray, None] = None,
        maturity: Union[float, np.ndarray, None] = None,
        volatility: Union[float, np.ndarray, None] = None,
        rate: Union[float, np.ndarray, None] = None,
        dividend: Union[float, np.ndarray] = 0.0,
        option_type: Union[str, np.ndarray] = "call",
        params: Optional[BSParameters] = None,
        out: Optional[np.ndarray] = None,
        **kwargs
    ) -> Union[float, np.ndarray]:
        """Highly optimized vectorized option pricing with memory reuse support."""
        if params is not None:
            spot = params.spot
            strike = params.strike
            maturity = params.maturity
            volatility = params.volatility
            rate = params.rate
            dividend = params.dividend

        if any(x is None for x in [spot, strike, maturity, volatility, rate]):
             raise TypeError("Missing required pricing parameters")
        
        # Performance optimization: Handle scalar cases separately to avoid array overhead
        if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(maturity) and \
           np.isscalar(volatility) and np.isscalar(rate) and np.isscalar(dividend) and \
           isinstance(option_type, str):
            S, K, T, sigma, r, q = float(spot), float(strike), float(maturity), \
                                  float(volatility), float(rate), float(dividend)
            is_call = option_type.lower() == "call"
            # We still use the jit function but with 1D arrays to leverage the logic
            # but we could also have a scalar JIT if needed.
            prices = batch_bs_price_jit(
                np.array([S]), np.array([K]), np.array([T]), 
                np.array([sigma]), np.array([r]), np.array([q]), 
                np.array([is_call]), out=out
            )
            return float(prices[0])

        # Optimized broadcasting: only broadcast if shapes differ
        inputs = [spot, strike, maturity, volatility, rate, dividend]
        if all(isinstance(x, np.ndarray) for x in inputs) and all(x.shape == inputs[0].shape for x in inputs):
            S, K, T, sigma, r, q = inputs
        else:
            S, K, T, sigma, r, q = np.broadcast_arrays(
                np.atleast_1d(spot).astype(np.float64),
                np.atleast_1d(strike).astype(np.float64),
                np.atleast_1d(maturity).astype(np.float64),
                np.atleast_1d(volatility).astype(np.float64),
                np.atleast_1d(rate).astype(np.float64),
                np.atleast_1d(dividend).astype(np.float64)
            )
        
        if isinstance(option_type, (str, np.str_)):
            is_call = np.full(S.shape, str(option_type).lower() == "call", dtype=bool)
        else:
            is_call = np.array([str(t).lower() == "call" for t in option_type], dtype=bool)
            
        return batch_bs_price_jit(S, K, T, sigma, r, q, is_call, out=out)

    @staticmethod
    def calculate_greeks_batch(
        spot: Union[float, np.ndarray, None] = None,
        strike: Union[float, np.ndarray, None] = None,
        maturity: Union[float, np.ndarray, None] = None,
        volatility: Union[float, np.ndarray, None] = None,
        rate: Union[float, np.ndarray, None] = None,
        dividend: Union[float, np.ndarray] = 0.0,
        option_type: Union[str, np.ndarray] = "call",
        params: Optional[BSParameters] = None,
        out_delta: Optional[np.ndarray] = None,
        out_gamma: Optional[np.ndarray] = None,
        out_vega: Optional[np.ndarray] = None,
        out_theta: Optional[np.ndarray] = None,
        out_rho: Optional[np.ndarray] = None,
        **kwargs
    ) -> Union[OptionGreeks, Dict[str, np.ndarray]]:
        """Highly optimized vectorized greeks calculation with memory reuse support."""
        if params is not None:
            spot = params.spot
            strike = params.strike
            maturity = params.maturity
            volatility = params.volatility
            rate = params.rate
            dividend = params.dividend

        if any(x is None for x in [spot, strike, maturity, volatility, rate]):
             raise TypeError("Missing required pricing parameters")
        
        # Handle scalar case
        if np.isscalar(spot) and isinstance(option_type, str):
            S, K, T, sigma, r, q = float(spot), float(strike), float(maturity), \
                                  float(volatility), float(rate), float(dividend)
            is_call = option_type.lower() == "call"
            delta, gamma, vega, theta, rho = batch_greeks_jit(
                np.array([S]), np.array([K]), np.array([T]), 
                np.array([sigma]), np.array([r]), np.array([q]), 
                np.array([is_call])
            )
            return OptionGreeks(
                delta=float(delta[0]),
                gamma=float(gamma[0]),
                vega=float(vega[0]),
                theta=float(theta[0]),
                rho=float(rho[0])
            )

        # Optimized broadcasting
        inputs = [spot, strike, maturity, volatility, rate, dividend]
        if all(isinstance(x, np.ndarray) for x in inputs) and all(x.shape == inputs[0].shape for x in inputs):
            S, K, T, sigma, r, q = inputs
        else:
            S, K, T, sigma, r, q = np.broadcast_arrays(
                np.atleast_1d(spot).astype(np.float64),
                np.atleast_1d(strike).astype(np.float64),
                np.atleast_1d(maturity).astype(np.float64),
                np.atleast_1d(volatility).astype(np.float64),
                np.atleast_1d(rate).astype(np.float64),
                np.atleast_1d(dividend).astype(np.float64)
            )
        
        if isinstance(option_type, (str, np.str_)):
            is_call = np.full(S.shape, str(option_type).lower() == "call", dtype=bool)
        else:
            is_call = np.array([str(t).lower() == "call" for t in option_type], dtype=bool)
            
        delta, gamma, vega, theta, rho = batch_greeks_jit(
            S, K, T, sigma, r, q, is_call,
            out_delta=out_delta, out_gamma=out_gamma, out_vega=out_vega,
            out_theta=out_theta, out_rho=out_rho
        )
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho
        }

def verify_put_call_parity(params: BSParameters, tolerance: float = 1e-4) -> bool:
    """Verify put-call parity for European options."""
    call_price = BlackScholesEngine.price_call(params)
    put_price = BlackScholesEngine.price_put(params)
    
    # C - P = S*e^(-qT) - K*e^(-rT)
    lhs = call_price - put_price
    rhs = params.spot * np.exp(-params.dividend * params.maturity) - \
          params.strike * np.exp(-params.rate * params.maturity)
          
    return bool(abs(lhs - rhs) < tolerance)


