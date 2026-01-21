import numpy as np
from scipy.stats import norm
from typing import Union, Optional, Dict
from src.pricing.models import BSParameters, OptionGreeks
from .base import PricingStrategy

import numpy as np
from scipy.stats import norm
from typing import Union, Optional, Dict, Any
from src.pricing.models import BSParameters, OptionGreeks
from .base import PricingStrategy

class BlackScholesEngine(PricingStrategy):
    @staticmethod
    def calculate_d1_d2(params: BSParameters):
        """
        Legacy method for calculating d1 and d2.
        Recommended to use internal JIT implementation for performance.
        """
        S = params.spot
        K = params.strike
        T = params.maturity
        r = params.rate
        sigma = params.volatility
        dividend = params.dividend

        if T <= 0:
            return 0.0, 0.0

        d1 = (np.log(S / K) + (r - dividend + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def price(self, params: Optional[BSParameters] = None, option_type: str = "call", **kwargs) -> Union[float, np.ndarray]:
        """Instance method for PricingStrategy interface. Delegates to optimized static method."""
        return self.price_options(params=params, option_type=option_type, **kwargs)

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Implementation of PricingStrategy.calculate_greeks. Delegates to optimized static method."""
        return self.calculate_greeks_batch(params=params, option_type=option_type)

    def calculate(self, params: BSParameters, type: str = 'call'):
        """
        Legacy method returning dict with price, delta, and gamma.
        Delegates to optimized methods.
        """
        price = self.price_options(params=params, option_type=type)
        greeks = self.calculate_greeks_batch(params=params, option_type=type)
        
        if isinstance(greeks, OptionGreeks):
            return {
                "price": price,
                "delta": greeks.delta,
                "gamma": greeks.gamma
            }
        else:
            return {
                "price": price,
                "delta": greeks['delta'],
                "gamma": greeks['gamma']
            }

    @staticmethod
    def price_call(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type='call'))

    @staticmethod
    def price_put(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type='put'))

    @staticmethod
    def calculate_greeks_static(params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Compatibility wrapper."""
        result = BlackScholesEngine.calculate_greeks_batch(params=params, option_type=option_type)
        if isinstance(result, dict):
             # This case shouldn't happen if params is a single object, but handling just in case
             return OptionGreeks(
                delta=float(result['delta'][0]),
                gamma=float(result['gamma'][0]),
                vega=float(result['vega'][0]),
                theta=float(result['theta'][0]),
                rho=float(result['rho'][0])
            )
        return result

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
        **kwargs
    ) -> Union[float, np.ndarray]:
        """Highly optimized vectorized option pricing."""
        from .quant_utils import batch_bs_price_jit
        
        if params is not None:
            spot = params.spot
            strike = params.strike
            maturity = params.maturity
            volatility = params.volatility
            rate = params.rate
            dividend = params.dividend

        if any(x is None for x in [spot, strike, maturity, volatility, rate]):
             raise TypeError("Missing required pricing parameters")
        
        # Broadcast all inputs to the same shape
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
            
        prices = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
        
        if np.isscalar(spot) and np.isscalar(strike) and np.isscalar(maturity) and \
           np.isscalar(volatility) and np.isscalar(rate) and np.isscalar(dividend) and \
           isinstance(option_type, str):
            return float(prices[0])
        return prices

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
        **kwargs
    ) -> Union[OptionGreeks, Dict[str, np.ndarray]]:
        """Highly optimized vectorized greeks calculation."""
        from .quant_utils import batch_greeks_jit
        
        if params is not None:
            spot = params.spot
            strike = params.strike
            maturity = params.maturity
            volatility = params.volatility
            rate = params.rate
            dividend = params.dividend

        if any(x is None for x in [spot, strike, maturity, volatility, rate]):
             raise TypeError("Missing required pricing parameters")
        
        # Broadcast all inputs to the same shape
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
            
        delta, gamma, vega, theta, rho = batch_greeks_jit(S, K, T, sigma, r, q, is_call)
        
        if np.isscalar(spot) and isinstance(option_type, str):
            return OptionGreeks(
                delta=float(delta[0]),
                gamma=float(gamma[0]),
                vega=float(vega[0]),
                theta=float(theta[0]),
                rho=float(rho[0])
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

def black_scholes(S, K, T, sigma, r, type='call', q=0.0):
    """Compatibility wrapper for black_scholes."""
    params = BSParameters(spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=q)
    engine = BlackScholesEngine()
    return engine.calculate(params, type)