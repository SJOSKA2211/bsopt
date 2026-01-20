import numpy as np
from scipy.stats import norm
from typing import Union, Optional, Dict
from src.pricing.models import BSParameters, OptionGreeks
from .base import PricingStrategy

class BlackScholesEngine(PricingStrategy):
    @staticmethod
    def calculate_d1_d2(params: BSParameters):
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
        """Instance method for PricingStrategy interface."""
        if params is None:
            params = BSParameters(**kwargs)
        return self.calculate(params, option_type)['price']




    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Implementation of PricingStrategy.calculate_greeks."""
        return self.calculate_greeks_static(params, option_type)

    def calculate(self, params: BSParameters, type: str = 'call'):
        S = params.spot
        K = params.strike
        T = params.maturity
        r = params.rate
        sigma = params.volatility
        dividend = params.dividend

        if np.any(T <= 0):
            prices = np.where(T <= 0, 
                              np.where(type == 'call', np.maximum(S - K, 0.0), np.maximum(K - S, 0.0)),
                              0.0)
            if np.all(T <= 0):
                return {"price": prices, "delta": np.zeros_like(S), "gamma": np.zeros_like(S)}


        d1, d2 = self.calculate_d1_d2(params)

        if type == 'call':
            price = S * np.exp(-dividend * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = np.exp(-dividend * T) * norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-dividend * T) * norm.cdf(-d1)
            delta = np.exp(-dividend * T) * (norm.cdf(d1) - 1)

        gamma = np.exp(-dividend * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return {
            "price": price,
            "delta": delta,
            "gamma": gamma
        }


    @staticmethod
    def price_call(params: BSParameters) -> float:
        engine = BlackScholesEngine()
        return engine.calculate(params, 'call')['price']

    @staticmethod
    def price_put(params: BSParameters) -> float:
        engine = BlackScholesEngine()
        return engine.calculate(params, 'put')['price']

    @staticmethod
    def _calculate_greeks_internal(params: BSParameters, option_type: str = "call") -> OptionGreeks:
        engine = BlackScholesEngine()
        res = engine.calculate(params, option_type)
        S = params.spot
        K = params.strike
        T = params.maturity
        r = params.rate
        sigma = params.volatility
        dividend = params.dividend
        
        d1, _ = BlackScholesEngine.calculate_d1_d2(params)
        
        if T > 0:
            vega = S * np.exp(-dividend * T) * norm.pdf(d1) * np.sqrt(T)
            
            if option_type == 'call':
                theta = -(S * sigma * np.exp(-dividend * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) - \
                        r * K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T)) + \
                        dividend * S * np.exp(-dividend * T) * norm.cdf(d1)
                rho = K * T * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))
            else:
                theta = -(S * sigma * np.exp(-dividend * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) + \
                        r * K * np.exp(-r * T) * norm.cdf(-(d1 - sigma * np.sqrt(T))) - \
                        dividend * S * np.exp(-dividend * T) * norm.cdf(-d1)
                rho = -K * T * np.exp(-r * T) * norm.cdf(-(d1 - sigma * np.sqrt(T)))
        else:
            vega = 0.0
            theta = 0.0
            rho = 0.0

        return OptionGreeks(
            delta=res['delta'],
            gamma=res['gamma'],
            vega=float(vega),
            theta=float(theta),
            rho=float(rho)
        )

    # Maintain calculate_greeks as static for existing callers
    @staticmethod
    def calculate_greeks_static(params: BSParameters, option_type: str = "call") -> OptionGreeks:
        return BlackScholesEngine._calculate_greeks_internal(params, option_type)

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