from dataclasses import dataclass
from typing import Union, Dict, Any, Optional
import numpy as np
from scipy.stats import norm

@dataclass
class BSParameters:
    spot: Union[float, np.ndarray]
    strike: Union[float, np.ndarray]
    maturity: Union[float, np.ndarray]
    volatility: Union[float, np.ndarray]
    rate: Union[float, np.ndarray]
    dividend: Union[float, np.ndarray] = 0.0

@dataclass
class OptionGreeks:
    delta: Union[float, np.ndarray]
    gamma: Union[float, np.ndarray]
    vega: Union[float, np.ndarray]
    theta: Union[float, np.ndarray]
    rho: Union[float, np.ndarray]

class BlackScholesEngine:
    @staticmethod
    def _price_options_static(params: BSParameters, option_type: Union[str, np.ndarray]) -> Union[float, np.ndarray]:
        S = np.array(params.spot, dtype=float)
        K = np.array(params.strike, dtype=float)
        T = np.array(params.maturity, dtype=float)
        r = np.array(params.rate, dtype=float)
        q = np.array(params.dividend, dtype=float)
        sigma = np.array(params.volatility, dtype=float)
        
        # Avoid division by zero warnings
        sigma_sqrt_T = sigma * np.sqrt(T)
        
        d1 = np.zeros_like(S)
        d2 = np.zeros_like(S)
        
        # Only compute where possible
        mask = (T > 0) & (sigma > 0)
        
        if np.any(mask):
            with np.errstate(divide='ignore', invalid='ignore'):
                 d1[mask] = (np.log(S[mask] / K[mask]) + (r[mask] - q[mask] + 0.5 * sigma[mask]**2) * T[mask]) / sigma_sqrt_T[mask]
                 d2[mask] = d1[mask] - sigma_sqrt_T[mask]
            
        exp_mqT = np.exp(-q * T)
        exp_mrT = np.exp(-r * T)
        
        is_call = (option_type == "call") if isinstance(option_type, str) else (np.array(option_type) == "call")
        
        # Calculate for Call
        price_call = S * exp_mqT * norm.cdf(d1) - K * exp_mrT * norm.cdf(d2)
        
        # Calculate for Put
        price_put = K * exp_mrT * norm.cdf(-d2) - S * exp_mqT * norm.cdf(-d1)
        
        if isinstance(option_type, str):
            res = price_call if option_type == "call" else price_put
        else:
             res = np.where(is_call, price_call, price_put)

        # Handle intrinsic value for T=0 or sigma=0
        if np.any(~mask):
            intrinsic_call = np.maximum(S - K, 0.0)
            intrinsic_put = np.maximum(K - S, 0.0)
            
            if isinstance(option_type, str):
                 res[~mask] = intrinsic_call[~mask] if option_type == "call" else intrinsic_put[~mask]
            else:
                 res[~mask] = np.where(is_call[~mask], intrinsic_call[~mask], intrinsic_put[~mask])

        if S.ndim == 0:
            return float(res)
        return res

    @staticmethod
    def price_options(
        params: Optional[BSParameters] = None,
        spot: Union[float, np.ndarray] = None,
        strike: Union[float, np.ndarray] = None,
        maturity: Union[float, np.ndarray] = None,
        volatility: Union[float, np.ndarray] = None,
        rate: Union[float, np.ndarray] = None,
        dividend: Union[float, np.ndarray] = 0.0,
        option_type: Union[str, np.ndarray] = "call",
        **kwargs
    ) -> Union[float, np.ndarray]:
        if params is None:
             # If any of the required fields are missing in args, check kwargs
             # But here we rely on the fact that numpy/float conversion will fail later if None
             # or we can construct safely.
             
             # Fallback to kwargs if explicit args are None
             spot = spot if spot is not None else kwargs.get("spot")
             strike = strike if strike is not None else kwargs.get("strike")
             maturity = maturity if maturity is not None else kwargs.get("maturity")
             volatility = volatility if volatility is not None else kwargs.get("volatility")
             rate = rate if rate is not None else kwargs.get("rate")
             dividend = dividend if dividend is not None else kwargs.get("dividend", 0.0)
             
             params = BSParameters(spot, strike, maturity, volatility, rate, dividend)
        
        return BlackScholesEngine._price_options_static(params, option_type)

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        # Instance method
        return self._calculate_greeks_static(params, option_type)
        
    def calculate_greeks_batch(self, 
        spot: Union[float, np.ndarray] = None,
        strike: Union[float, np.ndarray] = None,
        maturity: Union[float, np.ndarray] = None,
        volatility: Union[float, np.ndarray] = None,
        rate: Union[float, np.ndarray] = None,
        dividend: Union[float, np.ndarray] = 0.0,
        option_type: Union[str, np.ndarray] = "call",
        params: Optional[BSParameters] = None,
        **kwargs
    ) -> OptionGreeks:
        if params is None:
             params = BSParameters(spot, strike, maturity, volatility, rate, dividend)
        return self._calculate_greeks_static(params, option_type)

    @staticmethod
    def _calculate_greeks_static(params: BSParameters, option_type: Union[str, np.ndarray]) -> OptionGreeks:
        S = np.array(params.spot, dtype=float)
        K = np.array(params.strike, dtype=float)
        T = np.array(params.maturity, dtype=float)
        r = np.array(params.rate, dtype=float)
        q = np.array(params.dividend, dtype=float)
        sigma = np.array(params.volatility, dtype=float)

        sigma_sqrt_T = sigma * np.sqrt(T)
        
        d1 = np.zeros_like(S)
        d2 = np.zeros_like(S)
        
        mask = (T > 0) & (sigma > 0)
        
        if np.any(mask):
            with np.errstate(divide='ignore', invalid='ignore'):
                 d1[mask] = (np.log(S[mask] / K[mask]) + (r[mask] - q[mask] + 0.5 * sigma[mask]**2) * T[mask]) / sigma_sqrt_T[mask]
                 d2[mask] = d1[mask] - sigma_sqrt_T[mask]

        exp_mqT = np.exp(-q * T)
        exp_mrT = np.exp(-r * T)
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)
        
        is_call = (option_type == "call") if isinstance(option_type, str) else (np.array(option_type) == "call")

        # Initial greeks arrays
        delta = np.zeros_like(S)
        gamma = np.zeros_like(S)
        vega = np.zeros_like(S)
        theta = np.zeros_like(S)
        rho = np.zeros_like(S)
        
        if np.any(mask):
             # Delta
             if isinstance(option_type, str):
                 if option_type == "call":
                     delta[mask] = exp_mqT[mask] * cdf_d1[mask]
                 else:
                     delta[mask] = -exp_mqT[mask] * cdf_neg_d1[mask]
             else:
                 delta[mask] = np.where(is_call[mask], exp_mqT[mask] * cdf_d1[mask], -exp_mqT[mask] * cdf_neg_d1[mask])

             # Gamma (same)
             gamma[mask] = (exp_mqT[mask] * pdf_d1[mask]) / (S[mask] * sigma_sqrt_T[mask])
             
             # Vega (same)
             vega[mask] = S[mask] * exp_mqT[mask] * pdf_d1[mask] * np.sqrt(T[mask])
             
             # Theta
             term1 = -(S[mask] * sigma[mask] * exp_mqT[mask] * pdf_d1[mask]) / (2 * np.sqrt(T[mask]))
             
             if isinstance(option_type, str):
                 if option_type == "call":
                      theta[mask] = term1 - r[mask] * K[mask] * exp_mrT[mask] * cdf_d2[mask] + q[mask] * S[mask] * exp_mqT[mask] * cdf_d1[mask]
                 else:
                      theta[mask] = term1 + r[mask] * K[mask] * exp_mrT[mask] * cdf_neg_d2[mask] - q[mask] * S[mask] * exp_mqT[mask] * cdf_neg_d1[mask]
             else:
                  theta_call = term1 - r[mask] * K[mask] * exp_mrT[mask] * cdf_d2[mask] + q[mask] * S[mask] * exp_mqT[mask] * cdf_d1[mask]
                  theta_put = term1 + r[mask] * K[mask] * exp_mrT[mask] * cdf_neg_d2[mask] - q[mask] * S[mask] * exp_mqT[mask] * cdf_neg_d1[mask]
                  theta[mask] = np.where(is_call[mask], theta_call, theta_put)

             # Rho
             if isinstance(option_type, str):
                 if option_type == "call":
                      rho[mask] = K[mask] * T[mask] * exp_mrT[mask] * cdf_d2[mask]
                 else:
                      rho[mask] = -K[mask] * T[mask] * exp_mrT[mask] * cdf_neg_d2[mask]
             else:
                 rho[mask] = np.where(is_call[mask], K[mask] * T[mask] * exp_mrT[mask] * cdf_d2[mask], -K[mask] * T[mask] * exp_mrT[mask] * cdf_neg_d2[mask])

        # Handle scalar return
        if S.ndim == 0:
            return OptionGreeks(
                delta=float(delta),
                gamma=float(gamma),
                vega=float(vega),
                theta=float(theta),
                rho=float(rho)
            )
        
        return OptionGreeks(delta, gamma, vega, theta, rho)