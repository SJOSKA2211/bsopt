import numpy as np
from scipy.integrate import quad
from typing import Optional, Dict
import structlog
from dataclasses import dataclass
from functools import lru_cache

logger = structlog.get_logger()

@dataclass(frozen=True)
class HestonParams:
    """Immutable Heston parameters with validation."""
    v0: float      # Initial variance
    kappa: float   # Mean reversion speed
    theta: float   # Long-term variance
    sigma: float   # Vol of vol
    rho: float     # Correlation [-1, 1]

    def __post_init__(self):
        """Validate Feller condition and parameter bounds."""
        if not (2 * self.kappa * self.theta > self.sigma**2):
            raise ValueError(f"Feller condition violated: 2κθ={2*self.kappa*self.theta:.4f} <= σ²={self.sigma**2:.4f}")
        
        if not (-1 <= self.rho <= 1):
            raise ValueError(f"Correlation must be in [-1,1], got {self.rho}")
            
        if any(p <= 0 for p in [self.v0, self.kappa, self.theta, self.sigma]):
            raise ValueError("All parameters except rho must be positive")

class HestonModelFFT:
    """
    Heston Stochastic Volatility Model using Carr-Madan FFT.
    References:
    - Heston (1993): "A Closed-Form Solution for Options with Stochastic Volatility"
    - Carr & Madan (1999): "Option Valuation Using FFT"
    - Albrecher et al. (2007): "The Little Heston Trap"
    """
    # Numerical stability constants
    MAX_INTEGRATION_BOUND = 500.0
    MIN_PRICE = 1e-10
    MAX_CHAR_FUNC_ABS = 1e15

    def __init__(self, params: HestonParams, r: float, T: float):
        self.params = params
        self.r = r
        self.T = T
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate risk-free rate and time to maturity."""
        if not (-0.1 <= self.r <= 0.5):
            logger.warning("unusual_risk_free_rate", rate=self.r)
        if not (0 < self.T <= 10):
            raise ValueError(f"Time to maturity must be in (0, 10] years, got {self.T}")

    @lru_cache(maxsize=1024)
    def characteristic_func(self, u: complex) -> complex:
        """
        Heston characteristic function using JIT-optimized backend.
        """
        from src.pricing.quant_utils import heston_char_func_jit
        p = self.params
        
        try:
            phi = heston_char_func_jit(
                u, self.T, self.r, p.v0, p.kappa, p.theta, p.sigma, p.rho
            )
            
            # Sanity check
            if abs(phi) > self.MAX_CHAR_FUNC_ABS:
                return 0.0 + 0.0j
                
            return phi
        except Exception:
            return 0.0 + 0.0j

    def batch_characteristic_func(self, u_array: np.ndarray) -> np.ndarray:
        """
        Efficiently evaluate the characteristic function over an array of values.
        """
        # Vectorized evaluation using JIT backend
        from src.pricing.quant_utils import heston_char_func_jit
        p = self.params
        
        res = np.empty(len(u_array), dtype=np.complex128)
        for i in range(len(u_array)):
            res[i] = heston_char_func_jit(
                u_array[i], self.T, self.r, p.v0, p.kappa, p.theta, p.sigma, p.rho
            )
        return res

    def price_call(self, S0: float, K: float, alpha: float = 1.5) -> float:
        """
        Price European call using Carr-Madan FFT method.
        """
        if S0 <= 0 or K <= 0:
            raise ValueError(f"Prices must be positive: S0={S0}, K={K}")
            
        # For ITM call, use parity to price OTM put
        if K < S0:
            return self.price_put(S0, K, alpha=-2.5) + S0 - K * np.exp(-self.r * self.T)

        k = np.log(K / S0)
        
        def integrand(v: float) -> float:
            """Modified characteristic function for FFT."""
            try:
                cf = self.characteristic_func(v - (alpha + 1) * 1j)
                num = np.exp(-1j * v * k) * cf
                den = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
                if abs(den) < 1e-12:
                    return 0.0
                return np.real(num / den)
            except Exception as e:
                logger.warning("integrand_error", v=v, error=str(e))
                return 0.0

        try:
            integral, error_estimate = quad(
                integrand, 0, self.MAX_INTEGRATION_BOUND, 
                limit=100, epsabs=1e-8, epsrel=1e-6
            )
            if error_estimate > 1e-3:
                logger.warning("integration_error_high", error=error_estimate)
                
            price = (np.exp(-alpha * k) / np.pi) * integral
            discounted_price = np.exp(-self.r * self.T) * S0 * price
            
            # Apply floor and sanity bounds
            if discounted_price < self.MIN_PRICE:
                return self.MIN_PRICE
                
            # European call cannot exceed spot price
            if discounted_price > S0:
                logger.error("price_exceeds_spot", price=discounted_price, spot=S0)
                return max(S0 - K * np.exp(-self.r * self.T), self.MIN_PRICE)
                
            return discounted_price
        except Exception as e:
            logger.error("pricing_failed", error=str(e), S0=S0, K=K)
            # Fallback to intrinsic value
            return max(S0 - K * np.exp(-self.r * self.T), self.MIN_PRICE)

    def price_put(self, S0: float, K: float, alpha: float = -2.5) -> float:
        """Price European put. Uses parity for ITM puts."""
        if K > S0: # ITM Put
            return self.price_call(S0, K, alpha=1.5) - S0 + K * np.exp(-self.r * self.T)
            
        # OTM Put (K <= S0)
        k = np.log(K / S0)
        
        def integrand(v: float) -> float:
            try:
                # Same integrand structure works for puts with alpha < -1
                cf = self.characteristic_func(v - (alpha + 1) * 1j)
                num = np.exp(-1j * v * k) * cf
                den = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
                return np.real(num / den)
            except:
                return 0.0

        try:
            integral, _ = quad(integrand, 0, self.MAX_INTEGRATION_BOUND)
            price = (np.exp(-alpha * k) / np.pi) * integral
            discounted_price = np.exp(-self.r * self.T) * S0 * price
            return max(discounted_price, self.MIN_PRICE)
        except Exception as e:
            logger.error("put_pricing_failed", error=str(e), S0=S0, K=K)
            return max(K * np.exp(-self.r * self.T) - S0, self.MIN_PRICE)