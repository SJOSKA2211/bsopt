from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import threading

class ModelPool:
    """
    Thread-safe object pool for high-frequency pricing objects.
    Reduces object creation and GC pressure in tight loops.
    """
    def __init__(self):
        self._bs_pool = []
        self._greeks_pool = []
        self._lock = threading.Lock()
        self._max_size = 50000

    def get_bs_params(self, **kwargs) -> 'BSParameters':
        with self._lock:
            if self._bs_pool:
                obj = self._bs_pool.pop()
                for k, v in kwargs.items():
                    setattr(obj, k, v)
                return obj
        return BSParameters(**kwargs)

    def release_bs_params(self, obj: 'BSParameters'):
        with self._lock:
            if len(self._bs_pool) < self._max_size:
                self._bs_pool.append(obj)

    def get_greeks(self, **kwargs) -> 'OptionGreeks':
        with self._lock:
            if self._greeks_pool:
                obj = self._greeks_pool.pop()
                for k, v in kwargs.items():
                    setattr(obj, k, v)
                return obj
        return OptionGreeks(**kwargs)

    def release_greeks(self, obj: 'OptionGreeks'):
        with self._lock:
            if len(self._greeks_pool) < self._max_size:
                self._greeks_pool.append(obj)

# Global pool instance
global_model_pool = ModelPool()

@dataclass(slots=True)
class BSParameters:
    """
    Standard Black-Scholes model parameters with validation.
    """
    spot: float
    strike: float
    maturity: float
    volatility: float
    rate: float
    dividend: float = 0.0

    def __post_init__(self):
        # Validate parameters.
        _spot_arr = np.asanyarray(self.spot, dtype=np.float64)
        _strike_arr = np.asanyarray(self.strike, dtype=np.float64)
        _volatility_arr = np.asanyarray(self.volatility, dtype=np.float64)
        _maturity_arr = np.asanyarray(self.maturity, dtype=np.float64)
        _rate_arr = np.asanyarray(self.rate, dtype=np.float64)
        _dividend_arr = np.asanyarray(self.dividend, dtype=np.float64)

        if np.any(_spot_arr < 0) or np.any(_strike_arr < 0) or np.any(_volatility_arr < 0):
            raise ValueError("Spot, strike, and volatility must be non-negative")
        if np.any(_maturity_arr < 0):
            raise ValueError("Maturity cannot be negative")
        if np.any(_rate_arr < 0) or np.any(_dividend_arr < 0):
            raise ValueError("Rate and dividend cannot be negative")

@dataclass(slots=True)
class OptionGreeks:
    """
    Container for option sensitivity measures.
    """
    delta: Union[float, np.ndarray]
    gamma: Union[float, np.ndarray]
    theta: Union[float, np.ndarray]
    vega: Union[float, np.ndarray]
    rho: Union[float, np.ndarray]
    phi: Optional[Union[float, np.ndarray]] = None

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        raise TypeError(f"OptionGreeks indices must be strings, not {type(item).__name__}")

        def __contains__(self, item):

            return hasattr(self, item)

    

@dataclass(frozen=True, slots=True)
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

    