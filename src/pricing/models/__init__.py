from dataclasses import dataclass
from typing import Union, Optional
import numpy as np

@dataclass
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
            raise ValueError("Maturity must be non-negative")
        if np.any(_rate_arr < 0) or np.any(_dividend_arr < 0):
            raise ValueError("Rate and dividend must be non-negative")

@dataclass
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
        return getattr(self, item)