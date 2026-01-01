from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class BSParameters:
    """
    Standard Black-Scholes model parameters.
    """

    spot: float
    strike: float
    maturity: float
    volatility: float
    rate: float
    dividend: float = 0.0

    def __post_init__(self):
        # Convert to numpy arrays or floats for precision and vectorized validation
        self.spot = np.asanyarray(self.spot, dtype=np.float64)
        self.strike = np.asanyarray(self.strike, dtype=np.float64)
        self.maturity = np.asanyarray(self.maturity, dtype=np.float64)
        self.volatility = np.asanyarray(self.volatility, dtype=np.float64)
        self.rate = np.asanyarray(self.rate, dtype=np.float64)
        self.dividend = np.asanyarray(self.dividend, dtype=np.float64)

        if np.any(self.spot <= 0) or np.any(self.strike <= 0) or np.any(self.volatility <= 0):
            raise ValueError("Spot, strike, and volatility must be positive")


@dataclass
class OptionGreeks:
    """
    Container for option sensitivity measures.
    """

    delta: Union[float, np.ndarray]
    gamma: Union[float, np.ndarray]
    vega: Union[float, np.ndarray]
    theta: Union[float, np.ndarray]
    rho: Union[float, np.ndarray]

    def __getitem__(self, item):
        return getattr(self, item)
