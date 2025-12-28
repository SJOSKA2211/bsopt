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
        # Convert to float64 for precision and validate
        self.spot = float(self.spot)
        self.strike = float(self.strike)
        self.maturity = float(self.maturity)
        self.volatility = float(self.volatility)
        self.rate = float(self.rate)
        self.dividend = float(self.dividend)

        if self.spot <= 0:
            raise ValueError("Spot price must be positive")
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.maturity < 0:
            raise ValueError("Time to maturity must be non-negative")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")

        # Specific check for n_steps if called from Lattice
        if hasattr(self, "n_steps") and getattr(self, "n_steps") < 1:
            raise ValueError("Number of steps must be at least 1")


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
