import numpy as np
from typing import Dict

class MonteCarloEngine:
    """Classical Monte Carlo engine for option pricing."""
    
    def price_european(self, S0: float, K: float, T: float, r: float, sigma: float, num_simulations: int = 100000, **kwargs) -> Dict[str, float]:
        """Price European call option using classical Monte Carlo."""
        dt = T
        Z = np.random.standard_normal(num_simulations)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        payoffs = np.maximum(ST - K, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return {
            "price": float(price),
            "method": "classical_monte_carlo",
            "simulations": num_simulations
        }
