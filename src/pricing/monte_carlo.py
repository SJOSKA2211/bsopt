import numpy as np
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.stats import norm
from src.pricing.models import BSParameters, OptionGreeks
from src.pricing.quant_utils import jit_lsm_american, jit_mc_european_price, jit_mc_european_with_control_variate, _laguerre_basis_jit
from .base import PricingStrategy

@dataclass
class MCConfig:
    n_paths: int = 100000
    n_steps: int = 252
    antithetic: bool = True
    control_variate: bool = True
    seed: int = 42
    method: str = "monte_carlo"

    def __post_init__(self):
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        
        # If antithetic is True, ensure n_paths is even
        if self.antithetic and self.n_paths % 2 != 0:
            self.n_paths += 1
            
        if self.method == "sobol":
            # Sobol works best with powers of 2 for some implementations
            # The test expected rounding to next power of 2 for method='sobol'
            self.n_paths = 2**int(np.ceil(np.log2(self.n_paths)))

class MonteCarloEngine(PricingStrategy):
    """
    Advanced Monte Carlo engine for option pricing.
    Supports European and American options with variance reduction.
    """
    
    def __init__(self, config: Optional[MCConfig] = None):
        self.config = config or MCConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """Implementation of PricingStrategy.price."""
        price, _ = self.price_european(params, option_type)
        return price

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """
        Implementation of PricingStrategy.calculate_greeks using finite differences.
        Uses Common Random Numbers (CRN) to minimize variance in sensitivities.
        """
        seed = self.config.seed
        
        # Delta & Gamma (Central Difference)
        # Using a relatively large shift for stability in stochastic environments
        ds = max(params.spot * 0.001, 0.01)
        
        p_plus, _ = self.price_european(
            params.replace(spot=params.spot + ds), option_type, seed=seed
        )
        p_minus, _ = self.price_european(
            params.replace(spot=params.spot - ds), option_type, seed=seed
        )
        p_base, _ = self.price_european(params, option_type, seed=seed)
        
        delta = (p_plus - p_minus) / (2 * ds)
        gamma = (p_plus - 2 * p_base + p_minus) / (ds**2)
        
        # Vega (per 1% vol change)
        dvol = 0.001
        p_vol_plus, _ = self.price_european(
            params.replace(volatility=params.volatility + dvol), option_type, seed=seed
        )
        vega = (p_vol_plus - p_base) / (dvol * 100)
        
        # Theta (per day change)
        dt = 1.0 / 365.0
        if params.maturity > dt:
            p_theta, _ = self.price_european(
                params.replace(maturity=params.maturity - dt), option_type, seed=seed
            )
            theta = (p_theta - p_base)
        else:
            theta = 0.0 # Terminal theta handled by boundary checks
            
        # Rho (per 1% rate change)
        dr = 0.0001
        p_rho_plus, _ = self.price_european(
            params.replace(rate=params.rate + dr), option_type, seed=seed
        )
        rho = (p_rho_plus - p_base) / (dr * 100)
        
        return OptionGreeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho)
        )

    def price_european(
        self, 
        params: BSParameters, 
        option_type: str = "call",
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Price European option using JIT-optimized Monte Carlo.
        Returns (price, confidence_interval).
        """
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        if params.maturity <= 0:
            price = max(params.spot - params.strike, 0.0) if option_type == "call" else max(params.strike - params.spot, 0.0)
            return float(price), 0.0

        # Use the provided seed or the configured default
        run_seed = seed if seed is not None else self.config.seed
        np.random.seed(run_seed) # Set global seed for Numba's np.random

        if self.config.control_variate:
            price, std_err = jit_mc_european_with_control_variate(
                S0=float(params.spot),
                K=float(params.strike),
                T=float(params.maturity),
                r=float(params.rate),
                sigma=float(params.volatility),
                q=float(params.dividend),
                n_paths=self.config.n_paths,
                is_call=(option_type == "call"),
                antithetic=self.config.antithetic
            )
        else:
            price, std_err = jit_mc_european_price(
                S0=float(params.spot),
                K=float(params.strike),
                T=float(params.maturity),
                r=float(params.rate),
                sigma=float(params.volatility),
                q=float(params.dividend),
                n_paths=self.config.n_paths,
                is_call=(option_type == "call"),
                antithetic=self.config.antithetic
            )

        ci = 1.96 * std_err  # 95% confidence interval
        return float(price), float(ci)

    def _generate_random_normals(self, n_paths: int, n_steps: int) -> np.ndarray:
        """
        Generate random normal variables for simulation.
        Supports standard Monte Carlo and Sobol sequences.
        """
        if self.config.method == "sobol":
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_steps, scramble=True, seed=self.config.seed)
            sample = sampler.random(n_paths)
            return norm.ppf(sample)
        
        return self.rng.standard_normal((n_paths, n_steps))

    def price_american_lsm(self, params: BSParameters, option_type: str = "call") -> float:
        """
        Price American option using JIT-optimized Longstaff-Schwartz Least Squares Monte Carlo.
        """
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        is_call = (option_type == "call")
        
        if params.maturity <= 0:
            if is_call:
                return max(params.spot - params.strike, 0.0)
            else:
                return max(params.strike - params.spot, 0.0)

        return jit_lsm_american(
            S0=float(params.spot),
            K=float(params.strike),
            T=float(params.maturity),
            r=float(params.rate),
            sigma=float(params.volatility),
            q=float(params.dividend),
            n_paths=self.config.n_paths,
            n_steps=self.config.n_steps,
            is_call=is_call
        )

def _laguerre_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Generates basis functions based on test expectations.
    Wrapper around JIT implementation for compatibility.
    """
    return _laguerre_basis_jit(x, degree)

def geometric_asian_price(params: BSParameters, option_type: str, n_obs: int) -> float:
    """
    Calculates the price of a European Asian option with geometric average.
    Analytic solution for geometric Asian option.
    """
    if n_obs <= 0:
        raise ValueError("Number of observations must be positive.")

    T_prime = params.maturity * (n_obs + 1) / (2 * n_obs)
    sigma_prime = params.volatility * np.sqrt((n_obs + 1) * (2 * n_obs + 1) / (6 * n_obs**2))
    mu_prime = params.rate - 0.5 * params.volatility**2 + 0.5 * sigma_prime**2

    d1 = (np.log(params.spot / params.strike) + (mu_prime + 0.5 * sigma_prime**2) * T_prime) / (
        sigma_prime * np.sqrt(T_prime)
    )
    d2 = d1 - sigma_prime * np.sqrt(T_prime)

    if option_type == "call":
        price = params.spot * np.exp((mu_prime - params.rate) * T_prime) * norm.cdf(d1) - \
                params.strike * np.exp(-params.rate * T_prime) * norm.cdf(d2)
    elif option_type == "put":
        price = params.strike * np.exp(-params.rate * T_prime) * norm.cdf(-d2) - \
                params.spot * np.exp((mu_prime - params.rate) * T_prime) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

    return float(price)