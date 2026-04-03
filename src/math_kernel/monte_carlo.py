import dataclasses
from dataclasses import dataclass

import numpy as np

from src.config import settings
from src.math_kernel.base import PricingStrategy
from src.math_kernel.models import BSParameters, OptionGreeks
from src.math_kernel.quant_utils import (
    _laguerre_basis_jit,
    gpu_mc_european_price,
    jit_lsm_american,
    jit_mc_european_price,
    jit_mc_european_price_and_greeks,
    jit_mc_european_with_control_variate,
)

try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


@dataclass
class MCConfig:
    n_paths: int = 100000
    n_steps: int = 252
    antithetic: bool = True
    control_variate: bool = True
    seed: int = 42
    method: str = "monte_carlo"
    scheme: str = "euler"  # "euler" or "milstein"

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
            self.n_paths = 2 ** int(np.ceil(np.log2(self.n_paths)))


class MonteCarloEngine(PricingStrategy):  # optimized
    """
    Advanced Monte Carlo engine for option pricing.
    Supports European and American options with variance reduction techniques.
    """

    def __init__(self, config: MCConfig | None = None):
        self.config = config or MCConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def price_european(self, params: BSParameters, option_type: str = "call") -> float:
        """Implementation of PricingStrategy.price_european."""
        price, _ = self.simulate_european(params, option_type)
        return price

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """
        OPTIMIZATION: Unified pathwise sensitivities.
        Calculates price and Greeks in a single pass where possible.
        """
        # If using Sobol or Control Variate, fallback to FD for now (or implement PWM for them)
        # But for standard MC, use PWM.
        if (
            self.config.control_variate
            or self.config.method == "sobol"
            or self.config.scheme == "milstein"
        ):
            # Fallback to Finite Difference for complex methods not yet ported to PWM kernel
            return self._calculate_greeks_fd(params, option_type)

        price, delta, gamma, vega, rho = jit_mc_european_price_and_greeks(
            S0=float(params.spot),
            K=float(params.strike),
            T=float(params.maturity),
            r=float(params.rate),
            sigma=float(params.volatility),
            q=float(params.dividend),
            n_paths=self.config.n_paths,
            is_call=(option_type == "call"),
            antithetic=self.config.antithetic,
            scheme=self.config.scheme,
        )

        # Theta via PDE (consistent with other Greeks)
        theta_analytic = (
            params.rate * price
            - (params.rate - params.dividend) * params.spot * delta
            - 0.5 * (params.volatility**2) * (params.spot**2) * gamma
        )

        return OptionGreeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta_analytic / 365.0),  # Approximation via PDE
            rho=float(rho),
        )

    def _calculate_greeks_fd(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Fallback Finite Difference implementation."""
        seed = self.config.seed

        # Optimization: Pre-generate random numbers to reuse across all pricing calls
        # This significantly reduces computational overhead for finite difference
        # and ensures Common Random Numbers (CRN) are used for better stability.
        n_needed = self.config.n_paths // 2 if self.config.antithetic else self.config.n_paths
        run_seed = seed if seed is not None else self.config.seed

        if self.config.method == "sobol":
            # For European options, dimension d=1 (time step)
            z_innovations = self._generate_random_normals(n_needed, 1).flatten()
        else:
            # Standard MC uses global np.random inside JIT currently
            # We generate them here to freeze them for all perturbations
            np.random.seed(run_seed)
            z_innovations = np.random.standard_normal(n_needed)

        ds = max(params.spot * 0.001, 0.01)
        p_base, _ = self.simulate_european(
            params, option_type, seed=seed, z_innovations=z_innovations
        )
        p_plus, _ = self.simulate_european(
            dataclasses.replace(params, spot=params.spot + ds),
            option_type,
            seed=seed,
            z_innovations=z_innovations,
        )
        p_minus, _ = self.simulate_european(
            dataclasses.replace(params, spot=params.spot - ds),
            option_type,
            seed=seed,
            z_innovations=z_innovations,
        )
        delta = (p_plus - p_minus) / (2 * ds)
        gamma = (p_plus - 2 * p_base + p_minus) / (ds**2)

        dvol = 0.001
        p_vol_plus, _ = self.simulate_european(
            dataclasses.replace(params, volatility=params.volatility + dvol),
            option_type,
            seed=seed,
            z_innovations=z_innovations,
        )
        vega = (p_vol_plus - p_base) / (dvol * 100)

        dt = 1.0 / 365.0
        theta = 0.0
        if params.maturity > dt:
            p_theta, _ = self.simulate_european(
                dataclasses.replace(params, maturity=params.maturity - dt),
                option_type,
                seed=seed,
                z_innovations=z_innovations,
            )
            theta = p_theta - p_base

        dr = 0.0001
        p_rho_plus, _ = self.simulate_european(
            dataclasses.replace(params, rate=params.rate + dr),
            option_type,
            seed=seed,
            z_innovations=z_innovations,
        )
        rho = (p_rho_plus - p_base) / (dr * 100)

        return OptionGreeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
        )

    def price_batch(
        self,
        spots: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        vols: np.ndarray,
        rates: np.ndarray,
        dividends: np.ndarray,
        option_types: np.ndarray,
    ) -> np.ndarray:
        """Vectorized batch pricing for Monte Carlo."""
        is_call = (option_types == "call") | (option_types == "CALL")
        from src.math_kernel.quant_utils import batch_mc_european_price_and_greeks

        prices, _, _, _, _ = batch_mc_european_price_and_greeks(
            spots,
            strikes,
            maturities,
            rates,
            vols,
            dividends,
            self.config.n_paths,
            is_call,
            self.config.antithetic,
        )
        return prices

    def price_batch_greeks(
        self,
        spots: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        vols: np.ndarray,
        rates: np.ndarray,
        dividends: np.ndarray,
        option_types: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized batch Greeks for Monte Carlo."""
        is_call = (option_types == "call") | (option_types == "CALL")
        from src.math_kernel.quant_utils import batch_mc_european_price_and_greeks

        return batch_mc_european_price_and_greeks(
            spots,
            strikes,
            maturities,
            rates,
            vols,
            dividends,
            self.config.n_paths,
            is_call,
            self.config.antithetic,
        )

    def simulate_european(
        self,
        params: BSParameters,
        option_type: str = "call",
        seed: int | None = None,
        z_innovations: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """
        Price European option using JIT-optimized Monte Carlo.
        Returns (price, confidence_interval).
        """
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        if params.maturity <= 0:
            price = (
                max(params.spot - params.strike, 0.0)
                if option_type == "call"
                else max(params.strike - params.spot, 0.0)
            )
            return float(price), 0.0

        # Use the provided seed or the configured default
        run_seed = seed if seed is not None else self.config.seed
        np.random.seed(run_seed)  # Set global seed for Numba's np.random

        # High-performance path: Use GPU if n_paths is large and no custom innovations/control variates
        if (
            self.config.n_paths >= settings.MONTE_CARLO_GPU_THRESHOLD
            and not self.config.control_variate
            and not self.config.method == "sobol"
            and z_innovations is None
        ):
            gpu_res = gpu_mc_european_price(
                S0=float(params.spot),
                K=float(params.strike),
                T=float(params.maturity),
                r=float(params.rate),
                sigma=float(params.volatility),
                q=float(params.dividend),
                n_paths=self.config.n_paths,
                is_call=(option_type == "call"),
                antithetic=self.config.antithetic,
            )
            if gpu_res:
                price, std_err = gpu_res
                return float(price), float(1.96 * std_err)

        # Generate quasi-random numbers if method is 'sobol' and not provided
        if z_innovations is None and self.config.method == "sobol":
            n_needed = self.config.n_paths // 2 if self.config.antithetic else self.config.n_paths
            # For European options, dimension d=1 (time step)
            # Flatten to 1D array for JIT consumption
            z_innovations = self._generate_random_normals(n_needed, 1).flatten()

        if (
            CORE_AVAILABLE
            and not self.config.control_variate
            and self.config.method != "sobol"
            and z_innovations is None
        ):
            try:
                # Optimized Rust path (GIL-free)
                price = bsopt_core.monte_carlo_price(
                    float(params.spot),
                    float(params.strike),
                    float(params.maturity),
                    float(params.volatility),
                    float(params.rate),
                    float(params.dividend),
                    bool(option_type == "call"),
                    int(self.config.n_paths),
                )
                return float(price), 0.0  # std_err omitted in Rust impl for now
            except Exception:
                # logger.warning("rust_mc_failed_falling_back", error=str(e))
                pass

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
                antithetic=self.config.antithetic,
                z_innovations=z_innovations,
                scheme=self.config.scheme,
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
                antithetic=self.config.antithetic,
                z_innovations=z_innovations,
                scheme=self.config.scheme,
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

            from src.math_kernel.quant_utils import vectorized_fast_normal_ppf_v2

            sampler = qmc.Sobol(d=n_steps, scramble=True, seed=self.config.seed)
            sample = sampler.random(n_paths)
            # Use JIT accelerated inverse CDF
            return vectorized_fast_normal_ppf_v2(sample.ravel()).reshape(sample.shape)

        return self.rng.standard_normal((n_paths, n_steps))

    def price_american_lsm(self, params: BSParameters, option_type: str = "call") -> float:
        """
        Price American option using JIT-optimized Longstaff-Schwartz Least Squares Monte Carlo.
        """
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        is_call = option_type == "call"

        if params.maturity <= 0:
            if is_call:
                return max(params.spot - params.strike, 0.0)
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
            is_call=is_call,
            scheme=self.config.scheme,
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

    from src.math_kernel.quant_utils import fast_normal_cdf_v2

    if option_type == "call":
        price = params.spot * np.exp((mu_prime - params.rate) * T_prime) * fast_normal_cdf_v2(
            d1
        ) - params.strike * np.exp(-params.rate * T_prime) * fast_normal_cdf_v2(d2)
    elif option_type == "put":
        price = params.strike * np.exp(-params.rate * T_prime) * fast_normal_cdf_v2(
            -d2
        ) - params.spot * np.exp((mu_prime - params.rate) * T_prime) * fast_normal_cdf_v2(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

    return float(price)
