import numpy as np
import structlog

from src.pricing.models import HestonParams

logger = structlog.get_logger()

def _heston_integrand_vectorized(v, k, alpha, T, r, v0, kappa, theta, sigma, rho):
    """
    Vectorized Heston integrand using NumPy broadcasting.
    """
    v_grid = v.reshape(-1, 1)
    
    sig = np.maximum(sigma, 1e-6)
    u = v_grid - (alpha + 1) * 1j
    
    # Characteristic function calculation (vectorized across v and batch)
    xi = kappa - sig * rho * u * 1j
    d = np.sqrt(xi**2 + sig**2 * (u**2 + 1j * u))
    g = (xi + d) / (xi - d)
    
    # Numerical stability
    dT = d * T
    exp_dT = np.exp(np.clip(dT.real, -100, 100) + 1j * dT.imag)
    
    g_exp_dT = g * exp_dT
    
    # Stable formulation for A and B
    G = (1.0 - g_exp_dT) / np.maximum(1e-18, (1.0 - g))
    
    A = (kappa * theta / sig**2) * (
        (xi + d) * T - 2.0 * np.log(np.maximum(1e-18, G))
    )
    B = (v0 / sig**2) * (xi + d) * (1.0 - exp_dT) / np.maximum(1e-18, 1.0 - g_exp_dT)
    
    phi = np.exp(A + B)
    
    num = np.exp(-1j * v_grid * k) * phi
    den = alpha**2 + alpha - v_grid**2 + 1j * (2 * alpha + 1) * v_grid
    
    return np.real(num / den)

def batch_heston_price_jit(spots, strikes, maturities, rates, v0s, kappas, thetas, sigmas, rhos, is_calls, out):
    """
    God-Mode vectorized batch pricing. ZERO Python loops.
    """
    k = np.log(strikes / spots)
    # alpha = 1.5 for call, -2.5 for put usually, but let's stick to 1.5 and use parity
    alpha = np.where(is_calls, 1.5, 1.5)
    
    upper_bound = 250.0
    n_steps = 2000
    h = upper_bound / n_steps
    v = np.linspace(0, upper_bound, n_steps + 1)
    
    f_v = _heston_integrand_vectorized(v, k, alpha, maturities, rates, v0s, kappas, thetas, sigmas, rhos)
    
    weights = np.ones(n_steps + 1)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    weights = weights.reshape(-1, 1)
    
    integrals = (h / 3.0) * np.sum(f_v * weights, axis=0)
    
    price_vals = (np.exp(-alpha * k) / np.pi) * integrals
    discounted_prices = np.exp(-rates * maturities) * spots * price_vals
    
    # For puts, use put-call parity: P = C - S + K*exp(-rT)
    put_prices = discounted_prices - spots + strikes * np.exp(-rates * maturities)
    
    final_prices = np.where(is_calls, discounted_prices, put_prices)
    
    intrinsics = np.where(is_calls, np.maximum(spots - strikes, 0.0), np.maximum(strikes - spots, 0.0))
    out[:] = np.maximum(final_prices, intrinsics)

class HestonModelFFT:
    """
    Heston Model using vectorized FFT and Simpson integration.
    """
    MAX_INTEGRATION_BOUND = 250.0
    MIN_PRICE = 1e-10

    def __init__(self, params: HestonParams | None = None, r: float | None = None, T: float | None = None):
        self.params = params
        self.r = r
        self.T = T

    def price_surface_fft(self, S0: float, K_min: float, K_max: float, N: int = 1024) -> dict[float, float]:
        """
        O(N log N) multi-strike pricing using vectorized FFT.
        """
        if self.params is None or self.r is None or self.T is None:
            raise ValueError("Model must be initialized with params, r, and T for surface pricing.")
            
        p = self.params
        alpha = 1.5
        eta = 0.25 
        lambda_grid = (2 * np.pi) / (N * eta) 
        b = (N * lambda_grid) / 2 
        
        v = np.arange(N) * eta
        k_grid = -b + np.arange(N) * lambda_grid
        
        u = v - (alpha + 1) * 1j
        xi = p.kappa - p.sigma * p.rho * u * 1j
        d = np.sqrt(xi**2 + p.sigma**2 * (u**2 + 1j * u))
        g = (xi + d) / (xi - d)
        
        dT = d * self.T
        exp_dT = np.exp(np.clip(dT.real, -100, 100) + 1j * dT.imag)
        
        g_exp_dT = g * exp_dT
        G = (1.0 - g_exp_dT) / np.maximum(1e-18, (1.0 - g))
        
        A = (p.kappa * p.theta / p.sigma**2) * ((xi + d) * self.T - 2.0 * np.log(np.maximum(1e-18, G)))
        B = (p.v0 / p.sigma**2) * (xi + d) * (1.0 - exp_dT) / np.maximum(1e-18, 1.0 - g_exp_dT)
        
        phi = np.exp(A + B)
        psi = (np.exp(-self.r * self.T) * phi) / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
        
        w = (eta / 3.0) * (3 + (-1)**(np.arange(N)+1))
        w[0] = eta / 3.0
        
        phi_values = np.exp(1j * v * b) * psi * w
        x_fft = np.fft.fft(phi_values)
        
        prices = np.real(np.exp(-alpha * k_grid) / np.pi * x_fft) * S0
        strikes = S0 * np.exp(k_grid)
        
        mask = (strikes >= K_min) & (strikes <= K_max)
        filtered_strikes = strikes[mask]
        filtered_prices = np.maximum(prices[mask], self.MIN_PRICE)
        
        return dict(zip(filtered_strikes.tolist(), filtered_prices.tolist()))
