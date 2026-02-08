import numpy as np
import structlog

from src.pricing.models import HestonParams

logger = structlog.get_logger()

def _heston_integrand_vectorized(v, k, alpha, T, r, v0, kappa, theta, sigma, rho):
    """
    Vectorized Heston integrand using NumPy broadcasting.
    v: integration grid [N_steps]
    k: log-strikes [N_batch] or scalar
    Other params: scalars or arrays of shape [N_batch]
    """
    # Reshape for broadcasting
    # v becomes [N_steps, 1], others remain [N_batch] or scalar
    v_grid = v.reshape(-1, 1)
    
    sig = np.maximum(sigma, 1e-6)
    u = v_grid - (alpha + 1) * 1j
    
    # Characteristic function calculation (vectorized across v and batch)
    xi = kappa - sig * rho * u * 1j
    d = np.sqrt(xi**2 + sig**2 * (u**2 + 1j * u))
    g = (xi + d) / (xi - d)
    
    exp_dT = np.exp(d * T)
    
    A = (kappa * theta / sig**2) * (
        (xi + d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    B = (v0 / sig**2) * (xi + d) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    
    phi = np.exp(A + B)
    
    num = np.exp(-1j * v_grid * k) * phi
    den = alpha**2 + alpha - v_grid**2 + 1j * (2 * alpha + 1) * v_grid
    
    return np.real(num / den)

def batch_heston_price_jit(spots, strikes, maturities, rates, v0s, kappas, thetas, sigmas, rhos, is_calls, out):
    """
    God-Mode vectorized batch pricing. ZERO Python loops.
    """
    k = np.log(strikes / spots)
    # Note: Carr-Madan usually uses alpha=1.5 for calls. Puts handled via parity or alpha=-2.5
    # For batching, we'll simplify to alpha=1.5 and use parity for is_call=False later if needed
    alpha = np.where(is_calls, 1.5, -2.5)
    
    upper_bound = 250.0
    n_steps = 2000
    h = upper_bound / n_steps
    v = np.linspace(0, upper_bound, n_steps + 1)
    
    # Calculate all integrands in one large broadcasted array [N_steps, N_batch]
    f_v = _heston_integrand_vectorized(v, k, alpha, maturities, rates, v0s, kappas, thetas, sigmas, rhos)
    
    # Simpson weights [N_steps]
    weights = np.ones(n_steps + 1)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    weights = weights.reshape(-1, 1) # Broadcast weights across batch
    
    # Simpson integration across the v-axis (axis 0)
    integrals = (h / 3.0) * np.sum(f_v * weights, axis=0)
    
    price_vals = (np.exp(-alpha * k) / np.pi) * integrals
    discounted_prices = np.exp(-rates * maturities) * spots * price_vals
    
    # Arbitrage floor
    intrinsics = np.where(is_calls, np.maximum(spots - strikes, 0.0), np.maximum(strikes - spots, 0.0))
    out[:] = np.maximum(discounted_prices, intrinsics)

class HestonModelFFT:
    """
    Heston Model using vectorized FFT and Simpson integration.
    """
    MAX_INTEGRATION_BOUND = 250.0
    MIN_PRICE = 1e-10

    def __init__(self, params: HestonParams, r: float, T: float):
        self.params = params
        self.r = r
        self.T = T

    def price_surface_fft(self, S0: float, K_min: float, K_max: float, N: int = 1024) -> dict[float, float]:
        """
        O(N log N) multi-strike pricing using vectorized FFT.
        """
        p = self.params
        alpha = 1.5
        eta = 0.25 
        lambda_grid = (2 * np.pi) / (N * eta) 
        b = (N * lambda_grid) / 2 
        
        v = np.arange(N) * eta
        k_grid = -b + np.arange(N) * lambda_grid
        
        # Vectorized characteristic function across v-grid
        u = v - (alpha + 1) * 1j
        xi = p.kappa - p.sigma * p.rho * u * 1j
        d = np.sqrt(xi**2 + p.sigma**2 * (u**2 + 1j * u))
        g = (xi + d) / (xi - d)
        exp_dT = np.exp(d * self.T)
        
        A = (p.kappa * p.theta / p.sigma**2) * ((xi + d) * self.T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))
        B = (p.v0 / p.sigma**2) * (xi + d) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
        
        phi = np.exp(A + B)
        psi = (np.exp(-self.r * self.T) * phi) / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
        
        # Simpson weights
        w = (eta / 3.0) * (3 + (-1)**(np.arange(N)+1))
        w[0] = eta / 3.0
        
        phi_values = np.exp(1j * v * b) * psi * w
        x_fft = np.fft.fft(phi_values)
        
        prices = np.real(np.exp(-alpha * k_grid) / np.pi * x_fft) * S0
        strikes = S0 * np.exp(k_grid)
        
        # Boolean indexing for range filter (Vectorized)
        mask = (strikes >= K_min) & (strikes <= K_max)
        filtered_strikes = strikes[mask]
        filtered_prices = np.maximum(prices[mask], self.MIN_PRICE)
        
        return dict(zip(filtered_strikes.tolist(), filtered_prices.tolist()))
