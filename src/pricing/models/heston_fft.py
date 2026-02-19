from numba import njit, float64, complex128
import numpy as np
import structlog

from src.pricing.models import HestonParams

logger = structlog.get_logger()

@njit(complex128[:,:](float64[:], float64[:], float64, float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True, fastmath=True)
def _heston_cf_kernel(v, k, alpha, T, r, v0, kappa, theta, sigma, rho):
    """
    Fused Numba kernel for the Heston Characteristic Function.
    Avoids NumPy broadcasting overhead and uses machine-code complex math.
    """
    n_v = v.shape[0]
    n_batch = k.shape[0]
    res = np.zeros((n_v, n_batch), dtype=np.complex128)
    
    for i in range(n_v):
        u_v = v[i] - (alpha + 1) * 1j
        for j in range(n_batch):
            xi = kappa[j] - sigma[j] * rho[j] * u_v * 1j
            d = np.sqrt(xi**2 + sigma[j]**2 * (u_v**2 + 1j * u_v))
            g = (xi + d) / (xi - d)
            
            exp_dT = np.exp(d * T[j])
            G = (1.0 - g * exp_dT) / (1.0 - g)
            
            A = (kappa[j] * theta[j] / sigma[j]**2) * ((xi + d) * T[j] - 2.0 * np.log(G))
            B = (v0[j] / sigma[j]**2) * (xi + d) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
            
            phi = np.exp(A + B)
            
            num = np.exp(-1j * v[i] * k[j]) * phi
            den = alpha**2 + alpha - v[i]**2 + 1j * (2 * alpha + 1) * v[i]
            res[i, j] = num / den
            
    return res

def _heston_integrand_vectorized(v, k, alpha, T, r, v0, kappa, theta, sigma, rho):
    """
    Delegates to the JIT-compiled kernel.
    """
    return np.real(_heston_cf_kernel(v, k, alpha, T, r, v0, kappa, theta, sigma, rho))

def batch_heston_price_jit(spots, strikes, maturities, rates, v0s, kappas, thetas, sigmas, rhos, is_calls, out):
    """
    Advanced vectorized batch pricing. ZERO Python loops.
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

    def price_call(self, S0: float, K: float) -> float:
        """Single option pricing for backward compatibility."""
        out = np.zeros(1)
        batch_heston_price_jit(
            np.array([S0]), np.array([K]), np.array([self.T]), np.array([self.r]),
            np.array([self.params.v0]), np.array([self.params.kappa]),
            np.array([self.params.theta]), np.array([self.params.sigma]),
            np.array([self.params.rho]), np.array([True]), out
        )
        return float(out[0])

    def price_put(self, S0: float, K: float) -> float:
        """Single option pricing for backward compatibility."""
        out = np.zeros(1)
        batch_heston_price_jit(
            np.array([S0]), np.array([K]), np.array([self.T]), np.array([self.r]),
            np.array([self.params.v0]), np.array([self.params.kappa]),
            np.array([self.params.theta]), np.array([self.params.sigma]),
            np.array([self.params.rho]), np.array([False]), out
        )
        return float(out[0])

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
