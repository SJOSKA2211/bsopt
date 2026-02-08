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
    
    A = (kappa * theta / sig**2) * (
        (xi + d) * T - 2.0 * np.log(np.maximum(1e-12, (1.0 - g_exp_dT) / (1.0 - g)))
    )
    B = (v0 / sig**2) * (xi + d) * (1.0 - exp_dT) / (np.maximum(1e-12, 1.0 - g_exp_dT))
    
    phi = np.exp(A + B)
    
    num = np.exp(-1j * v_grid * k) * phi
    den = alpha**2 + alpha - v_grid**2 + 1j * (2 * alpha + 1) * v_grid
    
    return np.real(num / den)

def batch_heston_price_jit(spots, strikes, maturities, rates, v0s, kappas, thetas, sigmas, rhos, is_calls, out):
    """
    Prices a batch of Heston options using FFT-like grid integration.
    """
    n_batch = len(spots)
    if n_batch == 0:
        return
        
    # Integration parameters
    alpha = 1.5
    n_steps = 128
    v_max = 100.0
    v = np.linspace(0.0001, v_max, n_steps)
    dv = v[1] - v[0]
    
    for i in range(n_batch):
        S = spots[i]
        K = strikes[i]
        T = maturities[i]
        R = rates[i]
        v0 = v0s[i]
        kappa = kappas[i]
        theta = thetas[i]
        sigma = sigmas[i]
        rho = rhos[i]
        
        if T <= 1e-6:
            if is_calls[i]:
                out[i] = max(S - K, 0.0)
            else:
                out[i] = max(K - S, 0.0)
            continue
            
        k = np.log(K / S)
        
        # Grid integration (Trapezoidal rule)
        integrand = _heston_integrand_vectorized(v, k, alpha, T, R, v0, kappa, theta, sigma, rho)
        integral = np.sum(integrand) * dv
        
        price = (np.exp(-R * T) / np.pi) * integral * S
        
        if is_calls[i]:
            out[i] = max(0.0, price)
        else:
            # Put-Call Parity
            out[i] = max(0.0, price - S + K * np.exp(-R * T))

class HestonModelFFT:
    """Heston Pricing Engine using grid integration."""
    def price_batch(self, params_list: list[HestonParams], is_calls: np.ndarray) -> np.ndarray:
        n = len(params_list)
        spots = np.array([p.S for p in params_list])
        strikes = np.array([p.K for p in params_list])
        maturities = np.array([p.T for p in params_list])
        rates = np.array([p.r for p in params_list])
        v0s = np.array([p.v0 for p in params_list])
        kappas = np.array([p.kappa for p in params_list])
        thetas = np.array([p.theta for p in params_list])
        sigmas = np.array([p.sigma for p in params_list])
        rhos = np.array([p.rho for p in params_list])
        
        out = np.zeros(n)
        batch_heston_price_jit(spots, strikes, maturities, rates, v0s, kappas, thetas, sigmas, rhos, is_calls, out)
        return out
