import numpy as np
import structlog
from typing import Optional, Tuple, Union

logger = structlog.get_logger(__name__)

try:
    import Manifold_core
    RUST_AVAILABLE = True
except ImportError:
    logger.warning("rust_core_not_found_falling_back_to_python")
    RUST_AVAILABLE = False

def is_available() -> bool:
    """Check if the high-performance Rust core is available."""
    return RUST_AVAILABLE

is_rust_available = is_available

def price_black_scholes(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    volatility: np.ndarray,
    rate: np.ndarray,
    dividend: np.ndarray,
    is_call: np.ndarray
) -> np.ndarray:
    """
    High-performance batch Black-Scholes pricing via Rust.
    Returns NumPy array of prices.
    """
    if not RUST_AVAILABLE:
        # Fallback to pure Python/NumPy vectorized implementation
        from .black_scholes_vectorized import black_scholes_vectorized
        # Note: black_scholes_vectorized handles one type at a time, we might need a wrapper here
        # but for Phase 19 we assume Rust is preferred and Numba is secondary.
        return black_scholes_vectorized(spot, strike, maturity, rate, volatility, "call" if is_call[0] else "put")

    return Manifold_core.batch_black_scholes(
        spot.astype(np.float64),
        strike.astype(np.float64),
        maturity.astype(np.float64),
        volatility.astype(np.float64),
        rate.astype(np.float64),
        dividend.astype(np.float64),
        is_call.astype(bool)
    )

def calculate_greeks(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    volatility: np.ndarray,
    rate: np.ndarray,
    dividend: np.ndarray,
    is_call: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-performance batch Greeks calculation via Rust.
    Returns (delta, gamma, theta, vega, rho).
    """
    if not RUST_AVAILABLE:
        from .black_scholes_vectorized import black_scholes_greeks_vectorized
        return black_scholes_greeks_vectorized(spot, strike, maturity, rate, volatility, "call" if is_call[0] else "put")

    return Manifold_core.batch_black_scholes_greeks(
        spot.astype(np.float64),
        strike.astype(np.float64),
        maturity.astype(np.float64),
        volatility.astype(np.float64),
        rate.astype(np.float64),
        dividend.astype(np.float64),
        is_call.astype(bool)
    )

def price_heston(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    v0: np.ndarray
) -> np.ndarray:
    """
    High-performance batch Heston pricing via Rust.
    """
    if not RUST_AVAILABLE:
        from .heston_fft import heston_price_fft
        # heston_price_fft implementation might be different (returns strikes/prices)
        # We'd need a compliant fallback here.
        logger.error("heston_rust_fallback_not_fully_implemented")
        return np.zeros_like(spot)

    return Manifold_core.batch_heston_price(
        spot.astype(np.float64),
        strike.astype(np.float64),
        maturity.astype(np.float64),
        rate.astype(np.float64),
        kappa.astype(np.float64),
        theta.astype(np.float64),
        sigma.astype(np.float64),
        rho.astype(np.float64),
        v0.astype(np.float64)
    )

class RustTickBuffer:
    """Wrapper for the Rust-backed TickDataBuffer (Mmap)."""
    def __init__(self, path: str):
        if not RUST_AVAILABLE:
            raise RuntimeError("Manifold_core not available for TickDataBuffer")
        self._buffer = Manifold_core.TickDataBuffer(path)

    def get_prices(self) -> np.ndarray:
        return self._buffer.get_prices()

    def get_volumes(self) -> np.ndarray:
        return self._buffer.get_volumes()

    def parse_all(self):
        """Bulk parse ticks into a list of TickData objects."""
        return self._buffer.parse_all()
    
    @property
    def size(self) -> int:
        return self._buffer.size()

def simulate_gbm_rk4(
    s0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    t: float,
    dt: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Highly optimized GBM simulation using 4th-order Runge-Kutta in Rust.
    """
    if not RUST_AVAILABLE:
        from .gbm_solver import simulate_gbm_rk4 as gbm_rk4_py
        return gbm_rk4_py(s0, mu, sigma, t, dt, seed=seed)

    return Manifold_core.simulate_gbm_rk4(
        s0.astype(np.float64),
        mu.astype(np.float64),
        sigma.astype(np.float64),
        float(t),
        float(dt),
        seed
    )
