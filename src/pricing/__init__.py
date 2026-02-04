"""
Pricing Engine - Hybrid Architecture with Lazy Loading
Separates fast classical methods from slow quantum methods.
Performance Characteristics:
- Classical (Heston, Black-Scholes): Load in ~50ms
- Quantum (Qiskit): Load in ~2.5s (only loaded when needed)
"""
import sys
import os
from typing import TYPE_CHECKING, List
from src.utils.lazy_import import lazy_import, preload_modules

__all__ = [
    # Classical Pricing
    "HestonModelFFT", "HestonCalibrator", "BlackScholesEngine", "MonteCarloEngine",
    # Volatility Surface
    "SVISurface", "SABRModel",
    # Quantum Methods (Heavy!)
    "QuantumOptionPricer",
]

if TYPE_CHECKING:
    from .models.heston_fft import HestonModelFFT
    from .calibration.engine import HestonCalibrator
    from .black_scholes import BlackScholesEngine
    from .monte_carlo import MonteCarloEngine
    from .calibration.svi_surface import SVISurface
    from .vol_surface import SABRModel
    from .quantum_pricing import QuantumOptionPricer

_import_map = {
    # Classical (Fast - can preload)
    "HestonModelFFT": ".models.heston_fft",
    "HestonCalibrator": ".calibration.engine",
    "BlackScholesEngine": ".black_scholes",
    "MonteCarloEngine": ".monte_carlo",
    
    # Surface (Medium speed)
    "SVISurface": ".calibration.svi_surface",
    "SABRModel": ".vol_surface",
    
    # Quantum (Very slow - always lazy)
    "QuantumOptionPricer": ".quantum_pricing",
}

def __getattr__(name: str):
    return lazy_import(__name__, _import_map, name, sys.modules[__name__])

def __dir__() -> List[str]:
    return sorted(__all__)

def preload_classical_pricers():
    """Preload fast classical pricing methods."""
    fast_modules = {
        "HestonModelFFT",
        "HestonCalibrator",
        "BlackScholesEngine",
        "MonteCarloEngine",
        "SVISurface",
    }
    preload_modules(__name__, _import_map, fast_modules)
    
    # Warm up JIT compiled functions in quant_utils
    try:
        from .quant_utils import warmup_jit
        warmup_jit()
    except ImportError:
        pass

# Auto-preload in production
if os.getenv("ENVIRONMENT") == "production" and os.getenv("PRELOAD_PRICING") == "true":
    preload_classical_pricers()