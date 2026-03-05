"""
Pricing Engine - Hybrid Architecture with Lazy Loading
Separates fast classical methods from slow quantum methods.
Performance Characteristics:
- Classical (Heston, Black-Scholes): Load in ~50ms
- Quantum (Qiskit): Load in ~2.5s (only loaded when needed)
"""

import os
import sys
from typing import TYPE_CHECKING

from src.utils.lazy_import import lazy_import, preload_modules

__all__ = [
    # Classical Pricing
    "HestonModelFFT",
    "HestonCalibrator",
    "BlackScholesEngine",
    "MonteCarloEngine",
    # Volatility Surface
    "SVISurface",
    "SABRModel",
    # Quantum Methods (Heavy!)
    "QuantumOptionPricer",
    # High-Performance Kernels (quant_utils)
    "batch_bs_price_jit",
    "batch_greeks_jit",
    "scalar_bs_price_jit",
    "scalar_greeks_jit",
    "jit_cn_solver",
    "vectorized_newton_raphson_iv_jit",
]

if TYPE_CHECKING:
    from .black_scholes import BlackScholesEngine
    from .calibration.engine import HestonCalibrator
    from .calibration.svi_surface import SVISurface
    from .models.heston_fft import HestonModelFFT
    from .monte_carlo import MonteCarloEngine
    from .quant_utils import (
        batch_bs_price_jit,
        batch_greeks_jit,
        jit_cn_solver,
        scalar_bs_price_jit,
        scalar_greeks_jit,
        vectorized_newton_raphson_iv_jit,
    )
    from .quantum_pricing import QuantumOptionPricer
    from .vol_surface import SABRModel

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
    # Kernels
    "batch_bs_price_jit": ".quant_utils",
    "batch_greeks_jit": ".quant_utils",
    "scalar_bs_price_jit": ".quant_utils",
    "scalar_greeks_jit": ".quant_utils",
    "jit_cn_solver": ".quant_utils",
    "vectorized_newton_raphson_iv_jit": ".quant_utils",
}


def __getattr__(name: str):
    return lazy_import(__name__, _import_map, name, sys.modules[__name__])


def __dir__() -> list[str]:
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
