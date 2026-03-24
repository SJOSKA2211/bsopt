"""
Greek Engine — orchestrates real-time Greek calculations and mesh updates.
"""

from __future__ import annotations

import numpy as np
import structlog

from src.math_kernel.cuda_kernels import black_scholes_greeks_cupy
from src.shared.shm_mesh import GreeksBuffer, GreeksMesh

logger = structlog.get_logger(__name__)

class GreekEngine:
    """
    High-performance engine for Greek aggregation and SHM synchronization.
    """

    def __init__(self):
        self.mesh = GreeksMesh(create=True)
        self.buffer = GreeksBuffer(create=True)
        logger.info("greek_engine_initialized", mesh="shm_active")

    def update_symbol_greeks(
        self,
        symbol: str,
        s: float,
        k: float,
        t: float,
        sigma: float,
        r: float,
        q: float,
        is_call: bool,
    ) -> dict[str, float]:
        """
        Calculate and broadcast Greeks for a single instrument.
        """
        try:
            # Vectorized call for single element (high efficiency)
            greeks = black_scholes_greeks_cupy(
                np.array([s]),
                np.array([k]),
                np.array([t]),
                np.array([sigma]),
                np.array([r]),
                np.array([q]),
                np.array([is_call]),
            )

            res = {
                "delta": float(greeks["delta"][0]),
                "gamma": float(greeks["gamma"][0]),
                "theta": float(greeks["theta"][0]),
                "vega": float(greeks["vega"][0]),
                "rho": float(greeks["rho"][0]),
            }

            # Atomic broadcast to Shared Memory Mesh
            self.mesh.write(symbol, **res)
            self.buffer.write_greeks(symbol, **res)

            return res
        except Exception as e:
            logger.error("greek_calculation_failed", symbol=symbol, error=str(e))
            return {}

    def get_portfolio_greeks(self, symbols: list[str]) -> dict[str, float]:
        """
        O(1) retrieval from SHM mesh for portfolio aggregation.
        """
        total = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        for sym in symbols:
            data = self.mesh.read(sym)
            if data:
                for k in total:
                    total[k] += data.get(k, 0.0)
        return total
