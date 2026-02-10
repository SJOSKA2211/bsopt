
import os
import struct

import structlog

from src.pricing.factory import PricingEngineFactory
from src.pricing.models import BSParameters
from src.shared.observability import tune_gc
from src.shared.shm_mesh import SHM_NAME, GreeksBuffer, SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)

class GreekEngine:
    """
    The Oracle: High-Frequency Mathematical Feature Engine.
    Spins on the SHM Mesh, calculates Greeks via WASM, and writes to Greeks Mesh.
    Pinned to Core 12 for mathematical dominance.
    """
    def __init__(self):
        tune_gc()
        self.mesh = SharedMemoryRingBuffer(create=False)
        self.greeks = GreeksBuffer(create=False)
        self._last_head = 0
        self.engine = PricingEngineFactory.get_engine("black_scholes")

    def run(self, cpu_core: int = 12):
        """Hot loop: Zero-latency Greek calculations."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("greek_engine_pinned", core=cpu_core)
        except Exception as e:
            logger.error("greek_engine_pinning_failed", error=str(e))

        logger.info("greek_engine_spinning", shm=SHM_NAME)
        
        while True:
            # Poll tick head
            current_head = struct.unpack("q", self.mesh.buf[:8])[0]
            
            if current_head > self._last_head:
                # New tick detected!
                view, new_head = self.mesh.read_latest_view(self._last_head)
                
                for tick in view:
                    symbol = tick['symbol'].decode().strip('\x00')
                    price = float(tick['price'])
                    
                    #  VECTORIZED MATH: Calculate Greeks for this ticker
                    # In a true Advanced pass, we'd do the entire surface at once
                    params = BSParameters(S=price, K=100.0, T=0.1, sigma=0.2, r=0.05)
                    g = self.engine.calculate_greeks(params)
                    
                    # Write to Greeks Mesh (Direct silicon path to Core 2)
                    self.greeks.write_greeks(symbol, g.delta, g.gamma, g.theta, g.vega, g.rho)
                
                self._last_head = new_head
            else:
                os.sched_yield()

if __name__ == "__main__":
    ge = GreekEngine()
    ge.run()
