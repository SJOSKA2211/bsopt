import os
import struct

import structlog

from src.math_kernel.factory import PricingEngineFactory
from src.shared.observability import tune_gc
from src.shared.shm_mesh import SHM_NAME, GreeksBuffer, GreeksMesh, SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)


class GreekEngine:
    """
    The Oracle: High-Frequency Mathematical Feature Engine.
    Spins on the SHM Mesh, calculates Greeks via WASM/Rust, and writes to Greeks Mesh.
    Pinned to Core 12 for mathematical dominance.
    """

    def __init__(self):
        tune_gc()
        self.mesh = SharedMemoryRingBuffer(create=False)
        self.greeks_stream = GreeksBuffer(create=False)
        self.greeks_snapshot = GreeksMesh(create=False)
        self._last_head = 0
        self.engine = PricingEngineFactory.get_engine("black_scholes")

        # Pre-bind for hot loop speed
        self._struct_q = struct.Struct("q")
        self._head_mv = self.mesh.buf[:8]

    def run(self, cpu_core: int = 12):
        """Hot loop: Zero-latency Greek calculations."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("greek_engine_pinned", src.shared=cpu_core)
        except Exception:
            pass

        logger.info("greek_engine_spinning", shm=SHM_NAME)

        while True:
            # OPTIMIZED: Zero-copy head poll
            current_head = self._struct_q.unpack(self._head_mv)[0]

            if current_head > self._last_head:
                # 1. Zero-copy batch read
                slices, new_head = self.mesh.read_latest_slices(self._last_head)

                for chunk in slices:
                    # 2. Vectorized Batch Calculation
                    spots = chunk["price"]
                    # Call JIT-accelerated vectorized Greek kernel
                    # Note: We use a simplified constant for strike/maturity for demo
                    # In a real system, these would be matched per contract from a registry
                    deltas, gammas, thetas, vegas, rhos = self.engine.price_batch_greeks(
                        spots, 100.0, 0.1, 0.2, 0.05, 0.0
                    )

                    # 3. Write results back to Greeks Mesh (Stream + Snapshot)
                    for i in range(len(chunk)):
                        sym_bytes = chunk[i]["symbol"]
                        symbol = sym_bytes.decode("ascii").strip("\x00")

                        # Streaming update
                        self.greeks_stream.write_greeks_raw(
                            sym_bytes, deltas[i], gammas[i], thetas[i], vegas[i], rhos[i]
                        )

                        # Snapshot update
                        self.greeks_snapshot.write(
                            symbol, deltas[i], gammas[i], thetas[i], vegas[i], rhos[i]
                        )

                self._last_head = new_head
            else:
                os.sched_yield()


if __name__ == "__main__":
    ge = GreekEngine()
    ge.run()
