import os
import signal
import time
from typing import NoReturn

import structlog

from src.shared.observability import LATENCY_MS
from src.shared.shm_mesh import SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)

class HFTManifoldLauncher:
    """
    Institutional HFT Manifold Launcher.
    Orchestrates low-latency execution and shared-memory initialization.
    """
    def __init__(self, mesh_name: str = "bsopt_hft_mesh", size_mb: int = 128):
        self.mesh_name = mesh_name
        self.size_bytes = size_mb * 1024 * 1024
        self.shm = None
        self.running = True
        
        # Latency Thresholds (Institutional Standard)
        self.LATENCY_CRITICAL_MS = 50.0
        self.LATENCY_WARNING_MS = 10.0

    def initialize_mesh(self):
        """Atomic initialization of the shared memory mesh."""
        logger.info("initializing_shm_mesh", mesh=self.mesh_name, size_mb=self.size_bytes // 1024 // 1024)
        try:
            self.shm = SharedMemoryRingBuffer(self.mesh_name, self.size_bytes)
            logger.info("shm_mesh_ready")
        except Exception as e:
            logger.critical("shm_mesh_initialization_failed", error=str(e))
            raise

    def latency_sentinel_loop(self) -> NoReturn:
        """
        Sub-millisecond latency monitoring loop.
        Provides a real-time 'Execution Shield'.
        """
        logger.info("starting_latency_sentinel", critical_threshold=self.LATENCY_CRITICAL_MS)
        
        while self.running:
            start_time = time.perf_counter()
            
            # 1. Heartbeat/Ping Shared Memory
            try:
                # Simulation of a shared memory access/update
                # In a real scenario, this would check a 'last_update' timestamp in SHM
                _ = self.shm.get_latest_tick() if self.shm else None
            except Exception as e:
                logger.error("shm_heartbeat_failed", error=str(e))
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000.0 # ms
            
            LATENCY_MS.labels(path="shm_mesh").observe(latency)
            
            if latency > self.LATENCY_CRITICAL_MS:
                logger.critical("LATENCY_SHIELD_TRIGGERED", latency=latency)
                self.emergency_shutdown()
                
            elif latency > self.LATENCY_WARNING_MS:
                logger.warning("latency_jitter_detected", latency=latency)

            # High-fidelity sleep for the sentinel (100Hz)
            time.sleep(0.01)

    def emergency_shutdown(self):
        """Immediate shutdown of all HFT execution paths."""
        logger.critical("EMERGENCY_SHUTDOWN_INITIATED")
        self.running = False
        # 1. Cancel all outstanding orders (MOCK)
        # 2. Release SHM resources
        if self.shm:
            self.shm.close()
        os.kill(os.getpid(), signal.SIGTERM)

def handle_signal(signum, frame):
    logger.info("received_signal_shutting_down", signal=signum)
    exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    launcher = HFTManifoldLauncher()
    launcher.initialize_mesh()
    launcher.latency_sentinel_loop()
