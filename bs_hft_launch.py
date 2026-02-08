
import os
import sys
import threading
import time
import asyncio
import structlog
from src.streaming.ingestion_worker import IngestionWorker
from src.ml.reinforcement_learning.online_agent import OnlineRLAgent
from src.aiops.aiops_orchestrator import AIOpsOrchestrator
from src.config import settings

logger = structlog.get_logger(__name__)

def run_ingestion_worker(cpu_core: int):
    """Bridge to run the async IngestionWorker in a dedicated pinned thread."""
    worker = IngestionWorker()
    # Pinned thread needs its own event loop
    asyncio.run(worker.run(cpu_core=cpu_core))

def lock_memory():
    """Lock process memory to prevent swapping (Requires root/CAP_IPC_LOCK)."""
    try:
        import ctypes
        MCL_CURRENT = 1
        MCL_FUTURE = 2
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.mlockall(MCL_CURRENT | MCL_FUTURE) != 0:
            logger.warning("mlockall_failed", errno=ctypes.get_errno())
        else:
            logger.info("memory_locked_silicon")
    except Exception as e:
        logger.warning("memory_lock_not_available", error=str(e))

from src.trading.order_engine import OrderEngine
from src.monitoring.telemetry import TelemetryEngine
from src.shared.shm_mesh import OrderBuffer, ExecutionBuffer

def launch_manifold():
    """
    Orchestrates the high-frequency trading manifold.
    Assigns cores, locks memory, and starts the silicon swarm.
    """
    logger.info("launching_solenya_manifold", version="1.0.0-Singularity")
    
    # 1. Global Pre-flight
    lock_memory()
    # Initialize lock-free buffers
    _ = OrderBuffer(create=True)
    _ = ExecutionBuffer(create=True)
    
    # 2. Initialize Ingestion Swarm
    worker = IngestionWorker()
    
    # 2a. Pulse: XDP/SHM Path (Core 1)
    worker.xdp_ingester.start(cpu_core=1)
    
    # 2b. Voice: WS Broadcasting (Core 4)
    threading.Thread(
        target=lambda: asyncio.run(worker.run_broadcaster(cpu_core=4)),
        name="VoiceEngine", daemon=True
    ).start()
    
    # 2c. Scribe: DB Persistence (Core 5)
    threading.Thread(
        target=lambda: asyncio.run(worker.run_scribe(cpu_core=5)),
        name="ScribeEngine", daemon=True
    ).start()
    
    # 2d. Verve: Ingestion Dispatcher (Core 6)
    threading.Thread(
        target=lambda: asyncio.run(worker.run_dispatcher(cpu_core=6)),
        name="VerveEngine", daemon=True
    ).start()
    
    # 3. Start Agent (Core 2)
    agent = OnlineRLAgent(model_path="models/latest_td3.zip")
    threading.Thread(
        target=agent.run, 
        args=(2,), 
        name="AgentEngine", 
        daemon=True
    ).start()

    # 4. Start Order Engine (Core 7)
    oe = OrderEngine()
    threading.Thread(
        target=oe.run,
        args=(7,),
        name="OrderEngine",
        daemon=True
    ).start()

    # 5. Start Telemetry Engine (Core 8)
    te = TelemetryEngine()
    threading.Thread(
        target=te.run,
        args=(8,),
        name="TelemetryEngine",
        daemon=True
    ).start()
    
    # 4. Start AIOps Orchestrator (Core 3)
    # Runs in main thread to keep it free for global supervision
    orchestrator_config = {
        "prometheus_url": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        "api_service_name": "api",
        "latency_threshold": 0.5,
        "error_rate_threshold": 0.05,
        "predictive_scaling_enabled": True
    }
    
    # Pin orchestrator to Core 3
    try:
        os.sched_setaffinity(0, {3})
        logger.info("orchestrator_pinned", core=3)
    except Exception:
        pass
        
    orchestrator = AIOpsOrchestrator(orchestrator_config)
    
    logger.info("manifold_swarm_active")
    
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("manifold_shutdown_initiated")
        ingester.stop()
        # Agent thread is daemon, will exit with main

if __name__ == "__main__":
    launch_manifold()
