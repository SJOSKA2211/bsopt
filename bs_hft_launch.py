
import os
import sys
import threading
import time
import structlog
from src.data.xdp_ingest import XDPIngester
from src.ml.reinforcement_learning.online_agent import OnlineRLAgent
from src.aiops.aiops_orchestrator import AIOpsOrchestrator
from src.config import settings

logger = structlog.get_logger(__name__)

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

def launch_manifold():
    """
    Orchestrates the high-frequency trading manifold.
    Assigns cores, locks memory, and starts the silicon swarm.
    """
    logger.info("launching_solenya_manifold", version="1.0.0-Singularity")
    
    # 1. Global Pre-flight
    lock_memory()
    
    # 2. Start Ingester (Core 1)
    # Note: XDPIngester starts its own thread internally
    ingester = XDPIngester()
    ingester.start(cpu_core=1)
    
    # 3. Start Agent (Core 2)
    agent = OnlineRLAgent(model_path="models/latest_td3.zip")
    agent_thread = threading.Thread(
        target=agent.run, 
        args=(2,), 
        name="AgentEngine", 
        daemon=True
    )
    agent_thread.start()
    
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
