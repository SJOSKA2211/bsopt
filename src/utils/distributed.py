import os
import multiprocessing
import ray
import structlog
from typing import Optional, Dict, Any

logger = structlog.get_logger(__name__)

class RayOrchestrator:
    """
    SOTA: Distributed compute orchestrator for massive batch pricing.
    Optimizes Ray cluster parameters based on local hardware topology.
    """

    @staticmethod
    def pin_process_to_core(core_id: int):
        """🚀 SINGULARITY: Hard CPU affinity pinning."""
        try:
            os.sched_setaffinity(0, {core_id})
            logger.info("cpu_affinity_pinned", core=core_id)
        except Exception as e:
            logger.warning("cpu_pinning_failed", error=str(e))

    @staticmethod
    def get_optimal_core_for_numa(node_id: int) -> int:
        """SOTA: Find a physical core on the target NUMA node."""
        try:
            with open(f"/sys/devices/system/node/node{node_id}/cpulist", "r") as f:
                cores = f.read().strip().split(",")[0]
                return int(cores.split("-")[0])
        except:
            return node_id # Fallback to node_id as core_id

    @staticmethod
    def init(
        num_cpus: Optional[int] = None, 
        num_gpus: Optional[int] = 0, 
        object_store_memory_gb: Optional[float] = None,
        spill_dir: str = "/tmp/ray_spill"
    ):
        """🚀 SINGULARITY: Initialize Ray with optimal hardware settings."""
        if ray.is_initialized():
            logger.info("ray_already_initialized")
            return
            
        # 1. Hardware Detection
        detected_cpus = multiprocessing.cpu_count()
        actual_cpus = num_cpus if num_cpus is not None else detected_cpus
        
        # 2. Memory Optimization
        # Rule of thumb: Leave 30% for system and other services
        if object_store_memory_gb is None:
            total_ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
            object_store_memory = int(total_ram_gb * 0.3 * 1024**3)
        else:
            object_store_memory = int(object_store_memory_gb * 1024**3)
            
        # 3. Spill Path (prefer NVMe if possible, placeholder check)
        os.makedirs(spill_dir, exist_ok=True)
        
        logger.info(
            "initializing_ray_cluster", 
            cpus=actual_cpus, 
            gpus=num_gpus, 
            memory_gb=object_store_memory / 1024**3
        )
        
        # 🚀 SOTA: Initializing Ray with hardware-aware config
        ray.init(
            num_cpus=actual_cpus,
            num_gpus=num_gpus,
            object_store_memory=object_store_memory,
            _system_config={
                "object_spilling_config": {
                    "type": "filesystem",
                    "params": {"directory_path": spill_dir}
                }
            },
            ignore_reinit_error=True
        )
        
    @staticmethod
    def shutdown():
        if ray.is_initialized():
            ray.shutdown()
            logger.info("ray_cluster_shutdown")

if __name__ == "__main__":
    RayOrchestrator.init()
    print(f"Ray Nodes: {ray.nodes()}")
    RayOrchestrator.shutdown()