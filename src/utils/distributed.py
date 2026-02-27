import os
import sys

import ray
import structlog

logger = structlog.get_logger(__name__)


class RayOrchestrator:
    """
    Distributed compute orchestrator for option pricing tasks.
    Configures Ray cluster parameters based on local hardware topology.
    """

    @staticmethod
    def pin_process_to_core(core_id: int):
        """Pin the current process to a specific CPU core."""
        try:
            os.sched_setaffinity(0, {core_id})
            logger.info("cpu_affinity_set", core=core_id)
        except Exception as e:
            logger.warning("cpu_pinning_failed", error=str(e))

    @staticmethod
    def get_optimal_core_for_numa(node_id: int) -> int:
        """Identify a physical core on the target NUMA node."""
        try:
            with open(f"/sys/devices/system/node/node{node_id}/cpulist") as f:
                cores = f.read().strip().split(",")[0]
                return int(cores.split("-")[0])
        except Exception:
            return node_id  # Fallback to node_id as core_id

    @staticmethod
    def init(
        num_cpus: int | None = None,
        num_gpus: int | None = 0,
        object_store_memory_gb: float | None = None,
        spill_dir: str = "/tmp/ray_spill",
    ):
        """Initialize Ray with hardware-appropriate settings."""
        if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
            logger.info("ray_init_skipped_in_test")
            return

        if ray.is_initialized():
            logger.info("ray_already_active")
            return

        # OPTIMIZED: Docker-aware CPU count
        try:
            detected_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            import multiprocessing

            detected_cpus = multiprocessing.cpu_count()

        default_cpus = (
            min(detected_cpus, 2) if os.getenv("ENVIRONMENT") in ["dev", "test"] else detected_cpus
        )
        env_cpus = os.getenv("RAY_NUM_CPUS")
        actual_cpus = (
            int(env_cpus) if env_cpus else (num_cpus if num_cpus is not None else default_cpus)
        )

        total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        if object_store_memory_gb is None:
            limit = 0.1 if os.getenv("ENVIRONMENT") != "test" else 0.05
            object_store_memory = int(min(total_ram_gb * limit, 2.0) * 1024**3)
        else:
            object_store_memory = int(object_store_memory_gb * 1024**3)

        os.makedirs(spill_dir, exist_ok=True)

        logger.info(
            "initializing_ray",
            cpus=actual_cpus,
            gpus=num_gpus,
            memory_gb=round(object_store_memory / 1024**3, 2),
        )

        import json

        address = os.getenv("RAY_ADDRESS")
        if address:
            ray.init(address=address, ignore_reinit_error=True)
        else:
            ray.init(
                num_cpus=actual_cpus,
                num_gpus=num_gpus,
                object_store_memory=object_store_memory,
                _system_config={
                    "object_spilling_config": json.dumps(
                        {"type": "filesystem", "params": {"directory_path": spill_dir}}
                    )
                },
                ignore_reinit_error=True,
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
