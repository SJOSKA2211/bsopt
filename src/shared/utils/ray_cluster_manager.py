import os
from typing import Any

import ray
import structlog

logger = structlog.get_logger(__name__)

class RayClusterManager:
    """
    God-Tier Ray Cluster Orchestrator.
    Manages resource sharding, auto-scaling integration, and health monitoring.
    """

    @staticmethod
    def initialize(address: str | None = None, namespace: str = "bsopt") -> bool:
        """Initialize or connect to a Ray cluster with optimal settings."""
        if ray.is_initialized():
            logger.info("ray_already_initialized")
            return True

        try:
            #  HARDENED: Dynamic resource detection
            ray.init(
                address=address or os.getenv("RAY_ADDRESS"),
                namespace=namespace,
                ignore_reinit_error=True,
                include_dashboard=True,
                _temp_dir="/tmp/ray",  # Avoid permission issues in some environments
            )

            resources = ray.cluster_resources()
            logger.info(
                "ray_cluster_connected",
                cpus=resources.get("CPU", 0),
                gpus=resources.get("GPU", 0),
                memory_gb=resources.get("memory", 0) / 1e9,
                object_store_gb=resources.get("object_store_memory", 0) / 1e9,
                node_count=len(ray.nodes()),
            )
            return True
        except Exception as e:
            logger.error("ray_initialization_failed", error=str(e))
            return False

    @staticmethod
    def get_resource_stats() -> dict[str, Any]:
        """Fetch detailed resource utilization from the cluster."""
        if not ray.is_initialized():
            return {}
        return {
            "nodes": ray.nodes(),
            "available": ray.available_resources(),
            "total": ray.cluster_resources(),
        }

    @staticmethod
    def shutdown():
        """Gracefully shutdown the local Ray instance."""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("ray_shutdown_complete")
