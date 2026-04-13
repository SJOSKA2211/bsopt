"""
Distributed Computing Orchestration Layer.
"""

from src.shared.utils.ray_cluster_manager import RayClusterManager


class RayOrchestrator:
    """
    Convenience wrapper for Ray Cluster Management.
    Provides the standard init() interface used across the codebase.
    """

    @staticmethod
    def init(address: str | None = None, namespace: str = "bsopt") -> bool:
        """Initializes the Ray cluster connection."""
        return RayClusterManager.initialize(address=address, namespace=namespace)

    @staticmethod
    def get_stats():
        """Returns Ray cluster resource statistics."""
        return RayClusterManager.get_resource_stats()

    @staticmethod
    def shutdown():
        """Gracefully shuts down the Ray connection."""
        RayClusterManager.shutdown()