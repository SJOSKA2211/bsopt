import asyncio
import threading
from typing import Any, TypeVar

import ray
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RayActorPool:
    """
    High-Performance Ray Actor Pool: Handles round-robin load balancing and lifecycle management.
    OPTIMIZED: Dynamic scaling based on cluster resources.
    """

    def __init__(self, actor_class: type[T], count: int | None = None, name: str = "default"):
        self._actor_class = actor_class
        self._name = name
        self._count = count or self._detect_optimal_count()
        self._actors = [actor_class.remote() for _ in range(self._count)]
        self._index = 0
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        logger.info(
            "ray_actor_pool_initialized", name=name, count=self._count, actor=actor_class.__name__
        )

    def _detect_optimal_count(self) -> int:
        try:
            cpus = int(ray.cluster_resources().get("CPU", 2))
            return max(1, cpus - 1)  # Reserve one for driver
        except Exception:
            return 2

    async def get_actor(self) -> Any:
        """Async-safe round-robin actor retrieval."""
        async with self._lock:
            actor = self._actors[self._index % self._count]
            self._index += 1
            return actor

    def get_actor_sync(self) -> Any:
        """Thread-safe synchronous actor retrieval for legacy paths."""
        with self._sync_lock:
            actor = self._actors[self._index % self._count]
            self._index += 1
            return actor

    async def broadcast(self, method_name: str, *args, **kwargs) -> list[Any]:
        """Call a method on all actors in the pool simultaneously."""
        tasks = [getattr(actor, method_name).remote(*args, **kwargs) for actor in self._actors]
        return await asyncio.gather(
            *[asyncio.wrap_future(t.to_async()) if hasattr(t, "to_async") else t for t in tasks]
        )

    def shutdown(self):
        """Cleanly shutdown all actors."""
        for actor in self._actors:
            ray.kill(actor)
        self._actors = []
        logger.info("ray_actor_pool_shutdown", name=self._name)
