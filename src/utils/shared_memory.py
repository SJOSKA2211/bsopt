from __future__ import annotations

import atexit
import threading
from multiprocessing import shared_memory
from typing import Any

import structlog

logger = structlog.get_logger()


class SharedMemoryManager:
    """
    Manages a pool of pre-allocated shared memory segments to enable
    zero-allocation communication between processes.
    """

    _instance: SharedMemoryManager | None = None
    _lock = threading.Lock()

    def __init__(self, segment_size: int = 20 * 1024 * 1024, num_segments: int = 10) -> None:
        self.segment_size = segment_size
        self.num_segments = num_segments
        self.available_segments: list[str] = []
        self.all_segments: dict[str, shared_memory.SharedMemory] = {}
        self._pool_lock = threading.Lock()

        self._initialize_pool()

    @classmethod
    def get_instance(cls) -> SharedMemoryManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _initialize_pool(self) -> None:
        for i in range(self.num_segments):
            name = f"bsopt_shm_pool_{i}"
            try:
                # Aggressive cleanup of stale segments
                try:
                    temp = shared_memory.SharedMemory(name=name)
                    temp.close()
                    temp.unlink()
                    logger.info("cleaned_stale_shm", segment=name)
                except Exception:
                    pass

                shm = shared_memory.SharedMemory(name=name, create=True, size=self.segment_size)
                self.all_segments[name] = shm
                self.available_segments.append(name)
            except Exception as e:
                logger.error("shm_pool_init_failed", segment=name, error=str(e))

    def acquire(self) -> str | None:
        """Acquires a segment from the pool iteratively."""
        with self._pool_lock:
            while self.available_segments:
                seg_name = self.available_segments.pop()
                try:
                    shm = self.all_segments[seg_name]
                    # Probe health
                    buf = shm.buf
                    if buf is not None:
                        buf[0] = buf[0]
                    return seg_name
                except Exception as e:
                    logger.error("shm_segment_corrupt", segment=seg_name, error=str(e))
                    # Don't re-add corrupt segment to available

            logger.warning("shm_pool_exhausted")
            return None

    def release(self, name: str) -> None:
        """Releases a segment back to the pool."""
        with self._pool_lock:
            if name in self.all_segments and name not in self.available_segments:
                # OPTIMIZED: Zero out buffer on release for security/consistency
                shm = self.all_segments[name]
                buf = shm.buf
                if buf is not None:
                    buf[:] = b"\x00" * self.segment_size
                self.available_segments.append(name)
            else:
                logger.warning("shm_release_invalid", segment=name)

    def get_segment(self, name: str) -> shared_memory.SharedMemory | None:
        return self.all_segments.get(name)

    def cleanup(self, unlink: bool = False) -> None:
        """Close segment handles and optionally unlink (dangerous in multi-process)."""
        with self._pool_lock:
            for shm in self.all_segments.values():
                try:
                    shm.close()
                    if unlink:
                        shm.unlink()
                except Exception:
                    pass
            self.all_segments.clear()
            self.available_segments.clear()


# Global manager instance
shm_manager = SharedMemoryManager.get_instance()

# OPTIMIZED: Only close local handles on exit.
# Global unlinking should be handled by the orchestrator/launcher.
atexit.register(shm_manager.cleanup, unlink=False)
