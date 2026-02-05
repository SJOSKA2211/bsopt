import atexit
import threading
from multiprocessing import shared_memory

import structlog

logger = structlog.get_logger()

class SharedMemoryManager:
    """
    Manages a pool of pre-allocated shared memory segments to enable 
    zero-allocation communication between processes.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, segment_size: int = 20 * 1024 * 1024, num_segments: int = 10):
        self.segment_size = segment_size
        self.num_segments = num_segments
        self.available_segments: list[str] = []
        self.all_segments: dict[str, shared_memory.SharedMemory] = {}
        self._pool_lock = threading.Lock()
        
        self._initialize_pool()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _initialize_pool(self):
        for i in range(self.num_segments):
            name = f"bsopt_shm_pool_{i}"
            try:
                # 🚀 SOTA: Aggressive cleanup of stale segments
                try:
                    temp = shared_memory.SharedMemory(name=name)
                    temp.close()
                    temp.unlink()
                    logger.info("cleaned_stale_shm", segment=name)
                except Exception:
                    pass # nosec B110
                
                shm = shared_memory.SharedMemory(name=name, create=True, size=self.segment_size)
                self.all_segments[name] = shm
                self.available_segments.append(name)
            except Exception as e:
                logger.error("shm_pool_init_failed", segment=name, error=str(e))

    def acquire(self) -> str | None:
        """Acquires a segment from the pool with thread-safe pop."""
        with self._pool_lock:
            if not self.available_segments:
                logger.warning("shm_pool_exhausted")
                return None
            seg_name = self.available_segments.pop()
            # 🚀 OPTIMIZATION: Verify segment health before handover
            try:
                shm = self.all_segments[seg_name]
                _ = shm.buf[0] # Test access
                return seg_name
            except Exception as e:
                logger.error("shm_segment_corrupt", segment=seg_name, error=str(e))
                return self.acquire() # Recursive attempt

    def release(self, name: str):
        """Releases a segment back to the pool."""
        with self._pool_lock:
            if name in self.all_segments and name not in self.available_segments:
                self.available_segments.append(name)
            else:
                logger.warning("shm_release_invalid", segment=name)

    def get_segment(self, name: str) -> shared_memory.SharedMemory | None:
        return self.all_segments.get(name)

    def cleanup(self):
        with self._pool_lock:
            for shm in self.all_segments.values():
                try:
                    shm.close()
                    shm.unlink()
                except:
                    pass # nosec B110
            self.all_segments.clear()
            self.available_segments.clear()

# Global manager instance
shm_manager = SharedMemoryManager.get_instance()

# SOTA: Register cleanup with atexit to ensure segments are unlinked on process exit
atexit.register(shm_manager.cleanup)
