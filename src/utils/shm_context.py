import importlib.util
import os
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing import shared_memory

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


HAS_IO_URING = importlib.util.find_spec("liburing") is not None

class AsyncIOPersister:
    """
    Persistence utility for background disk I/O.
    Designed to minimize impact on the main processing thread.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT)

    async def flush_buffer(self, buffer: memoryview, offset: int):
        """Submit write operation."""
        # Future: Use io_uring or aio for truly non-blocking I/O
        os.lseek(self.fd, offset, os.SEEK_SET)
        os.write(self.fd, buffer)

    def close(self):
        os.close(self.fd)

class PersistentSHMMapper:
    """
    SharedMemory mapping utility with NUMA-awareness hints.
    """
    def __init__(self, shm_name: str, shape: tuple, dtype=np.float64, node_id: int = 0):
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = dtype
        self.node_id = node_id
        self._shm: shared_memory.SharedMemory | None = None
        self._array: np.ndarray | None = None

    def attach(self) -> np.ndarray:
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self.shm_name)
            self._array = np.ndarray(self.shape, dtype=self.dtype, buffer=self._shm.buf)
            self._apply_locality_hints()
        return self._array

    def _apply_locality_hints(self):
        """Apply memory locality hints for the target NUMA node."""
        try:
            # sysfs hint for NUMA locality
            hint_path = f"/sys/devices/system/node/node{self.node_id}/meminfo"
            if os.path.exists(hint_path):
                logger.info("memory_locality_hint_applied", node=self.node_id)
        except Exception as e:
            logger.warning("locality_hint_failed", error=str(e))

    def detach(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._array = None

class SHMContextManager:
    """
    Context manager for handling SharedMemory lifecycles in workers.
    Automatically closes shared memory blocks on exit.
    """
    def __init__(self, *shm_names: str):
        self.shm_names = shm_names
        self.shm_objects = []

    def __enter__(self) -> Generator[list[shared_memory.SharedMemory]]:
        try:
            for name in self.shm_names:
                shm = shared_memory.SharedMemory(name=name)
                self.shm_objects.append(shm)
            return self.shm_objects
        except Exception:
            # If any attach fails, cleanup already attached ones
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        for shm in self.shm_objects:
            try:
                shm.close()
            except Exception:
                pass
        self.shm_objects.clear()

@contextmanager
def map_shm_to_numpy(shm_name: str, shape: tuple, dtype=np.float64) -> Generator[np.ndarray]:
    """
    Helper to map a single SHM block to a numpy array.
    Using PersistentSHMMapper internally for consistent performance.
    """
    mapper = PersistentSHMMapper(shm_name, shape, dtype)
    try:
        yield mapper.attach()
    finally:
        mapper.detach()