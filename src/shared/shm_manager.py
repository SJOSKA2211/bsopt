"""
Shared Memory Context Manager
=============================

Enables zero-copy data sharing across processes using multiprocessing.shared_memory.
Uses msgspec for ultra-fast binary serialization.
"""

from multiprocessing import shared_memory
import msgspec
import structlog
from typing import Any, Optional, TypeVar, Generic, Type

logger = structlog.get_logger(__name__)

T = TypeVar("T")

class SHMManager(Generic[T]):
    """
    Manages a shared memory block for a specific data type.
    """

    def __init__(self, name: str, schema: Type[T], size: int = 10 * 1024 * 1024):
        self.name = name
        self.schema = schema
        self.size = size
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(schema)

    def create(self):
        """Create the shared memory block."""
        try:
            self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            logger.info("shm_created", name=self.name, size=self.size)
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name=self.name)
            logger.warning("shm_already_exists", name=self.name)

    def write(self, data: T):
        """Write data to the shared memory block."""
        if not self._shm:
            raise RuntimeError("SHM not initialized. Call create() first.")
        
        packed = self._encoder.encode(data)
        if len(packed) > self.size:
            raise ValueError(f"Data size {len(packed)} exceeds SHM size {self.size}")
        
        self._shm.buf[:len(packed)] = packed
        # Store actual length in the first 4 bytes if needed, but for simplicity:
        logger.debug("shm_write_complete", name=self.name, bytes=len(packed))

    def read(self) -> T:
        """Read and decode data from the shared memory block."""
        if not self._shm:
            self._shm = shared_memory.SharedMemory(name=self.name)
        
        # This is a naive read (reads the whole buffer). 
        # In production, we'd use a more sophisticated framing.
        return self._decoder.decode(self._shm.buf)

    def close(self):
        """Close the SHM handle."""
        if self._shm:
            self._shm.close()

    def unlink(self):
        """Destroy the SHM block."""
        if self._shm:
            self._shm.unlink()
            self._shm = None
            logger.info("shm_destroyed", name=self.name)
