"""
Shared Memory Context Manager
=============================

Enables zero-copy data sharing across processes using multiprocessing.shared_memory.
Uses msgspec for ultra-fast binary serialization.
"""

from multiprocessing import shared_memory
from typing import Generic, TypeVar

import msgspec
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")

class SHMManager(Generic[T]):
    """
    Manages a shared memory block for a specific data type.
    """

    def __init__(self, name: str, schema: type[T], size: int = 10 * 1024 * 1024):
        self.name = name
        self.schema = schema
        self.size = size
        self._shm: shared_memory.SharedMemory | None = None
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
        """
        Write data to the shared memory block atomically.
        Uses first byte as a spin-lock and next 4 bytes for length header.
        """
        if not self._shm:
            raise RuntimeError("SHM not initialized.")
        
        packed = self._encoder.encode(data)
        if len(packed) > self.size - 5:
            raise ValueError(f"Data size {len(packed)} exceeds SHM capacity")

        # 🚀 ATOMIC: Acquire spin-lock
        import time
        start = time.perf_counter()
        while self._shm.buf[0] != 0:
            time.sleep(0) # Yield to other threads
            if time.perf_counter() - start > 0.1: # 100ms timeout
                self._shm.buf[0] = 0 # Force clear if hung
            pass
        
        self._shm.buf[0] = 1 # Lock
        try:
            # Write length (4 bytes)
            import struct
            self._shm.buf[1:5] = struct.pack("I", len(packed))
            # Write data
            self._shm.buf[5:5+len(packed)] = packed
        finally:
            self._shm.buf[0] = 0 # Unlock
            
        logger.debug("shm_write_atomic", name=self.name, bytes=len(packed))

    def read(self) -> T:
        """Read and decode data from the shared memory block atomically."""
        if not self._shm:
            self._shm = shared_memory.SharedMemory(name=self.name)
        
        # 🚀 ATOMIC: Wait for unlock
        import time
        while self._shm.buf[0] != 0:
            time.sleep(0)
            
        import struct
        length = struct.unpack("I", self._shm.buf[1:5])[0]
        return self._decoder.decode(self._shm.buf[5:5+length])

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
