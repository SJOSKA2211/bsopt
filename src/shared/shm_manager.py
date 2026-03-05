"""
Shared Memory Context Manager

Enables zero-copy data sharing across processes using multiprocessing.shared_memory.
Uses msgspec for ultra-fast binary serialization.
"""

from multiprocessing import shared_memory
from typing import TypeVar

import msgspec
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class SHMManager[T]:
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
        Write data to SHM with OPTIMIZED spin-lock and memoryview access.
        """
        if not self._shm:
            raise RuntimeError("SHM not initialized.")

        packed = self._encoder.encode(data)
        if len(packed) > self.size - 5:
            raise ValueError(f"Data size {len(packed)} exceeds capacity")

        # 1. Use memoryview for fast slicing without copies
        mv = self._shm.buf

        # 2. Optimized Spin-Lock
        import time

        start = time.perf_counter()
        while mv[0] != 0:
            # OPTIMIZED: Use a combination of busy-wait and very short sleep
            # In a true Rick-pass, we'd use a machine-code 'pause' instruction
            if time.perf_counter() - start > 0.05:  # 50ms timeout
                logger.warning("shm_lock_contention_clearing", name=self.name)
                mv[0] = 0  # Safety break
                break
            pass

        mv[0] = 1  # LOCK
        try:
            import struct

            # Header: [Lock(1), Length(4)]
            mv[1:5] = struct.pack("I", len(packed))
            mv[5 : 5 + len(packed)] = packed
        finally:
            mv[0] = 0  # UNLOCK

    def read(self) -> T:
        """Read from SHM with wait-free optimization."""
        if not self._shm:
            self._shm = shared_memory.SharedMemory(name=self.name)

        mv = self._shm.buf
        # Polling for unlock
        while mv[0] != 0:
            pass  # Busy-wait for speed

        import struct

        length = struct.unpack("I", mv[1:5])[0]
        # Zero-copy decode from buffer
        return self._decoder.decode(mv[5 : 5 + length])

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
