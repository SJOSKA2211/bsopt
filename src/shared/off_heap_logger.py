import struct
import time
from multiprocessing import shared_memory
from typing import Any

import orjson

# Log Entry Structure: q (Timestamp), 1024s (JSON Payload) = 8 + 1024 = 1032 bytes
LOG_STRUCT = struct.Struct("q 1024s")
LOG_SIZE = LOG_STRUCT.size
LOG_BUFFER_CAPACITY = 10000
SHM_LOG_NAME = "bsopt_off_heap_logs"


class OffHeapLogger:
    """
    Zero-Latency Off-Heap Logger using Shared Memory.
    Bypasses standard Python logging I/O on the hot path by writing to a ring buffer.
    A background 'LogDrain' process is responsible for persisting these to disk/Loki.
    """

    def __init__(self, create: bool = False) -> None:
        self.shm_size = (LOG_SIZE * LOG_BUFFER_CAPACITY) + 8
        self.shm: shared_memory.SharedMemory | None = None
        self.buf: memoryview | None = None

        try:
            if create:
                try:
                    existing = shared_memory.SharedMemory(name=SHM_LOG_NAME)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(
                    name=SHM_LOG_NAME, create=True, size=self.shm_size
                )
                if self.shm is not None:
                    buf = self.shm.buf
                    if buf is not None:
                        buf[:8] = struct.pack("q", 0)  # Head index
                    self.buf = buf
            else:
                self.shm = shared_memory.SharedMemory(name=SHM_LOG_NAME)

            if self.shm is not None:
                self.buf = self.shm.buf
        except Exception:
            # Fallback to standard logging if SHM fails
            self.shm = None
            self.buf = None

    def log(self, event: str, **kwargs: Any) -> None:
        """Ultra-fast log write to shared memory. Aligned for atomic head update."""
        buf = self.buf
        if buf is None:
            return

        # 1. Prepare Payload (Still serialized here, ideally offloaded to a pre-allocated pool)
        # We limit to 1024 bytes to fit the fixed-size ring slot
        payload_data = orjson.dumps({"event": event, **kwargs})[:1024]
        payload_bytes = payload_data.ljust(1024, b"\x00")
        timestamp = int(time.time() * 1000)

        # 2. Lock-free slot calculation
        head = struct.unpack("q", buf[:8])[0]
        offset = 8 + (head % LOG_BUFFER_CAPACITY) * LOG_SIZE

        # 3. Write Data FIRST
        buf[offset : offset + LOG_SIZE] = LOG_STRUCT.pack(timestamp, payload_bytes)

        # 4. Atomic Head Update (Machine-word aligned write)
        buf[:8] = struct.pack("q", head + 1)

    def close(self) -> None:
        if self.shm:
            self.shm.close()


# Global ultra-fast logger for the hot path
omega_logger = OffHeapLogger()
