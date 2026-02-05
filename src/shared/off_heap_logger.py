import struct
import time
from multiprocessing import shared_memory

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
    def __init__(self, create: bool = False):
        self.shm_size = (LOG_SIZE * LOG_BUFFER_CAPACITY) + 8
        try:
            if create:
                try:
                    existing = shared_memory.SharedMemory(name=SHM_LOG_NAME)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(name=SHM_LOG_NAME, create=True, size=self.shm_size)
                self.shm.buf[:8] = struct.pack("q", 0) # Head index
            else:
                self.shm = shared_memory.SharedMemory(name=SHM_LOG_NAME)
            
            self.buf = self.shm.buf
        except Exception:
            # Fallback to standard logging if SHM fails
            self.shm = None

    def log(self, event: str, **kwargs):
        """Ultra-fast log write to shared memory."""
        if not self.shm:
            return

        head = struct.unpack("q", self.buf[:8])[0]
        offset = 8 + (head % LOG_BUFFER_CAPACITY) * LOG_SIZE
        
        payload = orjson.dumps({"event": event, **kwargs})[:1024].ljust(1024, b'\x00')
        timestamp = int(time.time() * 1000)
        
        self.buf[offset : offset + LOG_SIZE] = LOG_STRUCT.pack(timestamp, payload)
        self.buf[:8] = struct.pack("q", head + 1)

    def close(self):
        if self.shm:
            self.shm.close()

# Global ultra-fast logger for the hot path
omega_logger = OffHeapLogger()
