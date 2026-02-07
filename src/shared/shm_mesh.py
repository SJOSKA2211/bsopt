import os
import struct
from multiprocessing import Lock, shared_memory

import structlog

logger = structlog.get_logger()

# Market Tick Structure: 8s (Symbol), d (Price), q (Volume), d (Timestamp) = 8 + 8 + 8 + 8 = 32 bytes
TICK_STRUCT = struct.Struct("8s d q d")
TICK_SIZE = TICK_STRUCT.size
BUFFER_CAPACITY = 100000  # 100k ticks
SHM_NAME = "market_mesh_ring_buffer"


class SharedMemoryRingBuffer:
    """
    Ultra-high-performance Zero-Copy Ring Buffer using Shared Memory.
    Designed for single-writer (IngestionWorker), multi-reader (PricingEngine) access.
    """

    def __init__(self, create: bool = False):
        self.shm_size = (
            TICK_SIZE * BUFFER_CAPACITY
        ) + 8  # +8 bytes for the atomic head index
        self._lock = Lock()  # Inter-process lock
        self.shm = None
        self.buf = None

        try:
            if create:
                # Destroy existing if it exists to ensure fresh start
                try:
                    existing = shared_memory.SharedMemory(name=SHM_NAME)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(
                    name=SHM_NAME, create=True, size=self.shm_size
                )
                # Initialize head index to 0
                self.shm.buf[:8] = struct.pack("q", 0)
            else:
                try:
                    self.shm = shared_memory.SharedMemory(name=SHM_NAME)
                except FileNotFoundError:
                    # 🚀 SOTA: Graceful fallback for dev/test
                    if os.getenv("ENVIRONMENT") == "prod":
                        raise
                    logger.warning("shm_buffer_missing_using_dummy", name=SHM_NAME)
                    # Create a dummy buffer in process memory for safety
                    self.buf = bytearray(self.shm_size)
                    return

            self.buf = self.shm.buf
            logger.info(
                "shm_buffer_initialized",
                name=SHM_NAME,
                size=self.shm_size,
                create=create,
            )
        except Exception as e:
            logger.error("shm_initialization_failed", error=str(e))
            raise

    def write_tick(self, symbol: str, price: float, volume: int, timestamp: float):
        """Writer: IngestionWorker writes binary tick into the ring."""
        with self._lock:
            head = struct.unpack("q", self.buf[:8])[0]
            offset = 8 + (head % BUFFER_CAPACITY) * TICK_SIZE

            # Pack data into the buffer at the calculated offset
            # Truncate symbol to 8 chars
            sym_bytes = symbol.encode("ascii")[:8].ljust(8, b"\x00")
            self.buf[offset : offset + TICK_SIZE] = TICK_STRUCT.pack(
                sym_bytes, price, volume, timestamp
            )

            # Atomically (effectively) increment head
            self.buf[:8] = struct.pack("q", head + 1)

    def read_latest(self, last_head: int) -> tuple[list, int]:
        """Reader: PricingEngine reads all new ticks since last_head."""
        current_head = struct.unpack("q", self.buf[:8])[0]
        if current_head <= last_head:
            return [], last_head

        new_ticks = []
        # Optimization: limit read if we fell too far behind (overflow)
        start_idx = max(last_head, current_head - BUFFER_CAPACITY)

        for h in range(start_idx, current_head):
            offset = 8 + (h % BUFFER_CAPACITY) * TICK_SIZE
            data = TICK_STRUCT.unpack(self.buf[offset : offset + TICK_SIZE])
            symbol = data[0].decode("ascii").strip("\x00")
            new_ticks.append(
                {
                    "symbol": symbol,
                    "price": data[1],
                    "volume": data[2],
                    "timestamp": data[3],
                }
            )

        return new_ticks, current_head

    def read_latest_raw(self, last_head: int) -> tuple[bytes, int]:
        """
        Ultra-low latency reader: returns a single contiguous bytes object
        containing all new ticks in binary format.
        Perfect for direct WebSocket broadcasting.
        """
        current_head = struct.unpack("q", self.buf[:8])[0]
        if current_head <= last_head:
            return b"", last_head

        start_idx = max(last_head, current_head - BUFFER_CAPACITY)
        num_ticks = current_head - start_idx

        # Determine if we need to wrap around the ring
        start_off = 8 + (start_idx % BUFFER_CAPACITY) * TICK_SIZE
        end_off = 8 + (current_head % BUFFER_CAPACITY) * TICK_SIZE

        if start_off < end_off:
            # Contiguous read
            return bytes(self.buf[start_off:end_off]), current_head
        else:
            # Wrapped read: two fragments
            first_part = self.buf[start_off : 8 + BUFFER_CAPACITY * TICK_SIZE]
            second_part = self.buf[8:end_off]
            return bytes(first_part) + bytes(second_part), current_head

    def close(self):
        self.shm.close()

    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
