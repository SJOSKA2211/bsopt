import os
import struct
from multiprocessing import Lock, shared_memory

import msgspec
import numpy as np
import structlog

logger = structlog.get_logger()

# Market Tick Structure: 8s (Symbol), d (Price), q (Volume), d (Timestamp) = 32 bytes
TICK_DTYPE = np.dtype([
    ('symbol', 'S8'),
    ('price', 'f8'),
    ('volume', 'i8'),
    ('timestamp', 'f8')
])
TICK_SIZE = TICK_DTYPE.itemsize
BUFFER_CAPACITY = 100000 # 100k ticks
SHM_NAME = "market_mesh_ring_buffer"

class MarketTick(msgspec.Struct):
    symbol: str
    price: float
    volume: int
    timestamp: float

# Order Command: 8s (Symbol), d (Price), q (Quantity), i (Side: 1=Buy, -1=Sell) = 28 bytes
ORDER_DTYPE = np.dtype([
    ('symbol', 'S8'),
    ('price', 'f8'),
    ('quantity', 'i8'),
    ('side', 'i4')
])
ORDER_SIZE = ORDER_DTYPE.itemsize
ORDER_BUFFER_CAPACITY = 1000
SHM_ORDER_NAME = "order_command_buffer"

# Execution Status: q (OrderID), d (FillPrice), q (FillQty), i (Status) = 28 bytes
EXEC_DTYPE = np.dtype([
    ('order_id', 'i8'),
    ('fill_price', 'f8'),
    ('fill_qty', 'i8'),
    ('status', 'i4')
])
EXEC_SIZE = EXEC_DTYPE.itemsize
EXEC_BUFFER_CAPACITY = 1000
SHM_EXEC_NAME = "execution_status_buffer"

class OrderBuffer:
    """Lock-Free Order Command Buffer (Agent -> Engine)."""
    def __init__(self, create: bool = False):
        self.size = (ORDER_SIZE * ORDER_BUFFER_CAPACITY) + 8
        self.shm = shared_memory.SharedMemory(name=SHM_ORDER_NAME, create=create, size=self.size) if create else shared_memory.SharedMemory(name=SHM_ORDER_NAME)
        self.buf = self.shm.buf
        if create: self.buf[:8] = struct.pack("q", 0)
        self.view = np.frombuffer(self.buf, dtype=ORDER_DTYPE, offset=8, count=ORDER_BUFFER_CAPACITY)

    def write_order(self, symbol: str, price: float, qty: int, side: int):
        head = struct.unpack("q", self.buf[:8])[0]
        self.view[head % ORDER_BUFFER_CAPACITY] = (symbol.encode('ascii')[:8], price, qty, side)
        self.buf[:8] = struct.pack("q", head + 1)

class ExecutionBuffer:
    """Lock-Free Execution Status Buffer (Engine -> Agent)."""
    def __init__(self, create: bool = False):
        self.size = (EXEC_SIZE * EXEC_BUFFER_CAPACITY) + 8
        self.shm = shared_memory.SharedMemory(name=SHM_EXEC_NAME, create=create, size=self.size) if create else shared_memory.SharedMemory(name=SHM_EXEC_NAME)
        self.buf = self.shm.buf
        if create: self.buf[:8] = struct.pack("q", 0)
        self.view = np.frombuffer(self.buf, dtype=EXEC_DTYPE, offset=8, count=EXEC_BUFFER_CAPACITY)

    def write_exec(self, order_id: int, price: float, qty: int, status: int):
        head = struct.unpack("q", self.buf[:8])[0]
        self.view[head % EXEC_BUFFER_CAPACITY] = (order_id, price, qty, status)
        self.buf[:8] = struct.pack("q", head + 1)

class SharedMemoryRingBuffer:
    """
    Ultra-high-performance Lock-Free Zero-Copy Ring Buffer.
    Optimized for Single-Writer/Multi-Reader (SWMR) concurrency.
    The first 8 bytes are the 'head' index (uint64).
    """
    def __init__(self, create: bool = False):
        self.shm_size = (TICK_SIZE * BUFFER_CAPACITY) + 8 
        self.shm = None
        self.buf = None
        
        try:
            if create:
                try:
                    existing = shared_memory.SharedMemory(name=SHM_NAME)
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=self.shm_size)
                # Initialize head index to 0
                self.shm.buf[:8] = struct.pack("q", 0)
            else:
                try:
                    self.shm = shared_memory.SharedMemory(name=SHM_NAME)
                except FileNotFoundError:
                    if os.getenv("ENVIRONMENT") == "prod":
                        raise
                    logger.warning("shm_buffer_missing_using_dummy", name=SHM_NAME)
                    self.buf = bytearray(self.shm_size)
                    return
            
            self.buf = self.shm.buf
            # Create a numpy view of the entire tick buffer (skipping head index)
            self.data_view = np.frombuffer(self.buf, dtype=TICK_DTYPE, offset=8, count=BUFFER_CAPACITY)
            
            logger.info("shm_buffer_initialized_lock_free", name=SHM_NAME, size=self.shm_size, create=create)
        except Exception as e:
            logger.error("shm_initialization_failed", error=str(e))
            raise

    def write_tick(self, symbol: str, price: float, volume: int, timestamp: float):
        """Writer: Direct write into numpy view with atomic index update."""
        # 🚀 GOD MODE: No Lock. Single writer assumes exclusive access to the 'head' calculation.
        current_head = struct.unpack("q", self.buf[:8])[0]
        idx = current_head % BUFFER_CAPACITY
        
        # Zero-copy write via numpy view
        self.data_view[idx] = (symbol.encode('ascii')[:8], price, volume, timestamp)
        
        # Atomic head update (Readers will see the new head and read the data)
        self.buf[:8] = struct.pack("q", current_head + 1)

    def read_latest_view(self, last_head: int) -> tuple[np.ndarray, int]:
        """
        Reader: Lock-free polling of the head index.
        """
        current_head = struct.unpack("q", self.buf[:8])[0]
        if current_head <= last_head:
            return np.array([], dtype=TICK_DTYPE), last_head
        
        start_idx = max(last_head, current_head - BUFFER_CAPACITY)
        current_head - start_idx
        
        s = start_idx % BUFFER_CAPACITY
        e = current_head % BUFFER_CAPACITY
        
        if s < e:
            # Simple slice - zero copy
            return self.data_view[s:e], current_head
        else:
            # Wrap around - must concatenate (copy unavoidable here)
            return np.concatenate([self.data_view[s:], self.data_view[:e]]), current_head

    def read_latest_msgspec(self, last_head: int) -> tuple[list[MarketTick], int]:
        """High-level reader using msgspec for speed."""
        view, head = self.read_latest_view(last_head)
        # Faster than dictionary comprehension
        return [MarketTick(t['symbol'].decode().strip('\x00'), t['price'], t['volume'], t['timestamp']) for t in view], head

    def close(self):
        if hasattr(self, 'data_view'):
            del self.data_view
        if self.shm:
            self.shm.close()

    def unlink(self):
        if self.shm:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
