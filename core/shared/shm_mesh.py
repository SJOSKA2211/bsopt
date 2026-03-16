import os
import struct
import time
from multiprocessing import shared_memory

import msgspec
import numpy as np
import structlog

logger = structlog.get_logger()
memoryview_type = memoryview

# Market Tick Structure: 8s (Symbol), d (Price), q (Volume), d (Timestamp), q (receive_ts_ns) = 40 bytes
TICK_DTYPE = np.dtype(
    [
        ("symbol", "S8"),
        ("price", "f8"),
        ("volume", "i8"),
        ("timestamp", "f8"),
        ("receive_ts_ns", "i8"),
    ]
)
TICK_SIZE = TICK_DTYPE.itemsize
BUFFER_CAPACITY = 100000  # 100k ticks
SHM_NAME = "market_mesh_ring_buffer"


class MarketTick(msgspec.Struct):
    symbol: str
    price: float
    volume: int
    timestamp: float
    receive_ts_ns: int


# Order Command: 8s (Symbol), d (Price), q (Quantity), i (Side), d (Delta), d (Gamma), d (Vega), q (submit_ts_ns) = 60 bytes
ORDER_DTYPE = np.dtype(
    [
        ("symbol", "S8"),
        ("price", "f8"),
        ("quantity", "i8"),
        ("side", "i4"),
        ("delta", "f8"),
        ("gamma", "f8"),
        ("vega", "f8"),
        ("submit_ts_ns", "i8"),
    ]
)
ORDER_SIZE = ORDER_DTYPE.itemsize
ORDER_BUFFER_CAPACITY = 1000
SHM_ORDER_NAME = "order_command_buffer"

# Execution Status: q (OrderID), d (FillPrice), q (FillQty), i (Status), q (exec_ts_ns) = 36 bytes
EXEC_DTYPE = np.dtype(
    [
        ("order_id", "i8"),
        ("fill_price", "f8"),
        ("fill_qty", "i8"),
        ("status", "i4"),
        ("exec_ts_ns", "i8"),
    ]
)
EXEC_SIZE = EXEC_DTYPE.itemsize
EXEC_BUFFER_CAPACITY = 1000
SHM_EXEC_NAME = "execution_status_buffer"

# Risk State: d (CurrentDelta), d (CurrentGamma), d (CurrentVega), d (MaxDelta), d (MaxGamma), d (MaxVega), d (MarginUsage), q (last_sync_ts_ns) = 64 bytes
RISK_STATE_DTYPE = np.dtype(
    [
        ("current_delta", "f8"),
        ("current_gamma", "f8"),
        ("current_vega", "f8"),
        ("max_delta", "f8"),
        ("max_gamma", "f8"),
        ("max_vega", "f8"),
        ("margin_usage", "f8"),
        ("last_sync_ts_ns", "i8"),
    ]
)
SHM_RISK_NAME = "risk_state_buffer"


# Greeks State: 8s (Symbol), d (Delta), d (Gamma), d (Theta), d (Vega), d (Rho), q (calc_ts_ns) = 48 bytes
GREEKS_DTYPE = np.dtype(
    [
        ("symbol", "S8"),
        ("delta", "f8"),
        ("gamma", "f8"),
        ("theta", "f8"),
        ("vega", "f8"),
        ("rho", "f8"),
        ("calc_ts_ns", "i8"),
    ]
)
GREEKS_SIZE = GREEKS_DTYPE.itemsize
GREEKS_BUFFER_CAPACITY = 1000
SHM_GREEKS_NAME = "greeks_mesh_buffer"
# Map-based Greeks Snapshot: [Symbol(8s), Delta(d), Gamma(d), Theta(d), Vega(d), Rho(d), CalcTs(q)] * 2000 symbols
GREEKS_MAP_CAPACITY = 2000
GREEKS_MAP_SIZE = GREEKS_SIZE * GREEKS_MAP_CAPACITY


class GreeksMesh:
    """O(1) Map-based Greeks Mesh for instant lookup."""

    def __init__(self, create: bool = False):
        self.size = GREEKS_MAP_SIZE
        self.shm: shared_memory.SharedMemory | None = None
        self.buf: memoryview
        try:
            if create:
                try:
                    existing = shared_memory.SharedMemory(name="greeks_snapshot")
                    existing.close()
                    existing.unlink()
                except Exception:
                    pass
                sm = shared_memory.SharedMemory(name="greeks_snapshot", create=True, size=self.size)
                self.shm = sm
                self.buf = sm.buf
            else:
                sm = shared_memory.SharedMemory(name="greeks_snapshot")
                self.shm = sm
                self.buf = sm.buf

            self.view = np.frombuffer(self.buf, dtype=GREEKS_DTYPE, count=GREEKS_MAP_CAPACITY)

            # Fast index for symbol lookup: { "AAPL": index }
            self._symbol_to_idx = {}
            self._refresh_index()
        except Exception as e:
            if not create:
                logger.warning("greeks_snapshot_missing_using_dummy", error=str(e))
                self.buf = memoryview(bytearray(self.size))
                self.view = np.frombuffer(self.buf, dtype=GREEKS_DTYPE, count=GREEKS_MAP_CAPACITY)
            else:
                raise

    def _refresh_index(self):
        """Rebuild the local symbol-to-index map from SHM."""
        self._symbol_to_idx.clear()
        for i in range(GREEKS_MAP_CAPACITY):
            sym = self.view[i]["symbol"].decode("ascii").strip("\x00")
            if sym:
                self._symbol_to_idx[sym] = i

    def write(self, symbol: str, delta: float, gamma: float, theta: float, vega: float, rho: float):
        """Update the latest Greeks for a symbol."""
        idx = self._symbol_to_idx.get(symbol)
        if idx is None:
            # Find first empty slot or use a simple hash
            # For speed, we use a simple linear probe if collision
            h = hash(symbol) % GREEKS_MAP_CAPACITY
            for i in range(GREEKS_MAP_CAPACITY):
                probe_idx = (h + i) % GREEKS_MAP_CAPACITY
                existing = self.view[probe_idx]["symbol"].strip(b"\x00")
                if not existing or existing.decode("ascii") == symbol:
                    idx = probe_idx
                    self._symbol_to_idx[symbol] = idx
                    break

        if idx is not None:
            self.view[idx] = (
                symbol.encode("ascii")[:8],
                delta,
                gamma,
                theta,
                vega,
                rho,
                time.time_ns(),
            )

    def read(self, symbol: str) -> dict | None:
        """Instant O(1) Greek lookup."""
        idx = self._symbol_to_idx.get(symbol)
        if idx is None:
            # Try one full scan in case index is stale
            self._refresh_index()
            idx = self._symbol_to_idx.get(symbol)

        if idx is not None:
            data = self.view[idx]
            return {
                "delta": float(data["delta"]),
                "gamma": float(data["gamma"]),
                "theta": float(data["theta"]),
                "vega": float(data["vega"]),
                "rho": float(data["rho"]),
                "timestamp": int(data["calc_ts_ns"]),
            }
        return None


class GreeksBuffer:
    """High-Performance Greeks Mesh for real-time risk observability."""

    def __init__(self, create: bool = False):
        self.size = (GREEKS_SIZE * GREEKS_BUFFER_CAPACITY) + 8
        self.shm: shared_memory.SharedMemory | None = None
        self.buf: memoryview
        try:
            if create:
                try:
                    existing = shared_memory.SharedMemory(name=SHM_GREEKS_NAME)
                    existing.close()
                    existing.unlink()
                except Exception:
                    pass
                sm = shared_memory.SharedMemory(name=SHM_GREEKS_NAME, create=True, size=self.size)
                self.shm = sm
                self.buf = sm.buf
                struct.pack_into("q", self.buf, 0, 0)
            else:
                sm = shared_memory.SharedMemory(name=SHM_GREEKS_NAME)
                self.shm = sm
                self.buf = sm.buf

            self.view = np.frombuffer(
                self.buf, dtype=GREEKS_DTYPE, offset=8, count=GREEKS_BUFFER_CAPACITY
            )
        except Exception as e:
            if not create:
                logger.warning("greeks_shm_missing_using_dummy", error=str(e))
                self.buf = memoryview(bytearray(self.size))
                self.view = np.frombuffer(
                    self.buf, dtype=GREEKS_DTYPE, offset=8, count=GREEKS_BUFFER_CAPACITY
                )
            else:
                raise

    def write_greeks(
        self, symbol: str, delta: float, gamma: float, theta: float, vega: float, rho: float
    ):
        """Writer: Direct write into Greeks Mesh."""
        self.write_greeks_raw(symbol.encode("ascii")[:8], delta, gamma, theta, vega, rho)

    def write_greeks_raw(
        self, sym_bytes: bytes, delta: float, gamma: float, theta: float, vega: float, rho: float
    ):
        """Zero-copy writer for raw bytes symbol."""
        head = struct.unpack_from("q", self.buf, 0)[0]
        idx = head % GREEKS_BUFFER_CAPACITY
        self.view[idx] = (
            sym_bytes,
            delta,
            gamma,
            theta,
            vega,
            rho,
            time.time_ns(),
        )
        struct.pack_into("q", self.buf, 0, head + 1)


class RiskStateBuffer:
    """Zero-Latency Risk State Buffer for Engine-Worker Synchronization."""

    def __init__(self, create: bool = False):
        self.size = RISK_STATE_DTYPE.itemsize
        self.shm: shared_memory.SharedMemory | None = None
        self.buf: memoryview
        try:
            if create:
                sm = shared_memory.SharedMemory(name=SHM_RISK_NAME, create=True, size=self.size)
            else:
                sm = shared_memory.SharedMemory(name=SHM_RISK_NAME)
            self.shm = sm
            self.buf = sm.buf
            self.view = np.frombuffer(self.buf, dtype=RISK_STATE_DTYPE, count=1)
        except Exception as e:
            if not create:
                logger.warning("risk_shm_missing_using_dummy", error=str(e))
                local_data = bytearray(self.size)
                self.buf = memoryview(local_data)
                self.view = np.frombuffer(self.buf, dtype=RISK_STATE_DTYPE, count=1)
            else:
                raise

    def update(
        self,
        current_delta: float,
        current_gamma: float,
        current_vega: float,
        max_delta: float,
        max_gamma: float,
        max_vega: float,
        margin_usage: float = 0.0,
    ):
        if self.view is not None:
            self.view[0] = (
                current_delta,
                current_gamma,
                current_vega,
                max_delta,
                max_gamma,
                max_vega,
                margin_usage,
                time.time_ns(),
            )

    def read(self) -> tuple[float, float, float, float, float, float, float, int]:
        if self.view is not None:
            data = self.view[0]
            return (
                float(data["current_delta"]),
                float(data["current_gamma"]),
                float(data["current_vega"]),
                float(data["max_delta"]),
                float(data["max_gamma"]),
                float(data["max_vega"]),
                float(data["margin_usage"]),
                int(data["last_sync_ts_ns"]),
            )
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


class OrderBuffer:
    """Lock-Free Order Command Buffer (Agent -> Engine)."""

    def __init__(self, create: bool = False):
        self.size = (ORDER_SIZE * ORDER_BUFFER_CAPACITY) + 8
        self.shm: shared_memory.SharedMemory | None = None
        self.buf: memoryview
        try:
            if create:
                sm = shared_memory.SharedMemory(name=SHM_ORDER_NAME, create=True, size=self.size)
            else:
                sm = shared_memory.SharedMemory(name=SHM_ORDER_NAME)
            self.shm = sm
            self.buf = sm.buf
            if create:
                struct.pack_into("q", self.buf, 0, 0)
            self.view = np.frombuffer(
                self.buf, dtype=ORDER_DTYPE, offset=8, count=ORDER_BUFFER_CAPACITY
            )
        except Exception as e:
            if not create:
                logger.warning("order_shm_missing_using_dummy", error=str(e))
                self.buf = memoryview(bytearray(self.size))
                self.view = np.frombuffer(
                    self.buf, dtype=ORDER_DTYPE, offset=8, count=ORDER_BUFFER_CAPACITY
                )
            else:
                raise

    def write_order(
        self,
        symbol: str,
        price: float,
        qty: int,
        side: int,
        delta: float = 0.0,
        gamma: float = 0.0,
        vega: float = 0.0,
    ):
        # Extract head index from the first 8 bytes
        head = struct.unpack_from("q", self.buf, 0)[0]
        # Submitting order
        ts_ns = time.time_ns()
        self.view[head % ORDER_BUFFER_CAPACITY] = (
            symbol.encode("ascii")[:8],
            price,
            qty,
            side,
            delta,
            gamma,
            vega,
            ts_ns,
        )
        struct.pack_into("q", self.buf, 0, head + 1)


class ExecutionBuffer:
    """Lock-Free Execution Status Buffer (Engine -> Agent)."""

    def __init__(self, create: bool = False):
        self.size = (EXEC_SIZE * EXEC_BUFFER_CAPACITY) + 8
        self.shm: shared_memory.SharedMemory | None = None
        self.buf: memoryview
        try:
            if create:
                sm = shared_memory.SharedMemory(name=SHM_EXEC_NAME, create=True, size=self.size)
            else:
                sm = shared_memory.SharedMemory(name=SHM_EXEC_NAME)
            self.shm = sm
            self.buf = sm.buf
            if create:
                struct.pack_into("q", self.buf, 0, 0)
            self.view = np.frombuffer(
                self.buf, dtype=EXEC_DTYPE, offset=8, count=EXEC_BUFFER_CAPACITY
            )
        except Exception as e:
            if not create:
                logger.warning("exec_shm_missing_using_dummy", error=str(e))
                self.buf = memoryview(bytearray(self.size))
                self.view = np.frombuffer(
                    self.buf, dtype=EXEC_DTYPE, offset=8, count=EXEC_BUFFER_CAPACITY
                )
            else:
                raise

    def write_exec(self, order_id: int, price: float, qty: int, status: int):
        # Extract head index from the first 8 bytes
        head = struct.unpack_from("q", self.buf, 0)[0]
        # Executing response
        ts_ns = time.time_ns()
        self.view[head % EXEC_BUFFER_CAPACITY] = (order_id, price, qty, status, ts_ns)
        struct.pack_into("q", self.buf, 0, head + 1)


class SharedMemoryRingBuffer:
    """
    Ultra-high-performance Lock-Free Zero-Copy Ring Buffer.
    Optimized for Single-Writer/Multi-Reader (SWMR) concurrency.
    The first 8 bytes are the 'head' index (uint64).
    """

    def __init__(self, create: bool = False):
        self.shm_size = (TICK_SIZE * BUFFER_CAPACITY) + 8
        self.shm: shared_memory.SharedMemory | None = None
        self.buf: memoryview

        try:
            if create:
                try:
                    existing = shared_memory.SharedMemory(name=SHM_NAME)
                    existing.close()
                    existing.unlink()
                except Exception:
                    pass
                sm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=self.shm_size)
                self.shm = sm
                self.buf = sm.buf
                # Initialize head index to 0
                struct.pack_into("q", self.buf, 0, 0)
            else:
                try:
                    sm = shared_memory.SharedMemory(name=SHM_NAME)
                    self.shm = sm
                    self.buf = sm.buf
                except Exception:
                    if os.getenv("ENVIRONMENT") == "prod":
                        raise
                    logger.warning("shm_buffer_missing_using_dummy", name=SHM_NAME)
                    self.buf = memoryview(bytearray(self.shm_size))

            # Create a numpy view of the entire tick buffer (skipping head index)
            self.data_view = np.frombuffer(
                self.buf, dtype=TICK_DTYPE, offset=8, count=BUFFER_CAPACITY
            )

            logger.info(
                "shm_buffer_initialized_lock_free",
                name=SHM_NAME,
                size=self.shm_size,
                create=create,
            )
        except Exception as e:
            logger.error("shm_initialization_failed", error=str(e))
            raise

    def write_tick(self, symbol: str, price: float, volume: int, timestamp: float):
        """Writer: Direct write into numpy view with atomic index update."""
        # OPTIMIZED: Use pre-allocated bytes for symbol
        sym_bytes = symbol.encode("ascii")[:8]
        self.write_tick_raw(sym_bytes, price, volume, timestamp)

    def write_tick_raw(self, sym_bytes: bytes, price: float, volume: int, timestamp: float):
        """Zero-copy writer for raw bytes symbol."""
        # Lock-free head calculation from the prefix
        current_head = struct.unpack_from("q", self.buf, 0)[0]
        idx = current_head % BUFFER_CAPACITY

        receive_ts_ns = time.time_ns()

        # 1. Write Data FIRST (Data must be visible before head update)
        self.data_view[idx] = (sym_bytes, price, volume, timestamp, receive_ts_ns)

        # 2. Atomic Head update (Memory Barrier)
        struct.pack_into("q", self.buf, 0, current_head + 1)

    def read_latest_slices(self, last_head: int) -> tuple[list[np.ndarray], int]:
        """
        Reader: Yields 1 or 2 zero-copy slices to avoid concatenation allocation.
        """
        current_head = struct.unpack_from("q", self.buf, 0)[0]
        if current_head <= last_head:
            return [], last_head

        start_idx = max(last_head, current_head - BUFFER_CAPACITY)

        s = start_idx % BUFFER_CAPACITY
        e = current_head % BUFFER_CAPACITY

        if s < e:
            # Single slice - zero copy
            return [self.data_view[s:e]], current_head
        # Wrap around - return TWO slices to maintain zero-copy
        return [self.data_view[s:], self.data_view[:e]], current_head

    def read_latest_view(self, last_head: int) -> tuple[np.ndarray, int]:
        """
        Returns a single numpy view of the new ticks.
        Note: If the buffer wrapped, this will return a copy (concatenated).
        """
        slices, head = self.read_latest_slices(last_head)
        if not slices:
            return np.array([], dtype=TICK_DTYPE), head
        if len(slices) == 1:
            return slices[0], head

        # Concatenate is NOT zero-copy, but necessary for a single view if wrapped
        return np.concatenate(slices), head

    def read_latest_msgspec(self, last_head: int) -> tuple[list[MarketTick], int]:
        """High-level reader using msgspec for speed. Optimized with vectorized string decoding."""
        view, head = self.read_latest_view(last_head)
        if view is None or len(view) == 0:
            return [], head

        # Vectorized decoding of all symbols at once
        symbols = np.char.decode(view["symbol"], "ascii")
        prices = view["price"]
        volumes = view["volume"]
        timestamps = view["timestamp"]
        r_ts = view["receive_ts_ns"]

        # List comprehension is faster than append in a loop for msgspec creation
        ticks = [
            MarketTick(
                symbol=str(symbols[i]).strip("\x00"),
                price=float(prices[i]),
                volume=int(volumes[i]),
                timestamp=float(timestamps[i]),
                receive_ts_ns=int(r_ts[i]),
            )
            for i in range(len(view))
        ]
        return ticks, head

    def close(self):
        if hasattr(self, "data_view"):
            del self.data_view
        if self.shm is not None:
            self.shm.close()

    def unlink(self):
        if self.shm is not None:
            try:
                self.shm.unlink()
            except Exception:
                pass
