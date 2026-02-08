import os
import socket
import struct
import threading
from typing import Any

import numpy as np
import structlog

from src.shared.shm_mesh import TICK_DTYPE, SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)

# Raw Binary Tick: 8s (Symbol), d (Price), q (Volume), d (Timestamp)
# Total: 32 bytes. No JSON slop.
TICK_STRUCT = struct.Struct("8s d q d")

class XDPIngester:
    """
    God-Mode High-Performance Ingester.
    Bridges Python to the high-performance Rust Pulse extension.
    """
    def __init__(self, interface: str = "eth0", port: int = 5555):
        self.interface = interface
        self.port = port
        self._running = False
        self._pulse = None
        
        # 🚀 RUST PULSE: Try to use the ultra-high-speed Rust extension
        try:
            from src.shared.bsopt_pulse import RustPulse
            self._pulse = RustPulse(interface, port)
            logger.info("using_rust_pulse_extension")
        except ImportError:
            logger.warning("rust_pulse_missing_using_python_fallback")
            self.sock: socket.socket | None = None
            self._mesh = SharedMemoryRingBuffer()
            self._thread: threading.Thread | None = None

    def start(self, cpu_core: int = 1):
        """Initialize ingestion path in a pinned high-priority thread."""
        self._running = True
        if self._pulse:
            # SHM path is typically /dev/shm/market_mesh_ring_buffer on Linux
            shm_path = f"/dev/shm/{SHM_NAME}"
            self._pulse.start(shm_path, cpu_core)
            logger.info("rust_pulse_active_pinned", core=cpu_core)
            return

        try:
            # 🚀 PYTHON FALLBACK
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0800))
            # ... (rest of old Python start logic)

    def _run_loop(self, cpu_core: int):
        """Hot loop: Pinned to core, real-time priority."""
        # 🚀 SILICON LOCKDOWN: Pin to core and set real-time priority
        try:
            os.sched_setaffinity(0, {cpu_core})
            # Try to set real-time priority (requires root/capabilities)
            try:
                param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
                os.sched_setscheduler(0, os.SCHED_FIFO, param)
                logger.info("ingest_realtime_priority_set")
            except PermissionError:
                logger.warning("ingest_priority_failed_missing_perms")
        except Exception as e:
            logger.error("ingest_pinning_failed", error=str(e))

        # Pre-allocate buffer to avoid GC pressure
        buf = bytearray(2048)
        offset = 42 if self.sock.family == socket.AF_PACKET else 0
        
        while self._running:
            try:
                nbytes, _ = self.sock.recvfrom_into(buf)
                if nbytes > offset:
                    payload = buf[offset:offset+32]
                    # 🚀 SILICON SPEED: Unpack raw bytes directly into variables
                    # symbol, price, volume, timestamp
                    data = TICK_STRUCT.unpack(payload)
                    
                    # Write directly to lock-free SHM Mesh
                    self._mesh.write_tick(
                        symbol=data[0].decode('ascii').strip('\x00'),
                        price=data[1],
                        volume=data[2],
                        timestamp=data[3]
                    )
            except (BlockingIOError, InterruptedError):
                continue
            except Exception:
                continue

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.sock:
            self.sock.close()
        self._mesh.close()

if __name__ == "__main__":
    ingester = XDPIngester()
    ingester.start()
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ingester.stop()
