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
    Uses dedicated threading and raw binary struct mapping to eliminate Python overhead.
    Designed for future AF_XDP UMEMA integration.
    """
    def __init__(self, interface: str = "eth0", port: int = 5555):
        self.interface = interface
        self.port = port
        self.sock: socket.socket | None = None
        self._running = False
        self._mesh = SharedMemoryRingBuffer()
        self._thread: threading.Thread | None = None

    def start(self, cpu_core: int = 1):
        """Initialize ingestion path in a pinned high-priority thread."""
        self._running = True
        try:
            # 🚀 OPTIMIZATION: Use raw packet socket for zero-stack traversal
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0800))
            try:
                self.sock.bind((self.interface, 0))
            except PermissionError:
                logger.warning("using_udp_standard_socket_dev_mode")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.bind(("0.0.0.0", self.port))
            
            # Set high-performance socket options
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 16) # 16MB buffer
            
            self._thread = threading.Thread(target=self._run_loop, args=(cpu_core,), daemon=True, name="IngestEngine")
            self._thread.start()
            logger.info("ingester_active_pinned", core=cpu_core, interface=self.interface)
            
        except Exception as e:
            logger.error("ingester_init_failed", error=str(e))
            self._running = False

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
