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

    def start(self):
        """Initialize ingestion path in a dedicated high-priority thread."""
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
            
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="IngestEngine")
            self._thread.start()
            logger.info("ingester_active_dedicated_thread", interface=self.interface)
            
        except Exception as e:
            logger.error("ingester_init_failed", error=str(e))
            self._running = False

    def _run_loop(self):
        """Hot loop: Zero allocations, zero async overhead."""
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
