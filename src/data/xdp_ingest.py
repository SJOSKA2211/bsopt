import os
import socket
import struct
import threading

import structlog

from src.shared.shm_mesh import SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)

# Raw Binary Tick: 8s (Symbol), d (Price), q (Volume), d (Timestamp)
# Total: 32 bytes. No JSON slop.
TICK_STRUCT = struct.Struct("8s d q d")
SHM_NAME = "market_mesh_ring_buffer"

class XDPIngester:
    """
    High-Performance Ingester.
    Bridges Python to the high-performance Rust Pulse extension.
    """
    def __init__(self, interface: str = "eth0", port: int = 5555):
        self.interface = interface
        self.port = port
        self._running = False
        self._pulse = None
        
        #  RUST PULSE: Try to use the ultra-high-speed Rust extension
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
            shm_path = f"/dev/shm/{SHM_NAME}"
            self._pulse.start(shm_path, cpu_core)
            logger.info("rust_pulse_active_pinned", core=cpu_core)
            return

        try:
            #  PYTHON FALLBACK
            # Use AF_INET/UDP for generic portability if AF_PACKET fails
            try:
                self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0800))
                self.sock.bind((self.interface, 0))
            except (AttributeError, PermissionError):
                logger.warning("af_packet_failed_using_udp_fallback")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.bind(("", self.port))

            self._thread = threading.Thread(target=self._run_loop, args=(cpu_core,), daemon=True)
            self._thread.start()
            logger.info("python_ingest_started", core=cpu_core)
        except Exception as e:
            logger.error("ingest_start_failed", error=str(e))
            self._running = False

    def _run_loop(self, cpu_core: int):
        """Hot loop: Pinned to core, real-time priority."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            try:
                param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
                os.sched_setscheduler(0, os.SCHED_FIFO, param)
                logger.info("ingest_realtime_priority_set")
            except (PermissionError, AttributeError):
                logger.warning("ingest_priority_failed_missing_perms")
        except Exception as e:
            logger.error("ingest_pinning_failed", error=str(e))

        buf = bytearray(2048)
        # Offset 42 for Ethernet+IP+UDP header if using RAW AF_PACKET
        offset = 42 if hasattr(socket, 'AF_PACKET') and self.sock and self.sock.family == socket.AF_PACKET else 0
        
        while self._running:
            try:
                nbytes, _ = self.sock.recvfrom_into(buf)
                if nbytes > offset:
                    payload = buf[offset:offset+32]
                    if len(payload) < 32:
                        continue
                    data = TICK_STRUCT.unpack(payload)
                    
                    self._mesh.write_tick(
                        symbol=data[0].decode('ascii', errors='ignore').strip('\x00'),
                        price=data[1],
                        volume=data[2],
                        timestamp=data[3]
                    )
            except (BlockingIOError, InterruptedError):
                continue
            except Exception:
                # Non-critical error in loop
                continue

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.sock:
            self.sock.close()
        if hasattr(self, '_mesh'):
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
