import os
import socket
import struct

import structlog

from src.shared.shm_mesh import TICK_SIZE, SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)

#  SILICON SPEED: Pre-packed struct for UDP fire
# 8s (Symbol), d (Price), q (Volume), d (Timestamp), q (receive_ts_ns)
TICK_STRUCT = struct.Struct("8s d q d q")


class MeshBridge:
    """
    Advanced Inter-Node SHM Mirror.
    Uses high-speed UDP Multicast to synchronize SHM across machines.
    """

    def __init__(self, multicast_group: str = "239.0.0.1", port: int = 9999):
        self.multicast_group = multicast_group
        self.port = port
        self.running = False
        self.mesh = SharedMemoryRingBuffer(create=False)
        self._last_head = 0

    def run_broadcaster(self, cpu_core: int = 9):
        """Spins on local SHM and fires batches to the cluster."""
        self.running = True
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("mesh_broadcaster_pinned", core=cpu_core)
        except Exception:
            pass

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        # MTU-aware batching (32 ticks * 40 bytes = 1280 bytes, safe for 1500 MTU)
        BATCH_SIZE = 32

        while self.running:
            slices, new_head = self.mesh.read_latest_slices(self._last_head)

            if slices:
                for chunk in slices:
                    # Send in batches of BATCH_SIZE
                    for i in range(0, len(chunk), BATCH_SIZE):
                        batch = chunk[i : i + BATCH_SIZE]
                        # Use raw bytes from the numpy view slice
                        sock.sendto(batch.tobytes(), (self.multicast_group, self.port))

                self._last_head = new_head
            else:
                os.sched_yield()

    def run_listener(self, cpu_core: int = 10):
        """Listens for batch packets and writes them to local SHM."""
        self.running = True

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.multicast_group, self.port))
        mreq = struct.pack(
            "4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        # Buffer for MTU-sized packets
        buf = bytearray(2048)
        while self.running:
            try:
                nbytes, _ = sock.recvfrom_into(buf)
                if nbytes > 0:
                    # Process batch: each tick is TICK_SIZE (40) bytes
                    for i in range(0, nbytes, TICK_SIZE):
                        if i + TICK_SIZE > nbytes:
                            break

                        # Unpack raw bytes from buffer
                        data = TICK_STRUCT.unpack(buf[i : i + TICK_SIZE])
                        self.mesh.write_tick(
                            symbol=data[0]
                            .decode("ascii", errors="ignore")
                            .strip("\x00"),
                            price=data[1],
                            volume=data[2],
                            timestamp=data[3],
                        )
            except Exception as e:
                logger.error("mesh_listener_failed", error=str(e))
