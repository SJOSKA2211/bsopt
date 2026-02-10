
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
        """Spins on local SHM and fires packets to the cluster."""
        self.running = True
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("mesh_broadcaster_pinned", core=cpu_core)
        except Exception:
            pass

        # Setup Multicast Socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        
        logger.info("mesh_broadcaster_active", group=self.multicast_group)
        
        while self.running:
            # Poll SHM
            import struct
            current_head = struct.unpack("q", self.mesh.buf[:8])[0]
            
            if current_head > self._last_head:
                # Mirror the batches to the network
                view, new_head = self.mesh.read_latest_view(self._last_head)
                for tick in view:
                    #  ZERO-LATENCY FIRE
                    payload = TICK_STRUCT.pack(
                        tick['symbol'], tick['price'], tick['volume'], 
                        tick['timestamp'], tick['receive_ts_ns']
                    )
                    sock.sendto(payload, (self.multicast_group, self.port))
                self._last_head = new_head
            else:
                os.sched_yield()

    def run_listener(self, cpu_core: int = 10):
        """Listens for remote ticks and writes them to local SHM."""
        self.running = True
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("mesh_listener_pinned", core=cpu_core)
        except Exception:
            pass

        # Setup Multicast Listener
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        
        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        logger.info("mesh_listener_active", group=self.multicast_group)
        
        buf = bytearray(TICK_SIZE)
        while self.running:
            try:
                nbytes, _ = sock.recvfrom_into(buf)
                if nbytes >= TICK_SIZE:
                    # Unpack and write to local mesh
                    data = TICK_STRUCT.unpack(buf)
                    self.mesh.write_tick(
                        symbol=data[0].decode().strip('\x00'),
                        price=data[1],
                        volume=data[2],
                        timestamp=data[3]
                    )
                    # Note: write_tick will overwrite receive_ts_ns with local arrival
                    # This is intentional for local T2T tracking.
            except Exception as e:
                logger.error("mesh_listener_failed", error=str(e))
