import asyncio
import socket

import msgspec
import structlog

from src.shared.shm_mesh import MarketTick, SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)

# Constants for AF_XDP Simulation
ETH_P_IP = 0x0800

class XDPIngester:
    """
    High-performance ingestion simulator for market data.
    Uses msgspec for efficient zero-copy decoding.
    """
    def __init__(self, interface: str = "eth0", port: int = 5555):
        self.interface = interface
        self.port = port
        self.sock: socket.socket | None = None
        self._running = False
        self._mesh = SharedMemoryRingBuffer()
        self._decoder = msgspec.json.Decoder(MarketTick)

    async def start(self):
        """Initialize ingestion path."""
        try:
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_IP))
            try:
                self.sock.bind((self.interface, 0))
            except PermissionError:
                logger.warning("using_udp_fallback_dev_mode")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.bind(("0.0.0.0", self.port))
            
            self.sock.setblocking(False)
            self._running = True
            
            logger.info("ingester_active", interface=self.interface, port=self.port)
            
            loop = asyncio.get_event_loop()
            while self._running:
                try:
                    data, _ = await loop.sock_recvfrom(self.sock, 2048)
                    if data:
                        self._handle_packet(data)
                except (BlockingIOError, InterruptedError):
                    await asyncio.sleep(0.0001)
                except Exception as e:
                    logger.error("packet_error", error=str(e))
                
        except Exception as e:
            logger.error("ingester_init_failed", error=str(e))
            self._running = False

    def _handle_packet(self, data: bytes):
        """Decode and write to shared memory mesh."""
        try:
            # Skip headers if in packet mode
            offset = 42 if self.sock.family == socket.AF_PACKET else 0
            payload = data[offset:]
            
            tick = self._decoder.decode(payload)
            
            self._mesh.write_tick(
                symbol=tick.symbol,
                price=tick.price,
                volume=tick.volume,
                timestamp=tick.timestamp
            )
        except Exception:
            pass # Fast-path drop

    def stop(self):
        self._running = False
        if self.sock:
            self.sock.close()
        self._mesh.close()

if __name__ == "__main__":
    ingester = XDPIngester()
    asyncio.run(ingester.start())
