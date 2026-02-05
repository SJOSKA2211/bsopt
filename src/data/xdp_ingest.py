import socket
import struct
import os
import asyncio
import structlog
from typing import Optional
import msgspec
from src.shared.shm_mesh import MarketTick, SharedMemoryRingBuffer

logger = structlog.get_logger(__name__)

# Constants for AF_XDP Simulation
ETH_P_IP = 0x0800

class XDPIngester:
    """
    High-performance ingestion simulator for market data.
    Uses msgspec for ultra-fast zero-copy decoding.
    """
    def __init__(self, interface: str = "eth0", port: int = 5555):
        self.interface = interface
        self.port = port
        self.sock: Optional[socket.socket] = None
        self._running = False
        self._mesh = SharedMemoryRingBuffer()
        self._decoder = msgspec.json.Decoder(MarketTick)

    async def start(self):
        """🚀 SINGULARITY: Initialize High-speed ingestion path."""
        try:
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_IP))
            # Fallback for local testing where AF_PACKET might fail without root
            try:
                self.sock.bind((self.interface, 0))
            except PermissionError:
                logger.warning("insufficient_privileges_using_udp_fallback")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.bind(("0.0.0.0", self.port))
            
            self.sock.setblocking(False)
            self._running = True
            
            logger.info("ingester_started", interface=self.interface, port=self.port)
            
            loop = asyncio.get_event_loop()
            while self._running:
                try:
                    data, _ = await loop.sock_recvfrom(self.sock, 2048)
                    self._handle_packet(data)
                except Exception as e:
                    await asyncio.sleep(0.001) # Yield
                
        except Exception as e:
            logger.error("ingester_failed", error=str(e))
            self._running = False

    def _handle_packet(self, data: bytes):
        """🚀 SINGULARITY: msgspec zero-copy ingestion."""
        try:
            # If RAW socket, skip headers (simplified)
            payload = data[42:] if len(data) > 42 else data
            
            # 🚀 SOTA: msgspec decode is 10x faster than json.loads
            tick = self._decoder.decode(payload)
            
            self._mesh.write_tick(
                symbol=tick.symbol,
                price=tick.price,
                volume=tick.volume,
                timestamp=tick.timestamp
            )
        except Exception:
            pass # Fast-path error suppression

    def stop(self):
        self._running = False
        if self.sock:
            self.sock.close()
        self._mesh.close()

if __name__ == "__main__":
    ingester = XDPIngester()
    asyncio.run(ingester.start())
