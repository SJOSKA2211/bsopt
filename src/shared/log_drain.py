import struct
import time
import os
import sys
import asyncio
from multiprocessing import shared_memory
import structlog
import orjson
from src.shared.off_heap_logger import LOG_STRUCT, LOG_SIZE, LOG_BUFFER_CAPACITY, SHM_LOG_NAME
from src.utils.http_client import HttpClientManager

# Standard logging for the drainer itself
logger = structlog.get_logger()

class AsyncLogDrain:
    """
    SOTA: Asynchronous worker that drains the Off-Heap SHM log buffer and batches to Loki.
    Ensures that log persistence never touches the hot path.
    """
    def __init__(self, loki_url: str = None, batch_size: int = 1000, flush_interval: float = 5.0):
        try:
            self.shm = shared_memory.SharedMemory(name=SHM_LOG_NAME)
            self.buf = self.shm.buf
            # Initialize head to current head
            self.last_head = struct.unpack("q", self.buf[:8])[0]
            logger.info("log_drain_connected", head=self.last_head)
        except FileNotFoundError:
            logger.error("log_shm_not_found", reason="OffHeapLogger not initialized")
            sys.exit(1)
            
        self.loki_url = loki_url or os.getenv("LOKI_URL", "http://localhost:3100/loki/api/v1/push")
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue = []
        self._running = True

    async def _push_to_loki(self, batch):
        """🚀 SINGULARITY: Push batched logs to Loki using HTTP/2."""
        if not batch: return
        
        streams = []
        streams.append({
            "stream": {"job": "bsopt-off-heap", "env": os.getenv("ENVIRONMENT", "dev")},
            "values": [[str(ts * 1000000), line] for ts, line in batch] # nanoseconds
        })
        
        payload = {"streams": streams}
        client = HttpClientManager.get_client()
        
        try:
            response = await client.post(self.loki_url, content=orjson.dumps(payload), headers={"Content-Type": "application/json"})
            if response.status_code >= 400:
                logger.error("loki_push_failed", status=response.status_code, body=response.text)
        except Exception as e:
            logger.error("loki_push_error", error=str(e))

    async def run(self):
        """🚀 SOTA: Main async loop to drain and batch logs."""
        logger.info("async_log_drain_started", url=self.loki_url)
        last_flush = time.time()
        
        while self._running:
            current_head = struct.unpack("q", self.buf[:8])[0]
            
            if current_head > self.last_head:
                start_idx = max(self.last_head, current_head - LOG_BUFFER_CAPACITY)
                
                for h in range(start_idx, current_head):
                    offset = 8 + (h % LOG_BUFFER_CAPACITY) * LOG_SIZE
                    timestamp, payload = LOG_STRUCT.unpack(self.buf[offset : offset + LOG_SIZE])
                    
                    clean_json = payload.decode('utf-8').strip('\x00')
                    self.queue.append((timestamp, clean_json))
                
                self.last_head = current_head
            
            # Periodic or size-based flush
            if len(self.queue) >= self.batch_size or (time.time() - last_flush >= self.flush_interval and self.queue):
                batch_to_send = self.queue[:]
                self.queue = []
                asyncio.create_task(self._push_to_loki(batch_to_send))
                last_flush = time.time()
            
            await asyncio.sleep(0.1)

    def stop(self):
        self._running = False

if __name__ == "__main__":
    drain = AsyncLogDrain()
    try:
        asyncio.run(drain.run())
    except KeyboardInterrupt:
        logger.info("log_drain_stopped")
