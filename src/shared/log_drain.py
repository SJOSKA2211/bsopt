import asyncio
import struct
import time

import httpx
import structlog

from src.shared.off_heap_logger import (
    LOG_BUFFER_CAPACITY,
    LOG_SIZE,
    LOG_STRUCT,
    SHM_LOG_NAME,
)

# Standard logging for the drainer itself
logger = structlog.get_logger()

class AsyncLogDrain:
    """
    OPTIMIZED: Asynchronous worker that drains the Off-Heap SHM log buffer and batches to Loki.
    Ensures that log persistence never touches the hot path.
    """

    def __init__(
        self,
        loki_url: str | None = None,
        batch_size: int = 1000,
        flush_interval: float = 5.0,
    ) -> None:
        from multiprocessing import shared_memory

        self.loki_url = loki_url or "http://loki:3100/loki/api/v1/push"
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue: list[tuple[int, str]] = []
        self._running = True
        self.last_head = 0
        self._semaphore = asyncio.Semaphore(5)  # Limit in-flight pushes

        # Connect to SHM
        try:
            self._shm = shared_memory.SharedMemory(name=SHM_LOG_NAME)
            self.buf = self._shm.buf
        except Exception as e:
            logger.error("log_drain_shm_connect_failed", error=str(e))
            raise

    async def _push_to_loki(self, batch: list[tuple[int, str]]) -> None:
        """Push batched logs with semaphore protection."""
        async with self._semaphore:
            if not batch:
                return

            # Loki JSON format: { "streams": [ { "stream": { "label": "value" }, "values": [ [ "nanoseconds", "line" ] ] } ] }
            streams = {
                "streams": [
                    {
                        "stream": {"service": "bsopt", "source": "off_heap"},
                        "values": [[str(ts * 1000000), line] for ts, line in batch],
                    }
                ]
            }

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(self.loki_url, json=streams)
                    if resp.status_code != 204:
                        logger.warning("loki_push_partial_failure", status=resp.status_code)
            except Exception as e:
                logger.error("loki_push_failed", error=str(e))

    async def run(self) -> None:
        """Main async loop with optimized extraction."""
        logger.info("async_log_drain_started", url=self.loki_url)
        last_flush = time.time()
        mv = memoryview(self.buf)

        while self._running:
            # Atomic read of head pointer
            current_head = struct.unpack("q", mv[:8])[0]

            if current_head > self.last_head:
                
                # We still need to unpack for timestamps, but we use memoryview
                start_idx = max(self.last_head, current_head - LOG_BUFFER_CAPACITY)

                for h in range(start_idx, current_head):
                    offset = 8 + (h % LOG_BUFFER_CAPACITY) * LOG_SIZE
                    # Fast slice without copy
                    entry = mv[offset : offset + LOG_SIZE]
                    timestamp, payload = LOG_STRUCT.unpack(entry)

                    # Decoupled decoding
                    self.queue.append((timestamp, payload.decode("utf-8").rstrip("\x00")))

                self.last_head = current_head

            # Periodic or size-based flush
            if len(self.queue) >= self.batch_size or (
                time.time() - last_flush >= self.flush_interval and self.queue
            ):
                batch_to_send = self.queue[:]
                self.queue = []
                asyncio.create_task(self._push_to_loki(batch_to_send))
                last_flush = time.time()

            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False

if __name__ == "__main__":
    drain = AsyncLogDrain()
    try:
        asyncio.run(drain.run())
    except KeyboardInterrupt:
        logger.info("log_drain_stopped")
