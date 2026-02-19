import asyncio
import struct
import time

import structlog

from src.shared.off_heap_logger import (
    LOG_BUFFER_CAPACITY,
    LOG_SIZE,
    LOG_STRUCT,
)

# Standard logging for the drainer itself
logger = structlog.get_logger()


class AsyncLogDrain:
    """
    OPTIMIZED: Asynchronous worker that drains the Off-Heap SHM log buffer and batches to Loki.
    Ensures that log persistence never touches the hot path.
    """

    def __init__(
        self, loki_url: str = None, batch_size: int = 1000, flush_interval: float = 5.0
    ):
        # ... (init stays same)
        self._semaphore = asyncio.Semaphore(5)  # Limit in-flight pushes

    async def _push_to_loki(self, batch):
        """Push batched logs with semaphore protection."""
        async with self._semaphore:
            if not batch:
                return

            # ... (push logic stays same)

    async def run(self):
        """Main async loop with optimized extraction."""
        logger.info("async_log_drain_started", url=self.loki_url)
        last_flush = time.time()
        mv = memoryview(self.buf)

        while self._running:
            current_head = struct.unpack("q", self.buf[:8])[0]

            if current_head > self.last_head:
                # OPTIMIZED: Bulk grab from head difference
                # We still need to unpack for timestamps, but we use memoryview
                start_idx = max(self.last_head, current_head - LOG_BUFFER_CAPACITY)

                for h in range(start_idx, current_head):
                    offset = 8 + (h % LOG_BUFFER_CAPACITY) * LOG_SIZE
                    # Fast slice without copy
                    entry = mv[offset : offset + LOG_SIZE]
                    timestamp, payload = LOG_STRUCT.unpack(entry)

                    # Decoupled decoding
                    self.queue.append(
                        (timestamp, payload.decode("utf-8").rstrip("\x00"))
                    )

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

    def stop(self):
        self._running = False


if __name__ == "__main__":
    drain = AsyncLogDrain()
    try:
        asyncio.run(drain.run())
    except KeyboardInterrupt:
        logger.info("log_drain_stopped")
