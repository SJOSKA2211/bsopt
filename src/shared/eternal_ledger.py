import mmap
import os
import struct
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

class EternalLedger:
    """
    Advanced Zero-Copy Binary Persistence.
    Writes raw SHM ticks directly to a memory-mapped flat file.
    Designed for sub-microsecond logging.
    """

    # Pre-compiled structure for sub-microsecond packing
    _TICK_STRUCT = struct.Struct("8s d q d q")

    def __init__(self, file_path: str = "logs/eternal_ledger.bin", capacity: int = 1000000) -> None:
        self.file_path = file_path
        self.capacity = capacity
        self.entry_size = self._TICK_STRUCT.size
        self.file_size = self.entry_size * self.capacity

        # Ensure logs directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Initialize flat file
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(b"\x00" * self.file_size)

        self.file = open(file_path, "r+b")
        self.mmap = mmap.mmap(self.file.fileno(), self.file_size)
        self._offset = 0

        logger.info(
            "eternal_ledger_initialized",
            path=file_path,
            size_mb=self.file_size / 1024 / 1024,
        )

    def write_batch(self, batch: list[dict[str, Any]]) -> None:
        """HOT PATH: Optimized batch write to mmap."""
        try:
            
            # We use a memoryview of the mmap for faster slicing
            mv = memoryview(self.mmap)
            pack = self._TICK_STRUCT.pack

            for item in batch:
                if self._offset + self.entry_size > self.file_size:
                    self._offset = 0  # Ring-buffer wrap-around

                # Optimized packing using pre-bound method
                data = pack(
                    item["symbol"].encode("ascii")[:8],
                    item["price"],
                    item["volume"],
                    item["timestamp"],
                    item.get("receive_ts_ns", 0),
                )

                mv[self._offset : self._offset + self.entry_size] = data
                self._offset += self.entry_size

        except Exception as e:
            logger.error("ledger_batch_write_failed", error=str(e))

    def flush(self) -> None:
        self.mmap.flush()

    def close(self) -> None:
        self.mmap.close()
        self.file.close()
