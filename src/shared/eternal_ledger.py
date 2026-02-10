
import mmap
import os
import struct
from typing import Any

import structlog

from src.shared.shm_mesh import TICK_SIZE

logger = structlog.get_logger(__name__)

class EternalLedger:
    """
    Advanced Zero-Copy Binary Persistence.
    Writes raw SHM ticks directly to a memory-mapped flat file.
    Designed for sub-microsecond logging.
    """
    def __init__(self, file_path: str = "logs/eternal_ledger.bin", capacity: int = 1000000):
        self.file_path = file_path
        self.capacity = capacity
        self.entry_size = TICK_SIZE
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
        
        logger.info("eternal_ledger_initialized", path=file_path, size_mb=self.file_size/1024/1024)

    def write_batch(self, batch: list[Any]):
        """ HOT PATH: Write raw binary data directly to mmap."""
        for item in batch:
            if self._offset + self.entry_size > self.file_size:
                # Wrap around or rotate? For singularity, we rotate or expand.
                # For now, we wrap around like a true ring-ledger.
                self._offset = 0
            
            # For the ledger, we want the raw bytes. 
            # In a real Advanced pass, we'd copy directly from SHM memory addresses.
            # Here we pack the dict back to bytes (until we implement raw SHM copy).
            try:
                # 8s d q d q
                data = struct.pack("8s d q d q", 
                                   item['symbol'].encode('ascii')[:8],
                                   item['price'],
                                   item['volume'],
                                   item['timestamp'],
                                   item.get('receive_ts_ns', 0))
                self.mmap[self._offset:self._offset + self.entry_size] = data
                self._offset += self.entry_size
            except Exception as e:
                logger.error("ledger_write_failed", error=str(e))

    def flush(self):
        self.mmap.flush()

    def close(self):
        self.mmap.close()
        self.file.close()
