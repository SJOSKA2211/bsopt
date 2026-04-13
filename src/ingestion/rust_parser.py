"""
Rust-Accelerated Ingestion Parser

Zero-copy binary data parsing using the Production Rust core.
"""

import os
import struct

import numpy as np
import structlog

from src.shared.utils.binary_format import EQUA_MAGIC, EQUA_VERSION, HEADER_SIZE, EquaRecord

logger = structlog.get_logger(__name__)


class RustTickParser:
    """
    Python wrapper for the Rust TickDataBuffer.
    Handles memory-mapped binary tick files following the EQUA standard.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.buffer = None

        if not os.path.exists(file_path):
            logger.warning("mmap_file_not_found", path=file_path)
            return

        try:
            self._validate_header()
            import Manifold_core

            self.buffer = Manifold_core.TickDataBuffer(file_path)
            logger.info("rust_mmap_buffer_initialized", path=file_path, size=self.buffer.size())
        except ImportError:
            logger.error("rust_core_not_installed")
        except Exception as e:
            logger.error("rust_buffer_initialization_failed", error=str(e))

    def _validate_header(self):
        """Validates the EQUA header before passing to Rust."""
        try:
            with open(self.file_path, "rb") as f:
                header = f.read(HEADER_SIZE)
                if len(header) < HEADER_SIZE:
                    raise ValueError("File too small for EQUA header")

                magic, version, metadata_len = struct.unpack("<4sHH", header)
                if magic != EQUA_MAGIC:
                    raise ValueError(f"Invalid magic: {magic}")
                if version != EQUA_VERSION:
                    logger.warning("version_mismatch", expected=EQUA_VERSION, found=version)
        except Exception as e:
            logger.error("header_validation_failed", error=str(e))
            raise

    def get_views(self) -> dict[str, np.ndarray]:
        """
        Returns zero-copy NumPy views of the underlying mmap data.
        Calls Rust methods: get_symbols(), get_prices(), get_volumes(), get_timestamps().
        """
        if not self.buffer:
            return {}

        try:
            return {
                "symbols": self.buffer.get_symbols(),
                "prices": self.buffer.get_prices(),
                "volumes": self.buffer.get_volumes(),
                "timestamps": self.buffer.get_timestamps(),
            }
        except Exception as e:
            logger.error("rust_get_views_failed", error=str(e))
            return {}

    def to_pydantic(self, offset: int = 0, count: int = 100) -> list[EquaRecord]:
        """
        Converts a slice of the views into a list of EquaRecord for validated consumption.
        """
        views = self.get_views()
        if not views:
            return []

        prices = views.get("prices")
        if prices is None or len(prices) == 0:
            return []

        total = len(prices)
        if offset >= total:
            return []

        end = min(offset + count, total)

        symbols = views["symbols"]
        volumes = views["volumes"]
        timestamps = views["timestamps"]

        records = []
        for i in range(offset, end):
            # Decode symbol and strip nulls
            symbol_bytes = symbols[i]
            symbol = bytes(symbol_bytes).decode("utf-8").rstrip("\x00")

            records.append(
                EquaRecord(
                    symbol=symbol,
                    price=float(prices[i]),
                    volume=int(volumes[i]),
                    timestamp_ns=int(timestamps[i]),
                )
            )
        return records

    def get_total_ticks(self) -> int:
        if not self.buffer:
            return 0
        # Header is 8 bytes, each record is 32 bytes
        size = self.buffer.size()
        if size < 8:
            return 0
        return (size - 8) // 32