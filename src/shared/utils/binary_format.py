import csv
import struct
from collections.abc import Generator
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# EQUA Binary Format Constants
EQUA_MAGIC = b"EQUA"
EQUA_VERSION = 1
HEADER_FORMAT = "<4sHH"  # Magic (4s), Version (H), MetadataLen (H)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

RECORD_FORMAT = "<12sdiq"  # Symbol (12s), Price (d), Volume (i), Timestamp (q)
RECORD_SIZE = struct.calcsize(RECORD_FORMAT)

class EquaRecord(BaseModel):
    """
    Represents a single 32-byte record in the EQUA format.
    """

    symbol: str = Field(..., description="Symbol (max 12 bytes UTF-8)")
    price: float = Field(..., description="Price (double precision float)")
    volume: int = Field(..., description="Volume (4-byte signed integer)")
    timestamp_ns: int = Field(..., description="Timestamp (8-byte signed long long, nanoseconds)")

    @field_validator("symbol")
    @classmethod
    def validate_symbol_bytes(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 12:
            raise ValueError("Symbol must be at most 12 bytes when encoded in UTF-8")
        return v

    def pack(self) -> bytes:
        """Packs the record into exactly 32 bytes."""
        symbol_bytes = self.symbol.encode("utf-8")
        # struct.pack with 12s will null-pad or truncate.
        # We already validated length, so it will just null-pad if shorter.
        return struct.pack(RECORD_FORMAT, symbol_bytes, self.price, self.volume, self.timestamp_ns)

    @classmethod
    def unpack(cls, data: bytes) -> "EquaRecord":
        """Unpacks 32 bytes into an EquaRecord."""
        if len(data) != RECORD_SIZE:
            raise ValueError(f"Invalid record size: expected {RECORD_SIZE}, got {len(data)}")

        symbol_bytes, price, volume, timestamp_ns = struct.unpack(RECORD_FORMAT, data)
        # Decode and strip null padding
        symbol = symbol_bytes.decode("utf-8").rstrip("\x00")
        return cls(symbol=symbol, price=price, volume=volume, timestamp_ns=timestamp_ns)

class EquaWriter:
    """
    Writes records to a file in EQUA format.
    """

    def __init__(self, file_path: str | Path, metadata: bytes = b""):
        self.file_path = Path(file_path)
        self.metadata = metadata
        self.metadata_len = len(metadata)
        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, "wb")
        # Write header: Magic (4s), Version (H), MetadataLen (H)
        header = struct.pack(HEADER_FORMAT, EQUA_MAGIC, EQUA_VERSION, self.metadata_len)
        self.file.write(header)
        if self.metadata:
            self.file.write(self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write_record(self, record: EquaRecord):
        """Writes a single record to the file."""
        if not self.file:
            raise RuntimeError("EquaWriter must be used as a context manager")
        self.file.write(record.pack())

    def write_records(self, records: list[EquaRecord]):
        """Writes multiple records to the file."""
        for record in records:
            self.write_record(record)

class EquaReader:
    """
    Reads records from a file in EQUA format.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.header_info = None
        self.metadata = b""
        self.file = None

    def __enter__(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.file = open(self.file_path, "rb")
        # Read header
        header_data = self.file.read(HEADER_SIZE)
        if len(header_data) < HEADER_SIZE:
            raise ValueError("File too small to contain EQUA header")

        magic, version, metadata_len = struct.unpack(HEADER_FORMAT, header_data)
        if magic != EQUA_MAGIC:
            raise ValueError(f"Invalid magic: expected {EQUA_MAGIC}, got {magic}")

        self.header_info = {"version": version, "metadata_len": metadata_len}
        if metadata_len > 0:
            self.metadata = self.file.read(metadata_len)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def read_records(self) -> Generator[EquaRecord, None, None]:
        """Generates records from the file."""
        if not self.file:
            raise RuntimeError("EquaReader must be used as a context manager")

        while True:
            data = self.file.read(RECORD_SIZE)
            if not data:
                break
            if len(data) < RECORD_SIZE:
                # Potentially truncated file or trailing data
                break
            yield EquaRecord.unpack(data)

def csv_to_equa(csv_path: str | Path, equa_path: str | Path):
    """
    Converts a CSV file to EQUA format.
    CSV must have columns: symbol, price, volume, timestamp_ns
    """
    csv_path = Path(csv_path)
    equa_path = Path(equa_path)

    with open(csv_path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        with EquaWriter(equa_path) as writer:
            for row in reader:
                record = EquaRecord(
                    symbol=row["symbol"],
                    price=float(row["price"]),
                    volume=int(row["volume"]),
                    timestamp_ns=int(row["timestamp_ns"]),
                )
                writer.write_record(record)
