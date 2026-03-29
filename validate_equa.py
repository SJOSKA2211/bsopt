import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path("/home/h8t3dj4y/bsopt/src")))

from shared.utils.binary_format import EquaReader, EquaRecord, EquaWriter

def test_binary_format():
    test_file = "test_data.equa"
    records = [
        EquaRecord(
            symbol=f"SYM{i}", price=100.0 + i, volume=1000 + i, timestamp_ns=1600000000000 + i
        )
        for i in range(10)
    ]

    print(f"Writing {len(records)} records to {test_file}...")
    with EquaWriter(test_file) as writer:
        writer.write_records(records)

    print(f"Reading records back from {test_file}...")
    read_records = []
    with EquaReader(test_file) as reader:
        for record in reader.read_records():
            read_records.append(record)

    assert len(read_records) == len(records), (
        f"Expected {len(records)} records, got {len(read_records)}"
    )

    for i, (original, read) in enumerate(zip(records, read_records)):
        assert original.symbol == read.symbol, (
            f"Mismatch at index {i}: symbol {original.symbol} != {read.symbol}"
        )
        assert original.price == read.price, (
            f"Mismatch at index {i}: price {original.price} != {read.price}"
        )
        assert original.volume == read.volume, (
            f"Mismatch at index {i}: volume {original.volume} != {read.volume}"
        )
        assert original.timestamp_ns == read.timestamp_ns, (
            f"Mismatch at index {i}: timestamp {original.timestamp_ns} != {read.timestamp_ns}"
        )
        print(f"Record {i} verified: {read}")

    print("Validation successful: All records match exactly.")

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    try:
        test_binary_format()
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)
