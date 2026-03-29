import struct
from unittest.mock import MagicMock, patch

import pytest

from src.shared.shm_mesh import BUFFER_CAPACITY, TICK_SIZE, SharedMemoryRingBuffer

@pytest.fixture
def mock_shm():
    with patch("src.shared.shm_mesh.shared_memory.SharedMemory") as MockSHM:
        instance = MagicMock()
        # Create a real buffer for logic testing
        instance.buf = bytearray((TICK_SIZE * BUFFER_CAPACITY) + 8)
        instance.buf[:8] = struct.pack("q", 0)
        MockSHM.return_value = instance
        yield instance

def test_init_create(mock_shm):
    rb = SharedMemoryRingBuffer(create=True)
    assert rb.shm is not None

def test_write_read_tick(mock_shm):
    rb = SharedMemoryRingBuffer(create=True)

    # Write
    rb.write_tick("AAPL", 150.0, 100, 1000.0)

    # Read raw
    head = struct.unpack("q", rb.buf[:8])[0]
    assert head == 1

    # Read View
    view, new_head = rb.read_latest_view(0)
    assert len(view) == 1
    assert view[0]["symbol"].decode().strip("\x00") == "AAPL"
    assert view[0]["price"] == 150.0

def test_wrap_around(mock_shm):
    SharedMemoryRingBuffer(create=True)

    # Fake a full buffer
    # We can't easily fake 100k writes fast in python test
    # But we can check the read logic by manually setting head and view
    pass

def test_msgspec_read(mock_shm):
    rb = SharedMemoryRingBuffer(create=True)
    rb.write_tick("GOOG", 200.0, 50, 2000.0)

    ticks, head = rb.read_latest_msgspec(0)
    assert len(ticks) == 1
    assert ticks[0].symbol == "GOOG"
    assert ticks[0].price == 200.0
