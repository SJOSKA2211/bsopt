import numpy as np
import pytest
from src.shared.shm_mesh import SharedMemoryRingBuffer, MarketTick
import time

def test_shm_ring_buffer_write_read():
    # Use create=True to initialize
    mesh = SharedMemoryRingBuffer(create=True)
    
    symbol = "AAPL"
    price = 150.5
    volume = 1000
    timestamp = time.time()
    
    mesh.write_tick(symbol, price, volume, timestamp)
    
    # Read latest
    ticks, head = mesh.read_latest_msgspec(0)
    
    assert len(ticks) == 1
    assert ticks[0].symbol == symbol
    assert ticks[0].price == price
    assert ticks[0].volume == volume
    
    # Clear references
    del ticks
    
    mesh.close()
    mesh.unlink()

def test_shm_ring_buffer_view():
    mesh = SharedMemoryRingBuffer(create=True)
    
    for i in range(10):
        mesh.write_tick("MSFT", 300.0 + i, 100, time.time())
        
    view, head = mesh.read_latest_view(0)
    assert len(view) == 10
    assert view[0]['symbol'].decode().strip('\x00') == "MSFT"
    assert head == 10
    
    # Test offset read
    view2, head2 = mesh.read_latest_view(5)
    assert len(view2) == 5
    assert head2 == 10
    
    # 🚀 SOTA: Clear NumPy views before closing SHM to avoid BufferError
    del view
    del view2
    
    mesh.close()
    mesh.unlink()
