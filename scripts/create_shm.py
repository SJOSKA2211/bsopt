import time
import signal
import sys
from src.shared.shm_mesh import SharedMemoryRingBuffer

def signal_handler(sig, frame):
    print('Shutting down...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    shm = SharedMemoryRingBuffer(create=True)
    print("Market Mesh Ring Buffer created successfully.")
    print("Press Ctrl+C to exit and clean up.")
    while True:
        time.sleep(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
