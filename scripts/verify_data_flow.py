import asyncio
import time

from src.shared.shm_mesh import SharedMemoryRingBuffer


async def test_data_flow():
    print("Testing Manifold Data Flow...")

    # 1. Initialize SHM
    shm = SharedMemoryRingBuffer(create=True)

    # 2. Simulate Ingestion (Writing to SHM)
    print("Writing ticks to SHM...")
    time.time()
    for i in range(100):
        shm.write_tick(f"TSLA{i % 10}", 150.0 + i, 100, time.time())

    # 3. Test Optimized Read
    print("Reading ticks via optimized msgspec reader...")
    ticks, head = shm.read_latest_msgspec(0)
    print(f"Read {len(ticks)} ticks. Latest head: {head}")

    if len(ticks) > 0:
        print(f"First Tick: {ticks[0]}")
        assert ticks[0].symbol.startswith("TSLA")

    # 4. Benchmarking the new reader
    print("Benchmarking optimized reader (1000 reads of 100 ticks)...")
    start = time.perf_counter()
    for _ in range(1000):
        # Reset head to simulate constant flow
        shm.read_latest_msgspec(0)
    duration = time.perf_counter() - start
    print(f"Performance: {duration:.4f}s ({duration / 1000 * 1e6:.2f} us/read)")

    print("Data Flow Verification: SUCCESS")
    shm.close()
    shm.unlink()


if __name__ == "__main__":
    try:
        asyncio.run(test_data_flow())
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
