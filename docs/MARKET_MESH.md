# Architecture: Market Mesh (Shared Memory) 

## Overview
The "Market Mesh" is a low-latency IPC (Inter-Process Communication) system for zero-copy market data distribution. It uses POSIX shared memory to distribute market ticks and order updates across src (Scrapers, Trading Engine, ML Orchestrator) without the overhead of standard network or file-based IPC.

## Implementation Details
- **Core Manager**: `src/shared/shm_manager.py` (`SHMManager`).
- **Data Transport**: Uses `msgspec` and `orjson` for ultra-fast binary serialization within memory blocks.
- **Polling Strategy**: Implements **Adaptive IPC Polling** (1ms resolution during activity, backing off when idle) to minimize CPU usage while maintaining sub-millisecond delivery.
- **Buffer Initialization**: `src/scrapers/mesh_publisher.py`.
- **Buffer Size**: 50MB (Shared Memory block named "market_mesh").
- **Locking**: Low-latency **Spin-Locks** with memoryview access for atomic write synchronization.

## Data Flow
1. **Scraper/Ingester**: The `NSEScraper` or `XDPIngester` fetches/receives data and publishes it to the Market Mesh.
2. **Shared Memory**: Data is written directly to the mapped memory region using `write_tick_raw` to bypass string decoding overhead.
3. **Consumer**: The `OrderEngine` or `WebsocketManager` reads the state directly from memory, achieving sub-microsecond ingestion latency.

## Performance
- **Latency**: Sub-microsecond (Local memory access).
- **Throughput**: Limited only by memory bandwidth (~GB/s).
- **Serialization**: 10x faster than standard `json` via `msgspec` binary encoding.
- **CPU Pinning**: Often used in conjunction with `XDPIngester` to dedicate CPU cores to high-frequency ingestion and distribution.
