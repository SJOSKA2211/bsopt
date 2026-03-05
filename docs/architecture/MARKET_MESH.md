# Architecture: Market Mesh (Shared Memory)

## Overview
The "Market Mesh" is a low-latency IPC (Inter-Process Communication) system for zero-copy market data distribution. It uses POSIX shared memory to distribute market ticks and order updates across services (Scrapers, Trading Engine, ML Orchestrator) without the overhead of standard network or file-based IPC.

## Implementation Details
- **Core Manager**: `src/shared/shm_manager.py:L20` (`SHMManager`).
- **Buffer Initialization**: `src/scrapers/mesh_publisher.py:L22`.
- **Buffer Size**: 50MB (Shared Memory block named "market_mesh").
- **Data Structure**: Optimized `dict` mapping symbols to market state, serialized for shared memory access.

## Data Flow
1. **Scraper**: The `NSEScraper` (`src/scrapers/engine.py:L120`) fetches data and publishes it to the Market Mesh via `MeshPublisher`.
2. **Shared Memory**: Data is written directly to the mapped memory region.
3. **Consumer**: The `OrderEngine` (`src/trading/order_engine.py:L11`) or `TradingEnvironment` (`src/ml/reinforcement_learning/trading_env.py`) reads the state directly from memory, achieving sub-microsecond ingestion latency.

## Performance
- **Latency**: Sub-microsecond (Local memory access).
- **Throughput**: Limited only by memory bandwidth.
- **CPU Pinning**: Often used in conjunction with `XDPIngester` to dedicate CPU cores to high-frequency ingestion and distribution.
