# Architecture: Trading Engine Flow 

## Overview
The Trading Engine is a low-latency gateway designed for high-frequency execution across both centralized and decentralized (DeFi) venues. It bridges the Gap between the "Market Mesh" (Real-time data) and "Execution Venues" (Polygon/Exchanges).

## Data Flow
1. **Ingestion**: `XDPIngester` (`src/data/xdp_ingest.py`) receives binary market ticks over UDP/TCP.
2. **Distribution**: Data is published to the **Market Mesh** (Shared Memory) using `write_tick_raw` to eliminate string decoding overhead on the ingestion path.
3. **Gateway**: `OrderEngine` (`src/trading/order_engine.py`) monitors the mesh and strategy signals via high-precision adaptive polling (1ms).
4. **Risk Check**: Every order is validated via **Silicon Risk Kernels** (`src/trading/risk_kernels.py`) in < 300ns.
5. **Execution**:
    - **DeFi**: Orders are dispatched via `DeFiOptionsProtocol` (`src/blockchain/defi_options.py`) using Multicall3 for batching and EIP-1559 for gas management.
    - **Centralized**: Traditional API integration via the Trading Engine Gateway with Speculative Concurrency routing.

## High-Performance Performance
- **Shared Memory Buffers**: `OrderBuffer` and `ExecutionBuffer` facilitate lock-free, zero-copy communication between strategy and execution threads.
- **CPU Pinning**: The engine is designed to run on dedicated CPU cores with `os.sched_yield()` optimization to minimize context-switching.
- **Zero-Allocation Hot Loop**: The `OrderEngine` hot loop is re-engineered to operate without memory allocations, ensuring consistent latency profiles even under extreme load.
- **Rust Integration**: Leverages `bsopt_core` (Rust) for the fastest possible zero-GIL order loop execution when available.
