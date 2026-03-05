# Architecture: Trading Engine Flow

## Overview
The Trading Engine is a low-latency gateway designed for high-frequency execution across both centralized and decentralized (DeFi) venues. It bridges the Gap between the "Market Mesh" (Real-time data) and "Execution Venues" (Polygon/Exchanges).

## Data Flow
1. **Ingestion**: `XDPIngester` (`src/data/xdp_ingest.py`) receives binary market ticks.
2. **Distribution**: Data is published to the **Market Mesh** (Shared Memory).
3. **Gateway**: `OrderEngine` (`src/trading/order_engine.py`) monitors the mesh and strategy signals.
4. **Risk Check**: Every order is validated via **Vectorized Risk Kernels** (`src/trading/risk_kernels.py`).
5. **Execution**:
    - **DeFi**: Orders are dispatched via `DeFiOptionsProtocol` (`src/blockchain/defi_options.py:L13`) using Multicall3 for batching and EIP-1559 for gas management.
    - **Centralized**: Traditional API integration via the Trading Engine Gateway.

## Low-Latency Design
- **Shared Memory Buffers**: `OrderBuffer` and `ExecutionBuffer` facilitate lock-free communication between strategy and execution threads.
- **CPU Pinning**: The Engine is designed to run on dedicated CPU cores to minimize context-switching overhead.
- **Vectorized Logic**: All pre-trade calculations are performed using NumPy/Numba to ensure predictable, sub-microsecond response times.
