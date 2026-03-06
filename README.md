# BS-OPT: The Advanced Financial Manifold

##  Overview
BS-OPT is an advanced, high-performance financial platform for zero-latency derivative pricing, risk management, and autonomous trading. It integrates quantitative finance (Black-Scholes, Heston) with modern Machine Learning (Offline RL, Transformers) and low-latency systems engineering (eBPF/XDP, Shared Memory).

## 🏛️ Advanced Architecture

### The Brain: Offline Reinforcement Learning
- **Decision Transformer**: Implements offline RL using temporal observation windows for strategy optimization. 
- **Source**: `src/ml/reinforcement_learning/decision_transformer.py:L5`
- **Environment**: Gymnasium-based `TradingEnvironment` with Numba-accelerated state and reward kernels.

### The Wire: Low-Latency Ingestion
- **eBPF/XDP**: Silicon-level data ingestion for sub-microsecond latency.
- **Filter**: `scripts/hft/xdp_filter.c` (C/eBPF code).
- **Ingester**: `src/data/xdp_ingest.py:L18` (Python/Socket wrapper).

### The Mesh: Shared Memory Distribution
- **Market Mesh**: Zero-copy market data distribution via `SHMManager` providing high-throughput IPC.
- **Implementation**: `src/shared/shm_manager.py:L20` (Generic SHM management).
- **Buffer**: 50MB "market_mesh" initialized in `src/scrapers/mesh_publisher.py:L22`.

### The Body: Multi-tier Compute
- **WASM SIMD**: High-performance Rust/WASM implementation for browser-based pricing.
- **Source**: `src/wasm/src/simd_math.rs`
- **Numba JIT**: Vectorized risk validation and pricing kernels (e.g., `src/trading/risk_kernels.py:L5`).

### The Workers: Hybrid Distributed Architecture
- **Celery/Ray Hybrid**: Task queuing via Celery (`src/workers/math_worker.py:L58`) with heavy computation delegated to a pool of Ray `MathActor` instances.

## 🛠️ Prerequisites & Toolchains
The BS-OPT platform requires several low-level toolchains for its "Hardware-Fluid" features:
- **LLVM/Clang**: For compiling eBPF/XDP filters.
- **Rust & wasm-pack**: For compiling the WASM SIMD compute kernels.
- **Numba & LLVM**: For JIT-compiling Python-based mathematical kernels.
- **Docker**: For containerized orchestration.

## 🚀 Quick Start
```bash
# Start the Stack
make up

# Run Tests
make test-all

# Access CLI
make cli ARGS="status"
```

## 📜 Documentation
Detailed technical specifications are available in the `docs/` directory:
- [Market Mesh (Shared Memory)](docs/architecture/MARKET_MESH.md)
- [Hybrid Worker Architecture](docs/architecture/HYBRID_WORKERS.md)
- [Vectorized Risk Management](docs/architecture/VECTORIZED_RISK.md)
- [Trading Engine Flow](docs/architecture/TRADING_ENGINE.md)
- [Security Protocol](docs/SECURITY_PROTOCOL.md)
- [Anti-Freeze Guide (Build Optimization)](docs/mlops/anti-freeze.md)

## 🗄️ Database God Mode (v2.5)
The BS-OPT manifold is powered by a hyper-optimized **PostgreSQL 16 + TimescaleDB** backend:
- **JIT Acceleration**: Native JIT compilation enabled for complex analytical risk queries.
- **SIMD Compression**: TimescaleDB columnar compression with symbol-based segmenting.
- **Automated Maintenance**: background jobs for concurrent MV refreshes and statistics re-analysis.
- **Diagnostics Dashboard**: Real-time monitoring views (`db_health_overview`, `query_variance_report`).
- **Audit Tool**: In-built verification manifold for infra-structure integrity.

**Run Database Audit:**
```bash
python3 -m src.database.verify
```

---
*Created by the Joseph Kamau Maina Extension. Shut up and compute.*
