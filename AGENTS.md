# AGENTS.md - AI Agent Context for BSOPT

Welcome, fellow agent. This file provides the essential context and commands for developing and maintaining the BSOPT (Black-Scholes Optimization) system.

## Project Overview
BSOPT is a high-performance options pricing and risk management engine. It uses a hybrid Python/Rust architecture to achieve extreme throughput and low latency.

## Architecture
- **API (FastAPI)**: The main entry point for external requests. Located in `api/`.
- **Core Engine (Rust)**: High-performance math kernel. Located in `src/math_kernel/rust-core/`.
- **Pricing Logic (Python)**: Business logic and strategy implementations. Located in `src/quant/`.
- **Database (TimescaleDB)**: Optimized for time-series market data.
- **Cache (Redis)**: High-speed snapshot storage and shared memory mesh.

## Key Commands

### Development & Build
- **Build Rust Core**: 
  ```bash
  cd src/math_kernel/rust-core && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
  ```
- **Run API Locally**:
  ```bash
  ./run_api_local.sh
  ```
  *(Note: Requires core services like Postgres and Redis to be running)*

### Testing
- **Run All Tests**: `pytest`
- **Run Rust Benchmarks**: `cd src/math_kernel/rust-core && cargo bench`

### Observability
- **Health Check**: `GET /health`
- **Metrics (Prometheus)**: `GET /metrics`

## Agent Best Practices
1. **Performance First**: Always prefer the Rust core for heavy computations.
2. **Zero-Trust**: Adhere to the security layer (PKI/mTLS). Use `scripts/utils_env.sh` to load secrets.
3. **Idiomatic Rust/Python**: Follow existing patterns. Use `rayon` for parallelism in Rust and `msgspec` for JSON in Python.
4. **Documentation**: Keep `AGENTS.md` and `README.md` updated with any architectural changes.

## Environment Setup
- Ensure a virtual environment is active (usually `.venv`).
- Run `bootstrap.sh` for a full stack initialization (requires Docker/Podman).
- Use `scripts/setup_pki.sh` to regenerate certificates if needed.

## Performance Optimization
- When optimizing, monitor `/metrics` to see the impact on `manifold_latency_seconds`.
- Use SIMD where applicable in the Rust core.
- Prefer zero-copy operations when passing data between Python and Rust.
