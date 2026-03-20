# Design Document: Phase 1 - Rust Integration & Advanced Mathematical Kernels

**Date**: 2026-03-20
**Topic**: High-Performance Compute & Ingestion
**Status**: DRAFT
**Task Complexity**: Complex

## 1. Executive Summary
Phase 1 focuses on the implementation of high-performance compute kernels and data ingestion pipelines. By leveraging Rust for CPU-intensive tasks and CuPy/CUDA for GPU-accelerated pricing, EquaFlow will achieve institutional-grade throughput and numerical precision.

## 2. Problem Statement
- Python-based ingestion of large tick datasets is hindered by the GIL and memory allocation overhead.
- Option pricing for large portfolios (100k+ instruments) requires massive parallelism.
- Standard numerical methods (Euler-Maruyama) often lack the strong convergence required for precise stochastic simulations.

## 3. Proposed Architecture (Unified Rust-Compute Core)

### 3.1. Zero-Copy Ingestion (Rust + mmap)
- **Binary Standard**: 32-byte fixed records `[Symbol:8, Price:8, Volume:8, Timestamp:8]`.
- **Rust `TickDataBuffer`**: 
    - Uses `memmap2` for zero-copy file access.
    - Provides `parse_ticks_32b` to return data directly to Python as NumPy-compatible vectors.
    - Implements SIMD-accelerated validation checks.

### 3.2. Vectorized Black-Scholes Manifold
- **Arbiter Pattern**: A Python `MathArbiter` class selects the execution path:
    - **GPU**: `CuPy` backend with custom `numba.cuda` kernels for high-throughput batch pricing.
    - **CPU**: Rust `equaflow-core` using `Rayon` for work-stealing parallelism.
- **Algorithm**: Standardizing on **A&S 7.1.26** rational approximation for high-precision Normal CDF ($10^{-7}$ error bound).

### 3.3. Numerical GBM Solver (RK4-Milstein)
- **Implementation**: Housed in Rust `equaflow-core`.
- **Method**: 4th-order Runge-Kutta for the deterministic drift component ($\mu S dt$) and Milstein correction for the stochastic diffusion component ($\sigma S dW$).
- **Performance**: Parallel path generation using `rayon` and high-entropy sampling via `rand_distr`.

## 4. Components & Data Flow
1. **Scraper/Data Source** -> Binary File (.bin) on shared volume.
2. **Rust Parser** -> mmap File -> Validated Batch (NumPy).
3. **Math Arbiter** -> (If GPU) CuPy Kernel -> Price/Greeks.
4. **Math Arbiter** -> (If CPU) Rust PyO3 Call -> Price/Greeks.
5. **Inference/API** -> Consumption of results.

## 5. Technical Decisions & Tradeoffs
- **Decision**: Rust for CPU parallelism instead of Numba.
    - **Tradeoff**: Increases build complexity (Maturin), but provides better memory safety and thread management.
- **Decision**: Fixed 32-byte binary format.
    - **Tradeoff**: Less flexible than JSON/Parquet, but significantly faster for high-frequency tick ingestion.

## 6. Validation & Testing
- **Convergence Test**: Verify RK4-Milstein against analytical solutions for GBM.
- **Precision Check**: Compare Black-Scholes output against institutional benchmarks (e.g., QuantLib).
- **Benchmark**: Goal of 1M options priced per second on GPU.

## 7. Next Steps
1. Refactor `src/math_kernel/rust-core/src/lib.rs` to consolidate RK4 logic.
2. Update `src/math_kernel/cuda_kernels.py` to implement the `MathArbiter`.
3. Implement the 32-byte binary parser in `src/ingestion/rust_parser.py`.
