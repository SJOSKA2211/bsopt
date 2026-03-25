# Design Document: Phase 1 - Rust Integration & Advanced Mathematical Kernels

**Date**: 2026-03-20
**Topic**: High-Performance Compute & Ingestion
**Status**: DRAFT (v2.0 - Revised after Review)
**Task Complexity**: Complex

## 1. Executive Summary
Phase 1 focuses on the implementation of high-performance compute kernels and data ingestion pipelines. By leveraging Rust for CPU-intensive tasks and CuPy/CUDA for GPU-accelerated pricing, Manifold will achieve Production-grade throughput and numerical precision.

## 2. Problem Statement
- Current Rust parsing creates intermediary vectors, failing the zero-copy requirement.
- The EngineArbiter lacks GPU integration, leaving CUDA kernels unused.
- Numerical SDE solvers (RK4-Milstein) are less efficient than analytical solutions for GBM.
- The 32-byte binary format lacks robustness (precision and metadata).

## 3. Proposed Architecture (Unified Rust-Compute Core)

### 3.1. True Zero-Copy Ingestion (Rust + mmap)
- **Hardened Binary Standard**:
    - **Header**: Magic Bytes (`EQUA`), Version (u16), Metadata Length (u16).
    - **Record (32-byte)**: `[Symbol:12, Price:8 (f64), Volume:4 (i32), Timestamp:8 (i64 nanosecs)]`.
- **Rust `TickDataBuffer`**:
    - Uses `numpy` crate to return `PyArray` views directly onto the `memmap2` slice.
    - Zero allocations for data transfer to Python.

### 3.2. Vectorized Black-Scholes Manifold
- **EngineArbiter 2.0**:
    - **Priority 1: GPU**: `CuPy` backend if CUDA is available.
    - **Priority 2: Rust**: Parallelized CPU kernels via `Rayon`.
    - **Priority 3: WASM/NumPy**: Extreme fallback.
- **Precision**: Unified use of **A&S 7.1.26** for Normal CDF.

### 3.3. Exact GBM Path Generation
- **Implementation**: Housed in Rust `Manifold-core`.
- **Method**: Use the **Exact Analytical Solution**:
  $S_t = S_0 \exp\left( (\mu - \frac{1}{2}\sigma^2)t + \sigma W_t \right)$
- **Advantages**: Perfect strong and weak convergence, significantly faster than iterative numerical methods (RK4/Milstein).

## 4. Components & Data Flow
1. **Scraper** -> Hardened Binary File (.bin) with versioned header.
2. **Rust Parser** -> `numpy` view of mmap -> Validated NDArray.
3. **EngineArbiter** -> Selects best hardware path (CuPy vs Rust).
4. **Compute** -> Returns Prices/Greeks.

## 5. Technical Decisions & Tradeoffs
- **Decision**: Analytical GBM over RK4.
    - **Tradeoff**: Specific to GBM, but offers superior performance and accuracy for our core stochastic model.
- **Decision**: i64 Nanosecond Timestamps.
    - **Tradeoff**: Essential for HFT precision, avoids the drift issues of `f64`.

## 6. Validation & Testing
- **Zero-Copy Proof**: Monitor memory allocation during 1GB file ingestion (should be minimal).
- **GPU Throughput**: Target 5M options/sec on modern NVIDIA hardware.
- **Precision**: Bit-perfect match between CPU and GPU paths for identical seeds.

## 7. Next Steps
1. Update `src/math_kernel/rust-core/src/lib.rs` with `PyArray` zero-copy views and Exact GBM.
2. Refactor `src/math_kernel/arbiter.py` to include the `CuPy` path.
3. Update binary format generator and parser to use versioned headers and `i64` timestamps.
