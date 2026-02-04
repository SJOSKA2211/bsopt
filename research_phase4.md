# Research: Singularity Phase 4 (Gateway & WASM)

**Date**: 2026-02-04

## 1. Executive Summary
Audit of the API Gateway and WASM execution layers reveals high-performance foundations but significant orchestration and safety "slop". The Gateway lacks shared subgraph introspection across workers, and the WASM SIMD path is implemented using unsafe memory transmutations that risk runtime panics.

## 2. Technical Context
- **API Gateway**: `src/gateway/index.js:1` uses Fastify and Apollo Gateway.
- **WASM Rust**: `src/wasm/src/lib.rs:1` provides pricing kernels via `wasm-bindgen`.
- **SIMD Path**: `BlackScholesWASM.batch_calculate_simd` (`lib.rs:215`) uses `std::arch::wasm32` intrinsics.

## 3. Findings & Analysis
- **Gateway Redundancy**: When running in `cluster` mode (`index.js:100`), every worker process independently runs `IntrospectAndCompose`. This can be optimized by providing a pre-composed supergraph SDL via a shared volume or environment variable.
- **WASM Safety**: `batch_calculate_simd` uses `unsafe` and `std::mem::transmute` (`lib.rs:252`) to handle SIMD results. This is brittle. We should move to safer SIMD abstractions or at least wrap the memory access in a safer buffer management pattern.
- **WASM Feature Parity**: The SIMD path only calculates Price, then falls back to non-SIMD `calculate_greeks` (`lib.rs:261`), negating the performance benefits for Greek calculation batches.

## 4. Technical Constraints
- Gateway must maintain compatibility with Apollo Federation v2.
- WASM module must remain compatible with both Node.js and Browser runtimes.

## 5. Architecture Documentation
- **Gateway**: Acts as the "Federated Brain" of the platform.
- **WASM**: Serves as the "Computational Liver", processing heavy math with near-native speed.
EOF
