---
task_complexity: medium
topic: rust-core-health-and-telemetry
date: 2026-03-30
---

# Implementation Plan: Rust Core Health & Telemetry Revamp

This plan outlines the steps to instrument the Rust math kernel, expose metrics to Python via PyO3, and integrate them into the existing FastAPI `/metrics` endpoint.

## 1. Plan Overview
- **Total Phases**: 3
- **Agents Involved**: `coder`, `backend_specialist`, `performance_engineer`, `tester`
- **Estimated Effort**: Medium

## 2. Dependency Graph
```
Phase 1: Rust Foundation & Instrumentation (coder)
    |
    v
Phase 2: Python Integration & API Exposure (backend_specialist)
    |
    v
Phase 3: Validation, Performance & Documentation (tester, performance_engineer)
```

## 3. Execution Strategy Table

| Stage | Focus | Agent | Mode |
|-------|-------|-------|------|
| 1 | Rust Metrics Registry & PyO3 Bridge | `coder` | Sequential |
| 2 | Python Wrapper & FastAPI Integration | `backend_specialist` | Sequential |
| 3 | Verification & Benchmarking | `tester` | Sequential |

## 4. Phase Details

### Phase 1: Rust Foundation & Instrumentation
- **Objective**: Ensure Rust core health and implement internal metrics.
- **Agent**: `coder`
- **Files to Modify**:
    - `src/math_kernel/rust-core/Cargo.toml`: Add `prometheus = "0.13"` and `lazy_static = "1.4"`.
    - `src/math_kernel/rust-core/src/lib.rs`: 
        - Implement a global `ManifoldRegistry` using `lazy_static`.
        - Define `CALL_COUNTER` (CounterVec), `LATENCY_HISTOGRAM` (HistogramVec), and `RESOURCE_GAUGE` (Gauge).
        - Instrument `black_scholes_price`, `batch_black_scholes`, `batch_heston_price`, and `simulate_gbm_rk4`.
        - Implement `get_manifold_metrics() -> PyResult<String>` using `prometheus::TextEncoder`.
- **Validation**:
    - `cd src/math_kernel/rust-core && cargo test` (Requires Docker environment simulation or assuming availability).
    - `cargo clippy --manifest-path src/math_kernel/rust-core/Cargo.toml`.

### Phase 2: Python Integration & API Exposure
- **Objective**: Connect the Rust metrics to the Python API.
- **Agent**: `backend_specialist`
- **Files to Modify**:
    - `src/math_kernel/rust_engine.py`: Add `get_rust_metrics() -> str` that calls `Manifold_core.get_manifold_metrics()` with a try-except fallback.
    - `api/index.py`: Update the `/metrics` endpoint to call `rust_engine.get_rust_metrics()` and append the result to `generate_latest()`.
- **Validation**:
    - `uv run pytest tests/unit/test_core_engine.py`.
    - `curl http://localhost:8000/metrics` and verify `manifold_` metrics are present.

### Phase 3: Validation, Performance & Documentation
- **Objective**: Ensure NFRs are met and document the new system.
- **Agent**: `tester`, `performance_engineer`, `technical_writer`
- **Files to Modify**:
    - `README.md`: Document the new observability features and the `/metrics` endpoint.
- **Implementation Details**:
    - Create a benchmark script to compare pricing throughput with and without metrics (using a toggle if possible, or comparing against baseline).
- **Validation**:
    - Confirm SC-1 through SC-5.

## 5. File Inventory

| Phase | Path | Action | Purpose |
|-------|------|--------|---------|
| 1 | `src/math_kernel/rust-core/Cargo.toml` | Modify | Add prometheus/lazy_static dependencies. |
| 1 | `src/math_kernel/rust-core/src/lib.rs` | Modify | Metric registry and instrumentation. |
| 2 | `src/math_kernel/rust_engine.py` | Modify | Python-side bridge for metrics. |
| 2 | `api/index.py` | Modify | Expose Rust metrics in /metrics endpoint. |
| 3 | `README.md` | Modify | Update documentation. |

## 6. Risk Classification

| Phase | Risk | Level | Rationale |
|-------|------|-------|--|
| 1 | Rust Compilation Error | MEDIUM | New dependencies might conflict or require specific features. |
| 1 | Performance Overhead | MEDIUM | Histogram collection can be expensive in tight loops. |
| 2 | Python Scrape Failure | LOW | Simple string concatenation is robust. |

## 7. Execution Profile
- Total phases: 3
- Parallelizable phases: 0 (Sequential flow due to strong dependencies)
- Sequential-only phases: 3
- Estimated sequential wall time: 2-3 hours

## 8. Cost Summary

| Phase | Agent | Model | Est. Input | Est. Output | Est. Cost |
|-------|-------|-------|--|--|--|
| 1 | coder | Pro | 2000 | 1000 | $0.09 |
| 2 | backend_specialist | Pro | 1500 | 500 | $0.05 |
| 3 | tester | Pro | 1000 | 500 | $0.04 |
| **Total** | | | **4500** | **2000** | **$0.18** |
