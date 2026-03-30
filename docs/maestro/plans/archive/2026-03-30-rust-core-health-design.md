---
design_depth: deep
task_complexity: medium
topic: rust-core-health-and-telemetry
date: 2026-03-30
---

# Design Document: Rust Core Health & Telemetry Revamp

## 1. Problem Statement
The `Manifold_core` Rust math kernel is a critical component for high-performance financial pricing and simulations. Currently, its "health" is only monitored via a basic availability check (`is_rust_available`), which doesn't capture runtime issues like functional drift, performance degradation (latency), or resource exhaustion (memory/threads). This lack of observability makes it difficult to diagnose why the system might fall back to NumPy-based solvers or to optimize the kernel's performance in production. We need to "run rust-core until its healthy"—meaning functional, fast, and observable—and "revamp it" by implementing a robust, Prometheus-compatible telemetry system that reports detailed health and performance metrics through the Python engine.

## 2. Requirements

### Functional Requirements
- **REQ-1 (Core Health)**: Ensure all existing tests in `tests/test_core_engine.py` pass and the code complies with `clippy` and `rustfmt`.
- **REQ-2 (Internal Instrumentation)**: The Rust kernel must track functional metrics (total calls, failures) and performance metrics (latency histograms for pricing and simulation functions).
- **REQ-3 (Resource Monitoring)**: The Rust kernel must report system-level health, such as thread pool status (via `rayon`) and memory-mapped file usage (`TickDataBuffer`).
- **REQ-4 (Python Bridge)**: Expose a `get_prometheus_metrics()` function in Rust that returns a standard Prometheus-formatted text string.
- **REQ-5 (API Exposure)**: The Python engine must host a `/metrics` endpoint that aggregates internal Python metrics with the polled Rust metrics.

### Non-Functional Requirements
- **NFR-1 (Performance)**: Telemetry collection must have negligible impact on math kernel throughput (target < 1% overhead).
- **NFR-2 (Standardization)**: Use industry-standard libraries (`prometheus` crate in Rust, standard telemetry formats).
- **NFR-3 (Resilience)**: Failures in telemetry collection must not crash the functional path.

### Constraints
- **CON-1**: Must use `PyO3` for the Python-Rust boundary.
- **CON-2**: Must be compatible with the existing multi-stage Docker build process.

## 3. Approach
We will implement **Approach 1: Integrated Prometheus Bridge**. This approach leverages the `prometheus` crate in Rust to maintain an internal registry of metrics. The Python engine will periodically poll this registry through a specialized PyO3 function, ensuring a clean and low-overhead boundary.

### Selected Approach: Integrated Bridge
- **Implementation**: Create a global registry in `src/math_kernel/rust-core/src/lib.rs` and wrap pricing/simulation functions with metric increment/timing logic.
- **Rationale**: Best fit for existing PyO3 integration; high standardization with low effort. — *Traces to REQ-4, NFR-1.*

### Alternatives Considered
- **Sidecar Telemetry** (Rejected: disproportionate deployment complexity). — *Traces to NFR-2.*
- **Shared Memory Telemetry** (Rejected: complexity not justified by performance gains). — *Traces to NFR-1.*

### Decision Matrix Summary
The Integrated Bridge approach scored **4.8/5.0**, excelling in observability depth and implementation effort.

## 4. Architecture

### Key Components
1. **Rust Metric Registry**: A global static registry using the `prometheus` crate. Tracks counters (calls), histograms (latency), and gauges (resources).
2. **PyO3 Bridge**: `get_manifold_metrics() -> PyResult<String>` returns the scraped registry as text.
3. **Python Metrics Manager**: Utility in `rust_engine.py` to poll the bridge.
4. **API Endpoint**: FastAPI `/metrics` route serving aggregated metrics.

### Data Flow
1. **Functional Path**: App -> `rust_engine.py` -> Rust function -> metrics update -> Result.
2. **Observability Path**: Scraper -> FastAPI -> `rust_engine.py` -> PyO3 Bridge -> Registry Scrape -> Text.

## 5. Agent Team
- **coder (Rust)**: Implements instrumentation and PyO3 bridge.
- **backend_specialist (Python)**: Implements FastAPI endpoint and polling.
- **performance_engineer**: Validates < 1% overhead (NFR-1).
- **tester**: Verifies functional correctness and telemetry availability.
- **code_reviewer**: Final quality gate.

## 6. Risk Assessment
- **Instrumentation Overhead**: Mitigated by static registry and atomic counters.
- **Metrics Panic**: Mitigated by no-op patterns on uninitialized registries.
- **GIL Contention**: Mitigated by pre-formatted strings from Rust.
- **Build Failures**: Mitigated by local maturin/docker verification.

## 7. Success Criteria
- **SC-1 (Healthy Core)**: Green tests and clippy.
- **SC-2 (Telemetry Integrity)**: `/metrics` returns valid Prometheus data.
- **SC-3 (Functional Transparency)**: Real timing data in histograms.
- **SC-4 (Resilience Verified)**: System remains functional despite telemetry failures.
- **SC-5 (Performance Validated)**: Throughput overhead < 1%.
