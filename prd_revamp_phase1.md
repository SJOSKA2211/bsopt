# BS-OPT Revamp: Phase 1 PRD

## HR Eng

| BS-OPT Revamp Phase 1 |  | Comprehensive overhaul of the Hybrid Worker layer and Vectorized Risk Management to eliminate technical debt and optimize performance. |
| :---- | :---- | :---- |
| **Author**: High-Performance Engine **Contributors**: None **Intended audience**: Engineering | **Status**: Draft **Created**: 2026-03-05 | **Self Link**: N/A **Context**: Feature Revamp |

## Introduction
The current "Hybrid Worker" implementation is a mess of blocking calls and inefficient resource management. The "Vectorized Risk Management" is a good start but lacks the "God-tier" optimization needed for sub-microsecond trading. This revamp will modernize these core systems.

## Problem Statement
**Current Process:** 
- `math_worker.py` uses blocking `asyncio.run` calls inside Celery, leading to thread exhaustion.
- Ray delegation is inefficient and uses the `ActorPool` incorrectly.
- Risk kernels re-calculate portfolio deltas from scratch every time, wasting CPU cycles.

**Primary Users:** Quant Engineers, Traders.
**Pain Points:** Inconsistent latency, high CPU overhead, and potential for race conditions in shared state.
**Importance:** Critical for scaling the platform to handle 100k+ concurrent orders with sub-microsecond risk enforcement.

## Objective & Scope
**Objective:** Refactor the worker layer and optimize risk kernels for maximum efficiency and reliability.
**Ideal Outcome:** A clean, asynchronous worker layer and incremental, high-performance risk validation.

### In-scope or Goals
- Refactor `math_worker.py` to use proper asynchronous patterns.
- Implement an efficient Ray delegation strategy (e.g., using `ray.get` or `asyncio` with Ray).
- Implement an incremental Delta Tracker for the risk kernels.
- Add comprehensive unit tests for the new worker logic and risk kernels.

### Not-in-scope or Non-Goals
- Full rewrite of the ML/RL models (that's Phase 2, Assistant).
- Blockchain integration (Phase 3).

## Product Requirements
The system must be fully refactored and verified with automated tests.

### Critical User Journeys (CUJs)
1. **Symbol Recalibration**: A trader triggers a recalibration; the system delegates to Ray asynchronously and returns the result without blocking the Celery worker.
2. **Order Submission**: An order is submitted; the risk engine validates the order against an incremental delta tracker in < 500ns.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Asynchronous Ray Delegation | As an engineer, I want the Celery worker to handle multiple Ray tasks without blocking. |
| P0 | Incremental Delta Validation | As a trader, I want my risk checks to be as fast as possible by avoiding redundant calculations. |
| P1 | Error Handling & Retries | As a systems engineer, I want robust error handling and circuit breaking for the worker layer. |

## Assumptions
- Ray and Redis are available and correctly configured in the environment.
- The existing `HestonCalibrator` is functional and only needs better orchestration.

## Risks & Mitigations
- **Risk**: Asynchronous Celery workers can be tricky to configure. -> **Mitigation**: Use proper task base classes and ensure the event loop is correctly managed.
- **Risk**: Incremental trackers can drift. -> **Mitigation**: Implement a periodic "full-sync" check for the delta tracker.

## Tradeoff
- Complexity of asynchronous code vs. the simplicity of blocking code. We choose complexity for performance.

## Business Benefits/Impact/Metrics
**Success Metrics:**

| Metric | Current State (Benchmark) | Future State (Target) | Savings/Impacts |
| :---- | :---- | :---- | :---- |
| Calibration Latency | ~500ms (blocking) | ~200ms (non-blocking) | 60% reduction |
| Risk Check Latency | ~500ns | < 300ns | 40% reduction |
| Worker Throughput | Low | High (Async) | Scalable to 100k+ tasks |

## Stakeholders / Owners

| Name | Team/Org | Role | Note |
| :---- | :---- | :---- | :---- |
| High-Performance Engine | God Tier | Lead Architect | Performance audit active. |
