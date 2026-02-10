# BSOpt Phase 3: The Optimized PRD

## HR Eng

| Phase 3: The Optimized PRD |  | Summary: Achieving sub-microsecond latency through true kernel-bypass (AF_XDP), a lock-free shared memory mesh, and JIT-optimized RL state construction. |
| :---- | :---- | :---- |
| **Author**: Joseph Kamau Maina **Contributors**: The User (The User) **Intended audience**: Engineering | **Status**: Approved **Created**: 2026-02-08 | **Visibility**: Need to know |

## Introduction

Phase 2 unified the math. Phase 3 achieves total dominance. We are removing every single barrier between the wire and the model. No more kernel stacks, no more mutexes, no more Kafka overhead in the critical update loop.

## Problem Statement

**Current Process:** 
- `xdp_ingest.py` uses standard raw sockets (Fake XDP).
- `shm_mesh.py` uses `multiprocessing.Lock` (Huge bottleneck).
- `online_agent.py` uses Kafka (High latency).
- RL state construction uses unoptimized NumPy slices.

**Primary Users:** HFT Modules, Autonomous Market Makers, The Machine.
**Pain Points:** Packet-to-prediction latency in the milliseconds range (Unacceptable).
**Importance:** In HFT, being second is just a expensive way to lose money.

## Objective & Scope

**Objective:** Achieve <100µs end-to-end latency from packet arrival to RL action.
**Ideal Outcome:** True AF_XDP ingestion, a lock-free single-writer/multi-reader mesh, and a JIT-compiled state engine.

### In-scope or Goals
-   **True AF_XDP**: Implement a native AF_XDP reader (using `libxdp` bindings or raw syscalls) to bypass the kernel stack.
-   **Lock-Free Mesh**: Refactor `shm_mesh.py` to use atomic index updates (via `multiprocessing.Value` or raw memory offsets) instead of `multiprocessing.Lock`.
-   **Zero-Latency Agent**: Switch `online_agent.py` to read directly from the SHM Mesh instead of Kafka.
-   **Fused Kernels**: Use Numba to fuse `_get_state_vector` and indicator calculations into a single JIT-compiled function with AVX-512 support.

### Not-in-scope or Non-Goals
-   Rewriting the core RL model architecture (Transformer/GAT).
-   Changing the database schema.

## Product Requirements

### Critical User Journeys (CUJs)
1.  **HFT Update**: A price packet arrives on the NIC. AF_XDP dumps it into UMEMA. The Ingester writes it to the lock-free SHM Mesh. The Agent, spinning on the SHM head, detects the update, constructs the state vector via JIT kernel, and produces an action. Total time: <100µs.
2.  **Telemetry**: Performance metrics are pushed to the gateway *off the hot path* using a non-blocking queue.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | **Atomic Mesh** | As a writer, I don't want to wait for readers to finish. |
| P0 | **SHM Agent Loop** | As an agent, I want to see price updates the instant they hit memory. |
| P1 | **AF_XDP Ingest** | As a packet, I want to avoid the kernel's boring processing. |
| P1 | **JIT State Engine** | As a CPU, I want to use my AVX-512 units for state vector construction. |

## Assumptions
-   The NIC supports XDP (most modern ones do).
-   Python 3.13 stability for Numba JIT.

## Risks & Mitigations
-   **Risk**: Complexity of raw AF_XDP in Python. **Mitigation**: Use `pyelftools` and raw `setsockopt` if no library fits.
-   **Risk**: Race conditions in lock-free ring buffer. **Mitigation**: Strictly enforced Single-Writer/Multi-Reader (SWMR) pattern.

## Business Benefits/Impact/Metrics
-   **Metric**: Packet-to-Pricing Latency. **Target**: <100µs.
-   **Metric**: Throughput (Ticks/sec). **Target**: >1,000,000.
-   **Metric**: CPU Cache Efficiency. **Target**: >90% L1 Hit Rate for kernels.
