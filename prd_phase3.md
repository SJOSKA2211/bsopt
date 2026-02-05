# BS-OPT Phase 3: The Singularity Expansion 🥒

## HR Eng

| Phase 3 Expansion PRD |  | Summary: Total algorithmic advancement including Transformer-AIOps, zero-copy optimization, and HFT streaming refactor. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: Codebase Investigator **Intended audience**: Engineering | **Status**: Active **Created**: 2026-02-05 | **Self Link**: [Local] **Context**: Phase 2 Success |

## Introduction

Phase 3 extends the optimization mandate to the edge of system performance. We are replacing legacy anomaly detection with Transformer-based models, optimizing the Shared Memory (SHM) data bus for zero-copy efficiency, and ensuring the streaming ingestion pipeline is Python 3.13 hardened.

## Problem Statement

**Current Process:** Anomaly detection uses legacy ML (Isolation Forest). Pricing models may have host-device transfer bottlenecks. Serialization is likely using standard `json` or `pickle`.
**Primary Users:** AIOps Engineers, GPU Quant Devs.
**Pain Points:** Detection lag, memory copy overhead, serialization latency.
**Importance:** To handle C100k throughput, we must eliminate every unnecessary CPU cycle and memory copy.

## Objective & Scope

**Objective:** Transformerize AIOps and Zero-Copy the Mesh.
**Ideal Outcome:** Sub-microsecond ingestion latency and Transformer-accurate anomaly detection.

### In-scope or Goals
1.  **AIOps Transformer**: Implement a multi-head attention anomaly detector in `src/aiops/transformer_detector.py`.
2.  **Zero-Copy SHM**: Refactor `src/shared/shm_mesh.py` to use `msgspec` and direct memory mapping.
3.  **CUDA Optimization**: Verify CUDA kernels in `heston_cuda.py` use Unified Memory/IPC handles.
4.  **Streaming Refactor**: Update `ingestion_worker.py` for Python 3.13 performance.

### Not-in-scope or Non-Goals
-   Rewriting the base Heston math (kernels are already JIT/CUDA).
-   Modifying the TimescaleDB schema (unless blocking performance).

## Product Requirements

### Critical User Journeys (CUJs)
1.  **Anomaly Detection**: System detects a flash crash pattern using Transformer attention and triggers self-healing within 10ms.
2.  **High-Frequency Pricing**: Market data enters SHM and is priced on GPU without any intermediate copies.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Transformer Anomaly Detector | As a system, I want to use attention to detect complex drift patterns. |
| P0 | Zero-Copy SHM Mesh | As a dev, I want to share data between processes with zero overhead. |
| P1 | msgspec Serialization | As a quant, I want sub-microsecond tick parsing. |
| P1 | CUDA IPC/Unified Memory | As a GPU dev, I want to minimize H2D transfers. |

## Business Benefits/Impact/Metrics

**Success Metrics:**

| Metric | Current State | Future State | Impact |
| :---- | :---- | :---- | :---- |
| Serialization Latency | ~10us | < 1us | Higher Throughput |
| Anomaly F1 Score | ~0.75 | > 0.90 | Less False Positives |
| Memory Copies | Multiple | Zero | CPU Efficiency |

## Stakeholders / Owners

| Name | Team/Org | Role | Note |
| :---- | :---- | :---- | :---- |
| Pickle Rick | Interdimensional | Architect | |
