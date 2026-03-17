# BS-OPT Revamp: Phase 4 PRD (Quantum & Distributed Optimization)

## Introduction
Phase 4 is the final "God-tier" optimization phase. It focuses on two futuristic pillars:
1. **Quantum Realization**: Transitioning from analytical fallbacks to actual Quantum Amplitude Estimation (QAE) for option pricing.
2. **Distributed Ray Optimization**: Scaling RL training from a single node to a multi-node Ray cluster with high-performance data sharding.

## Problem Statement
- **Quantum Slop**: `quantum_pricing.py` contains the math but relies on simulation fallbacks. It needs to be verified against real quantum primitives (Qiskit 1.0+).
- **Training Scalability**: The current `BSOptDistributedTrainer` is localized. It needs to handle large-scale trajectory datasets across a distributed cluster without OOM or RPC bottlenecks.

## Objective
Finalize the quantum pricing engine and the distributed training orchestrator.

## Scope
- **Amplitude Estimation (QAE-v1)**: Implement Iterative Amplitude Estimation (IAE) using `qiskit_algorithms` for quadratic speedup over Monte Carlo.
- **Ray Cluster Orchestration**: Refactor `BSOptDistributedTrainer` to support automatic GPU/CPU resource negotiation and sharded data loading via `ray.data`.
- **Advanced Circuit Optimization**: Implement transpiler passes to minimize circuit depth for NISQ-era compatibility.

## Technical Requirements
- Ensure `QISKIT_AVAILABLE` is True in the environment.
- Use `ray.train.torch.prepare_data_loader` for DDP-compatible data sharding.
- Target < 500ms for a 5-qubit QAE execution on a simulator.
