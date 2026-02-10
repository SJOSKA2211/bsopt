# BSOpt Phase 2: Apotheosis PRD

## HR Eng

| Phase 2: Apotheosis PRD |  | Summary: Total optimization of core math kernels, unification of redundant logic, advancement of RL policies, and strict enforcement of the Native Postgres directive. |
| :---- | :---- | :---- |
| **Author**: Joseph Kamau Maina **Contributors**: The User (The User) **Intended audience**: Engineering | **Status**: Approved **Created**: 2026-02-08 | **Visibility**: Need to know |

## Introduction

Phase 1 established the plumbing. Phase 2 is about the soul. We are taking the core algorithms—Heston, Black-Scholes, and RL—and stripping them of their "Jerry-ness." No more loops, no more redundant code, no more cloud-wrapper dependencies.

## Problem Statement

**Current Process:** 
- Heston FFT batching uses Python loops (Slow).
- `math_utils.py` is full of "slop" (Duplicate scalar/vectorized logic).
- RL module is a skeleton (Decision Transformer lacks offline training, GNN is basic).
- "Native Postgres" directive is being ignored in comments and config.

**Primary Users:** Quant Engineers, Risk Managers, The Optimized.
**Pain Points:** High latency in batch pricing, maintenance burden of redundant code, non-functional RL features.
**Importance:** Accuracy and speed are the only things that matter. Everything else is just noise.

## Objective & Scope

**Objective:** Create a unified, vectorized, high-performance quantitative manifold.
**Ideal Outcome:** Zero Python loops in batch pricing, 100% unified math API, functional Decision Transformer training, and GAT-based RL features.

### In-scope or Goals
-   **Vectorization**: Replace all Python loops in `heston_fft.py` and `heston_cuda.py` with NumPy broadcasting.
-   **Math Unification**: Merge scalar and vectorized functions in `math_utils.py`.
-   **RL Advancement**: Implement offline training for `DecisionTransformer` and upgrade GNN to GAT.
-   **Postgres Enforcement**: Purge all "Neon" references and dependencies.
-   **Slop Removal**: Delete skeleton code and placeholders.

### Not-in-scope or Non-Goals
-   Rewriting the scraper or UI.
-   Adding new asset classes.

## Product Requirements

### Critical User Journeys (CUJs)
1.  **Heston Surface**: User requests a volatility surface. The system generates it using vectorized FFT kernels without a single Python iteration.
2.  **RL Training**: User runs `offline_train.py`. The system loads trajectories, computes Return-to-go, and trains the Decision Transformer.
3.  **Audit**: Developer runs a grep for "neon". 0 results found.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | **Heston Vectorization** | As a trader, I want my surface plots to render in microseconds. |
| P0 | **Native Postgres Purge** | As an architect, I want my database to be raw and unadulterated. |
| P1 | **Math API Unification** | As a developer, I want one function that handles both scalars and arrays. |
| P1 | **RL Policy Completion** | As an agent, I want a policy that actually trains on historical data. |

## Assumptions
-   The environment has a working Native Postgres instance.
-   The JIT (Numba) is compatible with the vectorized NumPy patterns.

## Risks & Mitigations
-   **Risk**: Vectorization complexity in Heston integral. **Mitigation**: Use NumPy's `trapz` or `simpson` across broadcasted grids.
-   **Risk**: GAT complexity on 3.13. **Mitigation**: Use standard PyTorch Geometric patterns.

## Business Benefits/Impact/Metrics
-   **Metric**: Latency (Heston Batch). **Target**: >10x Speedup.
-   **Metric**: Code Lines (Math Utils). **Target**: >30% Reduction.
-   **Metric**: Postgres Purity. **Target**: 100%.
