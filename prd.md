# Optimized Pipeline Optimization PRD

## HR Eng

| Optimized Pipeline Optimization PRD |  | Summary: Total refactor of the ML pipeline to eliminate temporal leakage, centralize tracking, and optimize execution speed. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: Morty (The User) **Intended audience**: Engineering | **Status**: Approved **Created**: 2026-02-08 | **Visibility**: Need to know |

## Introduction

The current ML pipeline is primitive. It suffers from data leakage due to random splitting on time-series data (`train_all.py`) and tracks experiments locally, making collaboration impossible. We are upgrading this to a "God-Mode" pipeline.

## Problem Statement

**Current Process:** 
- `train_test_split` with shuffling causes look-ahead bias (Validation Leakage).
- MLflow logs to local files, making it invisible to the cluster.
- Codebase contains "slop" (inefficient loops, lack of vectorization).

**Primary Users:** ML Engineers, The System itself (Self-Healing).
**Pain Points:** Invalid metrics, untracked experiments, slow iteration cycles.
**Importance:** We cannot price derivatives accurately if our models are training on the answers.

## Objective & Scope

**Objective:** Create a leak-free, centralized, high-performance ML pipeline.
**Ideal Outcome:** >97% Code Coverage, verified temporal splitting, centralized Native PostgreSQL-backed MLflow tracking.

### In-scope or Goals
-   **Temporal Validation**: Enforce strict sequential splitting for all time-series data.
-   **Centralized Tracking**: Configure MLflow to use `settings.DATABASE_URL` (Native Postgres).
-   **Audit & Refactor**: Scan `src/ml` and `src/models` for inefficiencies (loops vs vectorized).
-   **Coverage**: Add missing tests to ensure robust execution.

### Not-in-scope or Non-Goals
-   Rewriting the core math kernels (WASM/CUDA) unless blocking.
-   Any Cloud-specific wrappers (Neon, Supabase).

## Product Requirements

### Critical User Journeys (CUJs)
1.  **Training Run**: User executes `python src/ml/training/train_all.py`. The system splits data sequentially (no leakage), trains the model, and logs metrics/artifacts to the remote Postgres DB.
2.  **Audit**: Developer runs the test suite. Coverage is >97%. No "slop" code detected in critical paths.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | **Strict Temporal Split** | As a model, I want to learn from the past, not the future. |
| P0 | **Postgres MLflow Backend** | As an engineer, I want my experiments tracked in a real database, not `tmp/`. |
| P1 | **Vectorization Audit** | As a CPU, I want to process arrays, not iterate lists. |
| P1 | **97% Coverage** | As a deity, I demand perfection. |

## Assumptions
-   The `settings.DATABASE_URL` is available and writeable.
-   `src/ml/training/train_all.py` is the primary entry point.

## Risks & Mitigations
-   **Risk**: DB Connection failures. **Mitigation**: Retry logic in config.
-   **Risk**: Model performance drops after fixing leakage. **Mitigation**: Accept the truth; the previous metrics were lies.

## Business Benefits/Impact/Metrics
-   **Metric**: Model Validity (Leakage). **Target**: 0%.
-   **Metric**: Code Coverage. **Target**: >97%.
