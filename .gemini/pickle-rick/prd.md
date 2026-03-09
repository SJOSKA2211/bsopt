# BS-OPT Advancement & Native PostgreSQL Refactor PRD

## HR Eng

| BS-OPT Advancement PRD |  | Deep refactor to native PostgreSQL and ML pipeline optimization. |
| :---- | :---- | :---- |
| **Author**: High-Performance Engine **Contributors**: Assistant **Intended audience**: Engineering | **Status**: Draft **Created**: 2026-02-07 | **Context**: Full Codebase Optimization |

## Introduction
The current BS-OPT platform is built on Neon PostgreSQL and fragmented ML training scripts. This PRD outlines the transition to native PostgreSQL features (removing serverless dependencies) and the consolidation/optimization of ML pipelines for transdimensional pricing accuracy.

## Problem Statement

**Current Process:** 
- Database uses native PostgreSQL features.
- ML pipelines are fragmented and lack rigorous temporal validation.
- Test coverage is inconsistent across math kernels and ML serving logic.
**Primary Users:** Quant researchers, high-frequency traders.
**Pain Points:** Dependency lock-in, latency in model retraining, "Jerry-level" boilerplate.
**Importance:** High-performance trading requires total control over the data layer and maximum algorithmic efficiency.

## Objective & Scope

**Objective:** 
1. Refactor database layer to use native PostgreSQL features (TimeScaleDB-like partitioning, advanced indexing).
2. Consolidate ML pipelines into a unified, high-performance training/validation/evaluation loop.
3. Advance models (Transformers, RL) across the codebase.
4. Achieve >97% test coverage.

### In-scope or Goals
- Database: Replace Neon-specific logic with native pg_partman/partitioning.
- ML: Refactor `src/ml/pipeline.py` and `ModelTrainer`.
- Math: Optimize pricing kernels with Numba JIT and vectorized NumPy 2.0.
- Quality: Unit and integration tests for all core functions.

### Not-in-scope or Non-Goals
- Migrating to a different cloud provider (staying on existing infra but removing DB service dependencies).
- Changing the frontend UI (focus is on backend/math/ML).

## Product Requirements

### Critical User Journeys (CUJs)
1. **Native DB Migration**: Developer runs migration scripts that establish native partitioning and optimized indexes without Neon extensions.
2. **Unified Training**: Quant researcher triggers the ML pipeline which fetches data, performs HPO via Optuna/Ray, and registers a production-ready model in MLflow.
3. **High-Performance Pricing**: The system prices 100k options in <10ms using the optimized JIT-compiled kernels.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Native PostgreSQL Refactor | As an engineer, I want to use native PG features so I don't rely on third-party serverless DBs. |
| P0 | ML Pipeline Consolidation | As a researcher, I want a single source of truth for training and evaluation. |
| P1 | Numba/JIT Optimization | As a trader, I want sub-microsecond latency in pricing kernels. |
| P1 | 97% Test Coverage | As a maintainer, I want to ensure the mathematical validity of all kernels. |

## Assumptions
- PostgreSQL 16+ is available.
- CUDA environment is correctly set up for GPU-bound tasks.
- Python 3.13 venv is active.

## Risks & Mitigations
- **Risk**: Partitioning complexity -> **Mitigation**: Use `pg_partman` or robust native DDL scripts.
- **Risk**: Numba compilation overhead -> **Mitigation**: Use ahead-of-time (AOT) compilation or persistent cache.

## Business Benefits/Impact/Metrics
- **Latency**: Reduce pricing latency by 30%.
- **Reliability**: Increase test coverage to >97%.
- **Independence**: 0% dependency on Neon/Supabase.

## Stakeholders / Owners
| Name | Role |
| :---- | :---- |
| High-Performance Engine | High-Performance Architect |
| Assistant | Lead Compliance Officer |
