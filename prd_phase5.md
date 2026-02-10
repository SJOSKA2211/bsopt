# BSOpt Phase 5: Convergence PRD

## HR Eng

| Phase 5: Convergence PRD |  | Summary: Eliminating temporal leakage, unifying model evaluation, and centralizing experimental state. |
| :---- | :---- | :---- |
| **Author**: Joseph Kamau Maina **Contributors**: The User (The User) **Intended audience**: Engineering | **Status**: Approved **Created**: 2026-02-08 | **Visibility**: Need to know |

## Introduction

Phase 4 gave us speed and adaptability. Phase 5 gives us rigor. We are fixing the "Jerry-leaks" in our ML pipelines—specifically the temporal leakage in validation—and consolidating our fragmented training logic into a unified, traceable architecture.

## Problem Statement

**Current Process:** 
- `train_all.py` and other scripts use random shuffling for time-series data (Temporal Leakage).
- MLflow logs to local file systems, preventing distributed tracking.
- Evaluation metrics are scattered and don't provide a holistic view of trading vs. pricing risk.
- Multiple redundant training scripts (`train.py`, `train_v2.py`) coexist with duplicated logic.

**Primary Users:** Quant Researchers, ML Engineers, The God Module.
**Pain Points:** Over-optimistic backtests (due to leakage), lost experiment history, maintenance overhead.
**Importance:** In market data, if you look at the future to predict the past, you lose everything.

## Objective & Scope

**Objective:** Implement a theoretically sound, unified ML training and evaluation framework.
**Ideal Outcome:** Zero temporal leakage, 100% centralized experiment tracking, and a unified model scorecard.

### In-scope or Goals
-   **Temporal Validation**: Implement `TimeSeriesSplit` and sequential slicing across all training pipelines.
-   **Unified Scorecard**: Create a consolidated `ModelScorecard` in `src/ml/evaluation/metrics.py` that aggregates regression (MSE, MAE) and financial (Sharpe, Sortino, MaxDD) metrics.
-   **MLflow Centralization**: Configure MLflow to use the Postgres/Neon backend for all tracking.
-   **Pipeline Consolidation**: Consolidate redundant training scripts into a single, robust `src/ml/pipeline.py`.

### Not-in-scope or Non-Goals
-   Changing the neural network architectures (Task for Phase 6).
-   Modifying data ingestion logic.

## Product Requirements

### Critical User Journeys (CUJs)
1.  **Rigorous Validation**: A researcher trains a new model. The pipeline automatically applies Walk-Forward validation, ensuring no future data is leaked. The model performance is realistic.
2.  **Centralized Audit**: The Risk Orchestrator queries the centralized MLflow server to compare metrics across different model versions trained on different nodes.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | **TimeSeries Validation** | As a Quant, I want to trust that my model isn't cheating. |
| P0 | **Centralized MLflow** | As a Dev, I want to see experiments from all workers in one place. |
| P1 | **Unified Metrics** | As a Trader, I want to see regression accuracy and risk in one report. |
| P1 | **Script Consolidation** | As a Dev, I want one source of truth for training logic. |

## Business Benefits/Impact/Metrics
-   **Metric**: Backtest Realism. **Target**: <5% variance between backtest and paper trading.
-   **Metric**: Experiment Shareability. **Target**: 100% of runs in central MLflow.
-   **Metric**: Code Maintenance. **Target**: Remove 1,000+ lines of redundant slop.
