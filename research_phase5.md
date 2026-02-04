# Research: Singularity Phase 5 (ML Pipeline & Evaluation)

**Date**: 2026-02-04

## 1. Executive Summary
Audit of the ML training and evaluation modules identifies a critical "Jerry-leak": the use of random shuffling for time-series validation. Additionally, fragmented training scripts (`train.py`, `train_v2.py`, `train_all.py`) create maintenance overhead and potential inconsistencies in model tracking.

## 2. Technical Context
- **Master Trainer**: `src/ml/training/train_all.py:1` orchestrates XGBoost and NN training.
- **Evaluation**: `src/ml/evaluation/metrics.py:1` provides regression and financial metrics.
- **Tracking**: MLflow is used but defaults to local file storage (`train_all.py:100`).

## 3. Findings & Analysis
- **Temporal Leakage**: `train_all.py:82` uses `train_test_split` with a random seed. For market data, this is invalid as it violates temporal ordering. We must use `TimeSeriesSplit` or a custom Walk-Forward validation.
- **Metric Fragmentation**: `metrics.py` has good weighted MSE logic but lacks a unified "Model Scorecard" that combines regression accuracy with financial risk metrics (Sharpe, MaxDD).
- **Storage Slop**: MLflow is not configured to use the Neon backend for tracking, making distributed experiment management impossible.
- **Redundancy**: Multiple `train_*.py` scripts contain duplicated data preparation logic.

## 4. Technical Constraints
- Model training must remain reproducible.
- Evaluation metrics must support both pricing (regression) and trading (financial) performance.

## 5. Architecture Documentation
- **Current Pattern**: Fragmented scripts with local state.
- **Desired Pattern**: Unified Pipeline using a Shared Feature Store and Temporal Validation.
EOF
