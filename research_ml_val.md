# Research: Walk-Forward Validation for ML Tests

**Date**: 2026-02-04

## 1. Executive Summary
The current ML validation strategy is inconsistent. While some paths use `TimeSeriesSplit`, many still rely on static splits or random shuffling (in the case of the NN classifier). To achieve "Singularity" reliability, we must enforce a uniform Walk-Forward Validation strategy across all time-series models.

## 2. Technical Context
- **Optuna Path**: `src/ml/training/train.py:54` implements `TimeSeriesSplit` for XGBoost HPO.
- **Ray Path**: `src/ml/training/train.py:116` uses `train_test_split(..., shuffle=False)`, which is a single-fold temporal split.
- **TFT Path**: `tests/ml/test_tft_model.py:10` generates random noise for testing, which does not exercise the model's structural learning.

## 3. Findings & Analysis
- **Inconsistency**: There is no shared utility for temporal validation. Every script rolls its own logic.
- **Static Split Risk**: Single-fold splits (`shuffle=False`) are prone to local market regime bias. Walk-Forward (multi-fold sliding window) is required for robustness.
- **Mock Data Slop**: Unit tests use pure noise instead of structured synthetic data, leading to low-quality validation of the training logic.

## 4. Technical Constraints
- `TimeSeriesSplit` from sklearn is the standard for multi-fold temporal validation.
- Validation windows must ensure no data leakage (no overlap between train/test in any fold).

## 5. Architecture Documentation
- **Proposed Utility**: `src/ml/utils/validation.py` containing a `WalkForwardValidator` class.
- **Strategy**: 5-fold sliding window with expanding training set.
EOF
