# Implement Temporal Validation Implementation Plan

## Overview
Prevent data leakage in time-series models by replacing random data shuffling with strict sequential splitting in the master training pipeline.

## Current State Analysis
- **Master Trainer**: `src/ml/training/train_all.py:82` uses `train_test_split(..., random_state=42)`.
- **Leakage**: Random shuffling mixes future data points into the training set, leading to over-optimistic but invalid performance metrics.

## Implementation Approach
1. Refactor `train_all` to use sequential slicing for the train/test split.
2. Ensure data is sorted by timestamp (if not already handled by the pipeline).
3. Validate that the test set's temporal range starts after the training set's end.

## Phase 1: Sequential Splitting
### Overview
Remove random shuffling and implement index-based splitting.

### Changes Required:

#### 1. src/ml/training/train_all.py
**Changes**: Replace `train_test_split` with sequential slicing.
```python
    # 3. Train NN Classifier
    logger.info("Training NN Classifier...")
    # Prepare classification targets
    y_class = prepare_classification_data(X, feature_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 🚀 SINGULARITY: Strict Temporal Split (No Shuffling)
    test_size = 0.2
    split_idx = int(len(X_scaled) * (1 - test_size))
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    
    logger.info("temporal_split_complete", train_len=len(X_train), test_len=len(X_test))
```

### Success Criteria:
#### Automated:
- [ ] `pytest tests/test_aiops_foundation.py` (or relevant ml tests) pass.
#### Manual:
- [ ] Log output confirms the training set precedes the test set temporally (by checking metadata timestamps).

**Implementation Note**: Requires no new dependencies.

```