# Plan: MLOps Hardening

## Steps
1.  **Refactor Trainer**:
    -   Update `validate()` to return a metrics dict.
    -   Use `ModelScorecard` for full evaluation.
    -   Log metrics to MLflow at each epoch end.

2.  **Refactor Training Loop**:
    -   Update `train_v2.py` to use `WalkForwardValidator`.
    -   Implement a loop over folds.
    -   Report cross-validation average performance.

## Validation
-   Running `python src/ml/training/train_v2.py` should complete with logged metrics for each fold.
-   Check MLflow for the new metrics (Sharpe, R2, RMSE).

