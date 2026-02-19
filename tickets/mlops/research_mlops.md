# Research: MLOps Hardening

## Objectives
- Audit training loops.
- Implement validation metrics.
- Ensure cross-validation is temporal.

## Findings
- `train_v2.py` used a single random split, which is incorrect for time-series.
- `Trainer` only logged loss, missing financial metrics (Sharpe, etc.).
- `ModelScorecard` was available but not integrated.

## Strategy
- Integrate `ModelScorecard` into the `Trainer` validation loop.
- Use `WalkForwardValidator` in `train_v2.py` for all training runs.

