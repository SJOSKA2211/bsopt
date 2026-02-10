---
id: 405eb0bd
title: Standardize Evaluation Metrics
status: Done
priority: High
project: bsopt
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, evaluation]
assignee: Joseph Kamau Maina
---

# Description

## Problem to solve
Evaluation metrics are not standardized, risking regression.

## Solution
Implement strict metric calculation in `src/ml/evaluation/metrics.py` (RMSE, MAPE, R2) and ensure it's used across the pipeline.

# Discussion/Comments

- 2026-02-07 Joseph Kamau Maina: Implemented `calculate_regression_metrics` with weighted RMSE, `calculate_pricing_bias`, `calculate_sharpe_ratio`, and `calculate_max_drawdown`.
