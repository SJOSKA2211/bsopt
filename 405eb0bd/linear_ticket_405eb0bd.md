---
id: 405eb0bd
title: Fix Existing Failures (Batch 1 - ML/Trainer)
status: Done
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [bug, test-fix]
assignee: Joseph Kamau Maina
---

# Description

## Problem to solve
Hundreds of tests are failing or erroring.

## Solution
Fixed critical issues in `src/ml/`:
1.  `tft_model.py`: Missing imports.
2.  `trainer.py`: Missing base class, incorrect attribute access.
3.  `tracker.py` / Instrumentation: Fixed patching targets in tests.
4.  `mock_all.py`: Enhanced mocks for `train_test_split` and `optuna`.

# Discussion/Comments

- 2026-02-06 Joseph Kamau Maina: Pivoted to fix ML/Trainer/TFT first as they were causing the most noise and Ray issues. API/Auth/DB will be handled in Batch 2. ML module imports are now stable.
