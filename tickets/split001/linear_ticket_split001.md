---
id: split001
title: Implement Temporal Validation
status: Done
priority: Urgent
project: bsopt
created: 2026-02-08
updated: 2026-02-08
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, validation, leakage]
assignee: Pickle Rick
---

# Description

## Problem to solve
Random shuffling in `train_test_split` causes temporal data leakage, invalidating metrics.

## Solution
Refactor `src/ml/trainer.py` to use strict sequential slicing for train/test splits. Ensure data is sorted by time.

# Discussion
- 2026-02-08 Pickle Rick: Replaced `train_test_split(shuffle=True)` with explicit array slicing. Temporal integrity restored.
