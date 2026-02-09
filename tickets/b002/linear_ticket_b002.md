---
id: b002
title: Standardize InstrumentedTrainer Interface
status: Triage
priority: Urgent
project: bsopt
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, refactor, architecture]
assignee: Pickle Rick
---

# Description

## Problem to solve
Training logic is split between `src/ml/training/train.py` (XGBoost) and `train_nn.py` (NN), making it hard to maintain or swap models.

## Solution
Refactor `InstrumentedTrainer` in `src/ml/trainer.py` to be the single base class. Update `train.py` and `train_nn.py` to inherit from and strictly follow this interface.
