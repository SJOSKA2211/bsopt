---
id: 112a6bd0
title: Unify ML Training Pipeline
status: Done
priority: High
project: bsopt
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, refactor]
assignee: Pickle Rick
---

# Description

## Problem to solve
Training logic is split across `train.py`, `train_all.py` and others. It's messy and hard to maintain.

## Solution
Refactor into a single `AutonomousPipeline` class that handles data loading, training loops, and validation.

# Discussion/Comments

- 2026-02-07 Pickle Rick: Refactored `train_all.py` to use `AutonomousMLPipeline`. Deleted legacy logic. The pipeline now supports drift detection, database persistence, and optuna optimization.
