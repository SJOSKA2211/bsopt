---
id: ml002
title: "Unified ML Pipeline Consolidation"
status: Triage
priority: High
project: project
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, pipeline, refactor]
assignee: Pickle Rick
---

# Description

## Problem to solve
ML training, validation, and evaluation are spread across multiple scripts and files. This makes temporal validation inconsistent and model management difficult.

## Solution
1. Refactor `src/ml/pipeline.py` to be the single entry point.
2. Integrate `ModelTrainer` with Optuna/Ray for unified HPO.
3. Ensure all models are registered in a central MLflow registry.
4. Implement rigorous temporal cross-validation.
