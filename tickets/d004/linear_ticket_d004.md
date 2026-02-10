---
id: d004
title: Enhance Feature Engineering in DataPipeline
status: Triage
priority: High
project: bsopt
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [data, ml, features]
assignee: Joseph Kamau Maina
---

# Description

## Problem to solve
Feature engineering in `src/data/pipeline.py` is "very basic" (strike, maturity, implied vol). We need richer features for better model performance.

## Solution
Implement rolling statistics (mean, std), volatility surfaces, and lag features. Compute these efficiently (vectorized) within the pipeline or via optimized SQL.
