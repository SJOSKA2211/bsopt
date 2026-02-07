---
id: neural01
title: Implement Skeleton for Neural Pricing Engine
status: Backlog
priority: Medium
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [ml, neural, innovation]
assignee: Morty
---

# Description

## Problem to solve
XGBoost is limited for complex volatility surfaces. We need a Neural Network approach (e.g., Deep Hedging or Neural SDEs skeleton).

## Solution
1. Create `src/ml/models/neural_engine.py`.
2. Implement a PyTorch-based skeleton for option pricing.
3. Integrate with the `PricingEngineFactory`.
