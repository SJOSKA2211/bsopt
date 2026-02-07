---
id: 347d6575
title: Refactor Neural Strategy
status: Done
priority: Medium
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
The `NeuralStrategy` code is likely outdated or inefficient.

## Solution
Refactor `src/pricing/models/neural_strategy.py` to use cleaner architecture and better state management.

# Discussion/Comments

- 2026-02-07 Pickle Rick: Refactored `NeuralPricingStrategy` to use `onnxruntime` for high-performance inference. Added support for TensorRT/CUDA providers and fallback to Black-Scholes.
