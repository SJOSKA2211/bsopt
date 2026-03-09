---
id: mt003
title: "Math Kernel Optimization"
status: Triage
priority: Medium
project: project
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [math, pricing, optimization]
assignee: High-Performance Engine
---

# Description

## Problem to solve
Pricing kernels are not fully utilizing hardware-aware JIT or vectorized features of modern NumPy.

## Solution
1. Profile `src/pricing/` models.
2. Apply `@njit` and `@cuda.jit` where appropriate.
3. Migrate to NumPy 2.0 vectorized operations.
4. Verify results against analytical benchmarks.
