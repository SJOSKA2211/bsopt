---
id: mc01
title: Upgrade Monte Carlo Engine
status: Backlog
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [pricing, monte_carlo, innovation]
assignee: Morty
---

# Description

## Problem to solve
Standard MC is too slow and high-variance.

## Solution
1. Fully integrate **Sobol sequences** with Owen scrambling for all MC paths.
2. Implement the **Milstein scheme** for path generation to improve convergence order.
3. Optimize pathwise sensitivity (Greeks) calculations.
