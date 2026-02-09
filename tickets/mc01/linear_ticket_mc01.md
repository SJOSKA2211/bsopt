---
id: mc01
title: Upgrade Monte Carlo Engine
status: Done
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [pricing, monte_carlo, innovation]
assignee: Pickle Rick
---

# Description

## Problem to solve
Standard MC is too slow and high-variance.

## Solution
1. Fully integrate **Sobol sequences** with Owen scrambling for all MC paths.
2. Implement the **Milstein scheme** for path generation to improve convergence order.
3. Optimize pathwise sensitivity (Greeks) calculations.

# Discussion
- 2026-02-09 Pickle Rick: Fully upgraded the MC engine. Integrated Sobol sequences with Owen scrambling (via SciPy QMC). Implemented the Milstein scheme in `jit_generate_milstein_paths` for improved strong convergence. Optimized Greeks by implementing pathwise Gamma using the Likelihood Ratio Method (LRM) in the European pricing kernel. Verified with advanced convergence tests.
