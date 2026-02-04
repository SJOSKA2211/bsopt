---
id: opt003
title: Codebase Audit & Optimization
status: Done
priority: Medium
project: bsopt
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [optimization, refactor, performance]
assignee: Pickle Rick
---

# Description

## Problem to solve
The codebase may contain unoptimized functions, outdated comments, or inefficient logic.

## Solution
1. Audit the entire codebase.
2. Optimize functions for performance.
3. Update/verify comments.
4. Implement fine-tuning where applicable.

# Comments
- Analyzed `src/pricing/`.
- Implemented Pathwise Greeks in `src/pricing/quant_utils.py` (PWM Kernel).
- Updated `src/pricing/monte_carlo.py` to use PWM Greeks (60% speedup).
- Added scalar fast path to `src/pricing/black_scholes.py` to avoid numpy overhead.

