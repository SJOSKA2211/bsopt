---
id: 28bc6f22
title: Optimize Heston Pricing
status: Done
priority: Medium
project: bsopt
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [pricing, optimization]
assignee: Pickle Rick
---

# Description

## Problem to solve
Heston model pricing needs to be faster for high-frequency scenarios.

## Solution
Implement vectorized or JIT-compiled (Numba) versions of the Heston FFT pricing model in `src/pricing/models/`.

# Discussion/Comments

- 2026-02-07 Pickle Rick: Implemented `_heston_integrand_jit` and `batch_heston_price_jit` using Numba. Added FFT surface pricing O(N log N).
