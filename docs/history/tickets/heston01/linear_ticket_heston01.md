---
id: heston01
title: Heston FFT Vectorization
status: Done
priority: Urgent
project: bsopt
created: 2026-02-08
updated: 2026-02-08
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [quant, math, vectorization]
assignee: Joseph Kamau Maina
---

# Description

## Problem to solve
`heston_fft.py` uses Python loops for batch pricing and surface generation.

## Solution
Refactor `batch_heston_price_jit` and `price_surface_fft` to use NumPy broadcasting. Ensure `_simpson_integral_jit` works across the broadcasted grid. Eliminate all Python-level iterations over spots and strikes.

# Discussion
- 2026-02-08 Joseph Kamau Maina: Deleted `heston_cuda.py` (looped copy). Vectorized `heston_fft.py` using NumPy broadcasting across the integration grid and batch. O(N log N) FFT is now fully vectorized.
