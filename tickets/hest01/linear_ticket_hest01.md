---
id: hest01
title: Advance Heston Model Algorithms
status: Triage
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [pricing, heston, optimization]
assignee: Morty
---

# Description

## Problem to solve
The current Heston implementation is a standard Carr-Madan approach with non-vectorized characteristic function calls in the FFT path.

## Solution
1. Vectorize the characteristic function in `heston_fft.py`.
2. Implement the **Quadratic Exponential (QE)** scheme for Heston path simulation (more accurate than Euler-Maruyama).
3. Optimize the Simpson integration bound and step size.
