---
id: math01
title: Consolidate JIT Math Kernels
status: Triage
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [math, numba, refactor]
assignee: Morty
---

# Description

## Problem to solve
Math logic is scattered and partially duplicated between `black_scholes.py` and `math_utils.py`.

## Solution
Move all Black-Scholes and Greeks logic into `src/shared/math_utils.py` using `@njit` and `@vectorize`.
Ensure `math.sqrt` is used for scalars and `np.sqrt` for arrays.
