---
id: math01
title: Math API Unification
status: Done
priority: High
project: bsopt
created: 2026-02-08
updated: 2026-02-08
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [math, refactor, anti-slop]
assignee: Joseph Kamau Maina
---

# Description

## Problem to solve
`src/shared/math_utils.py` contains redundant scalar and vectorized implementations.

## Solution
Unify `calculate_price_scalar`/`calculate_price` and `calculate_greeks_scalar`/`calculate_greeks`. Use NumPy's ability to handle both scalars and arrays seamlessly. Delete the redundant "scalar" functions.

# Discussion
- 2026-02-08 Joseph Kamau Maina: Consolidated redundant math functions. The main functions now use NumPy broadcasting to handle both scalar and array inputs efficiently. Retained aliases for compatibility but purged the duplicate logic.
