---
id: quant01
title: Clean Up Quantum Pricing Mocks
status: Backlog
priority: Low
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [quantum, cleanup]
assignee: Morty
---

# Description

## Problem to solve
`quantum_pricing.py` is full of misleading "Speedup" calculations and mocks.

## Solution
1. Remove "SINGULARITY" and "SOTA".
2. Label the math fallback clearly.
3. Clean up the `MockClass` mess.
