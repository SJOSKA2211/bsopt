---
id: c003
title: Coverage & Linting Crusade (The "Make It Pretty" Ticket)
status: Todo
priority: Medium
order: 30
created: 2026-02-20
updated: 2026-02-20
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
---

# Description

## Problem to solve
The codebase is untested and messy. We need >=99% coverage to prove correctness, and 0 linting errors to prove discipline.

## Solution
Write tests for every uncovered line and fix every linter complaint.

## Implementation Details
1.  **Write Tests**: For `src/api/*`, `src/pricing/*`, `src/tasks/*`. Use `pytest-cov` to identify gaps.
2.  **Mock Everything**: Use `unittest.mock` to stub `redis`, `postgres`, and `kafka` if necessary to speed up tests and isolate units.
3.  **Fix Linting**: Run `make lint` and fix errors (e.g., `E501`, `F401`).
4.  **Format**: Run `make format`.
5.  **Achieve >=99% Coverage**: Update `coverage_report.txt` with proof.
