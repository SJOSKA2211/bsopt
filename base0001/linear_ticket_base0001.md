---
id: base0001
title: Baseline & Debug
status: Done
priority: Urgent
project: bsopt
created: 2026-02-10
updated: 2026-02-10
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [bug, maintenance, baseline]
assignee: Joseph Kamau Maina
---

# Description

## Problem to solve
We don't know the current state of the codebase. We need to run the full test suite with coverage, capture the errors, and fix them.

## Solution
1.  Run `pytest --cov=. --cov-report=term-missing`.
2.  Identify runtime errors.
3.  Fix errors.
4.  Record initial coverage %.

# Tasks
- [ ] Run test suite.
- [ ] Fix runtime errors.
- [ ] Report baseline coverage.
