---
id: cov0002
title: Coverage Injection
status: In Dev
priority: High
project: bsopt
created: 2026-02-10
updated: 2026-02-10
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [testing, coverage]
assignee: Pickle Rick
---

# Description

## Problem to solve
Coverage is at 11.14%. `aiops/` is at 0%. We need to inject tests.

## Solution
Targeting `aiops/drift_detector.py` first.

# Tasks
- [ ] Write unit tests for `aiops/drift_detector.py`.
- [ ] Verify coverage increase for `aiops/`.
