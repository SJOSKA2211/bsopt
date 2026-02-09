---
id: e005
title: Verify and Fix Python 3.13 Compatibility
status: Triage
priority: High
project: bsopt
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [maintenance, python, venv]
assignee: Pickle Rick
---

# Description

## Problem to solve
The user is running Python 3.13. We must ensure all dependencies and code paths are compatible, especially Numba and ML libraries which can be finicky.

## Solution
Audit `pyproject.toml` / `requirements.txt`. Update dependencies. Run tests in the 3.13 venv to verify stability.
