---
id: a001
title: Fix Dependencies & Python 3.13 Support
status: Triage
priority: Urgent
project: bsopt
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [infra, dependencies]
assignee: Joseph Kamau Maina
---

# Description
## Problem
`requirements.txt` is missing `torch`. Numba/Ray might fail on Python 3.13.

## Solution
1. Add `torch`, `torchvision` to requirements.
2. Verify install on Python 3.13.
3. Pin versions if necessary.
