---
id: opt001
title: "Global Codebase Optimization (Slop Purge)"
status: Done
priority: Medium
project: bsopt
created: 2026-02-03
updated: 2026-02-03
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [optimization, refactor]
assignee: Pickle Rick
---

# Description
## Problem to solve
Presence of iterrows() and manual loops in core services.

## Solution
Audit src/pricing and src/ml to vectorize loops and standardize comments.
