---
id: dep_clean_01
title: "Prune unused dependencies from requirements.txt"
status: Done
priority: Medium
project: bsopt
created: 2026-02-03
updated: 2026-02-03
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [cleanup, dependencies]
assignee: Pickle Rick
---

# Description
## Problem to solve
requirements.txt contains packages like transformers, authlib, etc., that might be unused.

## Solution
Audit dependencies and remove those not referenced in active code paths.

# Discussion/Comments
- [2026-02-03] Pickle Rick: Removed transformers, faust-streaming, authlib, and python-jose from requirements.txt.