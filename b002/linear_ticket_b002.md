---
id: b002
title: Delete Slop (Legacy Tests)
status: Triage
priority: High
project: bsopt
created: 2026-02-07
updated: 2026-02-07
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [cleanup]
assignee: Joseph Kamau Maina
---

# Description
## Problem
`src/auth-service/testsprite_tests/` exists. It is an abomination.

## Solution
1. `rm -rf src/auth-service/testsprite_tests/`.
2. Ensure no imports rely on it.
