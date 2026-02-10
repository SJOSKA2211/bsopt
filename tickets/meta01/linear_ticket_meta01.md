---
id: meta01
title: Final System-wide Metadata Wipe
status: Backlog
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [meta, cleanup]
assignee: The User
---

# Description

## Problem to solve
Stray "Optimized" strings remain in routes, config, and research docs.

## Solution
1. Grep and replace all remaining "OPTIMIZED" and "ADVANCED" occurrences in `src/`.
2. Ensure the "Refactored" headers are gone.
