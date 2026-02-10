---
id: db001
title: "Native PostgreSQL Refactor"
status: Done
priority: High
project: project
created: 2026-02-07
updated: 2026-02-10
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [db, native, refactor]
assignee: Pickle Rick
---

# Description

## Problem to solve
The project relies on Neon-specific PostgreSQL features and serverless abstractions. This needs to be replaced with native PostgreSQL 16+ features like native partitioning and standard indexing.

## Solution
1. Identify all Neon-specific DDL and code.
2. Replace with native PostgreSQL equivalents.
3. Update `src/shared/db.py` and `src/database/` scripts.
4. Ensure no regression in database performance.
