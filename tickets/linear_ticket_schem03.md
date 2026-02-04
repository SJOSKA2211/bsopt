---
id: schem03
title: Fix Market Mesh Schema Mismatches
status: Done
priority: High
project: BS-OPT
created: 2026-02-04
updated: 2026-02-04
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [bug, database, schema]
assignee: Pickle Rick
---

# Description

## Problem to solve
`tests/integration/database/test_market_mesh_schema.py` is failing. Code expects columns that DB doesn't have.

## Solution
Align the database migration/schema with the codebase.
