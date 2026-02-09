---
id: c003
title: Optimize Postgres Queries in Pipeliner
status: Triage
priority: Urgent
project: bsopt
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [database, performance, postgres]
assignee: Pickle Rick
---

# Description

## Problem to solve
`src/database/pipeliner.py` fetches training data synchronously, potentially blocking or performing unoptimized queries on large datasets.

## Solution
Analyze current queries (e.g., `SELECT * FROM market_data`) and optimize with indexes, partitioning, or window functions where appropriate. Use native Postgres features (CTE, lateral joins) if needed. Ensure async execution.
