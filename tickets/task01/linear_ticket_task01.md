---
id: task01
title: Optimize Celery Pricing Tasks
status: Backlog
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-06
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [celery, performance, refactor]
assignee: Morty
---

# Description

## Problem to solve
`pricing_tasks.py` creates new event loops and uses `nest_asyncio` in a tight loop. This is high-latency slop.

## Solution
1. Use a single long-lived event loop or refactor the cache lookup to be synchronous if possible (or use `asgiref.sync.async_to_sync`).
2. Use the new math kernels from `math01`.
3. Optimize the batch task to avoid manual dict construction where possible (use `msgspec` better).
