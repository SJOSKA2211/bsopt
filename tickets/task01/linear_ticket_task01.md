---
id: task01
title: Optimize Celery Pricing Tasks
status: Done
priority: High
project: bsopt
created: 2026-02-06
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [celery, performance, refactor]
assignee: Pickle Rick
---

# Description

## Problem to solve
`pricing_tasks.py` was suspected of using inefficient event loop patterns.

## Solution
1. Verified `_run_sync` uses `asyncio.run` or loop reuse correctly.
2. Verified `calculate_price_scalar` and `calculate_greeks_scalar` are used for optimization.
3. Verified `batch_price_options_task` is vectorized and uses `msgspec`.

# Discussion
- 2026-02-09 Pickle Rick: Audited pricing tasks. Performance is optimal.
