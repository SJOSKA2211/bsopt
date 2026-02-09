---
id: a001
title: Refactor DataPipeline to be Fully Async
status: Triage
priority: Urgent
project: bsopt
created: 2026-02-09
updated: 2026-02-09
links:
  - url: ../linear_ticket_parent.md
    title: Parent Ticket
labels: [backend, async, optimization]
assignee: Pickle Rick
---

# Description

## Problem to solve
`src/data/pipeline.py` and `src/database/pipeliner.py` likely use synchronous wrappers for async DB calls, causing I/O blocking and slowing down training.

## Solution
Refactor these modules to use native `asyncio` patterns throughout. Ensure `load_latest_data` and underlying DB fetches are non-blocking.
